"""2D Selective Scan (SS2D) operator used by WheatScopeNet.

Implements the state-space core described in paper Section 2.3.2, Eq. (1)-(5),
and illustrated in Fig. 4(c) (SS2D_OP Module).  The operator is a direct port of
the VMamba SS2D block (Y. Liu et al., 2024): the input is linearly projected and
split into a local branch and a gating branch (Eq. 1), the local branch is
enhanced by a depthwise convolution followed by SiLU (Eq. 2), scanned along four
directions by a selective state-space model (Eq. 3), aggregated and modulated by
the gating branch (Eq. 4), and finally projected back to the input channel
dimension (Eq. 5).

The selective scan itself is dispatched at import time: if the fused CUDA kernel
from ``mamba_ssm`` is installed it is used, otherwise the pure-PyTorch
``selective_scan_reference`` defined in this module is used automatically, so the
model stays runnable on CPU and for FLOPs profiling.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

__all__ = [
    "SELECTIVE_SCAN_CUDA_AVAILABLE",
    "selective_scan_reference",
    "SS2D",
    "SS2D_OP",
]


def selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation of the selective scan (paper Eq. 3).

    Numerically equivalent to ``mamba_ssm.ops.selective_scan_interface.
    selective_scan_ref`` for the real-valued, variable-B/C configuration used by
    :class:`SS2D`.  It is the fallback executed whenever the fused CUDA kernel is
    unavailable (CPU inference, FLOPs profiling, CUDA builds without
    ``mamba-ssm``).

    Args:
        u: Input sequence, shape ``(B, D, L)``.
        delta: Per-position step size, shape ``(B, D, L)``.
        A: State transition matrix, shape ``(D, N)``.
        B: Input projection, shape ``(B, G, N, L)`` with ``G`` groups (a
            ``(B, N, L)`` tensor, i.e. a single shared group, is also accepted).
            Grouped tensors are broadcast across the ``D`` dimension, each group
            covering ``D // G`` consecutive channels.
        C: Output projection, same shape convention as ``B``.
        D: Optional residual ("skip") weight, shape ``(D,)``.
        z: Optional gating branch, shape ``(B, D, L)``; when given the output is
            multiplied by ``silu(z)``.
        delta_bias: Optional bias added to ``delta`` BEFORE the softplus,
            shape ``(D,)``.
        delta_softplus: Whether to apply ``softplus`` to the (biased) step size.
        return_last_state: Accepted for call-signature compatibility with the
            fused kernel and ignored (this port never consumes the last state).

    Returns:
        The scanned output ``y`` of shape ``(B, D, L)`` in ``float32``.

    Note:
        The reference materialises ``(B, D, L, N)`` intermediates and is
        therefore memory hungry for long sequences; install ``mamba-ssm`` for
        full-resolution training.
    """
    del return_last_state  # only present for signature compatibility

    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()

    if delta_bias is not None:
        delta = delta + delta_bias.float()[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, seq_len = u.shape
    d_state = A.shape[1]

    # Broadcast grouped B/C across the channel dimension: group g covers the
    # channels [g * (D // G), (g + 1) * (D // G)).
    if B.dim() == 4:
        B = repeat(B, "b g n l -> b (g h) n l", h=dim // B.shape[1])
        b_seq = B.permute(0, 1, 3, 2)  # (batch, dim, L, N)
    else:
        b_seq = B.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, L, N) -> broadcast
    if C.dim() == 4:
        C = repeat(C, "b g n l -> b (g h) n l", h=dim // C.shape[1])
        c_seq = C.permute(0, 1, 3, 2)  # (batch, dim, L, N)
    else:
        c_seq = C.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, L, N) -> broadcast

    # Discretisation of the continuous-time system (paper Eq. 3).
    delta_a = torch.exp(delta.unsqueeze(-1) * A.view(1, dim, 1, d_state))
    delta_b_u = delta.unsqueeze(-1) * b_seq * u.unsqueeze(-1)

    state = u.new_zeros((batch, dim, d_state))
    ys = []
    for i in range(seq_len):
        state = delta_a[:, :, i] * state + delta_b_u[:, :, i]
        ys.append((state * c_seq[:, :, i]).sum(dim=-1))  # 'bdn,bdn->bd'
    y = torch.stack(ys, dim=2)  # (batch, dim, L)

    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(torch.float32)


try:  # pragma: no cover - depends on the optional CUDA extension
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    SELECTIVE_SCAN_CUDA_AVAILABLE: bool = True
except Exception as _import_error:  # pragma: no cover - CPU-only installations
    # A broken CUDA extension raises OSError/RuntimeError rather than
    # ImportError, and every such failure must degrade to the reference path.
    selective_scan_fn = selective_scan_reference
    SELECTIVE_SCAN_CUDA_AVAILABLE = False
    warnings.warn(
        "The fused mamba_ssm selective-scan kernel is unavailable "
        f"({type(_import_error).__name__}: {_import_error}); SS2D falls back to "
        "the pure-PyTorch selective_scan_reference. Results are equivalent but "
        "training and inference are considerably slower and use more memory. "
        "Install `mamba-ssm` and `causal-conv1d` for the fused CUDA kernel.",
        RuntimeWarning,
        stacklevel=2,
    )
    del _import_error


class SS2D(nn.Module):
    """2D selective scan block (paper Eq. 1-5, Fig. 4(c)).

    Ported from VMamba (Y. Liu et al., 2024) with the parameter registration
    order preserved (``in_proj``, ``conv2d``, ``x_proj_weight``,
    ``dt_projs_weight``, ``dt_projs_bias``, ``A_logs``, ``Ds``, ``out_norm``,
    ``out_proj``) so that the parameter count matches the paper exactly.

    Forward signature: ``(B, H, W, C) -> (B, H, W, C)``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Eq. (1): input projection -> local branch + gating branch.
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        # Eq. (2): depthwise convolution for local enhancement.
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # Eq. (3): per-direction (K=4) input-dependent B, C and step size.
        x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in x_proj], dim=0)
        )  # (K=4, N, inner)
        del x_proj

        dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_projs], dim=0)
        )  # (K=4, inner)
        del dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K*D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K*D)

        # Eq. (4) / Eq. (5): normalisation, gating and output projection.
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> nn.Linear:
        """Build and initialise one step-size projection (VMamba, verbatim)."""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(
        d_state: int,
        d_inner: int,
        copies: int = 1,
        device: Optional[torch.device] = None,
        merge: bool = True,
    ) -> nn.Parameter:
        """S4D-real initialisation of the state transition matrix (verbatim)."""
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(
        d_inner: int,
        copies: int = 1,
        device: Optional[torch.device] = None,
        merge: bool = True,
    ) -> nn.Parameter:
        """Initialise the residual "skip" parameter D (verbatim)."""
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def _selective_scan_2d(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Four-directional selective scan of a BCHW feature map (paper Eq. 3).

        The map is unrolled row-major and column-major, each of the two
        orderings additionally reversed, giving K=4 independent scan directions
        that share the same parameterisation.
        """
        B, _, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SS2D operator on a ``(B, H, W, C)`` tensor (paper Eq. 1-5)."""
        B, H, W, _ = x.shape

        # Eq. (1): linear projection -> local feature x and gating feature z.
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        # Eq. (2): depthwise convolution + SiLU local enhancement.
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)

        # Eq. (3): four-directional state-space scan.
        y1, y2, y3, y4 = self._selective_scan_2d(x)
        assert y1.dtype == torch.float32

        # Eq. (4): directional aggregation, normalisation and gating.
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)

        # Eq. (5): projection back to the input channel dimension.
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2D_OP(nn.Module):
    """SS2D_OP Module (paper Fig. 4(c)): channels-first wrapper around SS2D.

    The PH_SS2D branches operate on ``(B, C, H, W)`` tensors while SS2D expects
    ``(B, H, W, C)``; this module performs the two permutations and holds the
    SS2D weights.
    """

    def __init__(self, dim: int, d_state: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.ss2d = SS2D(d_model=dim, d_state=d_state, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``(B, C, H, W)`` tensor to ``(B, C, H, W)``."""
        assert x.shape[1] == self.dim, (
            f"SS2D_OP expects {self.dim} input channels, got {x.shape[1]}"
        )
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW -> BHWC
        x = self.ss2d(x)
        return x.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW

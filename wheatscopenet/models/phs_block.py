"""Parallel Hybrid Spatial (PHS) building blocks of WheatScopeNet.

Implements paper Section 2.3.2 / Figure 4:

* ``SimpleFusionGate`` -- the channel-wise, path-wise gate that mixes the
  parallel branches of the PH_SS2D module.
* ``PH_SS2D``          -- Fig. 4(b): multi-scale depthwise convolutions in
  parallel with the SS2D_OP state-space branch, fused by the gate.
* ``PHSBlock``         -- Fig. 4(a): the dual-branch residual block
  (PH_SS2D branch + MLP branch) used throughout the encoder, bottleneck and
  decoder of Fig. 3.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from timm.models.layers import DropPath
from torch import Tensor, nn

from .layers import LayerNorm, depthwise_conv, num_groups_for
from .ss2d import SS2D_OP

__all__ = ["SimpleFusionGate", "PH_SS2D", "PHSBlock"]


class SimpleFusionGate(nn.Module):
    """Channel-wise softmax gate over ``num_paths`` parallel branches.

    Squeeze-and-excitation style: global average pooling followed by a
    bottleneck 1x1-conv MLP that predicts ``num_paths * channels`` logits,
    normalised with a softmax over the path axis (paper Fig. 4(b)).
    """

    def __init__(self, channels: int, num_paths: int, reduction: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.num_paths = num_paths
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, num_paths * channels, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: [B, C, H, W] -> gate weights [num_paths, B, C, 1, 1]."""
        b, c = x.shape[0], x.shape[1]
        if c != self.channels:
            raise ValueError(
                f"SimpleFusionGate expects {self.channels} input channels, got {c}"
            )
        w = self.pool(x)                                        # [B, C, 1, 1]
        w = self.fc(w)                                          # [B, P*C, 1, 1]
        w = w.view(b, self.num_paths, self.channels, 1, 1)      # [B, P, C, 1, 1]
        w = self.softmax(w)                                     # softmax over paths
        return w.permute(1, 0, 2, 3, 4).contiguous()            # [P, B, C, 1, 1]


class PH_SS2D(nn.Module):
    """PH_SS2D module -- paper Fig. 4(b).

    Three parallel branches model the wheat canopy at complementary scopes:
    two local multi-scale depthwise convolution branches (3x3 and 5x5, each
    followed by Group Normalization and GELU) and one global SS2D_OP
    state-space branch (Fig. 4(c)). A :class:`SimpleFusionGate` predicts a
    per-channel weight for every branch, the weighted branch outputs are
    summed element-wise, and the result is projected by a 1x1 convolution.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: Sequence[int] = (3, 5),
        d_state: int = 16,
        dropout: float = 0.0,
        gate_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        num_gn_groups = num_groups_for(dim, 16)

        # 1. Local multi-scale depthwise convolution branches.
        self.conv_branches = nn.ModuleList(
            nn.Sequential(
                depthwise_conv(dim, kernel_size, bias=False),
                nn.GroupNorm(num_groups=num_gn_groups, num_channels=dim),
                nn.GELU(),
            )
            for kernel_size in self.kernel_sizes
        )

        # 2. Global state-space branch (SS2D_OP, paper Fig. 4(c)).
        self.ss2d_op = SS2D_OP(dim, d_state=d_state, dropout=dropout)
        self.ss2d_norm = nn.GroupNorm(num_groups=num_gn_groups, num_channels=dim)

        # 3. Gated fusion over the local and global paths.
        self.num_paths = len(self.conv_branches) + 1
        self.gate = SimpleFusionGate(dim, self.num_paths, reduction=gate_reduction)

        # 4. Output projection.
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: [B, C, H, W] -> [B, C, H, W]."""
        outputs: list[Tensor] = [branch(x) for branch in self.conv_branches]
        outputs.append(self.ss2d_norm(self.ss2d_op(x)))

        weights = self.gate(x)                          # [P, B, C, 1, 1]
        fused = outputs[0] * weights[0]
        for i in range(1, len(outputs)):
            fused = fused + outputs[i] * weights[i]

        return self.proj(fused)


class PHSBlock(nn.Module):
    """PHS Block -- paper Fig. 4(a).

    Dual-branch residual block: a normalised PH_SS2D branch followed by a
    channels-last MLP branch, both with LayerScale and stochastic depth.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: int = 2,
        ph_ss2d_cls: Callable[..., nn.Module] = PH_SS2D,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.ph_ss2d = ph_ss2d_cls(dim=dim)
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)

        self.gamma1 = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.gamma2 = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """``x``: [B, C, H, W] -> [B, C, H, W]."""
        # Branch 1: PH_SS2D (channels-first).
        residual = x
        hidden = self.ph_ss2d(self.norm1(x))
        if self.gamma1 is not None:
            hidden = self.gamma1.view(1, -1, 1, 1) * hidden
        x = residual + self.drop_path(hidden)

        # Branch 2: MLP (channels-last).
        residual = x
        hidden = x.permute(0, 2, 3, 1)                  # BCHW -> BHWC
        hidden = self.fc2(self.act(self.fc1(self.norm2(hidden))))
        if self.gamma2 is not None:
            hidden = self.gamma2 * hidden
        hidden = hidden.permute(0, 3, 1, 2)             # BHWC -> BCHW
        return residual + self.drop_path(hidden)

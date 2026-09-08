"""Shared primitive layers for WheatScopeNet.

This module implements the small building blocks that are reused across the
network described in the paper (Section 2.3, Fig. 3-6):

* :func:`num_groups_for` -- GroupNorm group selection helper used by every
  normalisation layer in the encoder/decoder and in the fusion modules.
* :func:`depthwise_conv` -- depthwise convolution factory used by the
  multi-scale DWConv branches of the PHS Block (Fig. 4(b)) and by the
  LightCSF fusion trunk (Fig. 6).
* :class:`LayerNorm` -- LayerNorm supporting both ``channels_last`` and
  ``channels_first`` tensor layouts, required because the PHS Block normalises
  a BCHW tensor before the spatial branch and a BHWC tensor before the MLP.
* :class:`SEBlock` -- Squeeze-and-Excitation channel recalibration used by the
  SCAB bridge (Fig. 5), by LightCSF (Fig. 6) and on the skip connections.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["num_groups_for", "depthwise_conv", "LayerNorm", "SEBlock"]


def num_groups_for(channels: int, max_groups: int = 16) -> int:
    """Return the largest divisor of ``channels`` that is <= ``max_groups``.

    GroupNorm requires ``num_channels % num_groups == 0``. The network uses
    channel widths that are not always divisible by a fixed group count, so the
    group count is derived per layer.

    Args:
        channels: Number of channels the GroupNorm layer will normalise.
        max_groups: Upper bound on the number of groups. WheatScopeNet always
            uses 16 (paper Section 2.3).

    Returns:
        The largest ``g`` with ``1 <= g <= min(channels, max_groups)`` such that
        ``channels % g == 0``. Returns ``1`` for non-positive ``channels``.
    """
    if channels <= 0:
        return 1
    limit = min(channels, max_groups)
    for groups in range(limit, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def depthwise_conv(dim: int, kernel_size: int, bias: bool = False) -> nn.Conv2d:
    """Build a depthwise (``groups=dim``) convolution with 'same' padding.

    Args:
        dim: Number of input and output channels.
        kernel_size: Spatial kernel size (odd values keep the resolution).
        bias: Whether to add a learnable bias term.

    Returns:
        A ``nn.Conv2d`` with ``groups=dim`` and padding ``(kernel_size - 1) // 2``.
    """
    return nn.Conv2d(
        dim,
        dim,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2,
        groups=dim,
        bias=bias,
    )


class LayerNorm(nn.Module):
    """LayerNorm that accepts ``channels_last`` (BHWC) or ``channels_first`` (BCHW).

    ``channels_last`` delegates to :func:`torch.nn.functional.layer_norm`.
    ``channels_first`` normalises over the channel axis explicitly so that the
    module can be applied directly to a BCHW feature map, which is what the PHS
    Block needs before its spatial branch (paper Fig. 4(a)).
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(
                f"Unsupported data_format: {data_format!r}. "
                "Expected 'channels_last' or 'channels_first'."
            )
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)

        self.normalized_shape = normalized_shape
        self.data_format = data_format
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            # Input: (N, ..., C)
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        # Input: (N, C, ...)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        view_shape = (1,) + self.normalized_shape + (1,) * (x.ndim - 2)
        return self.weight.view(view_shape) * x + self.bias.view(view_shape)

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"data_format={self.data_format}"
        )


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (1x1-convolution MLP form).

    Global average pooling squeezes each channel to a scalar descriptor, a
    bottleneck 1x1-conv MLP produces per-channel gates in ``(0, 1)``, and the
    input is rescaled channel-wise. Used on the SCAB outputs, inside the
    LightCSF fusion trunk and on the decoder skip connections.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Recalibrate a BCHW feature map; the output keeps the input shape."""
        weights = self.fc(self.pool(x))  # [B, C, 1, 1]
        return x * weights

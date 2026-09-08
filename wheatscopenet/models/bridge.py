"""Cross-scale bridge modules of WheatScopeNet (paper Section 2.3.3).

This module implements the two components that sit between the encoder and the
decoder of WheatScopeNet (paper Fig. 3(c)):

* :class:`SCAB` -- Spatial-Channel Attention Bridge (paper Fig. 5, Eq. 6-9),
  built from :class:`SpatialAttentionBridge` and :class:`ChannelAttentionBridge`.
* :class:`LightCSF` -- Light Cross-Scale Fusion (paper Fig. 6, Eq. 10-11).

Both modules consume and produce the ordered list of the five encoder skip
features ``[e1, e2, e3, e4, e5]`` with channels ``c_list[:5]`` and spatial
resolutions ``H, H/2, H/4, H/8, H/16``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import SEBlock, num_groups_for

__all__ = [
    "SpatialAttentionBridge",
    "ChannelAttentionBridge",
    "SCAB",
    "LightCSF",
]

_NUM_SCALES = 5


class SpatialAttentionBridge(nn.Module):
    """Shared spatial attention over the five skip features (paper Eq. 6).

    For every scale the channel-wise average and max responses are stacked into
    a two-channel descriptor and mapped by a single *shared* dilated convolution
    followed by a Sigmoid, producing one attention map ``[B, 1, H, W]`` per
    scale.  The 7x7 kernel with ``dilation=3`` and ``padding=9`` has an
    effective receptive field of 19x19 while keeping the spatial size unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        # Eq. (6): M_i = sigma(Conv([AvgPool_c(F_i); MaxPool_c(F_i)]))
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=9, dilation=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        assert len(features) == _NUM_SCALES, (
            f"SpatialAttentionBridge expects {_NUM_SCALES} features, got {len(features)}"
        )
        maps: list[Tensor] = []
        for feature in features:
            avg_out = feature.mean(dim=1, keepdim=True)
            max_out = feature.max(dim=1, keepdim=True)[0]
            descriptor = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
            maps.append(self.conv(descriptor))  # [B, 1, H, W]
        return maps


class ChannelAttentionBridge(nn.Module):
    """Joint channel attention over the five skip features (paper Eq. 8).

    The globally pooled descriptors of all five scales are concatenated into a
    single ``c_sum``-dimensional vector, which is mapped by one lightweight
    layer per scale and squashed by a Sigmoid, so that every scale is
    recalibrated with information coming from all the other scales.

    ``shared_conv`` (legacy ``get_all_att``) is constructed unconditionally.  It
    is only *used* by the ``split_att='conv'`` branch -- the paper uses
    ``split_att='fc'``; the ``'conv'`` variant is retained as a configuration
    option.  Its 744 parameters (a depthwise ``Conv1d`` over ``c_sum = 248``
    channels with kernel size 3) are nevertheless part of the 4.321 M parameter
    count reported in the paper, so it must not be removed or made conditional.
    """

    def __init__(self, c_list: Sequence[int], split_att: str = "fc") -> None:
        super().__init__()
        assert len(c_list) == _NUM_SCALES, (
            f"ChannelAttentionBridge expects {_NUM_SCALES} skip channels, got {len(c_list)}"
        )
        if split_att not in ("fc", "conv"):
            raise ValueError(f"split_att must be 'fc' or 'conv', got {split_att!r}")

        self.c_list = tuple(int(c) for c in c_list)
        self.split_att = split_att
        c_sum = sum(self.c_list)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Legacy `get_all_att`: depthwise Conv1d mixing neighbouring channel
        # descriptors. Used by the 'conv' branch; always built (see docstring).
        self.shared_conv = nn.Conv1d(c_sum, c_sum, 3, 1, 1, groups=c_sum, bias=False)
        self.att_layers = nn.ModuleList(
            [
                nn.Linear(c_sum, c) if split_att == "fc" else nn.Conv1d(c_sum, c, 1)
                for c in self.c_list
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        assert len(features) == _NUM_SCALES, (
            f"ChannelAttentionBridge expects {_NUM_SCALES} features, got {len(features)}"
        )
        # Eq. (8): global average pooling -> per-scale MLP -> Sigmoid.
        pooled = torch.cat([self.avgpool(f) for f in features], dim=1)  # [B, c_sum, 1, 1]
        if self.split_att == "fc":
            att_base = pooled.view(pooled.size(0), -1)  # [B, c_sum]
        else:
            att_base = self.shared_conv(pooled.squeeze(-1))  # [B, c_sum, 1]

        attentions: list[Tensor] = []
        for layer in self.att_layers:
            att = self.sigmoid(layer(att_base))
            if self.split_att == "fc":
                att = att.unsqueeze(-1).unsqueeze(-1)  # [B, c_i, 1, 1]
            else:
                att = att.unsqueeze(-1)  # [B, c_i, 1, 1]
            attentions.append(att)
        return attentions


class SCAB(nn.Module):
    """Spatial-Channel Attention Bridge (paper Fig. 5, Eq. 6-9).

    Spatial attention and channel attention are applied **sequentially**, each
    one followed by a residual addition of the *original* skip feature, and the
    result of every scale is finally recalibrated by its own SE Block with a
    reduction ratio of 4.  The order below is load-bearing for both the
    numerical behaviour and the reported parameter count -- do not reorder it.
    """

    def __init__(self, c_list: Sequence[int], split_att: str = "fc") -> None:
        super().__init__()
        assert len(c_list) == _NUM_SCALES, (
            f"SCAB expects {_NUM_SCALES} skip channels, got {len(c_list)}"
        )
        self.c_list = tuple(int(c) for c in c_list)
        self.spatial_att = SpatialAttentionBridge()
        self.channel_att = ChannelAttentionBridge(self.c_list, split_att=split_att)
        self.se = nn.ModuleList([SEBlock(c, reduction=4) for c in self.c_list])

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        assert len(features) == _NUM_SCALES, (
            f"SCAB expects {_NUM_SCALES} features, got {len(features)}"
        )
        residual = features

        # Eq. (6)-(7): spatial attention, then residual with the original input.
        spatial = self.spatial_att(features)
        fused_spatial = [spatial[i] * features[i] + residual[i] for i in range(_NUM_SCALES)]

        # Eq. (8)-(9): channel attention on the spatially enhanced features,
        # then residual with the ORIGINAL input again (not with fused_spatial).
        channel = self.channel_att(fused_spatial)
        fused_channel = [
            channel[i] * fused_spatial[i] + residual[i] for i in range(_NUM_SCALES)
        ]

        # Per-scale SE recalibration (reduction ratio 4).
        return [self.se[i](fused_channel[i]) for i in range(_NUM_SCALES)]


class LightCSF(nn.Module):
    """Light Cross-Scale Fusion (paper Fig. 6, Eq. 10-11).

    Four-stage pipeline -- compression, fusion, enhancement, redistribution:

    1. Each scale is projected to ``max(1, c // embed_ratio)`` channels by a
       1x1 convolution (compression).
    2. All projected features are resized to the resolution of scale index 2
       (the ``H/4`` skip) and concatenated (alignment + fusion).
    3. The concatenated tensor is processed by a 3x3 depthwise convolution,
       GroupNorm, GELU, an SE Block (Eq. 10) and a 1x1 pointwise convolution
       (enhancement).
    4. The result is split back along the channel dimension according to the
       per-scale compressed widths, resized to each original resolution,
       expanded by a 1x1 convolution and added to the corresponding original
       skip feature as a residual (Eq. 11, redistribution).
    """

    def __init__(
        self,
        c_list: Sequence[int],
        embed_ratio: int = 2,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        assert len(c_list) == _NUM_SCALES, (
            f"LightCSF expects {_NUM_SCALES} skip channels, got {len(c_list)}"
        )
        assert embed_ratio >= 1, "embed_ratio must be >= 1"

        self.c_list = tuple(int(c) for c in c_list)
        self.reduced = tuple(max(1, c // embed_ratio) for c in self.c_list)
        self.total = sum(self.reduced)
        # Reference scale for the fusion stage (legacy `ri = 2`): the H/4 skip.
        self.ref_index = 2

        # Stage 1 -- compression (1x1 convolutions, no bias).
        self.down = nn.ModuleList(
            [
                nn.Conv2d(self.c_list[i], self.reduced[i], 1, bias=False)
                for i in range(_NUM_SCALES)
            ]
        )
        # Stage 4 -- expansion back to the original channel width.
        self.up = nn.ModuleList(
            [
                nn.Conv2d(self.reduced[i], self.c_list[i], 1, bias=False)
                for i in range(_NUM_SCALES)
            ]
        )

        # Stages 2-3 -- depthwise fusion + SE enhancement + pointwise mixing.
        self.fuse = nn.Sequential(
            nn.Conv2d(self.total, self.total, 3, 1, 1, groups=self.total, bias=False),
            nn.GroupNorm(num_groups_for(self.total, 16), self.total),
            nn.GELU(),
            SEBlock(self.total, reduction=reduction),  # Eq. (10)
            nn.Conv2d(self.total, self.total, 1, bias=False),
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        assert len(features) == _NUM_SCALES, (
            f"LightCSF expects {_NUM_SCALES} features, got {len(features)}"
        )

        original_sizes = [f.shape[-2:] for f in features]

        # Stage 1 -- compress every scale into the shared low-dimensional space.
        projected = [down(f) for f, down in zip(features, self.down)]

        # Stage 2 -- align all scales to the reference resolution and concatenate.
        ref_h, ref_w = projected[self.ref_index].shape[-2:]
        aligned = [
            p
            if p.shape[-2:] == (ref_h, ref_w)
            else F.interpolate(p, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            for p in projected
        ]
        fused = self.fuse(torch.cat(aligned, dim=1))  # Stage 3 -- enhancement.

        # Stage 4 -- redistribute to each scale and residually fuse (Eq. 11).
        parts = torch.split(fused, list(self.reduced), dim=1)
        outputs: list[Tensor] = []
        for part, up, size, feature in zip(parts, self.up, original_sizes, features):
            if part.shape[-2:] != size:
                part = F.interpolate(part, size=size, mode="bilinear", align_corners=False)
            outputs.append(feature + up(part))
        return outputs

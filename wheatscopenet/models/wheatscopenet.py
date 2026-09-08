"""WheatScopeNet: the organ-level segmentation network of the WheatScoper framework.

This module implements the full network of paper Figure 3:

* Fig. 3(a) -- encoder: two plain conv stages followed by three PHS-Block stages.
* Fig. 3(b) -- bottleneck: PHS Blocks at the deepest resolution (H/32).
* Fig. 3(c) -- cross-scale bridge: SCAB (Fig. 5) -> LightCSF (Fig. 6) -> per-skip SE.
* Fig. 3(d) -- decoder: four PHS-Block stages plus a light conv stage, each fused with
  its skip connection through bilinear upsampling.

The reference configuration (paper Section 2.3) is
``c_list=(8, 16, 32, 64, 128, 256)`` with ``depths=(2, 2, 2, 2)``, which yields
4.321 M parameters and 51.93 G FLOPs at 2048x2048 input.
"""

from __future__ import annotations

from functools import partial
from typing import Any, List, Sequence

import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch import Tensor, nn

from .bridge import SCAB, LightCSF
from .layers import SEBlock, num_groups_for
from .phs_block import PH_SS2D, PHSBlock

__all__ = ["WheatScopeNet", "build_wheatscopenet"]


class WheatScopeNet(nn.Module):
    """Lightweight organ-level segmentation network (paper Fig. 3).

    Args:
        num_classes: number of output classes (paper: 3 -- background, spike, leaf).
        input_channels: number of input image channels (RGB -> 3).
        c_list: the six stage widths ``(c0, ..., c5)``.
        depths: number of PHS Blocks in encoder stages 3/4/5 and in the bottleneck.
            The decoder mirrors these depths.
        d_state: SS2D state dimension used inside every PH_SS2D module.
        kernel_sizes: depthwise kernel sizes of the multi-scale branch (paper: 3 and 5).
        mlp_ratio: expansion ratio of the PHS-Block MLP.
        drop_path_rate: maximum stochastic-depth rate; linearly ramped over all blocks.
        layer_scale_init_value: LayerScale initialisation; ``<= 0`` disables LayerScale.
        split_att: channel-attention head of SCAB, ``"fc"`` in the paper.
        csf_embed_ratio: channel-reduction ratio of LightCSF.

    Forward input is ``[B, input_channels, H, W]`` and the output is a raw logit map
    ``[B, num_classes, H, W]`` -- no sigmoid or softmax is applied.
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 3,
        c_list: Sequence[int] = (8, 16, 32, 64, 128, 256),
        depths: Sequence[int] = (2, 2, 2, 2),
        d_state: int = 16,
        kernel_sizes: Sequence[int] = (3, 5),
        mlp_ratio: int = 2,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        split_att: str = "fc",
        csf_embed_ratio: int = 2,
    ) -> None:
        super().__init__()

        c_list = tuple(int(c) for c in c_list)
        depths = tuple(int(d) for d in depths)
        if len(c_list) != 6:
            raise ValueError(f"c_list must contain exactly 6 stage widths, got {len(c_list)}")
        if len(depths) != 4:
            raise ValueError(f"depths must contain exactly 4 stage depths, got {len(depths)}")
        if any(d < 0 for d in depths):
            raise ValueError(f"depths must be non-negative, got {depths}")

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.c_list = c_list
        self.depths = depths
        self.d_state = d_state
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.split_att = split_att
        self.csf_embed_ratio = csf_embed_ratio

        # Stochastic depth: the encoder stages and their decoder mirrors share one
        # linearly increasing budget, hence 2 * sum(depths) blocks in total.
        total_blocks = 2 * sum(depths)
        dpr: List[float] = (
            torch.linspace(0.0, drop_path_rate, total_blocks).tolist() if total_blocks > 0 else []
        )
        dpr_idx = 0

        # Pre-bind the PH_SS2D hyper-parameters so that PHSBlock only has to pass `dim`.
        configured_ph_ss2d = partial(
            PH_SS2D,
            kernel_sizes=self.kernel_sizes,
            d_state=d_state,
        )
        make_block = partial(
            PHSBlock,
            ph_ss2d_cls=configured_ph_ss2d,
            layer_scale_init_value=layer_scale_init_value,
            mlp_ratio=mlp_ratio,
        )

        def group_norm(channels: int) -> nn.GroupNorm:
            return nn.GroupNorm(num_groups_for(channels, 16), channels)

        # ------------------------------------------------------------------
        # Fig. 3(a) -- encoder
        # ------------------------------------------------------------------
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, 1, 1),
            group_norm(c_list[0]),
            nn.GELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, 1, 1),
            group_norm(c_list[1]),
            nn.GELU(),
        )
        self.encoder3 = nn.Sequential(
            *[make_block(dim=c_list[1], drop_path=dpr[dpr_idx + j]) for j in range(depths[0])],
            nn.Conv2d(c_list[1], c_list[2], 3, 1, 1),
        )
        dpr_idx += depths[0]
        self.norm3 = group_norm(c_list[2])

        self.encoder4 = nn.Sequential(
            *[make_block(dim=c_list[2], drop_path=dpr[dpr_idx + j]) for j in range(depths[1])],
            nn.Conv2d(c_list[2], c_list[3], 3, 1, 1),
        )
        dpr_idx += depths[1]
        self.norm4 = group_norm(c_list[3])

        self.encoder5 = nn.Sequential(
            *[make_block(dim=c_list[3], drop_path=dpr[dpr_idx + j]) for j in range(depths[2])],
            nn.Conv2d(c_list[3], c_list[4], 3, 1, 1),
        )
        dpr_idx += depths[2]
        self.norm5 = group_norm(c_list[4])

        # ------------------------------------------------------------------
        # Fig. 3(b) -- bottleneck
        # ------------------------------------------------------------------
        self.bottleneck = nn.Sequential(
            *[make_block(dim=c_list[4], drop_path=dpr[dpr_idx + j]) for j in range(depths[3])],
            nn.Conv2d(c_list[4], c_list[5], 3, 1, 1),
            group_norm(c_list[5]),
            nn.GELU(),
        )
        dpr_idx += depths[3]

        # ------------------------------------------------------------------
        # Fig. 3(c) -- cross-scale bridge on the five skip connections
        # ------------------------------------------------------------------
        skip_channels = c_list[:5]
        self.scab = SCAB(skip_channels, split_att=split_att)
        self.lightcsf = LightCSF(skip_channels, embed_ratio=csf_embed_ratio)
        self.skip_se = nn.ModuleList([SEBlock(c, reduction=4) for c in skip_channels])

        # ------------------------------------------------------------------
        # Fig. 3(d) -- decoder (mirrors the encoder depths, 3x3 convs without bias)
        # ------------------------------------------------------------------
        self.decoder1 = nn.Sequential(
            *[make_block(dim=c_list[5], drop_path=dpr[dpr_idx + j]) for j in range(depths[3])],
            nn.Conv2d(c_list[5], c_list[4], 3, 1, 1, bias=False),
        )
        dpr_idx += depths[3]
        self.dnorm1 = group_norm(c_list[4])

        self.decoder2 = nn.Sequential(
            *[make_block(dim=c_list[4], drop_path=dpr[dpr_idx + j]) for j in range(depths[2])],
            nn.Conv2d(c_list[4], c_list[3], 3, 1, 1, bias=False),
        )
        dpr_idx += depths[2]
        self.dnorm2 = group_norm(c_list[3])

        self.decoder3 = nn.Sequential(
            *[make_block(dim=c_list[3], drop_path=dpr[dpr_idx + j]) for j in range(depths[1])],
            nn.Conv2d(c_list[3], c_list[2], 3, 1, 1, bias=False),
        )
        dpr_idx += depths[1]
        self.dnorm3 = group_norm(c_list[2])

        self.decoder4 = nn.Sequential(
            *[make_block(dim=c_list[2], drop_path=dpr[dpr_idx + j]) for j in range(depths[0])],
            nn.Conv2d(c_list[2], c_list[1], 3, 1, 1, bias=False),
        )
        dpr_idx += depths[0]
        self.dnorm4 = group_norm(c_list[1])

        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, 1, 1),
            group_norm(c_list[0]),
            nn.GELU(),
        )

        # Segmentation head: 1x1 conv producing raw class logits.
        self.head = nn.Conv2d(c_list[0], num_classes, 1)

        self.apply(self._init_weights)

    # ----------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Truncated-normal for linear layers, Kaiming fan-out for convolutions."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Run the network and return raw logits of shape ``[B, num_classes, H, W]``."""
        # --- Fig. 3(a) encoder -------------------------------------------------
        e1 = self.encoder1(x)  # c0, H
        t1 = F.max_pool2d(e1, 2, 2)  # c0, H/2
        e2 = self.encoder2(t1)  # c1, H/2
        t2 = F.max_pool2d(e2, 2, 2)  # c1, H/4
        e3 = F.gelu(self.norm3(self.encoder3(t2)))  # c2, H/4
        t3 = F.max_pool2d(e3, 2, 2)  # c2, H/8
        e4 = F.gelu(self.norm4(self.encoder4(t3)))  # c3, H/8
        t4 = F.max_pool2d(e4, 2, 2)  # c3, H/16
        e5 = F.gelu(self.norm5(self.encoder5(t4)))  # c4, H/16
        t5 = F.max_pool2d(e5, 2, 2)  # c4, H/32

        # --- Fig. 3(c) cross-scale bridge -------------------------------------
        skips: List[Tensor] = [e1, e2, e3, e4, e5]
        skips = self.scab(skips)                                    # Fig. 5
        skips = self.lightcsf(skips)                                # Fig. 6
        skips = [se(skip) for se, skip in zip(self.skip_se, skips)]  # SE recalibration

        # --- Fig. 3(b) bottleneck ---------------------------------------------
        b = self.bottleneck(t5)  # c5, H/32

        # --- Fig. 3(d) decoder -------------------------------------------------
        d1 = F.interpolate(
            self.dnorm1(self.decoder1(b)),
            size=skips[4].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = F.gelu(d1 + skips[4])

        d2 = F.interpolate(
            self.dnorm2(self.decoder2(x)),
            size=skips[3].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = F.gelu(d2 + skips[3])

        d3 = F.interpolate(
            self.dnorm3(self.decoder3(x)),
            size=skips[2].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = F.gelu(d3 + skips[2])

        d4 = F.interpolate(
            self.dnorm4(self.decoder4(x)),
            size=skips[1].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = F.gelu(d4 + skips[1])

        d5 = F.interpolate(
            self.decoder5(x),
            size=skips[0].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = d5 + skips[0]  # no GELU on the last fusion

        return self.head(x)  # raw logits, no sigmoid/softmax


def build_wheatscopenet(cfg: Any) -> WheatScopeNet:
    """Instantiate :class:`WheatScopeNet` from a configuration object.

    Every field is read with ``getattr`` so that a partially populated config (for
    example an ``argparse.Namespace``) still produces the paper's default model.
    """
    return WheatScopeNet(
        num_classes=getattr(cfg, "num_classes", 3),
        input_channels=getattr(cfg, "input_channels", 3),
        c_list=getattr(cfg, "c_list", (8, 16, 32, 64, 128, 256)),
        depths=getattr(cfg, "depths", (2, 2, 2, 2)),
        d_state=getattr(cfg, "d_state", 16),
        kernel_sizes=getattr(cfg, "kernel_sizes", (3, 5)),
        mlp_ratio=getattr(cfg, "mlp_ratio", 2),
        drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
        layer_scale_init_value=getattr(cfg, "layer_scale_init_value", 1e-6),
        split_att=getattr(cfg, "split_att", "fc"),
        csf_embed_ratio=getattr(cfg, "csf_embed_ratio", 2),
    )

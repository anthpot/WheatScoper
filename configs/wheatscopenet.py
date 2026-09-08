"""Reference configuration of WheatScopeNet.

All values are transcribed from the manuscript "WheatScoper: A lightweight
organ-based framework for multi-view wheat phenotyping using time-series RGB
images" (paper Section 2.2 for the data settings, Section 2.3 for the network
architecture and Section 2.4 for the optimisation protocol).

The class is deliberately a plain attribute container: ``tools/train.py`` and
``tools/predict.py`` instantiate it and override individual attributes from the
command line, so nothing here depends on the runtime environment.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any


class Config:
    """Paper configuration of WheatScopeNet (organ-level, three classes)."""

    # ------------------------------------------------------------------
    # Model (paper Section 2.3; 4.321 M parameters, 51.93 G FLOPs at 2048 x 2048)
    # ------------------------------------------------------------------
    num_classes = 3                       # background + spike + leaf
    class_names = ("background", "spike", "leaf")
    input_channels = 3                    # RGB input (paper Section 2.2)
    c_list = (8, 16, 32, 64, 128, 256)    # stage widths c0..c5 (paper Fig. 3)
    depths = (2, 2, 2, 2)                 # PHS Blocks per deep stage (Fig. 3(a))
    d_state = 16                          # SS2D state dimension (Section 2.3.1)
    kernel_sizes = (3, 5)                 # multi-scale DWConv (Section 2.3.2)
    mlp_ratio = 2                         # PHS Block channel-MLP ratio (Fig. 4(a))
    drop_path_rate = 0.0                  # no stochastic depth in the paper
    layer_scale_init_value = 1e-6         # PHS Block layer-scale init
    split_att = "fc"                      # SCAB channel attention (paper Eq. 8)
    csf_embed_ratio = 2                   # LightCSF compression ratio (Fig. 6)
    ignore_index = 255                    # label value excluded from loss/metrics

    # ------------------------------------------------------------------
    # Data (paper Section 2.2)
    # ------------------------------------------------------------------
    data_root = "data/wheat_canopy"       # <root>/images/<split>, <root>/masks/<split>
    input_size = (2048, 2048)             # images are resized to 2048 x 2048

    # ------------------------------------------------------------------
    # Training (paper Sections 2.4 and 3.2.1)
    # ------------------------------------------------------------------
    seed = 42
    epochs = 300
    batch_size = 2
    val_batch_size = 2
    num_workers = 4
    amp = True                            # automatic mixed precision (Section 2.4)
    amp_dtype = "fp16"                    # "fp16" or "bf16"

    # AdamW: initial learning rate 1e-3, weight decay 1e-2 (paper Section 2.4).
    opt = "AdamW"
    lr = 1e-3
    weight_decay = 1e-2
    betas = (0.9, 0.999)
    eps = 1e-8

    # Cosine annealing from 1e-3 down to 1e-6 over the whole schedule.
    sch = "CosineAnnealingLR"
    eta_min = 1e-6

    # Combined cross-entropy + Dice loss, both weights 1.0 (paper Eq. 12).
    w_ce = 1.0
    w_dice = 1.0

    # Checkpoint selection and output location.
    metric_to_monitor = "mdice"
    work_dir = "results"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def build_work_dir(root: str = work_dir) -> str:
        """Return a timestamped run directory ``<root>/wheatscopenet_<stamp>``.

        The timestamp is evaluated when this method is called, never at class
        definition time, so importing the module has no side effects.
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(root, f"wheatscopenet_{stamp}")

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serialisable dictionary."""
        settings: dict[str, Any] = {}
        for name in sorted(dir(self)):
            if name.startswith("_"):
                continue
            value = getattr(self, name)
            if callable(value):
                continue
            settings[name] = list(value) if isinstance(value, tuple) else value
        return settings

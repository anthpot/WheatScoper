"""Training utilities for WheatScopeNet (paper Section 2.4).

Reproducibility helpers, logging, and the optimiser / scheduler builders for the
exact training recipe reported in the paper: AdamW (lr 1e-3, weight decay 1e-2,
betas (0.9, 0.999), eps 1e-8) with a cosine-annealing learning-rate schedule
(T_max = epochs, eta_min = 1e-6).
"""

from __future__ import annotations

import inspect
import logging
import logging.handlers
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

__all__ = [
    "set_seed",
    "get_logger",
    "build_optimizer",
    "build_scheduler",
    "log_config",
    "count_parameters",
]

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# The paper trains with AdamW only (Section 2.4).
_SUPPORTED_OPTIMIZER = "adamw"
# The paper uses a single cosine-annealing schedule (Section 2.4).
_SUPPORTED_SCHEDULER = "cosineannealinglr"


def set_seed(seed: int) -> None:
    """Seed every RNG used during training and make cuDNN deterministic."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_logger(name: str, log_dir: str) -> logging.Logger:
    """Return a logger writing to ``<log_dir>/<name>.info.log`` and to stdout.

    Handlers are attached only once, so repeated calls with the same ``name``
    return the already configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        os.path.join(log_dir, f"{name}.info.log"),
        when="D",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def build_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Build the AdamW optimiser described in paper Section 2.4.

    Raises:
        ValueError: If ``config.opt`` requests anything other than AdamW.
    """
    opt_name = str(getattr(config, "opt", "AdamW"))
    if opt_name.lower() != _SUPPORTED_OPTIMIZER:
        raise ValueError(
            f"Unsupported optimizer '{opt_name}'. WheatScopeNet is trained with "
            "AdamW only (paper Section 2.4)."
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable_params,
        lr=float(config.lr),
        betas=tuple(getattr(config, "betas", (0.9, 0.999))),
        eps=float(getattr(config, "eps", 1e-8)),
        weight_decay=float(getattr(config, "weight_decay", 1e-2)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Build the cosine-annealing LR schedule described in paper Section 2.4.

    Raises:
        ValueError: If ``config.sch`` requests anything other than CosineAnnealingLR.
    """
    sch_name = str(getattr(config, "sch", "CosineAnnealingLR"))
    if sch_name.lower() != _SUPPORTED_SCHEDULER:
        raise ValueError(
            f"Unsupported scheduler '{sch_name}'. WheatScopeNet is trained with "
            "CosineAnnealingLR only (paper Section 2.4)."
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config.epochs),
        eta_min=float(getattr(config, "eta_min", 1e-6)),
    )


def log_config(config, logger: logging.Logger) -> None:
    """Log every public attribute of the configuration object."""
    logger.info("#---------- Config info ----------#")
    items = []
    for name, value in inspect.getmembers(config):
        if name.startswith("_"):
            continue
        if (
            inspect.ismodule(value)
            or inspect.isfunction(value)
            or inspect.ismethod(value)
            or inspect.isclass(value)
        ):
            continue
        try:
            value_str = repr(value)
        except Exception:  # pragma: no cover - defensive, repr should not fail
            value_str = f"<{type(value).__name__} object (repr unavailable)>"
        if len(value_str) > 200:
            value_str = f"<{type(value).__name__} object (repr too long)>"
        items.append((name, value_str))

    if not items:
        logger.warning("No configuration entries could be extracted.")
    for name, value_str in sorted(items):
        logger.info(f"{name}: {value_str}")
    logger.info("#---------------------------------#")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters of ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

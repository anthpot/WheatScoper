"""Training and evaluation loops for WheatScopeNet (paper Section 2.4).

`train_one_epoch` runs one optimisation pass over the training set and, in the
same pass, accumulates the confusion-matrix metrics (no second forward pass).
`evaluate` runs a full inference pass over an evaluation split and
reports the metrics of paper Section 3.2.2 / Fig. 11.

Mixed precision follows the paper's protocol: `torch.amp.autocast('cuda', ...)`
with FP16 by default (`config.amp`, `config.amp_dtype`).
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import SegmentationMetrics

__all__ = ["train_one_epoch", "evaluate"]

_AMP_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

_BF16_FALLBACK_WARNED = False


def _resolve_amp_dtype(config) -> torch.dtype:
    """Map ``config.amp_dtype`` ('fp16' | 'bf16') to a torch dtype."""
    global _BF16_FALLBACK_WARNED
    name = str(getattr(config, "amp_dtype", "fp16")).lower()
    if name not in _AMP_DTYPES:
        raise ValueError(
            f"Unsupported amp_dtype '{name}'. Supported values: 'fp16', 'bf16'."
        )
    dtype = _AMP_DTYPES[name]
    if dtype is torch.bfloat16 and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            if not _BF16_FALLBACK_WARNED:
                warnings.warn(
                    "BF16 autocast is not supported by this GPU; falling back to FP16.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _BF16_FALLBACK_WARNED = True
            dtype = torch.float16
    return dtype


def _class_names(config, num_classes: int) -> Sequence[str]:
    return getattr(
        config, "class_names", tuple(f"class{i}" for i in range(num_classes))
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config,
    logger,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, object]:
    """Run one training epoch and return the epoch's loss and metrics.

    Raises:
        ValueError: If the training loss becomes NaN or Inf.
    """
    device = torch.device(device)
    model.train()

    amp_enabled = bool(getattr(config, "amp", False)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(config)

    num_classes = int(config.num_classes)
    metrics = SegmentationMetrics(
        num_classes, ignore_index=int(getattr(config, "ignore_index", 255))
    )

    loss_list: list[float] = []
    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", ncols=120, leave=False)

    for it, (images, targets) in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        loss_item = loss.item()
        if not np.isfinite(loss_item):
            logger.error(
                f"Non-finite training loss (NaN/Inf) at epoch {epoch}, iteration {it}. "
                "Stopping training."
            )
            raise ValueError("Non-finite training loss (NaN/Inf).")

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_list.append(loss_item)

        # Metrics are accumulated from the same forward pass; no extra inference.
        with torch.no_grad():
            metrics.update(torch.argmax(logits.detach(), dim=1), targets)

        now_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss_item:.4f}", lr=f"{now_lr:.6f}")

    avg_loss = float(np.mean(loss_list)) if loss_list else 0.0
    final_lr = float(optimizer.param_groups[0]["lr"])

    results = metrics.compute()
    results["loss"] = avg_loss
    results["lr"] = final_lr

    logger.info(
        f"[Train] Epoch {epoch} finished. Avg loss: {avg_loss:.4f} | LR: {final_lr:.6f}"
    )
    logger.info(
        f"[Train] Epoch {epoch} mIoU={results['miou']:.4f}, "
        f"mDice={results['mdice']:.4f}, PA={results['pa']:.4f}"
    )
    logger.info("\n" + metrics.format_table(_class_names(config, num_classes)))
    return results


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config,
    logger,
    stage: str = "val",
) -> Dict[str, object]:
    """Run a full inference pass over a split and return loss + metrics.

    Args:
        stage: Split name used for logging, e.g. ``"val"``.
    """
    device = torch.device(device)
    model.eval()

    amp_enabled = bool(getattr(config, "amp", False)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(config)

    num_classes = int(config.num_classes)
    metrics = SegmentationMetrics(
        num_classes, ignore_index=int(getattr(config, "ignore_index", 255))
    )

    label = stage.capitalize()
    loss_list: list[float] = []
    pbar = tqdm(loader, desc=f"[{label}]", ncols=120, leave=False)

    for it, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        loss_item = loss.item()
        if not np.isfinite(loss_item):
            logger.warning(
                f"Non-finite {stage} loss (NaN/Inf) at batch {it}. "
                "Skipping this batch."
            )
            continue
        loss_list.append(loss_item)
        pbar.set_postfix(loss=f"{loss_item:.4f}")

        # softmax is monotonic, so argmax over the logits gives the same labels.
        metrics.update(torch.argmax(logits, dim=1), targets)

    avg_loss = float(np.mean(loss_list)) if loss_list else float("nan")

    results = metrics.compute()
    results["loss"] = avg_loss

    logger.info(
        f"[{label}] Finished. Avg loss: {avg_loss:.4f} | "
        f"mIoU={results['miou']:.4f}, mDice={results['mdice']:.4f}, "
        f"mPrecision={results['mprecision']:.4f}, mRecall={results['mrecall']:.4f}, "
        f"PA={results['pa']:.4f}, aAcc={results['aacc']:.4f}"
    )
    logger.info("\n" + metrics.format_table(_class_names(config, num_classes)))
    return results

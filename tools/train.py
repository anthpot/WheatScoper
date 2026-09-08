#!/usr/bin/env python3
"""Training driver for WheatScopeNet (paper Sections 2.4 and 3.2.1).

Run from the repository root, e.g.::

    python tools/train.py --data-root data/wheat_canopy
    python tools/train.py --data-root data/wheat_canopy --epochs 300 --batch-size 2

Single-GPU (or CPU) training only: no distributed data parallel, no image
dumping. Checkpoints, the resolved configuration and the per-epoch metric
history are written into the run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# Make "import wheatscopenet" and "from configs... import Config" resolve when
# this file is executed as "python tools/train.py" from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.wheatscopenet import Config  # noqa: E402
from wheatscopenet.data import WheatCanopyDataset  # noqa: E402
from wheatscopenet.engine import evaluate, train_one_epoch  # noqa: E402
from wheatscopenet.losses import CrossEntropyDiceLoss  # noqa: E402
from wheatscopenet.models import build_wheatscopenet  # noqa: E402
from wheatscopenet.utils import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    count_parameters,
    get_logger,
    log_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train WheatScopeNet for organ-level wheat canopy segmentation.",
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset root holding images/<split> and masks/<split>.")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Run directory. Default: a timestamped folder under config.work_dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of segmentation classes (paper uses 3).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume optimisation from.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def resolve_device(name: str | None) -> torch.device:
    """Return the requested device, defaulting to CUDA when it is available."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_config(args: argparse.Namespace, device: torch.device) -> Config:
    """Instantiate the paper configuration and apply the command-line overrides."""
    cfg = Config()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
        cfg.val_batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.num_classes is not None and args.num_classes != cfg.num_classes:
        cfg.num_classes = args.num_classes
        # The paper's three class names no longer apply to a different label set.
        cfg.class_names = tuple(f"class{i}" for i in range(args.num_classes))
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_amp or device.type != "cuda":
        cfg.amp = False
    return cfg


def build_loaders(cfg: Config, device: torch.device) -> tuple[DataLoader, DataLoader]:
    """Create the training and validation data loaders (paper Section 2.2)."""
    train_set = WheatCanopyDataset(cfg.data_root, split="train", input_size=tuple(cfg.input_size))
    val_set = WheatCanopyDataset(cfg.data_root, split="val", input_size=tuple(cfg.input_size))
    common: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=False, **common)
    val_loader = DataLoader(val_set, batch_size=cfg.val_batch_size, shuffle=False,
                            drop_last=False, **common)
    return train_loader, val_loader


def scalar_metrics(stats: dict[str, Any], prefix: str) -> dict[str, float]:
    """Keep only the scalar entries of a metric dict (per-class arrays are dropped)."""
    flat: dict[str, float] = {}
    for key, value in stats.items():
        if isinstance(value, str) or hasattr(value, "__len__"):
            continue
        try:
            flat[f"{prefix}{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return flat


def read_history(work_dir: str, before_epoch: int) -> list[dict[str, float]]:
    """Reload the metric rows of previous epochs so a resumed run keeps its history."""
    path = os.path.join(work_dir, "metrics_history.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return [row for row in rows if float(row.get("epoch", 0)) < before_epoch]


def write_history(work_dir: str, history: list[dict[str, float]]) -> None:
    """Persist the per-epoch metric history as both JSON and CSV."""
    with open(os.path.join(work_dir, "metrics_history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    if not history:
        return
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(os.path.join(work_dir, "metrics_history.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(path: str, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any, scaler: torch.amp.GradScaler, epoch: int,
                    best_metric: float, best_epoch: int, cfg: Config) -> None:
    """Write a checkpoint holding the weights and the full optimisation state."""
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "monitor": cfg.metric_to_monitor,
            "config": cfg.to_dict(),
        },
        path,
    )


def load_resume_state(path: str, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      scheduler: Any, scaler: torch.amp.GradScaler,
                      device: torch.device) -> tuple[int, float, int]:
    """Restore model/optimiser state and return (start_epoch, best_metric, best_epoch)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_metric = float(checkpoint.get("best_metric", float("-inf")))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    return start_epoch, best_metric, best_epoch


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    cfg = build_config(args, device)

    work_dir = args.work_dir or Config.build_work_dir(root=cfg.work_dir)
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    cfg.work_dir = work_dir

    set_seed(cfg.seed)
    logger = get_logger("train", work_dir)
    logger.info(f"WheatScopeNet training | device: {device}")
    log_config(cfg, logger)
    with open(os.path.join(work_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)

    train_loader, val_loader = build_loaders(cfg, device)
    logger.info(f"Training images: {len(train_loader.dataset)} | validation images: {len(val_loader.dataset)}")

    model = build_wheatscopenet(cfg).to(device)
    n_params = count_parameters(model)
    startup = f"WheatScopeNet trainable parameters: {n_params:,} ({n_params / 1e6:.3f} M)"
    print(startup)
    logger.info(startup)

    criterion = CrossEntropyDiceLoss(w_ce=cfg.w_ce, w_dice=cfg.w_dice, ignore_index=cfg.ignore_index)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp))

    monitor = cfg.metric_to_monitor
    higher_is_better = monitor != "loss"
    best_metric = float("-inf") if higher_is_better else float("inf")
    best_epoch, start_epoch = 0, 1
    if args.resume:
        start_epoch, best_metric, best_epoch = load_resume_state(
            args.resume, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, device=device)
        logger.info(f"Resumed from {args.resume}: starting at epoch {start_epoch}, "
                    f"best {monitor}={best_metric:.4f} (epoch {best_epoch})")

    history: list[dict[str, float]] = read_history(work_dir, start_epoch)
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            started = time.time()
            train_stats = train_one_epoch(model, train_loader, criterion, optimizer,
                                          device, epoch, cfg, logger, scaler)
            val_stats = evaluate(model, val_loader, criterion, device, cfg, logger, stage="val")
            scheduler.step()

            score = float(val_stats[monitor])
            improved = score > best_metric if higher_is_better else score < best_metric
            row = {"epoch": float(epoch), "lr": float(optimizer.param_groups[0]["lr"]),
                   "epoch_time_s": time.time() - started}
            row.update(scalar_metrics(train_stats, "train_"))
            row.update(scalar_metrics(val_stats, "val_"))
            history.append(row)
            write_history(work_dir, history)

            summary = (f"Epoch {epoch}/{cfg.epochs} | train loss {train_stats['loss']:.4f} | "
                       f"val loss {val_stats['loss']:.4f} | val mIoU {val_stats['miou']:.4f} | "
                       f"val mDice {val_stats['mdice']:.4f} | val PA {val_stats['pa']:.4f}")
            print(summary)
            logger.info(summary)

            if improved:
                best_metric, best_epoch = score, epoch
                save_checkpoint(os.path.join(checkpoint_dir, "best.pth"), model=model,
                                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                                epoch=epoch, best_metric=best_metric, best_epoch=best_epoch,
                                cfg=cfg)
                logger.info(f"New best {monitor}={best_metric:.4f} at epoch {epoch}; saved best.pth")
            save_checkpoint(os.path.join(checkpoint_dir, "latest.pth"), model=model,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, best_metric=best_metric, best_epoch=best_epoch,
                            cfg=cfg)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by the user; latest.pth holds the last finished epoch.")

    final = (f"Training finished. Best {monitor}={best_metric:.4f} at epoch {best_epoch}. "
             f"Checkpoints in {checkpoint_dir}")
    print(final)
    logger.info(final)


if __name__ == "__main__":
    main()

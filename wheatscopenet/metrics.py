"""Confusion-matrix segmentation metrics for WheatScopeNet.

Implements the evaluation protocol reported in the paper (Section 3.2.2, Fig. 11):
per-class IoU / Dice / Precision / Recall / Accuracy plus the aggregate
mIoU, mDice, mPrecision, mRecall, PA and aAcc scores.

All statistics are derived from a single int64 confusion matrix that is
accumulated incrementally over batches, so the scores are dataset-level
(micro-accumulated) rather than an average of per-batch values.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

__all__ = ["SegmentationMetrics"]

# Numerical guard used in every metric definition (identical to the legacy engine).
EPS: float = 1e-6

_TABLE_RULE = (
    "+-------------+-------+-------+-------+--------+-----------+--------+"
)
_TABLE_HEADER = (
    "|    Class    |  IoU  |  Acc  |  Dice | Fscore | Precision | Recall |"
)


class SegmentationMetrics:
    """Accumulate a confusion matrix and derive segmentation metrics from it.

    Args:
        num_classes: Number of semantic classes (3 for the paper: background,
            spike, leaf).
        ignore_index: Label value that is excluded from the statistics.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def reset(self) -> None:
        """Zero the accumulated confusion matrix."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    @torch.no_grad()
    def update(self, pred_labels: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            pred_labels: Predicted class indices ``[B, H, W]`` (argmax output,
                **not** raw logits).
            target: Ground-truth class indices ``[B, H, W]``.
        """
        if pred_labels.shape != target.shape:
            raise ValueError(
                "pred_labels and target must have the same shape, got "
                f"{tuple(pred_labels.shape)} and {tuple(target.shape)}"
            )

        valid = target != self.ignore_index
        pred_flat = pred_labels[valid].detach().cpu().numpy().astype(np.int64)
        target_flat = target[valid].detach().cpu().numpy().astype(np.int64)
        if pred_flat.size == 0:
            return

        for label_name, labels in (("target", target_flat), ("pred_labels", pred_flat)):
            if labels.min() < 0 or labels.max() >= self.num_classes:
                raise ValueError(
                    f"{label_name} contains a value outside [0, {self.num_classes}) "
                    "while updating the confusion matrix. Check the dataset "
                    "annotations and the ignore_index setting."
                )

        indices = self.num_classes * target_flat + pred_flat
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(
            self.num_classes, self.num_classes
        ).astype(np.int64)

    def compute(self) -> Dict[str, object]:
        """Return per-class arrays and aggregate scalars.

        Definitions (``TP`` = diagonal, ``FP`` = column sum - TP,
        ``FN`` = row sum - TP)::

            iou       = TP / (TP + FP + FN + eps)
            dice      = 2 TP / (2 TP + FP + FN + eps)
            precision = TP / (TP + FP + eps)
            recall    = accuracy = TP / (TP + FN + eps)
            pa        = aacc = TP.sum() / conf.sum()
        """
        conf = self.confusion_matrix.astype(np.float64)
        tp = np.diag(conf)
        fp = conf.sum(axis=0) - tp
        fn = conf.sum(axis=1) - tp

        iou = tp / (tp + fp + fn + EPS)
        dice = 2.0 * tp / (2.0 * tp + fp + fn + EPS)
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        accuracy = recall  # per-class pixel accuracy == recall
        overall = tp.sum() / (conf.sum() + EPS)

        return {
            # per-class arrays
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            # aggregate scalars
            "miou": float(iou.mean()),
            "mdice": float(dice.mean()),
            "mprecision": float(precision.mean()),
            "mrecall": float(recall.mean()),
            "pa": float(overall),
            "aacc": float(overall),
        }

    def format_table(self, class_names: Sequence[str]) -> str:
        """Render the per-class metrics as an ASCII table (values in percent)."""
        if len(class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries but the metric was "
                f"built for {self.num_classes} classes"
            )

        results = self.compute()
        iou = results["iou"]
        acc = results["accuracy"]
        dice = results["dice"]
        precision = results["precision"]
        recall = results["recall"]

        def safe(value: float) -> float:
            value = float(value)
            return value if np.isfinite(value) else 0.0

        lines = [_TABLE_RULE, _TABLE_HEADER, _TABLE_RULE]
        for idx, name in enumerate(class_names):
            fscore = safe(dice[idx]) * 100  # F-score equals Dice for binary-per-class
            lines.append(
                f"| {str(name):^11} "
                f"| {safe(iou[idx]) * 100:5.2f} "
                f"| {safe(acc[idx]) * 100:5.2f} "
                f"| {fscore:5.2f} "
                f"| {fscore:6.2f} "
                f"| {safe(precision[idx]) * 100:9.2f} "
                f"| {safe(recall[idx]) * 100:6.2f} |"
            )
        lines.append(_TABLE_RULE)
        return "\n".join(lines)

"""Segmentation loss of WheatScopeNet (paper Eq. (12)).

Implements the combined cross-entropy + soft-Dice objective used to train the
organ-level segmentation network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CrossEntropyDiceLoss"]


class CrossEntropyDiceLoss(nn.Module):
    """Combined cross-entropy and soft-Dice loss (paper Eq. (12)).

    ``L = w_ce * L_CE + w_dice * L_Dice`` with ``w_ce = w_dice = 1.0`` in the
    paper's configuration (Section 2.4).

    The Dice term is computed on softmax probabilities against the one-hot
    encoded targets, aggregated over the batch and spatial dimensions
    ``(0, 2, 3)`` so it yields one score per class, and returned as
    ``1 - dice.mean()`` (the mean includes the background class). Pixels
    labelled ``ignore_index`` are excluded from both terms.

    Args:
        w_ce: Weight of the cross-entropy term (paper: 1.0).
        w_dice: Weight of the Dice term (paper: 1.0). ``0`` disables it.
        ignore_index: Target value marking pixels excluded from the loss.
        smooth: Numerical stabiliser added to the Dice numerator/denominator.
    """

    def __init__(
        self,
        w_ce: float = 1.0,
        w_dice: float = 1.0,
        ignore_index: int = 255,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Raw network outputs ``[B, C, H, W]``.
            targets: Class indices ``[B, H, W]`` (``ignore_index`` allowed).

        Returns:
            Scalar loss tensor.
        """
        loss_ce = self.ce(logits, targets)

        if self.w_dice > 0:
            probs = torch.softmax(logits, dim=1)

            # Mask out ignored pixels before one-hot encoding: they are mapped to
            # class 0 first so one_hot stays in range, then zeroed out below.
            valid = targets != self.ignore_index
            targets_masked = torch.where(valid, targets, torch.zeros_like(targets))
            targets_onehot = F.one_hot(targets_masked, num_classes=logits.size(1))
            targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

            valid_mask = valid.unsqueeze(1).float()
            probs = probs * valid_mask
            targets_onehot = targets_onehot * valid_mask

            dims = (0, 2, 3)
            intersection = torch.sum(probs * targets_onehot, dims)
            cardinality = torch.sum(probs + targets_onehot, dims)
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            loss_dice = 1 - dice.mean()  # averaged over all classes, background included
        else:
            loss_dice = torch.zeros((), dtype=loss_ce.dtype, device=loss_ce.device)

        return self.w_ce * loss_ce + self.w_dice * loss_dice

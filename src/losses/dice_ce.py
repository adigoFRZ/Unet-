"""Baseline v1 loss: CrossEntropy + multiclass soft Dice.

Deliberately plain. No class weighting, no Focal, no Tversky, no boundary or
Hausdorff term -- all of those are separate later experiments, and mixing them in
now would make the baseline impossible to attribute.

The Dice term is a *foreground macro average* over classes 1, 2 and 3 only.
Background (class 0) is excluded on purpose: it occupies >99% of the crop, so
including it would push the Dice term toward ~1 from the very first step and
drown out the signal from the structures we actually care about.

    dice_loss = 1 - mean_over_{c in {STN,SN,RN}} ( 2*sum(p_c * g_c) + eps
                                                  / (sum(p_c) + sum(g_c) + eps) )

where ``p`` is the softmax probability and ``g`` the one-hot ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import crop_spec as cs


@dataclass
class DiceCELossConfig:
    """Weights and numerics for :class:`DiceCELoss`."""

    ce_weight: float = 1.0
    dice_weight: float = 1.0
    #: classes averaged into the Dice term; background is intentionally absent
    dice_classes: Sequence[int] = cs.FOREGROUND_CLASSES
    smooth: float = 1e-5
    #: optional per-class weights for the CE term (None = unweighted, Baseline v1)
    ce_class_weights: Sequence[float] | None = None


class DiceCELoss(nn.Module):
    """Weighted sum of cross-entropy and foreground-macro soft Dice."""

    def __init__(self, config: DiceCELossConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiceCELossConfig()
        weights = None
        if self.config.ce_class_weights is not None:
            weights = torch.tensor(list(self.config.ce_class_weights), dtype=torch.float32)
        self.register_buffer("ce_class_weights", weights if weights is not None else None,
                             persistent=False)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(total_loss, components)``.

        ``logits``: (N, C, D, H, W) raw scores.
        ``target``: (N, D, H, W) integer labels.
        """
        if logits.dim() != 5 or target.dim() != 4:
            raise ValueError(
                f"expected logits (N,C,D,H,W) and target (N,D,H,W), got "
                f"{tuple(logits.shape)} and {tuple(target.shape)}"
            )
        # target is (N, D, H, W) -- its spatial dims are shape[1:], not shape[2:].
        if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
            raise ValueError(
                f"batch/spatial mismatch: logits {tuple(logits.shape)} vs "
                f"target {tuple(target.shape)}"
            )

        ce = F.cross_entropy(logits, target.long(), weight=self.ce_class_weights)
        dice = self.soft_dice_loss(logits, target)
        total = self.config.ce_weight * ce + self.config.dice_weight * dice

        return total, {
            "loss": total.detach(),
            "ce": ce.detach(),
            "dice": dice.detach(),
        }

    def per_case_per_class_dice(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Soft Dice as an ``(N, n_dice_classes)`` matrix.

        The reductions run over **D, H, W only** -- never over the batch. Summing
        across the batch would implicitly weight each case by its target volume,
        so a case with large structures would dominate the loss while a case with
        small ones barely contributed. Evaluation is a per-case mean, so the
        training objective has to be per-case too, or the two disagree.
        """
        probabilities = F.softmax(logits, dim=1)
        per_class: list[torch.Tensor] = []

        for class_id in self.config.dice_classes:
            # Indexing out the class dim leaves (N, D, H, W); the spatial dims of
            # THIS tensor are (1, 2, 3). Reducing over dim 0 would sum the batch,
            # which is exactly the pooling this method exists to avoid.
            prob_c = probabilities[:, class_id]              # (N, D, H, W)
            true_c = (target == class_id).to(prob_c.dtype)   # (N, D, H, W)
            spatial_dims = tuple(range(1, prob_c.ndim))      # (1, 2, 3)
            numerator = 2.0 * torch.sum(prob_c * true_c, dim=spatial_dims)
            denominator = (
                torch.sum(prob_c, dim=spatial_dims)
                + torch.sum(true_c, dim=spatial_dims)
            )
            per_class.append(
                (numerator + self.config.smooth) / (denominator + self.config.smooth)
            )

        if not per_class:
            return logits.new_zeros((logits.shape[0], 0))
        return torch.stack(per_class, dim=1)  # (N, n_dice_classes)

    def soft_dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 - mean soft Dice over cases x foreground classes.

        Background is excluded: it is >99% of the crop, so including it would pin
        the term near 1 from the first step and swamp the signal from the
        structures actually being segmented.
        """
        dice_matrix = self.per_case_per_class_dice(logits, target)
        if dice_matrix.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return 1.0 - dice_matrix.mean()


def soft_dice_per_case(
    logits: torch.Tensor,
    target: torch.Tensor,
    classes: Sequence[int] = cs.FOREGROUND_CLASSES,
) -> list[dict[int, float]]:
    """Per-case, per-class soft Dice for monitoring (not part of the loss).

    Returns one dict per batch element, so a caller can see individual cases
    rather than a batch-pooled number.
    """
    with torch.no_grad():
        criterion = DiceCELoss(DiceCELossConfig(dice_classes=tuple(classes)))
        matrix = criterion.per_case_per_class_dice(logits, target)  # (N, C)
        return [
            {class_id: float(matrix[row, col]) for col, class_id in enumerate(classes)}
            for row in range(matrix.shape[0])
        ]

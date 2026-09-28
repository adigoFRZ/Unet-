"""Experiment G1 loss: cross-entropy + foreground soft Dice on a soft STN target.

Baseline v1 is `CE + foreground macro soft Dice` over STN/SN/RN against a hard
one-hot label. Experiment G1 changes exactly one thing: for the training split
the STN component of the *target* becomes the interface-centred soft map built by
:mod:`data.interface_soft_target`. The objective is otherwise untouched -- same
CE, same Dice, same foreground class set, same smoothing, same averaging, same
weights. Nothing here re-weights anything.

Two dispatch paths, and the difference between them matters
-----------------------------------------------------------
* **Hard target** ``(N, D, H, W)`` integer -> the whole call is delegated to the
  baseline's own :class:`DiceCELoss`. Not reimplemented, not approximated: the
  object is called. That is what makes "the baseline is unchanged" a structural
  property rather than a promise, and it is why this class composes rather than
  inherits.

* **Soft target** ``(N, C, D, H, W)`` float -> soft-label CE and soft-target Dice.
  The Dice form is the baseline's own, with the one-hot mask replaced by the soft
  target::

      numerator   = 2 * sum(p_c * q_c)
      denominator =     sum(p_c) + sum(q_c)

  which reduces exactly to the baseline's ``2*sum(p*g) / (sum(p)+sum(g))`` when
  ``q`` is one-hot.

Why the CE has to change form
-----------------------------
``F.cross_entropy`` accepts either class indices or a probability target, but not
a target that is hard for some classes and soft for one. Since only STN is
softened, the whole target arrives as probabilities and the CE is written out
explicitly. The reduction is ``mean`` over batch and spatial positions, matching
``F.cross_entropy``'s default, so the two paths stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import crop_spec as cs
from losses.dice_ce import DiceCELoss, DiceCELossConfig


@dataclass
class InterfaceSoftDiceCELossConfig:
    """Weights and numerics. Defaults reproduce the Baseline v1 objective."""

    ce_weight: float = 1.0
    dice_weight: float = 1.0
    smooth: float = 1e-5


class InterfaceSoftDiceCELoss(nn.Module):
    """CE + foreground soft Dice, accepting either a hard or a soft target."""

    def __init__(self, config: InterfaceSoftDiceCELossConfig | None = None) -> None:
        super().__init__()
        self.config = config or InterfaceSoftDiceCELossConfig()
        # Composition rather than inheritance: the hard-target path is produced
        # by the baseline's own code, so it cannot drift away from it.
        #
        # Built with the FULL weights, not just the Dice weight, because the hard
        # path delegates the entire call to it -- CE included. Constructing it the
        # way Experiment C does (ce_weight=0.0) would silently drop the CE term
        # from every hard-target run.
        self._baseline = DiceCELoss(DiceCELossConfig(
            ce_weight=self.config.ce_weight,
            dice_weight=self.config.dice_weight,
            smooth=self.config.smooth,
        ))

    # ------------------------------------------------------------------ #
    # soft path
    # ------------------------------------------------------------------ #

    def per_case_per_class_dice_soft(
        self, logits: torch.Tensor, soft_target: torch.Tensor
    ) -> torch.Tensor:
        """Soft Dice as an ``(N, n_dice_classes)`` matrix, soft-target form.

        Reduced over **D, H, W only**, never over the batch -- batch pooling would
        weight each case by its target volume, and evaluation is a per-case mean.
        """
        # Forced to fp32 deliberately. Under AMP the logits arrive as fp16, and
        # the reductions below each sum ~295k values; letting those accumulate in
        # fp16 risks overflow. The baseline path is untouched by this because it
        # runs through DiceCELoss, not through here.
        probabilities = F.softmax(logits.float(), dim=1)
        per_class: list[torch.Tensor] = []

        for class_id in cs.FOREGROUND_CLASSES:
            prob_c = probabilities[:, class_id]                 # (N, D, H, W)
            # Contiguous so the reduction runs over the same layout as the hard
            # path's `(target == class_id)` mask, which is freshly allocated.
            true_c = soft_target[:, class_id].to(prob_c.dtype).contiguous()
            spatial_dims = tuple(range(1, prob_c.ndim))         # (1, 2, 3)
            numerator = 2.0 * torch.sum(prob_c * true_c, dim=spatial_dims)
            denominator = (
                torch.sum(prob_c, dim=spatial_dims)
                + torch.sum(true_c, dim=spatial_dims)
            )
            per_class.append(
                (numerator + self.config.smooth) / (denominator + self.config.smooth)
            )

        return torch.stack(per_class, dim=1)

    def soft_ce(self, logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
        """``-sum_c q_c log softmax(logits)_c``, averaged over batch x spatial.

        The reduction is ``mean`` over batch and spatial positions, matching
        ``F.cross_entropy``'s default so the two paths stay comparable. Computed
        in fp32 for the reason given in ``per_case_per_class_dice_soft``.
        """
        log_probabilities = F.log_softmax(logits.float(), dim=1)
        return -(soft_target.float() * log_probabilities).sum(dim=1).mean()

    # ------------------------------------------------------------------ #
    # dispatch
    # ------------------------------------------------------------------ #

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(total_loss, components)``.

        ``target`` is either ``(N, D, H, W)`` integer labels -- in which case the
        baseline criterion handles the whole call -- or ``(N, C, D, H, W)`` float
        probabilities.
        """
        if logits.dim() != 5:
            raise ValueError(f"expected logits (N,C,D,H,W), got {tuple(logits.shape)}")

        if target.dim() == 4:
            # Hard target: the baseline's own criterion, called directly.
            return self._baseline(logits, target)

        if target.dim() != 5:
            raise ValueError(
                f"expected target (N,D,H,W) hard or (N,C,D,H,W) soft, got "
                f"{tuple(target.shape)}"
            )
        if target.shape != logits.shape:
            raise ValueError(
                f"soft target shape {tuple(target.shape)} != logits shape "
                f"{tuple(logits.shape)}"
            )
        if not target.is_floating_point():
            raise ValueError(f"soft target must be floating point, got {target.dtype}")

        ce = self.soft_ce(logits, target)
        dice = 1.0 - self.per_case_per_class_dice_soft(logits, target).mean()
        total = self.config.ce_weight * ce + self.config.dice_weight * dice

        return total, {
            "loss": total.detach(),
            "ce": ce.detach(),
            "dice": dice.detach(),
        }

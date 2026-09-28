"""Experiment B loss: CrossEntropy + a class-*hybrid* foreground overlap term.

Baseline v1 uses one symmetric soft Dice over STN/SN/RN. The baseline error
analysis found STN is the odd one out -- it is systematically **over-segmented**
(predicted/GT volume ratio 1.134, FP 2427 > FN 1792, precision 0.6999 < recall
0.7687), while SN (0.966) and RN (1.035) have no systematic volume bias. So this
loss leaves SN and RN exactly as they were and re-weights only STN, toward
"predict less of it":

    overlap = mean over {STN, SN, RN} of  per-case overlap index
              STN -> asymmetric Tversky index, alpha=0.6, beta=0.4
              SN  -> soft Dice  } identical implementation to Baseline v1
              RN  -> soft Dice  }

    total   = ce_weight * CE + overlap_weight * (1 - overlap)

PARAMETER CONVENTION -- READ THIS BEFORE CHANGING ANYTHING
---------------------------------------------------------
**alpha weights FALSE POSITIVES. beta weights FALSE NEGATIVES.**

        Tversky = (TP + eps) / (TP + alpha*FP + beta*FN + eps)

This is the OPPOSITE naming from Salehi et al.'s focal-Tversky paper, where the
symbols are conventionally swapped (their alpha sits on FN). The two conventions
are silently interchangeable -- both produce a number in [0, 1] and both "work" --
so a mix-up would not raise, it would just quietly move the loss in the wrong
direction. That is why the convention is pinned by tests in
``tests/test_stn_tversky_v1.py`` rather than only by this docstring: the tests
assert that raising alpha makes FP hurt more, and that swapping alpha/beta
reverses which error is penalised.

With alpha=0.6 > beta=0.4 an extra false positive costs more than an extra false
negative, which is the whole point of the experiment: push the STN boundary in.

Per-case reduction
------------------
Every index is reduced over **D, H, W only** and only then averaged over
(case, class). Pooling voxels across the batch would weight each case by its
target volume, while evaluation is an unweighted per-case mean -- the two would
disagree, and training would optimise something other than what is reported.
This mirrors ``losses.dice_ce`` exactly; see the note there.

Why this module *composes* rather than re-implements Dice
--------------------------------------------------------
:class:`HybridOverlapCELoss` holds an internal :class:`~losses.dice_ce.DiceCELoss`
and delegates the SN/RN columns to it, so those two columns are the baseline's own
code producing bit-identical values. Re-deriving the same formula here would let
the two drift apart while every test still passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import crop_spec as cs
from losses.dice_ce import DiceCELoss, DiceCELossConfig

#: Class id for STN. Derived from the frozen ``data.crop_spec.CLASS_NAMES``
#: mapping rather than restated as a bare literal, so the id keeps exactly one
#: source of truth. (crop_spec is frozen and must not gain constants for this.)
STN: Final[int] = next(c for c, name in cs.CLASS_NAMES.items() if name == "STN")


def tversky_index_from_counts(
    tp: torch.Tensor | float,
    fp: torch.Tensor | float,
    fn: torch.Tensor | float,
    alpha: float,
    beta: float,
    smooth: float = 1e-5,
) -> torch.Tensor | float:
    """``(TP + smooth) / (TP + alpha*FP + beta*FN + smooth)`` -- the core formula.

    ``alpha`` weights FALSE POSITIVES and ``beta`` weights FALSE NEGATIVES. Kept
    as a free function over *counts* so the convention can be tested directly,
    without a softmax or a model in the way.

    Note on ``smooth``: written this way, the index equals the baseline's soft
    Dice when ``alpha = beta = 0.5`` only in the ``smooth -> 0`` limit. The
    baseline uses ``(2*TP + eps) / (2*TP + FP + FN + eps)``, which in this
    normalisation is ``smooth = eps/2``; the two differ by ``O(eps)``, about
    1e-8 on the volumes here. The formula above is the one the experiment
    protocol specifies, so it is the one implemented -- the divergence is
    recorded by a test rather than papered over.
    """
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


@dataclass
class HybridOverlapCELossConfig:
    """Weights and numerics for :class:`HybridOverlapCELoss`."""

    ce_weight: float = 1.0
    #: weight of the whole hybrid overlap term (STN + SN + RN), not of Dice alone
    overlap_weight: float = 1.0
    #: class that gets the asymmetric Tversky index (default STN = 1)
    tversky_class: int = STN
    #: weight on FALSE POSITIVES for the Tversky class
    tversky_alpha: float = 0.6
    #: weight on FALSE NEGATIVES for the Tversky class
    tversky_beta: float = 0.4
    smooth: float = 1e-5
    #: classes that keep the baseline soft Dice. ``None`` -> every foreground
    #: class except ``tversky_class``, which is the only value that can produce
    #: the (STN, SN, RN) column order the loss assumes.
    dice_classes: Sequence[int] | None = None

    def resolved_dice_classes(self) -> tuple[int, ...]:
        if self.dice_classes is not None:
            return tuple(self.dice_classes)
        return tuple(c for c in cs.FOREGROUND_CLASSES if c != self.tversky_class)


class HybridOverlapCELoss(nn.Module):
    """Cross-entropy plus per-class hybrid overlap (Tversky on STN, Dice on SN/RN).

    Returns ``(total_loss, components)`` like :class:`~losses.dice_ce.DiceCELoss`,
    so it is a drop-in for the training loop.
    """

    def __init__(self, config: HybridOverlapCELossConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridOverlapCELossConfig()

        self.dice_classes: tuple[int, ...] = self.config.resolved_dice_classes()
        # The overlap matrix is built by concatenating the Tversky columns in
        # front of the Dice columns, so column j only means the class we think it
        # does if (tversky_class, *dice_classes) == FOREGROUND_CLASSES. A mean is
        # order-invariant, which means a wrong order would leave the *loss*
        # unchanged and only silently scramble the per-class diagnostics -- so it
        # is checked here rather than left to a comment.
        expected = (self.config.tversky_class, *self.dice_classes)
        if expected != tuple(cs.FOREGROUND_CLASSES):
            raise ValueError(
                f"overlap classes must partition the foreground classes in the "
                f"order {tuple(cs.FOREGROUND_CLASSES)}; got tversky_class="
                f"{self.config.tversky_class} and dice_classes="
                f"{tuple(self.dice_classes)} -> {expected}"
            )
        if self.config.tversky_class not in cs.FOREGROUND_CLASSES:
            raise ValueError(f"tversky_class {self.config.tversky_class} is not a "
                             f"foreground class {tuple(cs.FOREGROUND_CLASSES)}")
        if min(self.config.tversky_alpha, self.config.tversky_beta) < 0:
            raise ValueError("tversky alpha/beta must be non-negative")

        # Composition, not inheritance: the SN/RN columns are produced by the
        # baseline's own code path. `ce_weight=0` because the CE term of the
        # inner module is never used -- this module computes CE itself.
        self._dice = DiceCELoss(DiceCELossConfig(
            ce_weight=0.0,
            dice_weight=1.0,
            dice_classes=self.dice_classes,
            smooth=self.config.smooth,
        ))

    # ------------------------------------------------------------------ #
    # overlap indices
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(logits: torch.Tensor, target: torch.Tensor) -> None:
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

    def per_case_per_class_tversky(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Asymmetric Tversky index as an ``(N, n_tversky_classes)`` matrix.

        ``tp = sum(p*g)``, ``fp = sum(p*(1-g))``, ``fn = sum((1-p)*g)`` on the
        softmax probabilities, summed over **D, H, W only** -- never over the
        batch (see the module docstring).
        """
        self._validate(logits, target)
        probabilities = F.softmax(logits, dim=1)
        per_class: list[torch.Tensor] = []

        for class_id in (self.config.tversky_class,):
            prob_c = probabilities[:, class_id]              # (N, D, H, W)
            true_c = (target == class_id).to(prob_c.dtype)   # (N, D, H, W)
            spatial_dims = tuple(range(1, prob_c.ndim))      # (1, 2, 3)
            tp = torch.sum(prob_c * true_c, dim=spatial_dims)
            fp = torch.sum(prob_c * (1.0 - true_c), dim=spatial_dims)
            fn = torch.sum((1.0 - prob_c) * true_c, dim=spatial_dims)
            per_class.append(tversky_index_from_counts(
                tp, fp, fn,
                alpha=self.config.tversky_alpha,
                beta=self.config.tversky_beta,
                smooth=self.config.smooth,
            ))

        if not per_class:
            return logits.new_zeros((logits.shape[0], 0))
        return torch.stack(per_class, dim=1)

    def per_case_per_class_dice(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Baseline v1 soft Dice, ``(N, n_dice_classes)``. Delegated verbatim.

        Deliberately a one-line pass-through: this is the guarantee that the
        SN/RN term is the baseline's implementation rather than a copy of it.
        """
        return self._dice.per_case_per_class_dice(logits, target)

    def per_case_per_class_overlap(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """``(N, n_foreground_classes)`` matrix, columns in ``FOREGROUND_CLASSES``
        order -- i.e. (STN, SN, RN) for the default configuration."""
        return torch.cat(
            [self.per_case_per_class_tversky(logits, target),
             self.per_case_per_class_dice(logits, target)],
            dim=1,
        )

    def overlap_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``1 - mean`` over the (cases x classes) overlap matrix.

        The two-step reduction is the point: compute each (case, class) index,
        then macro-average. It is written as ``1 - matrix.mean()`` rather than as
        the mean of three separate ``1 - index`` terms because those differ in
        floating point; this is the form the protocol specifies.
        """
        matrix = self.per_case_per_class_overlap(logits, target)
        if matrix.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return 1.0 - matrix.mean()

    # ------------------------------------------------------------------ #
    # total loss
    # ------------------------------------------------------------------ #

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(total_loss, components)``.

        ``logits``: (N, C, D, H, W) raw scores. ``target``: (N, D, H, W) labels.
        """
        self._validate(logits, target)

        # Written out rather than delegated: with `weight=None` this is literally
        # the baseline's CE expression, and keeping it explicit makes that
        # checkable by a test.
        ce = F.cross_entropy(logits, target.long(), weight=None)
        overlap = self.overlap_loss(logits, target)
        total = self.config.ce_weight * ce + self.config.overlap_weight * overlap

        with torch.no_grad():
            tversky_matrix = self.per_case_per_class_tversky(logits, target)
            dice_matrix = self.per_case_per_class_dice(logits, target)
        per_class: dict[str, torch.Tensor] = {}
        # Iterated in FOREGROUND_CLASSES order so the keys come out as
        # (stn_tversky_loss, sn_dice_loss, rn_dice_loss), matching the per-class
        # ordering used everywhere else in the repo. Note the STN term is named
        # `stn_tversky_loss`, never `stn_dice_loss`: it is not a Dice.
        for class_id in cs.FOREGROUND_CLASSES:
            name = cs.CLASS_NAMES[class_id].lower()
            if class_id == self.config.tversky_class:
                per_class[f"{name}_tversky_loss"] = (
                    1.0 - tversky_matrix[:, 0].mean()).detach()
            else:
                column = self.dice_classes.index(class_id)
                per_class[f"{name}_dice_loss"] = (
                    1.0 - dice_matrix[:, column].mean()).detach()

        components = {"loss": total.detach(), "ce": ce.detach(),
                      "overlap": overlap.detach()}
        components.update(per_class)

        return total, components

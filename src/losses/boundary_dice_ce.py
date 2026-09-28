"""Experiment C loss: Baseline v1 loss + an STN signed-distance boundary term.

Baseline v1 is CE + foreground soft Dice over STN/SN/RN. Experiments A
(augmentation) and B (asymmetric Tversky on STN) both successfully moved the
predicted/GT volume ratio towards 1.0 but paid for it with recall, and neither
improved STN Dice -- the error analysis had already shown STN's dominant error is
*boundary* error (Dice vs HD95 correlate at r = -0.656), not a volume bias. So
this loss keeps the baseline objective completely intact and adds a term that
supervises the boundary directly:

    total = ce_weight * CE
          + dice_weight * foreground soft Dice (STN, SN, RN -- baseline's own)
          + boundary_weight * mean(p_STN * phi)

where ``phi`` is the signed distance map of the *ground-truth* STN mask, negative
inside and positive outside (see ``scripts/build_boundary_cache.py``).

Why the sign works out
----------------------
    inside GT   : phi < 0  ->  raising p_STN LOWERS the term
    outside GT  : phi > 0  ->  raising p_STN RAISES the term
    near the boundary : |phi| ~ 0  ->  the voxel barely matters

and because ``phi`` is the *distance to* the boundary rather than an indicator of
it, voxels further from the boundary carry proportionally more weight. That is
the property a plain Dice/CE term does not have: both of those weight every
misclassified voxel equally regardless of how far it sits from the true edge.

Relationship to the published boundary loss
-------------------------------------------
This is Kervadec et al.'s boundary loss (the level-set form
``integral of phi_G * s_theta``) with two deviations the experiment protocol fixes
deliberately: the integral is replaced by a **mean** over voxels, and distances
are **clipped to +/-10 mm and divided by 10**. The clipping matters here -- this
crop is 64 mm across while STN has an equivalent radius of ~3.4 mm, so without it
the term would be dominated by voxels tens of millimetres away. Measured on a
real case at initialisation, the term still supplies ~1.9% of the total gradient
norm at ``boundary_weight=0.10``, against ~2.4% for the whole Dice term.

Note that the mean-vs-integral choice is a pure scale factor (the crop is the same
size for every case), so it is absorbed by ``boundary_weight``; it is not a
modelling difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import crop_spec as cs
from losses.dice_ce import DiceCELoss, DiceCELossConfig

#: Class id for STN. ``data.crop_spec`` is frozen and carries no STN constant, so
#: it is derived from that module's own CLASS_NAMES rather than restated as a
#: bare literal -- there is still only one place the mapping is written down.
STN: Final[int] = next(c for c, name in cs.CLASS_NAMES.items() if name == "STN")


@dataclass
class BoundaryDiceCELossConfig:
    """Weights and numerics for :class:`BoundaryDiceCELoss`."""

    ce_weight: float = 1.0
    #: weight of the foreground soft-Dice term (unchanged from Baseline v1)
    dice_weight: float = 1.0
    #: weight of the STN signed-distance term. 0.0 recovers Baseline v1 exactly.
    boundary_weight: float = 0.10
    #: class whose boundary is supervised
    boundary_class: int = STN
    smooth: float = 1e-5


class BoundaryDiceCELoss(nn.Module):
    """CE + Baseline foreground soft Dice + STN signed-distance boundary term.

    ``forward`` takes an optional third argument. CE and the Dice term are the
    baseline's own computations, so the only thing under test is the new term.
    """

    def __init__(self, config: BoundaryDiceCELossConfig | None = None) -> None:
        super().__init__()
        self.config = config or BoundaryDiceCELossConfig()
        if self.config.boundary_class not in cs.FOREGROUND_CLASSES:
            raise ValueError(
                f"boundary_class {self.config.boundary_class} is not a foreground "
                f"class {tuple(cs.FOREGROUND_CLASSES)}"
            )
        # Composition rather than inheritance: the Dice term is produced by the
        # baseline's own code path, so it cannot drift from it.
        self._dice = DiceCELoss(DiceCELossConfig(
            ce_weight=0.0,
            dice_weight=1.0,
            smooth=self.config.smooth,
        ))

    # ------------------------------------------------------------------ #
    # the new term
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.dim() != 5 or target.dim() != 4:
            raise ValueError(
                f"expected logits (N,C,D,H,W) and target (N,D,H,W), got "
                f"{tuple(logits.shape)} and {tuple(target.shape)}"
            )
        if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
            raise ValueError(
                f"batch/spatial mismatch: logits {tuple(logits.shape)} vs "
                f"target {tuple(target.shape)}"
            )

    def per_case_per_class_dice(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Baseline v1 soft Dice, ``(N, n_classes)``. Delegated verbatim.

        A one-line pass-through on purpose: it is the guarantee that the Dice
        term is the baseline's implementation rather than a copy of it.
        """
        return self._dice.per_case_per_class_dice(logits, target)

    def boundary_loss(self, logits: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """``mean`` over voxels of ``p_STN * phi``.

        Reduced over D, H, W and the batch together, exactly as written in the
        protocol. Unlike the Dice term this is not a per-case ratio, so pooling
        over the batch is not a weighting distortion here -- every case
        contributes its own fixed-size crop, so the mean is already an unweighted
        average over cases.
        """
        if phi.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"boundary map shape {tuple(phi.shape)} does not match logits "
                f"spatial shape (N,D,H,W)={tuple(logits.shape[:1] + logits.shape[2:])}"
            )
        probability = F.softmax(logits, dim=1)[:, self.config.boundary_class]
        return (probability * phi.to(probability.dtype)).mean()

    # ------------------------------------------------------------------ #
    # total loss
    # ------------------------------------------------------------------ #

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        boundary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(total_loss, components)``.

        ``boundary`` is the STN signed-distance map, (N, D, H, W), negative
        inside the GT. It is required whenever ``boundary_weight`` is non-zero:
        training without it would silently drop the one thing this experiment
        exists to test, and the run would look like a clean null result.
        """
        self._validate(logits, target)
        if self.config.boundary_weight != 0.0 and boundary is None:
            raise ValueError(
                "boundary map is required when boundary_weight != 0; got None. "
                "Check that the boundary cache is configured and loaded."
            )

        ce = F.cross_entropy(logits, target.long(), weight=None)
        dice = self._dice.soft_dice_loss(logits, target)

        if boundary is None:
            boundary_term = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            boundary_term = self.boundary_loss(logits, boundary)

        total = (self.config.ce_weight * ce
                 + self.config.dice_weight * dice
                 + self.config.boundary_weight * boundary_term)

        return total, {
            "loss": total.detach(),
            "ce": ce.detach(),
            "dice": dice.detach(),
            "boundary": boundary_term.detach(),
        }

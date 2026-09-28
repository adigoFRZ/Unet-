"""Experiment G1-MC: per-case mass-conserving interface soft target.

PROTOTYPE / AUDIT ONLY. Not wired into the trainer, not used by any config.

WHAT THIS CHANGES, AND WHAT IT DELIBERATELY DOES NOT
----------------------------------------------------
The G1 audit measured a systematic **+4.104% net STN target mass** in every one
of the 160 development-train cases: the interface ramp is symmetric *per voxel*,
but a thin structure has more interface-adjacent voxels outside it than inside
(25,519 vs 19,192, a factor of 1.33), so the outside gain beats the inside loss.

This module removes that bias by rescaling the **outside** shell only, per case::

    gamma_i = R_i / A_i
        R_i = sum over GT-STN voxels of (1 - q_STN)      mass taken off inside
        A_i = sum over non-STN voxels  of      q_STN     mass placed outside

    q'_STN = q_STN                     where GT is STN     (UNCHANGED)
    q'_STN = gamma_i * q_STN           elsewhere, in the G1 soft band

so the delivered outside mass becomes ``gamma_i * A_i == R_i`` and the case total
is exactly ``H_i``::

    sum q'_STN = (H_i - R_i) + R_i = H_i

Everything geometric is inherited untouched from
:mod:`data.interface_soft_target`: the interface definition, the signed distance,
the band, the ramp, the face-count tie rule, the class mapping, the crop and the
axis order. Only the *magnitude* of the outside shell moves.

Outside scaling is uniform within a case, so ``q'_STN`` stays monotonic in
``|d_surface|`` -- the two outside levels keep their order.

The mass removed from an outside voxel is returned to **that voxel's own hard
class** and nowhere else. A non-STN voxel's residual is never redistributed
between anatomical classes.

``gamma`` is computed from the case's own mass balance and from nothing else. It
is deliberately not a parameter: there is no argument, config key or default that
can set it, because tuning it would break the exact conservation this module
exists to provide.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from data import crop_spec as cs
from data.interface_soft_target import (
    BAND_MM,
    RESIDUAL_CLASSES,
    STN,
    interface_soft_target,
)


def conservation_terms(q_stn: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """``(removed, added, gamma)`` for one case.

    ``removed`` is the STN target mass taken off voxels that are STN in the
    ground truth; ``added`` is the mass placed on voxels that are not.
    """
    removed = float((1.0 - q_stn[mask]).sum())
    added = float(q_stn[~mask].sum())
    if added <= 0.0:
        raise ValueError(
            f"no STN target mass outside the mask (added={added}); gamma is "
            f"undefined. The case has {int(mask.sum())} GT STN voxels."
        )
    return removed, added, removed / added


def mass_conserving_soft_target(
    label: np.ndarray,
    band_mm: float = BAND_MM,
    expected_shape: tuple[int, int, int] | None = cs.CROP_SHAPE_DHW,
) -> np.ndarray:
    """The per-case mass-conserving ``(C, D, H, W)`` soft target.

    Guarantees, per case: ``sum_v q'_STN(v) == sum_v 1[y(v) == STN]``, every value
    in ``[0, 1]``, and ``sum_c q'_c == 1`` at every voxel -- all to floating-point
    accumulation accuracy. Voxels outside the band stay bit-exactly one-hot.
    """
    base = interface_soft_target(
        label, band_mm=band_mm, expected_shape=expected_shape
    ).astype(np.float64)

    mask = label == STN
    q_stn = base[STN]
    removed, added, gamma = conservation_terms(q_stn, mask)

    # Inside the structure the definition is untouched; outside it is scaled so
    # the case as a whole delivers exactly the mass it takes away.
    q_new = np.where(mask, q_stn, gamma * q_stn)

    target = base.copy()
    target[STN] = q_new
    reduction = q_stn - q_new          # zero wherever GT is STN
    for class_id in RESIDUAL_CLASSES:
        target[class_id] += np.where((~mask) & (label == class_id), reduction, 0.0)

    if not np.isfinite(target).all():
        raise ValueError("mass-conserving target contains NaN/Inf")
    if target.min() < -1e-12 or target.max() > 1.0 + 1e-12:
        raise ValueError(
            f"target left [0, 1]: min {target.min():.3e}, max {target.max():.3e}"
        )
    return np.ascontiguousarray(target, dtype=np.float32)

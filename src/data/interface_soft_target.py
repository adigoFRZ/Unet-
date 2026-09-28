"""Experiment G1: interface-centered soft boundary target for STN.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is **not** partial-volume ground truth. The dataset carries no sub-voxel
annotation, so no true occupancy fraction exists to supervise against. What this
module builds is a *modelling assumption*:

    the STN/non-STN boundary lies on the shared face between a 6-connected pair
    of voxels that differ in class,

which places it exactly half a voxel spacing from each of those two centres.
Nothing measures that; it is a definition imposed on the discrete mask.

The resulting target encodes a training hypothesis: that supervising the shells
either side of that interface at 0.75 / 0.25 rather than 1 / 0 represents the
boundary better than a hard one-hot label does, and may therefore reduce the
one-voxel STN extent error Experiment G0 measured. Whether it does is an
empirical question this experiment exists to answer.

WHY NOT THE OBVIOUS SIGNED DISTANCE
-----------------------------------
The natural-looking ``edt(~mask) - edt(mask)`` puts its zero level on the *voxel
centres*, not on the interface between them. On this grid the smallest non-zero
``|d|`` it can produce is exactly one in-plane spacing (0.6666667 mm), so a band
of half-width ``w = 0.6666667`` would contain **no voxel centres at all** and the
"soft" target would collapse onto the hard one-hot label -- measured: 0 voxels
differ across all 40 development-val cases. The interface-centred distance below
puts the zero level on the face instead, which moves the nearest shell to ``s/2``
and makes the band non-degenerate.

GEOMETRY
--------
Distances are exact, not approximated. A face can only lie within ``band`` of a
voxel centre if its normal is in-plane *and* its D-offset is zero, because the
smallest through-plane contribution is already ``s_d/2 = 1.0 mm > band``. That
leaves 12 candidate faces per voxel (2 W-faces x 3 H-offsets, and the transpose),
and the achievable ``|d|`` below 1 mm are exactly two values::

    s/2                 = 0.333333350          in-plane adjacent
    hypot(s/2, s/2)     = 0.471404544          in-plane diagonal

Both are *constants* here because ``s_h == s_w``, so "which faces are nearest" is
decided by exact ``==`` comparison between computed constants -- no floating
point tolerance is involved anywhere. ``d_surface`` saturates at ``+/-s_d/2``,
which is the true value for a D-adjacent voxel and harmless beyond it because the
target is already saturated there.

RESIDUAL ALLOCATION
-------------------
``q_STN`` is the ramp value; the remaining mass ``r = 1 - q_STN`` goes to the
non-STN classes, allocated by **face count** among the nearest equidistant faces,
never by a class priority and never by defaulting to background::

    n_c(v) = number of nearest (equidistant) faces whose non-STN side is class c
    q_c(v) = r(v) * n_c(v) / sum_j n_j(v)

For a voxel whose hard label is already non-STN the residual goes to that label,
which is what the ground truth asserts about the tissue at that location.

Because every in-band face is either W-normal (spanning ``s_h x s_d``) or
H-normal (spanning ``s_w x s_d``) and ``s_h == s_w``, **all in-band faces have
identical physical area**, so "by face count" and "by physical face area" are
exactly equivalent in this experiment. D-normal faces never enter the band.

This allocation is local discrete-interface bookkeeping. It is not an anatomical
partial-volume fraction and must not be reported as one.
"""

from __future__ import annotations

from math import hypot
from typing import Final

import numpy as np

from data import crop_spec as cs

#: Class id for STN. ``data.crop_spec`` is frozen and carries no STN constant, so
#: it is derived from that module's own CLASS_NAMES rather than restated as a
#: bare literal -- there is still only one place the mapping is written down.
STN: Final[int] = next(c for c, name in cs.CLASS_NAMES.items() if name == "STN")

#: Non-STN classes the residual may be allocated to, in a fixed order.
#:
#: Built from the full class range rather than from ``CLASS_NAMES``: background
#: (class 0) is deliberately absent from ``CLASS_NAMES`` because it is not a
#: segmentation *target*, but it is very much a residual class here -- it is in
#: fact the largest one. Deriving this from ``CLASS_NAMES`` silently dropped it.
RESIDUAL_CLASSES: Final[tuple[int, ...]] = tuple(
    c for c in range(int(cs.NUM_CLASSES)) if c != STN
)

#: Pre-registered half-width of the linear ramp, in millimetres: one in-plane
#: voxel spacing. Fixed before training; never adjusted from validation data.
BAND_MM: Final[float] = 0.6666667


def _faces(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane interface faces, recorded at the index of their *high* voxel.

    ``face_w[d, h, w]`` is the face between ``w-1`` and ``w``; ``face_h`` is the
    same for the H axis. Both keep the label-map shape so a candidate can be
    gathered with a single shift.
    """
    d, h, w = mask.shape
    face_w = np.zeros((d, h, w), dtype=bool)
    face_w[:, :, 1:] = mask[:, :, :-1] != mask[:, :, 1:]
    face_h = np.zeros((d, h, w), dtype=bool)
    face_h[:, 1:, :] = mask[:, :-1, :] != mask[:, 1:, :]
    return face_w, face_h


def _other_class(label: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """Class of the non-STN voxel sharing each face, indexed like the face array.

    ``axis`` is 2 for W-normal faces and 1 for H-normal faces. Where the indexed
    voxel is itself non-STN its own label is the answer; otherwise the answer is
    the neighbour on the low side. Exactly one of the two is non-STN wherever a
    face exists, so this is unambiguous.
    """
    other = np.zeros(label.shape, dtype=np.int8)
    outside = ~mask
    other[outside] = label[outside]

    low = np.zeros_like(label)
    if axis == 1:
        low[:, 1:, :] = label[:, :-1, :]
    else:
        low[:, :, 1:] = label[:, :, :-1]

    shape = [1, 1, 1]
    shape[axis] = label.shape[axis]
    has_low = np.arange(label.shape[axis]).reshape(shape) > 0

    take_low = mask & has_low
    other[take_low] = low[take_low]
    return other


def _candidate_offsets() -> list[tuple[float, bool, int, int]]:
    """The 12 in-plane candidates as ``(distance_mm, is_w_face, dw, dh)``."""
    _, s_h, s_w = cs.SPACING_DHW_MM
    out: list[tuple[float, bool, int, int]] = []
    for dw in (0, 1):                       # W-normal faces
        for dh in (-1, 0, 1):
            dx = abs(0.5 - dw) * s_w
            dy = max(0, abs(dh) - 0.5) * s_h
            out.append((dx if dy == 0.0 else hypot(dx, dy), True, dw, dh))
    for dh in (0, 1):                       # H-normal faces
        for dw in (-1, 0, 1):
            dx = max(0, abs(dw) - 0.5) * s_w
            dy = abs(0.5 - dh) * s_h
            out.append((dy if dx == 0.0 else hypot(dx, dy), False, dw, dh))
    return out


def _candidates() -> list[tuple[float, bool, int, int]]:
    """Module-level cache: the candidate set is a constant of the frozen crop."""
    global _CANDIDATES
    if _CANDIDATES is None:
        _CANDIDATES = _candidate_offsets()
    return _CANDIDATES


_CANDIDATES: list[tuple[float, bool, int, int]] | None = None


def _gather(array: np.ndarray, dw: int, dh: int, fill: int | bool) -> np.ndarray:
    """``out[v] = array[v + (0, dh, dw)]``, filled where that falls off the edge."""
    h, w = array.shape[1], array.shape[2]
    out = np.full(array.shape, fill, dtype=array.dtype)
    h_lo, h_hi = max(0, -dh), h - max(0, dh)
    w_lo, w_hi = max(0, -dw), w - max(0, dw)
    out[:, h_lo:h_hi, w_lo:w_hi] = array[:, h_lo + dh:h_hi + dh, w_lo + dw:w_hi + dw]
    return out


def nearest_face_counts(
    label: np.ndarray,
    expected_shape: tuple[int, int, int] | None = cs.CROP_SHAPE_DHW,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """``(d_surface, {class: nearest-face count})`` for one ``(D, H, W)`` label map.

    ``d_surface`` is negative inside STN and positive outside, exact for
    ``|d| < s_d/2`` and saturated at ``+/-s_d/2`` beyond. The counts enumerate the
    faces achieving that minimum, which is what the residual allocation divides.

    ``expected_shape`` guards against being handed a mis-cropped or wrongly
    transposed volume, which is the failure this whole module is most exposed to.
    The geometry itself is shape-agnostic, so tests pass ``None`` to exercise it
    on small synthetic masks.
    """
    if label.ndim != 3:
        raise ValueError(f"expected a (D, H, W) label map, got shape {label.shape}")
    if expected_shape is not None and tuple(label.shape) != tuple(expected_shape):
        raise ValueError(
            f"label shape {tuple(label.shape)} != expected {tuple(expected_shape)}"
        )

    mask = label == STN
    s_d = cs.SPACING_DHW_MM[0]
    face_w, face_h = _faces(mask)
    faces = {True: face_w, False: face_h}
    others = {
        True: _other_class(label, mask, axis=2),
        False: _other_class(label, mask, axis=1),
    }

    # Saturation value: the true distance for a D-adjacent voxel. Anything past
    # it is already saturated in the target, so it never has to be known.
    d_surface_mag = np.full(label.shape, s_d / 2.0, dtype=np.float64)
    gathered: list[tuple[float, np.ndarray, np.ndarray]] = []
    for distance, is_w, dw, dh in _candidates():
        exists = _gather(faces[is_w], dw, dh, False)
        other = _gather(others[is_w], dw, dh, 0)
        gathered.append((distance, exists, other))
        closer = exists & (distance < d_surface_mag)
        d_surface_mag = np.where(closer, distance, d_surface_mag)

    counts = {c: np.zeros(label.shape, dtype=np.float64) for c in RESIDUAL_CLASSES}
    for distance, exists, other in gathered:
        # Exact comparison -- both sides are one of the two analytic constants.
        # A tolerance here would merge genuinely different faces into a tie.
        on_nearest = exists & (distance == d_surface_mag)
        for c in RESIDUAL_CLASSES:
            counts[c] += (on_nearest & (other == c)).astype(np.float64)

    d_surface = np.where(mask, -d_surface_mag, d_surface_mag)
    return d_surface, counts


#: Voxels that can be softened lie within two in-plane steps of an STN voxel (a
#: face is adjacent to one, and a softened voxel is within one step of the face).
#: The window below pads that by one more on every axis for safety. Proved
#: equivalent to the whole-volume computation on all 40 development-val cases.
WINDOW_MARGIN: Final[int] = 3


def _stn_window(mask: np.ndarray, margin: int = WINDOW_MARGIN) -> tuple[slice, ...]:
    """Smallest box around the STN voxels, padded by ``margin`` and clamped."""
    index = np.argwhere(mask)
    lo = np.maximum(index.min(axis=0) - margin, 0)
    hi = np.minimum(index.max(axis=0) + margin + 1, np.array(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def soft_target_from_distance(
    label: np.ndarray,
    d_surface: np.ndarray,
    counts: dict[int, np.ndarray],
    band_mm: float = BAND_MM,
) -> np.ndarray:
    """Assemble ``(C, D, H, W)`` from a distance map and its nearest-face counts.

    Split out so the windowed and whole-volume paths can be compared directly:
    this function has no notion of a crop, so feeding it whole-volume inputs must
    reproduce whole-volume results exactly.
    """
    n_classes = int(cs.NUM_CLASSES)
    mask = label == STN

    # The linear ramp, centred on the interface. Symmetric about 0.5 by
    # construction: q(-d) + q(+d) == 1.
    q_stn = np.clip(0.5 - d_surface / (2.0 * band_mm), 0.0, 1.0)
    residual = 1.0 - q_stn
    total = sum(counts[c] for c in RESIDUAL_CLASSES)
    # The split applies only to STN voxels. Guarding on `mask` is load-bearing:
    # without it a non-STN voxel would get 1.0 for its own class *and* a non-zero
    # split on the others, so sum_c q_c would exceed 1.
    split_applies = mask & (total > 0)

    target = np.zeros((n_classes, *label.shape), dtype=np.float64)
    target[STN] = q_stn
    for c in RESIDUAL_CLASSES:
        split = np.where(
            split_applies, counts[c] / np.where(total > 0, total, 1.0), 0.0
        )
        # GT is not STN: the residual belongs to that label.
        own = (~mask) & (label == c)
        target[c] = residual * np.where(own, 1.0, split)
    return target


def interface_soft_target(
    label: np.ndarray,
    band_mm: float = BAND_MM,
    expected_shape: tuple[int, int, int] | None = cs.CROP_SHAPE_DHW,
) -> np.ndarray:
    """The ``(C, D, H, W)`` interface-centred soft target for one case.

    Guaranteed: every value lies in ``[0, 1]`` and ``sum_c q_c == 1`` at every
    voxel, to floating-point exactness. Voxels outside the band come back exactly
    one-hot on their hard label.

    The geometry is computed on a small window around the STN rather than the
    whole crop: the structure occupies ~200 voxels of a 294912-voxel crop, and
    the whole-volume path costs ~105 ms/case against ~3 ms windowed. Voxels
    outside the window cannot be softened, so they take the one-hot baseline
    exactly.
    """
    if band_mm <= 0.0:
        raise ValueError(f"band_mm must be positive, got {band_mm}")
    label = np.asarray(label)
    if label.ndim != 3:
        raise ValueError(f"expected a (D, H, W) label map, got shape {label.shape}")
    if expected_shape is not None and tuple(label.shape) != tuple(expected_shape):
        raise ValueError(
            f"label shape {tuple(label.shape)} != expected {tuple(expected_shape)}"
        )

    n_classes = int(cs.NUM_CLASSES)
    # One-hot baseline, exact for every voxel that cannot be softened.
    target = np.eye(n_classes, dtype=np.float64)[label].transpose(3, 0, 1, 2)

    mask = label == STN
    if mask.any():
        window = _stn_window(mask)
        sub_label = label[window]
        d_surface, counts = nearest_face_counts(sub_label, expected_shape=None)
        target[(slice(None),) + window] = soft_target_from_distance(
            sub_label, d_surface, counts, band_mm
        )

    if not np.isfinite(target).all():
        raise ValueError("soft target contains NaN/Inf")
    return np.ascontiguousarray(target, dtype=np.float32)

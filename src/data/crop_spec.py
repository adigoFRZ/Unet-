"""The frozen Baseline v1 spatial crop, and the NIfTI -> PyTorch axis convention.

This module is the single source of truth for both. Everything downstream
(cache, dataset, model, metrics) imports from here so the convention cannot drift
between components -- an axis-order mistake between the image and the label would
silently train a network on misaligned data.

THE CROP IS FROZEN
------------------
Derived from the development-train spatial envelope only (see
``results/spatial_roi_design/``). It must not be moved per case, adjusted from a
case's own GT, or revised using internal_test / challenge_test.

AXIS CONVENTION
---------------
nibabel volumes are ``(X, Y, Z)`` with spacing ``(0.667, 0.667, 2.0)`` mm and
orientation LAS. PyTorch ``Conv3d`` expects ``(N, C, D, H, W)``.

The physical axes are mapped so that the *anisotropic* axis stays the depth axis::

    D = Z   (inferior-superior, spacing 2.0 mm -- the thick-slice direction)
    H = Y   (posterior-anterior, spacing 0.667 mm)
    W = X   (right-left,          spacing 0.667 mm)

so the tensor spacing, in ``(D, H, W)`` order, is ``(2.0, 0.667, 0.667)`` mm.

The transform is a pure axis permutation ``(X,Y,Z) -> (Z,Y,X)``. It is a
*transpose, not a flip*: no axis is reversed, so left/right, anterior/posterior
and inferior/superior are all preserved. The anisotropic kernels in the model are
written for exactly this layout.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# --------------------------------------------------------------------------- #
# Frozen crop
# --------------------------------------------------------------------------- #

#: Shape of the raw NIfTI array in (X, Y, Z) order, before the crop.
#:
#: This is the grid the crop indices below are expressed in, so it is the only
#: place that number is written down. Every case shares it: the dataset audit
#: found one unique shape, one unique spacing and one unique (zero-translation)
#: affine across all 500 cases x 5 volumes. Experiment E's coordinate channels
#: are normalised against this grid *before* the crop, so they must not be
#: re-normalised afterwards -- doing so would silently change their meaning.
RAW_SHAPE_XYZ: Final[tuple[int, int, int]] = (300, 300, 70)

#: NIfTI array slice bounds, half-open, applied as ``vol[X0:X1, Y0:Y1, Z0:Z1]``.
CROP_X: Final[tuple[int, int]] = (103, 199)
CROP_Y: Final[tuple[int, int]] = (103, 199)
CROP_Z: Final[tuple[int, int]] = (8, 40)

#: Resulting shape in NIfTI (X, Y, Z) order.
CROP_SHAPE_XYZ: Final[tuple[int, int, int]] = (
    CROP_X[1] - CROP_X[0],
    CROP_Y[1] - CROP_Y[0],
    CROP_Z[1] - CROP_Z[0],
)

#: Resulting shape in PyTorch (D, H, W) order -- (Z, Y, X).
CROP_SHAPE_DHW: Final[tuple[int, int, int]] = (
    CROP_SHAPE_XYZ[2],
    CROP_SHAPE_XYZ[1],
    CROP_SHAPE_XYZ[0],
)

#: Inclusive voxel-index bbox in NIfTI order, for reporting/QA.
CROP_BBOX_INCLUSIVE_XYZ: Final[tuple[tuple[int, int, int], tuple[int, int, int]]] = (
    (CROP_X[0], CROP_Y[0], CROP_Z[0]),
    (CROP_X[1] - 1, CROP_Y[1] - 1, CROP_Z[1] - 1),
)

#: Number of input modalities and output classes.
IN_CHANNELS: Final[int] = 3
NUM_CLASSES: Final[int] = 4

#: Channel order of the image tensor. Fixed.
CHANNEL_ORDER: Final[tuple[str, ...]] = ("T1", "QSM", "NM")

#: Class ids. 0 is background and is NOT a target.
BACKGROUND: Final[int] = 0
FOREGROUND_CLASSES: Final[tuple[int, ...]] = (1, 2, 3)
CLASS_NAMES: Final[dict[int, str]] = {1: "STN", 2: "SN", 3: "RN"}

#: Voxel spacing in (D, H, W) order, millimetres. Matches CROP_SHAPE_DHW axes.
SPACING_DHW_MM: Final[tuple[float, float, float]] = (2.0, 0.6666667, 0.6666667)

#: Voxel spacing in NIfTI (X, Y, Z) order, millimetres.
SPACING_XYZ_MM: Final[tuple[float, float, float]] = (0.6666667, 0.6666667, 2.0)

#: Dtype the cache stores images as.
IMAGE_DTYPE: Final[str] = "float32"
#: Dtype the cache stores labels as (integer values 0..3).
LABEL_DTYPE: Final[str] = "uint8"


def crop_xyz(volume: np.ndarray) -> np.ndarray:
    """Apply the frozen crop to a ``(X, Y, Z)`` array."""
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    return volume[CROP_X[0]:CROP_X[1], CROP_Y[0]:CROP_Y[1], CROP_Z[0]:CROP_Z[1]]


def xyz_to_dhw(array: np.ndarray) -> np.ndarray:
    """Reorder a cropped ``(X, Y, Z)`` array to PyTorch ``(D, H, W)`` = ``(Z, Y, X)``.

    A permutation only -- no axis is flipped. ``np.transpose`` is used rather than
    ``np.moveaxis`` for clarity about the exact permutation.
    """
    if array.ndim != 3:
        raise ValueError(f"expected a 3D array, got shape {array.shape}")
    return np.transpose(array, (2, 1, 0))


def to_tensor_layout(volume_xyz: np.ndarray) -> np.ndarray:
    """Full NIfTI -> tensor transform: crop, then reorder.

    Input  (X, Y, Z)          e.g. (300, 300, 70)
    Output (Z, Y, X)          e.g. (32, 96, 96)
    """
    return np.ascontiguousarray(xyz_to_dhw(crop_xyz(volume_xyz)))


def assert_crop_shape(shape: tuple[int, ...]) -> None:
    """Raise unless ``shape`` is the frozen crop shape in DHW order."""
    if tuple(shape) != CROP_SHAPE_DHW:
        raise ValueError(
            f"tensor shape {tuple(shape)} does not match the frozen crop "
            f"{CROP_SHAPE_DHW} (DHW)"
        )

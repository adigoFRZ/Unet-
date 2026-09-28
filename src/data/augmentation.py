"""Conservative augmentation for Experiment A (Augmentation v1).

Design rules, all of which are enforced here rather than left to convention:

1. **One spatial transform per case, applied to everything.** The three
   modalities and the label share a single sampled affine. Applying independent
   transforms per channel would destroy the inter-modal alignment that the whole
   project verified in the alignment stage, and would do it silently.

2. **Interpolation order follows the data type.** Images use linear
   interpolation (order=1); the label uses nearest neighbour (order=0), so it
   stays a discrete 0/1/2/3 map. Any interpolated label value would be an
   invalid class.

3. **Background stays exactly zero.** The normalised images have a
   mathematically meaningful zero background, and downstream code (and the
   background invariant established during preprocessing) depends on it. The
   affine fills with ``cval=0``, and the intensity transforms are applied only to
   non-zero voxels, so a zero voxel can never be shifted off zero.

4. **No gamma on QSM.** QSM is signed (it has negative values throughout the
   brain), and ``x ** gamma`` is undefined for negative ``x``. Gamma is therefore
   restricted to the strictly non-negative modalities, and even there it is
   applied to the positive part only, leaving non-positive values untouched.

5. **Deliberately conservative.** No flips, no elastic deformation, no cutout, no
   mixup, no large rotations. The rotation is in-plane only, because the
   through-plane spacing is 2.0 mm versus 0.667 mm in-plane and a rotation about
   X or Y would resample that thick axis far more aggressively than it deserves.

Axis convention: tensors are (D, H, W) = (Z, Y, X). Configuration is expressed in
anatomical XYZ because that is how the ranges are specified, and converted here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

LOGGER = logging.getLogger(__name__)

#: modalities that may receive gamma. QSM is signed and is excluded.
GAMMA_SAFE_MODALITIES: tuple[str, ...] = ("T1", "NM")


@dataclass
class SpatialAugmentConfig:
    """Ranges for the affine transform. Voxel units, anatomical XYZ order."""

    #: max absolute translation per anatomical axis, in voxels
    translation_x: float = 4.0
    translation_y: float = 4.0
    translation_z: float = 1.0
    #: max absolute in-plane rotation (about the Z / D axis), degrees
    rotation_z_deg: float = 5.0
    #: uniform isotropic scaling range
    scale: tuple[float, float] = (0.95, 1.05)
    #: probability that a case receives any spatial transform at all
    probability: float = 0.5
    #: set False to disable spatial augmentation entirely
    enabled: bool = True


@dataclass
class IntensityAugmentConfig:
    """Ranges for the per-modality intensity transform."""

    scale: tuple[float, float] = (0.90, 1.10)
    shift: tuple[float, float] = (-0.10, 0.10)
    gamma: tuple[float, float] = (0.9, 1.1)
    probability: float = 0.5
    enabled: bool = True
    #: modalities allowed to receive gamma (see module docstring)
    gamma_modalities: tuple[str, ...] = GAMMA_SAFE_MODALITIES


@dataclass
class AugmentSummary:
    """What the last call actually did, for logging and tests."""

    spatial_applied: bool = False
    spatial_params: dict[str, float] = field(default_factory=dict)
    intensity_applied: dict[str, dict[str, float]] = field(default_factory=dict)


class SegmentationAugmentor:
    """Applies conservative augmentation to a (C, D, H, W) sample.

    Deterministic given ``seed``: the RNG lives in the instance, so two
    augmentors built with the same seed produce the same sequence of samples.
    """

    def __init__(
        self,
        spatial: SpatialAugmentConfig | None = None,
        intensity: IntensityAugmentConfig | None = None,
        channel_order: Sequence[str] = ("T1", "QSM", "NM"),
        seed: int = 42,
    ) -> None:
        self.spatial = spatial or SpatialAugmentConfig()
        self.intensity = intensity or IntensityAugmentConfig()
        self.channel_order = tuple(channel_order)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.last_summary = AugmentSummary()

    # ---- sampling ---------------------------------------------------------- #

    def _sample_spatial(self) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Return (matrix, offset, params) mapping OUTPUT coords to INPUT coords.

        ``scipy.ndimage.affine_transform`` evaluates
        ``output[i] = input[matrix @ i + offset]``, so this matrix is the inverse
        of the visual transform. The composition below is built around the volume
        centre so that rotation and scaling do not also translate the anatomy.
        """
        rng = self._rng
        theta = np.deg2rad(rng.uniform(-self.spatial.rotation_z_deg,
                                       self.spatial.rotation_z_deg))
        scale = rng.uniform(*self.spatial.scale)

        # Translation is sampled in anatomical XYZ, then reordered to (D, H, W).
        t_x = rng.uniform(-self.spatial.translation_x, self.spatial.translation_x)
        t_y = rng.uniform(-self.spatial.translation_y, self.spatial.translation_y)
        t_z = rng.uniform(-self.spatial.translation_z, self.spatial.translation_z)
        translation_dhw = np.array([t_z, t_y, t_x], dtype=np.float64)

        cos, sin = np.cos(theta), np.sin(theta)
        # In-plane rotation: mixes the H (Y) and W (X) axes, leaving D (Z) alone.
        rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos, -sin],
            [0.0, sin, cos],
        ], dtype=np.float64)

        matrix = scale * rotation
        params = {
            "rotation_z_deg": float(np.rad2deg(theta)),
            "scale": float(scale),
            "translation_x_vox": float(t_x),
            "translation_y_vox": float(t_y),
            "translation_z_vox": float(t_z),
        }
        return matrix, translation_dhw, params

    # ---- transforms -------------------------------------------------------- #

    def apply_affine(
        self,
        image: np.ndarray,
        label: np.ndarray,
        matrix: np.ndarray,
        translation_dhw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply one affine to the image stack and the label, identically.

        Images: order=1 (linear). Label: order=0 (nearest neighbour). Both fill
        with 0, which is background for the images and class 0 for the label.
        """
        centre = (np.asarray(label.shape, dtype=np.float64) - 1.0) / 2.0
        offset = centre + translation_dhw - matrix @ centre

        augmented_image = np.empty_like(image, dtype=np.float32)
        for channel in range(image.shape[0]):
            augmented_image[channel] = ndimage.affine_transform(
                image[channel].astype(np.float32), matrix, offset=offset,
                order=1, mode="constant", cval=0.0, prefilter=False,
            )
        augmented_label = ndimage.affine_transform(
            label.astype(np.uint8), matrix, offset=offset,
            order=0, mode="constant", cval=0, prefilter=False,
        )
        return augmented_image, augmented_label

    def apply_intensity(
        self, image: np.ndarray, modality: str
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Per-modality intensity transform, applied to foreground only.

        Only non-zero voxels are touched, so the exactly-zero background stays
        exactly zero and the "background is 0" invariant survives augmentation.
        """
        rng = self._rng
        scale = rng.uniform(*self.intensity.scale)
        shift = rng.uniform(*self.intensity.shift)
        use_gamma = modality in self.intensity.gamma_modalities
        gamma = rng.uniform(*self.intensity.gamma) if use_gamma else 1.0

        out = image.copy()
        mask = image != 0
        if mask.any():
            values = image[mask].astype(np.float32)
            values = values * scale + shift
            if use_gamma and gamma != 1.0:
                # Positive part only: x**gamma is undefined for x <= 0, and these
                # volumes are z-scored so roughly half the foreground is negative.
                positive = values > 0
                values[positive] = np.power(values[positive], gamma)
            out[mask] = values

        params = {"scale": float(scale), "shift": float(shift)}
        if use_gamma:
            params["gamma"] = float(gamma)
        return out, params

    # ---- entry point ------------------------------------------------------- #

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment one sample. ``image`` is (C, D, H, W), ``label`` is (D, H, W)."""
        summary = AugmentSummary()
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label)

        if self.spatial.enabled and self._rng.random() < self.spatial.probability:
            matrix, translation, params = self._sample_spatial()
            image, label = self.apply_affine(image, label, matrix, translation)
            summary.spatial_applied = True
            summary.spatial_params = params

        if self.intensity.enabled:
            for channel, modality in enumerate(self.channel_order):
                if self._rng.random() < self.intensity.probability:
                    image[channel], params = self.apply_intensity(image[channel], modality)
                    summary.intensity_applied[modality] = params

        self.last_summary = summary
        return image, label


# --------------------------------------------------------------------------- #
# Quality checks
# --------------------------------------------------------------------------- #


def label_is_valid(label: np.ndarray) -> bool:
    """True when the label contains only the four legal class ids."""
    return set(np.unique(label).tolist()).issubset({0, 1, 2, 3})


def touches_border(label: np.ndarray, class_id: int, margin: int = 0) -> bool:
    """True if the class reaches within ``margin`` voxels of any crop face.

    Used to detect augmentation that pushes anatomy out of the fixed crop: once a
    structure touches the border it is being clipped, and the sample is teaching
    the network a truncated shape.
    """
    mask = label == class_id
    if not mask.any():
        return False
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        profile = np.flatnonzero(mask.any(axis=other))
        if profile.size == 0:
            continue
        if profile[0] <= margin or profile[-1] >= mask.shape[axis] - 1 - margin:
            return True
    return False

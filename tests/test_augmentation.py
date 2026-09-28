"""Unit tests for Experiment A augmentation.

The load-bearing test here is synchronisation. A transform applied to the images
but not the label (or applied with different parameters per modality) still
produces a well-formed, trainable tensor -- it just trains the network on
mislabelled data. Nothing crashes; the only symptom is a worse metric months
later. So the synchronisation test is written to actually detect that, by
checking that the image content and the label still agree after augmentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import crop_spec as cs  # noqa: E402
from data.augmentation import (  # noqa: E402
    GAMMA_SAFE_MODALITIES,
    IntensityAugmentConfig,
    SegmentationAugmentor,
    SpatialAugmentConfig,
    label_is_valid,
    touches_border,
)

CACHE_DIR = PROJECT_ROOT / "cache" / "baseline_v1"


def synthetic_sample(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A bright blob in every channel with the label sitting exactly on it.

    The label IS the blob, so image/label agreement is directly measurable.
    """
    rng = np.random.default_rng(seed)
    shape = cs.CROP_SHAPE_DHW
    image = np.zeros((3, *shape), dtype=np.float32)
    label = np.zeros(shape, dtype=np.uint8)

    d, h, w = shape
    label[d // 2 - 3:d // 2 + 3, h // 2 - 6:h // 2 + 6, w // 2 - 6:w // 2 + 6] = 1
    label[d // 2 - 6:d // 2 - 3, h // 2 + 6:h // 2 + 12, w // 2 - 4:w // 2 + 4] = 2
    label[d // 2 + 3:d // 2 + 6, h // 2 - 12:h // 2 - 6, w // 2 + 4:w // 2 + 12] = 3

    for channel in range(3):
        image[channel] = rng.normal(0.0, 0.02, shape).astype(np.float32)
        image[channel][label > 0] += 5.0
        image[channel][label == 0] = 0.0
    return image, label


def contrast_inside_vs_outside(image: np.ndarray, label: np.ndarray) -> float:
    """Mean image intensity inside the label minus outside it."""
    inside = label > 0
    if not inside.any() or inside.all():
        return float("nan")
    return float(image[inside].mean() - image[~inside].mean())


# --------------------------------------------------------------------------- #
# 1. spatial synchronisation
# --------------------------------------------------------------------------- #


def test_image_and_label_stay_synchronised() -> None:
    """The label must move exactly with the image, not independently.

    Before augmentation the label sits on a bright blob, so the contrast between
    inside and outside the label is large. That contrast must survive: if the
    label were transformed with different parameters than the image, the blob and
    the label would drift apart and the contrast would collapse.
    """
    image, label = synthetic_sample()
    before = contrast_inside_vs_outside(image[0], label)
    assert before > 1.0, "test fixture is broken: no contrast to begin with"

    augmentor = SegmentationAugmentor(
        spatial=SpatialAugmentConfig(probability=1.0),   # force a transform
        intensity=IntensityAugmentConfig(enabled=False),
        channel_order=cs.CHANNEL_ORDER,
        seed=3,
    )
    for trial in range(8):
        aug_image, aug_label = augmentor(image.copy(), label.copy())
        assert augmentor.last_summary.spatial_applied
        after = contrast_inside_vs_outside(aug_image[0], aug_label)
        assert after > 0.5 * before, (
            f"trial {trial}: label/image contrast collapsed from {before:.2f} to "
            f"{after:.2f} — the label is not following the image"
        )


def test_all_modalities_share_one_transform() -> None:
    """All three channels must be warped by the same affine.

    Each channel carries the same blob in the same place, so after a shared
    transform their foreground supports must still coincide. Per-channel
    transforms would separate the supports relative to each other.
    """
    image, label = synthetic_sample()
    image[1] = image[0].copy()          # put the identical blob in every channel
    image[2] = image[0].copy()

    augmentor = SegmentationAugmentor(
        spatial=SpatialAugmentConfig(probability=1.0),
        intensity=IntensityAugmentConfig(enabled=False),
        channel_order=cs.CHANNEL_ORDER,
        seed=11,
    )
    for _ in range(6):
        aug_image, _ = augmentor(image.copy(), label.copy())
        supports = [aug_image[c] != 0 for c in range(3)]
        for a in range(3):
            for b in range(a + 1, 3):
                disagree = int(np.count_nonzero(supports[a] != supports[b]))
                assert disagree == 0, (
                    f"channels {a} and {b} have {disagree} voxels that disagree "
                    f"about foreground — they were not transformed identically"
                )


# --------------------------------------------------------------------------- #
# 2. shapes
# --------------------------------------------------------------------------- #


def test_shapes_unchanged() -> None:
    image, label = synthetic_sample()
    augmentor = SegmentationAugmentor(channel_order=cs.CHANNEL_ORDER, seed=5)
    for _ in range(10):
        aug_image, aug_label = augmentor(image.copy(), label.copy())
        assert aug_image.shape == (3, *cs.CROP_SHAPE_DHW)
        assert aug_label.shape == cs.CROP_SHAPE_DHW
        assert aug_image.dtype == np.float32


# --------------------------------------------------------------------------- #
# 3. label remains discrete
# --------------------------------------------------------------------------- #


def test_label_stays_discrete() -> None:
    """Nearest-neighbour resampling must never invent an interpolated class."""
    image, label = synthetic_sample()
    augmentor = SegmentationAugmentor(channel_order=cs.CHANNEL_ORDER, seed=7)
    for _ in range(15):
        _, aug_label = augmentor(image.copy(), label.copy())
        assert aug_label.dtype == np.uint8
        assert label_is_valid(aug_label), (
            f"augmented label has values {sorted(np.unique(aug_label).tolist())}"
        )


# --------------------------------------------------------------------------- #
# 4. validation is never augmented
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not CACHE_DIR.is_dir(), reason="cache not built")
def test_validation_split_refuses_augmentation() -> None:
    from data.segmentation_dataset import SegmentationDataset

    with pytest.raises(ValueError, match="training"):
        SegmentationDataset(PROJECT_ROOT, "val", augment=True)


@pytest.mark.skipif(not CACHE_DIR.is_dir(), reason="cache not built")
def test_validation_samples_are_deterministic() -> None:
    """Re-reading the same val sample must give byte-identical tensors."""
    from data.segmentation_dataset import SegmentationDataset

    dataset = SegmentationDataset(PROJECT_ROOT, "val", augment=False)
    first = dataset[0]["image"]
    second = dataset[0]["image"]
    assert torch_equal(first, second)
    assert dataset.augment is False
    assert dataset.augmentor is None


def torch_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a, b))


# --------------------------------------------------------------------------- #
# 5. reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_is_reproducible() -> None:
    image, label = synthetic_sample()
    first = SegmentationAugmentor(channel_order=cs.CHANNEL_ORDER, seed=99)
    second = SegmentationAugmentor(channel_order=cs.CHANNEL_ORDER, seed=99)

    for _ in range(5):
        a_image, a_label = first(image.copy(), label.copy())
        b_image, b_label = second(image.copy(), label.copy())
        assert np.array_equal(a_image, b_image)
        assert np.array_equal(a_label, b_label)


def test_different_seed_differs() -> None:
    image, label = synthetic_sample()
    first = SegmentationAugmentor(channel_order=cs.CHANNEL_ORDER, seed=1)
    second = SegmentationAugmentor(channel_order=cs.CHANNEL_ORDER, seed=2)
    a_image, _ = first(image.copy(), label.copy())
    b_image, _ = second(image.copy(), label.copy())
    assert not np.array_equal(a_image, b_image)


# --------------------------------------------------------------------------- #
# 6. QSM never receives gamma
# --------------------------------------------------------------------------- #


def test_qsm_is_excluded_from_gamma() -> None:
    assert "QSM" not in GAMMA_SAFE_MODALITIES
    assert "QSM" not in IntensityAugmentConfig().gamma_modalities


def test_qsm_intensity_params_never_include_gamma() -> None:
    """QSM is signed, so x**gamma is undefined; gamma must never be sampled."""
    image, label = synthetic_sample()
    image[1] -= 2.0     # make the QSM channel genuinely signed, as the real data is
    augmentor = SegmentationAugmentor(
        spatial=SpatialAugmentConfig(enabled=False),
        intensity=IntensityAugmentConfig(probability=1.0),   # always apply
        channel_order=cs.CHANNEL_ORDER,
        seed=13,
    )
    qsm_seen = 0
    for _ in range(25):
        augmentor(image.copy(), label.copy())
        applied = augmentor.last_summary.intensity_applied
        if "QSM" in applied:
            qsm_seen += 1
            assert "gamma" not in applied["QSM"], (
                f"QSM received gamma={applied['QSM'].get('gamma')}"
            )
        if "T1" in applied:
            assert "gamma" in applied["T1"]
    assert qsm_seen > 0, "QSM was never augmented; the test proved nothing"


def test_qsm_gamma_would_produce_nan_but_is_not_applied() -> None:
    """Document why QSM is excluded: the operation is mathematically invalid."""
    values = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        naive = np.power(values, 1.1)
    assert not np.isfinite(naive[0]), "expected NaN for a negative base"

    # The augmentor's positive-only rule keeps everything finite.
    image = np.zeros((3, 4, 4, 4), dtype=np.float32)
    image[1].ravel()[:] = values.repeat(16)
    augmentor = SegmentationAugmentor(
        intensity=IntensityAugmentConfig(probability=1.0), seed=4,
    )
    out, _ = augmentor(image, np.zeros((4, 4, 4), dtype=np.uint8))
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# background invariant
# --------------------------------------------------------------------------- #


def test_intensity_augmentation_preserves_zero_background() -> None:
    """The normalised data has a meaningful zero background; it must survive."""
    image, label = synthetic_sample()
    augmentor = SegmentationAugmentor(
        spatial=SpatialAugmentConfig(enabled=False),
        intensity=IntensityAugmentConfig(probability=1.0),
        channel_order=cs.CHANNEL_ORDER, seed=17,
    )
    for _ in range(10):
        aug_image, _ = augmentor(image.copy(), label.copy())
        for channel in range(3):
            zeros = image[channel] == 0
            assert np.all(aug_image[channel][zeros] == 0), (
                f"channel {channel}: a zero background voxel became non-zero"
            )


@pytest.mark.skipif(not CACHE_DIR.is_dir(), reason="cache not built")
def test_augmented_train_samples_are_valid_on_real_data() -> None:
    from data.segmentation_dataset import SegmentationDataset

    dataset = SegmentationDataset(PROJECT_ROOT, "train", augment=True, augment_seed=42)
    assert dataset.augmentor is not None
    for index in range(6):
        sample = dataset[index]
        assert sample["image"].shape == (3, *cs.CROP_SHAPE_DHW)
        assert sample["label"].shape == cs.CROP_SHAPE_DHW
        unique = set(sample["label"].unique().tolist())
        assert unique.issubset({0, 1, 2, 3})
        for class_id in cs.FOREGROUND_CLASSES:
            assert (sample["label"] == class_id).any(), (
                f"{cs.CLASS_NAMES[class_id]} missing after augmentation"
            )

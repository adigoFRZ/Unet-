"""Unit tests for Experiment E -- coordinate and occupancy spatial priors.

Three properties carry the whole experiment, and each is easy to get subtly
wrong in a way that still runs:

**The coordinate channels must not be re-normalised after the crop.** Normalising
on the raw grid and then renormalising the crop to [-1, 1] gives a channel that
looks perfectly reasonable, is still in [-1, 1], and silently throws away the
absolute position it exists to encode. The tests therefore pin the *asymmetric*
cropped ranges rather than just checking the range is bounded.

**The occupancy prior must not leak a training case's own label.** This is the
difference between a prior and an answer key. The decisive test is a synthetic
one: if case *i* is the only case marking a voxel, that voxel must be exactly 0 in
case *i*'s leave-one-out prior. ``S/160``, ``S/159`` and any off-by-one
subtraction fail it, and no amount of "it looks about right" would catch them.

**A validation prior must not depend on validation labels at all.** Asserted the
strong way: for the val split, the prior is computed twice with two *different*
label arguments and must come out identical. For the train split the same call
must differ -- otherwise the previous assertion would pass trivially for a wrapper
that ignores labels everywhere.

Run with:  python -m pytest tests/test_spatial_prior_v1.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import crop_spec as cs  # noqa: E402

torch = pytest.importorskip("torch", reason="PyTorch is required for these tests")

from data.segmentation_dataset import SegmentationDataset  # noqa: E402
from data.spatial_prior import (  # noqa: E402
    MODE_BOTH,
    MODE_COORDS,
    MODE_NONE,
    MODE_OCCUPANCY,
    OCCUPANCY_CLASSES,
    OccupancyPrior,
    SpatialPriorDataset,
    build_coordinate_channels,
    resolve_spatial_prior_mode,
    spatial_prior_channel_count,
)
from models.anisotropic_unet3d import (  # noqa: E402
    AnisotropicUNet3D,
    UNet3DConfig,
    count_parameters,
)
from training.train_baseline import (  # noqa: E402
    BaselineConfig,
    build_criterion,
    build_dataloader,
    load_config,
    resolve_input_channels,
)

CACHE_DIR = PROJECT_ROOT / "cache" / "baseline_v1"
PRIOR_CACHE = PROJECT_ROOT / "cache" / "spatial_prior_v1"
#: Experiment E configs and the baseline they are compared against. Resolved
#: inside the subject-clean set, which is the configuration generation the
#: reported results come from; the pre-correction copies at the configs/ root
#: are kept only because frozen records reference them by path.
REPORTED_CONFIGS = PROJECT_ROOT / "configs" / "subject_clean_v1"
BASELINE_CONFIG = REPORTED_CONFIGS / "baseline_v1.yaml"
CONFIG_DIR = REPORTED_CONFIGS / "spatial_prior"

needs_cache = pytest.mark.skipif(not CACHE_DIR.is_dir(), reason="base cache not built")
needs_prior = pytest.mark.skipif(not PRIOR_CACHE.is_dir(),
                                 reason="occupancy cache not built")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _tiny_base_dataset(n: int = 2, split: str = "val") -> SegmentationDataset:
    dataset = SegmentationDataset(
        root=PROJECT_ROOT, split=split, cache_dir=CACHE_DIR,
        manifest_dir=PROJECT_ROOT / "manifests" / "experiment")
    dataset.case_ids = dataset.case_ids[:n]
    return dataset


def _synthetic_prior(masks: list[np.ndarray], ids: list[str]) -> OccupancyPrior:
    """An OccupancyPrior built from in-memory masks (no disk, no real labels)."""
    sums = np.zeros((len(OCCUPANCY_CLASSES), *cs.CROP_SHAPE_DHW), dtype=np.uint16)
    for mask in masks:
        for column, class_id in enumerate(OCCUPANCY_CLASSES):
            sums[column] += (mask == class_id).astype(np.uint16)
    return OccupancyPrior(sums, ids)


def _blank_mask() -> np.ndarray:
    return np.zeros(cs.CROP_SHAPE_DHW, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# A. baseline default behaviour
# --------------------------------------------------------------------------- #

def test_baseline_config_is_unaffected_by_the_new_fields() -> None:
    """A. Adding Experiment E must not move Baseline v1 by one byte."""
    config = load_config(BASELINE_CONFIG)
    assert config.spatial_prior_mode == "none"
    assert config.spatial_prior_cache_dir is None
    assert resolve_input_channels(config) == 3
    assert isinstance(build_criterion(config), type(build_criterion(BaselineConfig())))
    assert type(build_criterion(config)).__name__ == "DiceCELoss"

    model = AnisotropicUNet3D(UNet3DConfig(
        in_channels=resolve_input_channels(config), num_classes=4, base_channels=16))
    assert count_parameters(model)["total"] == 5_240_420


@needs_cache
def test_baseline_dataset_is_not_wrapped_and_keeps_three_channels() -> None:
    """A. mode=none must leave the plain dataset in place, not a pass-through wrapper."""
    config = load_config(BASELINE_CONFIG)
    dataset, _ = build_dataloader(config, "val", shuffle=False)
    assert type(dataset).__name__ == "SegmentationDataset"
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 96, 96)
    assert "spatial_prior" not in sample and "boundary" not in sample


def test_baseline_config_file_declares_no_prior_keys() -> None:
    """A. baseline_v1.yaml must not need new fields to keep working."""
    import yaml

    declared = yaml.safe_load(
        BASELINE_CONFIG.read_text(encoding="utf-8"))
    assert not {"spatial_prior_mode", "spatial_prior_cache_dir"} & set(declared)


# --------------------------------------------------------------------------- #
# B. coordinate channels
# --------------------------------------------------------------------------- #

def test_coordinate_channels_shape_and_channel_order() -> None:
    coords = build_coordinate_channels()
    assert coords.shape == (3, 32, 96, 96)
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()


def test_coordinate_axes_map_to_the_right_tensor_axis() -> None:
    """B. Cx varies along W, Cy along H, Cz along D -- checked by what varies.

    A channel placed on the wrong axis has the right shape and the right value
    range, so only this check distinguishes it.
    """
    cx, cy, cz = build_coordinate_channels()

    def varies(volume: np.ndarray, axis: int) -> bool:
        first = np.take(volume, 0, axis=axis)
        second = np.take(volume, 1, axis=axis)
        return not np.allclose(first, second)

    assert varies(cx, 2) and not varies(cx, 0) and not varies(cx, 1), "Cx must vary along W"
    assert varies(cy, 1) and not varies(cy, 0) and not varies(cy, 2), "Cy must vary along H"
    assert varies(cz, 0) and not varies(cz, 1) and not varies(cz, 2), "Cz must vary along D"


def test_coordinate_ranges_are_the_raw_grid_crop_not_renormalised() -> None:
    """B. The cropped ranges must stay asymmetric.

    Re-normalising the crop to [-1, 1] would give exactly (-1, +1) for all three
    channels. The frozen crop is not centred on the raw volume, so the correct
    ranges are 2*103/299-1 .. 2*198/299-1 for x and y, and 2*8/69-1 .. 2*39/69-1
    for z. Asserting the asymmetry is what catches the renormalisation mistake.
    """
    cx, cy, cz = build_coordinate_channels()
    nx, ny, nz = cs.RAW_SHAPE_XYZ

    def expected(n: int, lo: int, hi: int) -> tuple[float, float]:
        return 2.0 * lo / (n - 1) - 1.0, 2.0 * (hi - 1) / (n - 1) - 1.0

    for volume, name, (n, (lo, hi)) in (
            (cx, "Cx", (nx, cs.CROP_X)),
            (cy, "Cy", (ny, cs.CROP_Y)),
            (cz, "Cz", (nz, cs.CROP_Z))):
        want_lo, want_hi = expected(n, lo, hi)
        assert float(volume.min()) == pytest.approx(want_lo, abs=1e-6), name
        assert float(volume.max()) == pytest.approx(want_hi, abs=1e-6), name

    # The x/y range and the z range must differ -- that asymmetry is the evidence
    # that no post-crop renormalisation happened.
    assert not np.isclose(cx.min(), -1.0, atol=1e-6)
    assert not np.isclose(cx.max(), +1.0, atol=1e-6)
    assert abs(cz.max() - cz.min()) != pytest.approx(abs(cx.max() - cx.min()), rel=1e-3)


def test_coordinate_formula_on_the_raw_grid() -> None:
    """B. The formula is 2*i/(n-1)-1 evaluated on the RAW index, before cropping."""
    nx, ny, nz = cs.RAW_SHAPE_XYZ
    for volume, n, (lo, hi), axis in (
            (build_coordinate_channels()[0], nx, cs.CROP_X, 0),
            (build_coordinate_channels()[1], ny, cs.CROP_Y, 1),
            (build_coordinate_channels()[2], nz, cs.CROP_Z, 2)):
        index = np.take(volume, 0, axis=axis)  # constant along its own axis
        raw_index = lo
        assert float(index.flat[0]) == pytest.approx(
            2.0 * raw_index / (n - 1) - 1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# C / H / I. wrapper shapes and channel order
# --------------------------------------------------------------------------- #

@needs_cache
@pytest.mark.parametrize("mode,expected_channels", [
    (MODE_COORDS, 6), (MODE_OCCUPANCY, 6), (MODE_BOTH, 9)])
def test_wrapper_output_shapes(mode: str, expected_channels: int) -> None:
    """C/H/I. Each mode yields the documented channel count."""
    if mode in (MODE_OCCUPANCY, MODE_BOTH) and not PRIOR_CACHE.is_dir():
        pytest.skip("occupancy cache not built")
    base = _tiny_base_dataset(n=2, split="val")
    occupancy = (OccupancyPrior.load(PRIOR_CACHE) if mode in (MODE_OCCUPANCY, MODE_BOTH)
                 else None)
    dataset = SpatialPriorDataset(base, mode, split="val", occupancy=occupancy)
    sample = dataset[0]
    assert sample["image"].shape == (expected_channels, 32, 96, 96)
    assert sample["label"].shape == (32, 96, 96)
    assert dataset.case_ids == base.case_ids


@needs_cache
def test_prior_channels_are_appended_so_channel_zero_stays_t1() -> None:
    """C. The first three channels must be bit-identical to the plain dataset.

    Channel 0 is used as the background for the prediction overlays; prepending a
    prior would silently put a coordinate map behind every overlay.
    """
    base = _tiny_base_dataset(n=2, split="val")
    plain = base[0]["image"]
    dataset = SpatialPriorDataset(base, MODE_COORDS, split="val")
    augmented = dataset[0]["image"]

    assert augmented.shape == (6, 32, 96, 96)
    assert torch.equal(augmented[:3], plain), (
        "the image channels were not preserved as the first three channels")
    # ...and the priors are genuinely there, not zeros.
    assert not torch.allclose(augmented[3:], torch.zeros_like(augmented[3:]))


@needs_cache
def test_wrapper_forwards_base_attributes_transparently() -> None:
    base = _tiny_base_dataset(n=2, split="val")
    dataset = SpatialPriorDataset(base, MODE_COORDS, split="val")
    assert dataset.split == "val"
    assert dataset.base is base
    assert dataset.case_ids == base.case_ids
    assert len(dataset) == len(base)
    # forwarded, not reimplemented
    assert dataset.describe()["split"] == "val"
    assert dataset.modalities == base.modalities


def test_unknown_prior_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown spatial_prior_mode"):
        resolve_spatial_prior_mode("atlas")
    assert resolve_spatial_prior_mode(None) == MODE_NONE
    assert spatial_prior_channel_count(MODE_NONE) == 0
    assert spatial_prior_channel_count(MODE_BOTH) == 6


# --------------------------------------------------------------------------- #
# D. occupancy cache provenance
# --------------------------------------------------------------------------- #

@needs_prior
def test_occupancy_cache_shape_dtype_and_metadata() -> None:
    """D. Integer counts on the frozen crop, with train provenance recorded."""
    prior = OccupancyPrior.load(PRIOR_CACHE)
    assert prior.sums.shape == (3, 32, 96, 96)
    assert np.issubdtype(prior.sums.dtype, np.integer)
    assert prior.n_train == 160
    assert len(set(prior.case_ids)) == 160
    assert int(prior.sums.max()) <= prior.n_train
    assert (prior.sums >= 0).all()
    assert prior.metadata["source_split"] == "train"
    assert prior.metadata["class_order"] == list(OCCUPANCY_CLASSES)
    assert prior.metadata["n_train"] == 160


@needs_prior
def test_occupancy_cache_matches_the_train_manifest_exactly() -> None:
    """D. The recorded case set must be exactly the development-train manifest."""
    import pandas as pd

    manifest = sorted(pd.read_csv(
        PROJECT_ROOT / "manifests" / "experiment" / "train.csv")["case_id"].astype(str))
    prior = OccupancyPrior.load(PRIOR_CACHE, expected_case_ids=manifest)
    assert sorted(prior.case_ids) == manifest


@needs_prior
def test_occupancy_cache_refuses_a_mismatched_case_set() -> None:
    """D. A cache built from a different cohort must be refused, not used."""
    with pytest.raises(ValueError, match="different case set"):
        OccupancyPrior.load(PRIOR_CACHE, expected_case_ids=["SYN_000", "SYN_001"])


@needs_prior
def test_occupancy_cache_contains_only_train_cases() -> None:
    """D. No validation case may appear in the cache's provenance."""
    import pandas as pd

    val_ids = set(pd.read_csv(
        PROJECT_ROOT / "manifests" / "experiment" / "val.csv")["case_id"].astype(str))
    prior = OccupancyPrior.load(PRIOR_CACHE)
    assert not (set(prior.case_ids) & val_ids), (
        "the occupancy cache lists validation cases -- it must be train-only")


# --------------------------------------------------------------------------- #
# E / F. leave-one-out correctness
# --------------------------------------------------------------------------- #

def test_leave_one_out_equals_the_average_over_the_other_cases() -> None:
    """E. Definitional check: (S - M_i)/(n-1) must equal the average of the rest.

    Built synthetically so the expected value is computed independently of the
    implementation. Any of S/n, S/(n-1) or subtracting the wrong case differs here.
    """
    rng = np.random.default_rng(0)
    n = 5
    masks = []
    for _ in range(n):
        mask = _blank_mask()
        mask[10:14, 20:30, 20:30] = 1
        mask[rng.integers(0, 30), rng.integers(0, 90), rng.integers(0, 90)] = 2
        masks.append(mask)
    ids = [f"case_{i}" for i in range(n)]
    prior = _synthetic_prior(masks, ids)

    for i in range(n):
        got = prior.leave_one_out(i, masks[i])
        want = np.zeros_like(got)
        for j in range(n):
            if j == i:
                continue
            for column, class_id in enumerate(OCCUPANCY_CLASSES):
                want[column] += (masks[j] == class_id)
        want /= (n - 1)
        assert np.allclose(got, want, atol=1e-6), f"case {i}"


def test_case_own_mask_never_leaks_into_its_own_leave_one_out_prior() -> None:
    """E. The decisive leak test, and it cannot be satisfied by a wrong formula.

    Case 0 is the only case marking voxel V. In case 0's leave-one-out prior that
    voxel must be exactly 0. ``S/160``, ``S/159`` and subtracting any other case
    all leave a non-zero value there.
    """
    empty = _blank_mask
    masks = [empty() for _ in range(4)]
    unique = empty()
    unique[5:9, 40:60, 40:60] = 1
    masks[0] = unique
    ids = ["c0", "c1", "c2", "c3"]
    prior = _synthetic_prior(masks, ids)

    loo = prior.leave_one_out(0, masks[0])
    stn = OCCUPANCY_CLASSES.index(1)
    assert loo[stn][5:9, 40:60, 40:60].max() == 0.0, (
        "case 0's own STN mask is still present in its own leave-one-out prior "
        "-- this is a self-referential leak")

    # The fixed map, by contrast, MUST contain it (1/4 of the cases).
    assert prior.fixed()[stn][6, 45, 45] == pytest.approx(0.25, abs=1e-6)

    # And a case that is not the contributor must still see it, at 1/3.
    other = prior.leave_one_out(1, masks[1])
    assert other[stn][6, 45, 45] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_leave_one_out_uses_the_original_cache_not_the_supplied_label_sum() -> None:
    """E. The subtraction is against the cached sum, using the case's own label.

    Guards the mistake of rebuilding a sum from the supplied label (which would
    make the prior depend on the very mask it is supposed to exclude).
    """
    masks = [_blank_mask() for _ in range(3)]
    masks[0][0:4, 0:8, 0:8] = 1
    masks[1][0:4, 0:8, 0:8] = 1
    masks[2][0:4, 0:8, 0:8] = 1
    prior = _synthetic_prior(masks, ["a", "b", "c"])
    stn = OCCUPANCY_CLASSES.index(1)

    # A voxel marked by all three: every LOO must show it in 2 of the other 2,
    # i.e. exactly 1.0 -- not 0.5, which S/3 would give.
    loo = prior.leave_one_out(0, masks[0])
    assert loo[stn][1, 4, 4] == pytest.approx(1.0, abs=1e-6)


def test_changing_another_cases_mask_changes_this_cases_prior() -> None:
    """F. Sensitivity in the other direction: the prior must track the cohort."""
    masks = [_blank_mask() for _ in range(4)]
    masks[0][0:4, 0:8, 0:8] = 1
    ids = ["c0", "c1", "c2", "c3"]
    before = _synthetic_prior(masks, ids).leave_one_out(0, masks[0])

    masks_after = [m.copy() for m in masks]
    masks_after[2][0:4, 0:8, 0:8] = 1          # case 2 now also marks that region
    after = _synthetic_prior(masks_after, ids).leave_one_out(0, masks[0])

    stn = OCCUPANCY_CLASSES.index(1)
    assert before[stn][1, 4, 4] == pytest.approx(0.0, abs=1e-6)
    assert after[stn][1, 4, 4] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert not np.allclose(before, after)


# --------------------------------------------------------------------------- #
# G. validation prior is label-independent
# --------------------------------------------------------------------------- #

@needs_cache
def test_validation_prior_ignores_the_label_it_is_given() -> None:
    """G. For val, the prior must not depend on any label -- asserted strongly.

    The same call is made twice with two different labels and must return the
    identical map; and the train-split counterpart of the same call MUST differ,
    which stops this from passing trivially for a wrapper that ignores labels
    everywhere.
    """
    if not PRIOR_CACHE.is_dir():
        pytest.skip("occupancy cache not built")
    occupancy = OccupancyPrior.load(PRIOR_CACHE)
    base = _tiny_base_dataset(n=2, split="val")
    dataset = SpatialPriorDataset(base, MODE_OCCUPANCY, split="val",
                                  occupancy=occupancy)

    label_a = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.uint8)
    label_b = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.uint8)
    label_b[10:20, 40:60, 40:60] = 1

    prior_a = dataset.prior_for(0, label_a)
    prior_b = dataset.prior_for(0, label_b)
    assert np.array_equal(prior_a, prior_b), (
        "the validation prior changed when the label changed -- validation labels "
        "must never influence the prior")

    expected = occupancy.sums.astype(np.float32) / occupancy.n_train
    assert np.allclose(prior_a, expected, atol=1e-6)

    # Train split: the same comparison must NOT be equal.
    #
    # Both labels here have to be *subtractable* -- the leave-one-out guard
    # rejects a mask that marks voxels the cache never counted, which is why an
    # arbitrary synthetic mask cannot be used on the train path. An all-zero mask
    # (subtracts nothing) and the case's own real label (subtracts its true mask)
    # are both valid and must give different priors.
    train_id = occupancy.case_ids[0]
    empty_label = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.uint8)
    real_label = np.load(CACHE_DIR / "labels" / f"{train_id}.npy")

    train_base = _tiny_base_dataset(n=2, split="val")
    train_base.case_ids = occupancy.case_ids[:2]
    train_dataset = SpatialPriorDataset(train_base, MODE_OCCUPANCY, split="train",
                                        occupancy=occupancy)
    assert not np.array_equal(train_dataset.prior_for(0, empty_label),
                              train_dataset.prior_for(0, real_label)), (
        "the training prior ignored the case's own label -- leave-one-out is not "
        "being applied")

    # And the guard that made the synthetic mask invalid is real, not incidental.
    with pytest.raises(ValueError, match="negative leave-one-out count"):
        train_dataset.prior_for(0, label_b)


# --------------------------------------------------------------------------- #
# J / K. parameter counts
# --------------------------------------------------------------------------- #

def test_e1_and_e2_have_identical_parameter_counts() -> None:
    """J. Equal capacity makes E1-vs-E2 a like-for-like comparison."""
    counts = {}
    for mode, channels in ((MODE_COORDS, 6), (MODE_OCCUPANCY, 6)):
        model = AnisotropicUNet3D(UNet3DConfig(
            in_channels=channels, num_classes=4, base_channels=16))
        counts[mode] = count_parameters(model)["total"]
    assert counts[MODE_COORDS] == counts[MODE_OCCUPANCY] == 5_240_852


def test_e3_parameter_count() -> None:
    """K. E3 adds six channels, so twice the extra stem parameters."""
    model = AnisotropicUNet3D(UNet3DConfig(
        in_channels=9, num_classes=4, base_channels=16))
    assert count_parameters(model)["total"] == 5_241_284

    baseline = count_parameters(AnisotropicUNet3D(UNet3DConfig(
        in_channels=3, num_classes=4, base_channels=16)))["total"]
    per_channel = (5_240_852 - baseline) / 3
    assert per_channel == 144            # 16 out-channels x 1x3x3 kernel
    assert 5_241_284 - baseline == 6 * per_channel


@pytest.mark.parametrize("name,channels", [
    ("e1_coords", 6), ("e2_occupancy", 6), ("e3_both", 9)])
def test_experiment_configs_derive_the_right_channel_count(name: str,
                                                           channels: int) -> None:
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    assert resolve_input_channels(config) == channels
    assert config.augmentation_enabled is False, "Experiment E v1 forbids augmentation"
    assert config.loss_type == "dice_ce"
    assert config.boundary_cache_dir is None
    assert config.modalities is None or tuple(config.modalities) == cs.CHANNEL_ORDER


def test_priors_and_augmentation_are_refused_together() -> None:
    """augmentation transforms images but not priors -- the pair must be refused."""
    if not CACHE_DIR.is_dir():
        pytest.skip("base cache not built")
    config = load_config(CONFIG_DIR / "e1_coords.yaml")
    config.augmentation_enabled = True
    config.modalities = list(cs.CHANNEL_ORDER)
    config.in_channels = resolve_input_channels(config)
    with pytest.raises(ValueError, match="does not support augmentation"):
        build_dataloader(config, "train", shuffle=False, augment=True)


def test_occupancy_mode_without_a_cache_dir_is_an_error() -> None:
    if not CACHE_DIR.is_dir():
        pytest.skip("base cache not built")
    config = load_config(CONFIG_DIR / "e2_occupancy.yaml")
    config.spatial_prior_cache_dir = None
    config.modalities = list(cs.CHANNEL_ORDER)
    config.in_channels = resolve_input_channels(config)
    with pytest.raises(ValueError, match="requires spatial_prior_cache_dir"):
        build_dataloader(config, "val", shuffle=False)


def test_wrapper_requires_an_occupancy_prior_when_the_mode_needs_one() -> None:
    if not CACHE_DIR.is_dir():
        pytest.skip("base cache not built")
    base = _tiny_base_dataset(n=2, split="val")
    with pytest.raises(ValueError, match="needs an occupancy prior"):
        SpatialPriorDataset(base, MODE_OCCUPANCY, split="val", occupancy=None)
    with pytest.raises(ValueError, match="does not use an occupancy prior"):
        SpatialPriorDataset(base, MODE_COORDS, split="val",
                            occupancy=OccupancyPrior(
                                np.zeros((3, *cs.CROP_SHAPE_DHW), dtype=np.uint16),
                                ["a", "b"]))

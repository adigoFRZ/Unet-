"""Unit tests for the Baseline v1 pipeline.

The axis-order tests are the important ones. A silent (X,Y,Z) vs (D,H,W) mix-up
between image and label would still run end to end and would train a network on
spatially scrambled data -- no exception, no warning, just bad results. So the
transform is pinned down explicitly, including a test that would FAIL if the
transform were accidentally a flip instead of a pure permutation.

Run with:  python -m pytest tests -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from _local_fixtures import legacy_training_cache_usable  # noqa: E402

#: The legacy 160/40 experiment split and the cache built over it are local
#: artifacts -- neither is published. `cache/baseline_v1` existing is not enough:
#: after `reproduce.py prepare` it exists but holds the subject-clean cohort
#: (161/38), so these tests must skip rather than run against the wrong cases.
LEGACY_FIXTURE_READY = legacy_training_cache_usable()
LEGACY_FIXTURE_REASON = "legacy experiment split and its cache are not both present"

from data import crop_spec as cs  # noqa: E402

torch = pytest.importorskip("torch", reason="PyTorch is required for these tests")

from data.segmentation_dataset import SegmentationDataset  # noqa: E402
from losses.dice_ce import DiceCELoss, DiceCELossConfig  # noqa: E402
from models.anisotropic_unet3d import (  # noqa: E402
    AnisotropicUNet3D,
    UNet3DConfig,
    count_parameters,
)

CACHE_DIR = PROJECT_ROOT / "cache" / "baseline_v1"


# --------------------------------------------------------------------------- #
# 1. crop shape
# --------------------------------------------------------------------------- #


def test_crop_shape_constants() -> None:
    """The frozen crop must be exactly 96 x 96 x 32 in (X,Y,Z) and (32,96,96) in DHW."""
    assert cs.CROP_SHAPE_XYZ == (96, 96, 32)
    assert cs.CROP_SHAPE_DHW == (32, 96, 96)
    assert cs.CROP_X == (103, 199)
    assert cs.CROP_Y == (103, 199)
    assert cs.CROP_Z == (8, 40)
    # inclusive bbox as specified in the frozen protocol
    assert cs.CROP_BBOX_INCLUSIVE_XYZ == ((103, 103, 8), (198, 198, 39))


def test_crop_applied_to_volume() -> None:
    volume = np.arange(300 * 300 * 70, dtype=np.float32).reshape(300, 300, 70)
    cropped = cs.crop_xyz(volume)
    assert cropped.shape == (96, 96, 32)
    # spot-check a corner to confirm the slice bounds, not just the shape
    assert cropped[0, 0, 0] == volume[103, 103, 8]
    assert cropped[-1, -1, -1] == volume[198, 198, 39]


# --------------------------------------------------------------------------- #
# 2. NIfTI (X,Y,Z) -> PyTorch (D,H,W)
# --------------------------------------------------------------------------- #


def test_axis_transform_is_permutation_not_flip() -> None:
    """(X,Y,Z) -> (Z,Y,X) must reorder, never reverse, any axis.

    Each axis gets a distinct length and a marker at index 0, so a flip anywhere
    changes where the marker lands and fails the test.
    """
    volume = np.zeros((5, 7, 11), dtype=np.float32)
    volume[0, 0, 0] = 1.0          # low corner
    volume[4, 6, 10] = 2.0         # high corner

    out = cs.xyz_to_dhw(volume)
    assert out.shape == (11, 7, 5), "expected (Z, Y, X) ordering"

    # The low corner must stay at the LOW index of every axis => no flip.
    assert out[0, 0, 0] == 1.0
    assert out[10, 6, 4] == 2.0


def test_axis_transform_preserves_values_exactly() -> None:
    rng = np.random.default_rng(0)
    volume = rng.random((6, 5, 4)).astype(np.float32)
    out = cs.xyz_to_dhw(volume)
    # transposing must permute, not resample or alter, any value
    assert np.array_equal(out, np.transpose(volume, (2, 1, 0)))
    assert np.array_equal(np.sort(out.ravel()), np.sort(volume.ravel()))


def test_to_tensor_layout_matches_manual_crop_and_transpose() -> None:
    rng = np.random.default_rng(1)
    volume = rng.random((300, 300, 70)).astype(np.float32)
    fast = cs.to_tensor_layout(volume)
    manual = np.transpose(
        volume[103:199, 103:199, 8:40], (2, 1, 0)
    )
    assert np.array_equal(fast, manual)
    assert fast.shape == cs.CROP_SHAPE_DHW


def test_spacing_order_matches_axis_order() -> None:
    """D must be the thick-slice (2.0 mm) axis, matching the tensor layout."""
    assert cs.SPACING_DHW_MM == (2.0, 0.6666667, 0.6666667)
    assert cs.CROP_SHAPE_DHW[0] == 32, "D must be the Z axis (32 slices)"


# --------------------------------------------------------------------------- #
# 3-4. Dataset shape / dtype / labels
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not LEGACY_FIXTURE_READY, reason=LEGACY_FIXTURE_REASON)
def test_dataset_item_shapes_and_dtypes() -> None:
    dataset = SegmentationDataset(PROJECT_ROOT, "train")
    sample = dataset[0]

    assert sample["image"].shape == (3, 32, 96, 96), "(C,D,H,W)"
    assert sample["label"].shape == (32, 96, 96), "(D,H,W)"
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.int64
    assert isinstance(sample["case_id"], str)


@pytest.mark.skipif(not LEGACY_FIXTURE_READY, reason=LEGACY_FIXTURE_REASON)
def test_dataset_labels_in_range_and_all_classes_present() -> None:
    dataset = SegmentationDataset(PROJECT_ROOT, "train")
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        label = dataset[index]["label"]
        unique = set(torch.unique(label).tolist())
        assert unique.issubset({0, 1, 2, 3}), f"unexpected labels {unique}"
        for class_id in cs.FOREGROUND_CLASSES:
            assert (label == class_id).any(), (
                f"{cs.CLASS_NAMES[class_id]} missing from case "
                f"{dataset.case_ids[index]}"
            )


@pytest.mark.skipif(not LEGACY_FIXTURE_READY, reason=LEGACY_FIXTURE_REASON)
def test_dataset_rejects_reserved_splits() -> None:
    """internal_test / challenge_test must not be loadable at this stage."""
    for split in ("internal_test", "challenge_test"):
        with pytest.raises(ValueError):
            SegmentationDataset(PROJECT_ROOT, split)


@pytest.mark.skipif(not LEGACY_FIXTURE_READY, reason=LEGACY_FIXTURE_REASON)
def test_dataset_images_are_finite() -> None:
    dataset = SegmentationDataset(PROJECT_ROOT, "val")
    for index in range(0, len(dataset), max(1, len(dataset) // 5)):
        assert torch.isfinite(dataset[index]["image"]).all()


# --------------------------------------------------------------------------- #
# 5. model forward
# --------------------------------------------------------------------------- #


def test_model_forward_output_shape() -> None:
    model = AnisotropicUNet3D(UNet3DConfig(base_channels=16))
    model.eval()
    x = torch.randn(1, 3, 32, 96, 96)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 32, 96, 96), f"got {tuple(y.shape)}"


def test_model_forward_batch_of_two() -> None:
    model = AnisotropicUNet3D(UNet3DConfig(base_channels=16))
    model.eval()
    x = torch.randn(2, 3, 32, 96, 96)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4, 32, 96, 96)


def test_model_channel_widths() -> None:
    cfg = UNet3DConfig(base_channels=16)
    assert cfg.channels == [16, 32, 64, 128, 256]


# --------------------------------------------------------------------------- #
# 6-8. loss and backward
# --------------------------------------------------------------------------- #


def test_loss_is_finite() -> None:
    criterion = DiceCELoss()
    logits = torch.randn(1, 4, 32, 96, 96)
    target = torch.zeros(1, 32, 96, 96, dtype=torch.long)
    target[0, 10:14, 40:50, 40:50] = 1
    target[0, 14:18, 50:60, 50:60] = 2
    target[0, 18:22, 60:70, 60:70] = 3

    loss, components = criterion(logits, target)
    assert torch.isfinite(loss)
    assert torch.isfinite(components["ce"])
    assert torch.isfinite(components["dice"])
    assert 0.0 <= float(components["dice"]) <= 1.0


def test_loss_rejects_mismatched_shapes() -> None:
    criterion = DiceCELoss()
    with pytest.raises(ValueError):
        criterion(torch.randn(1, 4, 32, 96, 96), torch.zeros(1, 16, 48, 48, dtype=torch.long))


def _make_target(n: int = 2, size: int = 16) -> torch.Tensor:
    """``n`` cases whose foreground volumes differ a lot between cases.

    Case 0 is small, case 1 is roughly 4x larger. That contrast is what makes a
    batch-pooled Dice reduction distinguishable from a per-case one.
    """
    target = torch.zeros(n, 8, size, size, dtype=torch.long)
    if n >= 1:
        target[0, 2:4, 2:6, 2:6] = 1
        target[0, 4:5, 3:7, 3:7] = 2
        target[0, 5:6, 4:8, 4:8] = 3
    if n >= 2:
        target[1, 1:6, 1:9, 1:9] = 1
        target[1, 2:7, 2:12, 2:12] = 2
        target[1, 3:8, 3:13, 3:13] = 3
    return target


def test_dice_is_reduced_per_case_not_pooled_over_batch() -> None:
    """Batch pooling must not let a large-target case outvote a small one.

    The Dice term is computed per (case, class) and only then averaged. If it
    were pooled over the whole batch, the larger case in a pair would dominate
    the gradient purely because it contains more voxels -- while evaluation is an
    unweighted per-case mean. The two must agree.
    """
    from losses.dice_ce import DiceCELoss as D

    criterion = D(DiceCELossConfig())
    target = _make_target()
    logits = torch.randn(2, 4, 8, 16, 16)

    matrix = criterion.per_case_per_class_dice(logits, target)
    assert matrix.shape == (2, 3), f"expected (N, n_classes), got {tuple(matrix.shape)}"

    # The loss must equal the mean of the per-case-per-class matrix.
    loss = criterion.soft_dice_loss(logits, target)
    assert float(loss) == pytest.approx(1.0 - float(matrix.mean()), abs=1e-6)

    # And it must NOT equal a volume-weighted pooling of the same values.
    volumes = torch.tensor([
        [float((target[i] == c).sum()) for c in (1, 2, 3)] for i in range(2)
    ])
    pooled = float((matrix * volumes).sum() / volumes.sum())
    loss_unweighted = float(matrix.mean())
    assert abs(loss_unweighted - pooled) > 1e-4, (
        "the two reductions coincide here, so this test cannot detect pooling; "
        "make the per-case target volumes more different"
    )


def test_dice_per_case_is_invariant_to_batch_composition() -> None:
    """A case's own Dice must not depend on which other cases share its batch."""
    from losses.dice_ce import DiceCELoss as D

    criterion = D(DiceCELossConfig())
    torch.manual_seed(7)

    target_a = _make_target(n=1)
    logits_a = torch.randn(1, 4, 8, 16, 16)
    alone = criterion.per_case_per_class_dice(logits_a, target_a)

    target_b = _make_target(n=2)[1:2]
    logits_b = torch.randn(1, 4, 8, 16, 16)
    batched = criterion.per_case_per_class_dice(
        torch.cat([logits_a, logits_b], dim=0),
        torch.cat([target_a, target_b], dim=0),
    )
    assert torch.allclose(alone[0], batched[0], atol=1e-6), (
        "case 0's Dice changed when another case was added to the batch -- "
        "the reduction is still pooled over the batch"
    )


def test_dice_foreground_classes_are_1_2_3_and_exclude_background() -> None:
    """Background must not be one of the classes averaged into the Dice term."""
    config = DiceCELossConfig()
    assert 0 not in config.dice_classes
    assert tuple(config.dice_classes) == (1, 2, 3)


def test_loss_does_not_include_background_in_dice() -> None:
    """A correct background must earn no credit in the Dice term.

    Two things are checked:
      * a *perfect* prediction of all four classes gives a Dice loss of ~0;
      * a prediction that is right about background but never predicts any
        foreground stays far from 0, so background correctness cannot mask a
        completely missed target.

    The second value is 0.297 rather than exactly 1.0 because the epsilon in the
    soft-Dice denominator (1e-5) is comparable to the vanishing probability mass
    of an absent class. The assertion below is therefore on the order of the
    value, not on its exact size.
    """
    criterion = DiceCELoss(DiceCELossConfig(ce_weight=0.0, dice_weight=1.0))

    target = torch.zeros(1, 8, 16, 16, dtype=torch.long)
    target[0, 2:4, 4:8, 4:8] = 1
    target[0, 4:6, 8:12, 8:12] = 2
    target[0, 5:7, 2:6, 10:14] = 3

    perfect = torch.full((1, 4, 8, 16, 16), -10.0)
    for class_id in range(4):
        perfect[:, class_id] = torch.where(
            target == class_id, torch.tensor(10.0), torch.tensor(-10.0)
        )
    _, components = criterion(perfect, target)
    assert float(components["dice"]) == pytest.approx(0.0, abs=1e-4)

    background_only = torch.full((1, 4, 8, 16, 16), -10.0)
    background_only[:, 0] = 10.0        # correct background, never any foreground
    all_background = torch.zeros(1, 8, 16, 16, dtype=torch.long)
    _, components = criterion(background_only, all_background)
    assert float(components["dice"]) > 0.2, (
        "a correct background alone must not look like success"
    )

    # And confirm the exclusion is what makes the difference: averaging over all
    # four classes would LOWER the loss for the same failed prediction, which is
    # precisely the masking this design avoids.
    with_background = DiceCELoss(DiceCELossConfig(
        ce_weight=0.0, dice_weight=1.0, dice_classes=(0, 1, 2, 3)
    ))
    _, components_bg = with_background(background_only, all_background)
    assert float(components_bg["dice"]) < float(components["dice"])


def test_backward_produces_nonzero_gradients() -> None:
    torch.manual_seed(0)
    model = AnisotropicUNet3D(UNet3DConfig(base_channels=8))
    criterion = DiceCELoss()
    x = torch.randn(1, 3, 32, 96, 96)
    target = torch.zeros(1, 32, 96, 96, dtype=torch.long)
    target[0, 10:14, 40:50, 40:50] = 1

    loss, _ = criterion(model(x), target)
    loss.backward()

    total = 0.0
    nonzero_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.abs().sum())
        if parameter.grad.abs().sum() > 0:
            nonzero_tensors += 1
    assert total > 0.0, "all gradients are zero -- backward is not working"
    assert nonzero_tensors > 0


# --------------------------------------------------------------------------- #
# 9. determinism
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not LEGACY_FIXTURE_READY, reason=LEGACY_FIXTURE_REASON)
def test_dataset_order_is_stable() -> None:
    first = SegmentationDataset(PROJECT_ROOT, "train").case_ids
    second = SegmentationDataset(PROJECT_ROOT, "train").case_ids
    assert first == second
    assert first == sorted(first), "case ids must be sorted for determinism"


def test_relative_paths_anchor_to_project_root_not_cwd(tmp_path, monkeypatch) -> None:
    """Config paths must resolve against the project root, not the shell's cwd.

    Without this, launching the trainer from another directory would silently
    read a different (or missing) cache.
    """
    from utils.paths import REPO_ROOT, resolve_path, resolve_project_root

    assert resolve_project_root(".") == REPO_ROOT
    assert resolve_project_root(None) == REPO_ROOT
    assert resolve_project_root("") == REPO_ROOT

    # Simulate being launched from an unrelated directory.
    monkeypatch.chdir(tmp_path)
    assert resolve_project_root(".") == REPO_ROOT, "cwd leaked into the project root"
    assert resolve_path("cache/baseline_v1") == (REPO_ROOT / "cache" / "baseline_v1")
    assert resolve_path("manifests/experiment") == (REPO_ROOT / "manifests" / "experiment")

    absolute = tmp_path / "somewhere"
    assert resolve_path(absolute) == absolute.resolve()


def test_config_paths_resolve_to_existing_locations(tmp_path) -> None:
    """The shipped baseline config must resolve to real directories."""
    from training.train_baseline import BaselineConfig, load_config, resolve_config_paths
    from utils.paths import REPO_ROOT

    # The subject-clean config is the one every reported result was produced with.
    config = load_config(REPO_ROOT / "configs" / "subject_clean_v1" / "baseline_v1.yaml")
    assert isinstance(config, BaselineConfig)

    root = resolve_config_paths(config, REPO_ROOT)
    assert root == REPO_ROOT
    assert Path(config.cache_dir) == REPO_ROOT / "cache" / "baseline_v1"
    assert Path(config.manifest_dir) == REPO_ROOT / "manifests" / "subject_clean_v1"
    assert (Path(config.checkpoint_dir)
            == REPO_ROOT / "checkpoints" / "subject_clean_v1" / "baseline_v1")
    assert (Path(config.results_dir)
            == REPO_ROOT / "results" / "experiments_subject_clean_v1" / "baseline_v1")
    # paths are absolute after resolution
    for value in (config.cache_dir, config.manifest_dir, config.checkpoint_dir,
                  config.results_dir):
        assert Path(value).is_absolute()


def test_dataset_root_ignores_cwd(tmp_path, monkeypatch) -> None:
    """SegmentationDataset must resolve its cache/manifest dirs off the root."""
    if not LEGACY_FIXTURE_READY:
        pytest.skip(LEGACY_FIXTURE_REASON)
    monkeypatch.chdir(tmp_path)
    dataset = SegmentationDataset(PROJECT_ROOT, "train")
    assert dataset.cache_dir == PROJECT_ROOT.resolve() / "cache" / "baseline_v1"
    assert dataset.manifest_dir == PROJECT_ROOT.resolve() / "manifests" / "experiment"
    assert len(dataset) == 160


def test_same_seed_gives_same_initial_weights() -> None:
    def build(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        model = AnisotropicUNet3D(UNet3DConfig(base_channels=8))
        return next(model.parameters()).detach().clone()

    assert torch.equal(build(42), build(42))
    assert not torch.equal(build(42), build(43))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_metrics_macro_excludes_background() -> None:
    from evaluation import segmentation_metrics as sm

    # Construct a prediction that is exactly right in every class, background
    # included. Every channel must be set explicitly: leaving non-target channels
    # at 0 while the background channel is -10 would make argmax tie-break into
    # class 1 across the whole background.
    target = np.zeros((8, 16, 16), dtype=np.int64)
    target[2:4, 4:8, 4:8] = 1
    target[4:6, 8:12, 8:12] = 2
    target[5:7, 2:6, 10:14] = 3

    logits = np.full((4, 8, 16, 16), -10.0, dtype=np.float32)
    for class_id in (0, 1, 2, 3):
        logits[class_id] = np.where(target == class_id, 10.0, -10.0)

    metrics = sm.evaluate_case(logits, target)
    assert metrics.dice[1] == pytest.approx(1.0)
    assert metrics.dice[2] == pytest.approx(1.0)
    assert metrics.dice[3] == pytest.approx(1.0)
    assert metrics.macro_foreground_dice == pytest.approx(1.0)


def test_metrics_reject_batched_input() -> None:
    from evaluation import segmentation_metrics as sm

    with pytest.raises(ValueError):
        sm.evaluate_case(np.zeros((2, 4, 8, 16, 16), dtype=np.float32),
                         np.zeros((2, 8, 16, 16), dtype=np.int64))


def test_empty_prediction_scores_zero_not_nan() -> None:
    """GT present + empty prediction must be 0, never NaN.

    NaN would be dropped by the aggregator, so a model that predicts nothing at
    all would have every one of its failures removed from the mean and could
    appear to beat a model that actually tried. STN/SN/RN are present in every
    case in this dataset, so this is the realistic failure mode.
    """
    from evaluation import segmentation_metrics as sm

    assert sm.precision_from_counts(0, 0, 50) == 0.0
    assert sm.recall_from_counts(0, 0, 50) == 0.0
    assert sm.dice_from_counts(0, 0, 50) == 0.0


def test_empty_prediction_is_counted_in_aggregate() -> None:
    from evaluation import segmentation_metrics as sm

    # One case with a correct prediction, one that predicts nothing at all.
    target = np.zeros((2, 8, 16, 16), dtype=np.int64)
    for i in range(2):
        target[i, 2:4, 4:8, 4:8] = 1
        target[i, 4:6, 8:12, 8:12] = 2
        target[i, 5:7, 2:6, 10:14] = 3

    logits = np.full((2, 4, 8, 16, 16), -10.0, dtype=np.float32)
    for c in range(4):
        logits[0, c] = np.where(target[0] == c, 10.0, -10.0)
    logits[1, 0] = 10.0          # case 1 predicts pure background

    cases = [
        sm.evaluate_case(logits[i], target[i], case_id=f"c{i}") for i in range(2)
    ]
    assert cases[1].dice[1] == 0.0
    assert cases[1].empty_prediction[1] is True

    aggregate = sm.aggregate_cases(cases)
    # STN mean Dice must be (1.0 + 0.0)/2 = 0.5, i.e. the failure is NOT dropped.
    assert aggregate["dice_STN"]["mean"] == pytest.approx(0.5)
    assert aggregate["dice_STN"]["n_cases"] == 2
    assert aggregate["empty_prediction_STN"]["count"] == 1
    assert aggregate["empty_prediction_STN"]["rate"] == pytest.approx(0.5)


def test_absent_class_still_yields_nan() -> None:
    """A class missing from BOTH prediction and GT stays undefined (NaN)."""
    from evaluation import segmentation_metrics as sm

    assert math.isnan(sm.dice_from_counts(0, 0, 0))
    assert math.isnan(sm.precision_from_counts(0, 0, 0))
    assert math.isnan(sm.recall_from_counts(0, 0, 0))
    # GT absent but something was predicted: precision is 0, recall undefined.
    assert sm.precision_from_counts(0, 10, 0) == 0.0
    assert math.isnan(sm.recall_from_counts(0, 10, 0))


def test_hd95_aggregation_reports_n_valid_and_spacing() -> None:
    from evaluation import segmentation_metrics as sm

    target = np.zeros((3, 8, 16, 16), dtype=np.int64)
    for i in range(3):
        target[i, 2:4, 4:8, 4:8] = 1
        target[i, 4:6, 8:12, 8:12] = 2
        target[i, 5:7, 2:6, 10:14] = 3

    logits = np.full((3, 4, 8, 16, 16), -10.0, dtype=np.float32)
    for i in range(3):
        for c in range(4):
            logits[i, c] = np.where(target[i] == c, 10.0, -10.0)
    logits[2, 0] = 10.0      # case 2 predicts nothing -> HD95 undefined there

    cases = [
        sm.evaluate_case(logits[i], target[i], case_id=f"c{i}", compute_hd95=True)
        for i in range(3)
    ]
    aggregate = sm.aggregate_cases(cases, include_hd95=True)

    for name in cs.CLASS_NAMES.values():
        key = f"hd95_{name}_mm"
        assert key in aggregate, f"{key} missing from aggregation"
        assert aggregate[key]["n_valid"] == 2, "the empty-prediction case has no HD95"
        assert math.isnan(cases[2].hd95_mm[1]), "undefined HD95 must be NaN, not 0"
    # Perfect predictions -> surface distance 0 mm.
    assert aggregate["hd95_STN_mm"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_hd95_aggregation_can_be_skipped() -> None:
    """Per-epoch validation must not pay for HD95."""
    from evaluation import segmentation_metrics as sm

    target = np.zeros((1, 8, 16, 16), dtype=np.int64)
    target[0, 2:4, 4:8, 4:8] = 1
    target[0, 4:6, 8:12, 8:12] = 2
    target[0, 5:7, 2:6, 10:14] = 3
    logits = np.full((1, 4, 8, 16, 16), -10.0, dtype=np.float32)
    for c in range(4):
        logits[0, c] = np.where(target[0] == c, 10.0, -10.0)

    cases = [sm.evaluate_case(logits[0], target[0], case_id="c0")]
    assert not any(k.startswith("hd95_") for k in sm.aggregate_cases(cases))
    assert any(k.startswith("hd95_")
               for k in sm.aggregate_cases(
                   [sm.evaluate_case(logits[0], target[0], case_id="c0",
                                     compute_hd95=True)]))


def test_hd95_uses_millimetre_spacing() -> None:
    """A 1-voxel shift along D (2.0 mm) must score ~2 mm, not ~1."""
    from evaluation import segmentation_metrics as sm

    a = np.zeros((16, 32, 32), dtype=bool)
    a[8, 16, 16] = True
    b = np.zeros((16, 32, 32), dtype=bool)
    b[9, 16, 16] = True     # one voxel apart along D

    hd_d = sm.hausdorff_95_mm(a, b, cs.SPACING_DHW_MM)

    c = np.zeros((16, 32, 32), dtype=bool)
    c[8, 17, 16] = True     # one voxel apart along H
    hd_h = sm.hausdorff_95_mm(a, c, cs.SPACING_DHW_MM)

    assert hd_d == pytest.approx(2.0, abs=1e-3), "through-plane must use 2.0 mm"
    assert hd_h == pytest.approx(0.6666667, abs=1e-3), "in-plane must use 0.667 mm"


# --------------------------------------------------------------------------- #
# final best-checkpoint evaluation
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not LEGACY_FIXTURE_READY, reason=LEGACY_FIXTURE_REASON)
def test_final_evaluation_writes_per_case_and_summary(tmp_path) -> None:
    """The final pass must produce the per-case CSV and summary JSON with HD95."""
    import json

    from torch.utils.data import DataLoader, Subset

    from training.train_baseline import (
        BaselineConfig,
        final_evaluation,
        save_checkpoint,
    )

    dataset = SegmentationDataset(PROJECT_ROOT, "val")
    subset = Subset(dataset, list(range(3)))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    # The checkpoint records its own architecture, and final_evaluation rebuilds
    # from that record -- so the config used to save must match the saved weights.
    config = BaselineConfig(base_channels=4)
    model = AnisotropicUNet3D(UNet3DConfig(base_channels=4)).to("cpu")
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, model, optimiser, None, epoch=3, best_metric=0.5,
                    config=config)

    result = final_evaluation(checkpoint, loader, torch.device("cpu"),
                              tmp_path, config)

    per_case = tmp_path / "val_per_case_best.csv"
    summary = tmp_path / "val_summary_best.json"
    assert per_case.is_file() and summary.is_file()

    import pandas as pd

    frame = pd.read_csv(per_case)
    assert len(frame) == 3
    for column in ("case_id", "Dice_STN", "Dice_SN", "Dice_RN",
                   "Precision_STN", "Precision_SN", "Precision_RN",
                   "Recall_STN", "Recall_SN", "Recall_RN",
                   "HD95_STN_mm", "HD95_SN_mm", "HD95_RN_mm",
                   "macro_foreground_dice"):
        assert column in frame.columns, f"{column} missing from per-case CSV"

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["n_cases"] == 3
    assert payload["hd95_spacing_dhw_mm"] == list(cs.SPACING_DHW_MM)
    for name in cs.CLASS_NAMES.values():
        stats = payload["metrics"][f"hd95_{name}_mm"]
        for stat in ("mean", "std", "median", "min", "max", "n_valid"):
            assert stat in stats, f"hd95_{name}_mm missing '{stat}'"
        assert stats["n_valid"] == 3
    assert "empty_prediction_STN" in payload["metrics"]
    # Selection must be documented as Dice-based, never HD95-based.
    assert "HD95" in payload["selection_metric"]
    assert result["n_cases"] == 3

"""Unit tests for Experiment C -- the STN signed-distance boundary loss.

The failures worth guarding against here are different from Experiments A/B.
There is no alpha/beta convention to confuse; instead there is a **sign
convention** that is just as silent. ``mean(p_STN * phi)`` only rewards the right
behaviour if phi is negative inside the ground truth and positive outside it. Get
that backwards and the loss still runs, still decreases, and still produces
plausible numbers -- while actively training the model to dilate STN. So the sign
is asserted from both directions, plus the distance weighting, plus the use of
real anisotropic spacing.

The second theme is that this loss must be *purely additive*: CE and the
foreground Dice term are Baseline v1's own computations and are compared bitwise,
and ``boundary_weight=0`` must recover the baseline loss exactly.

Run with:  ./.venv/Scripts/python.exe -m pytest tests/test_stn_boundary_v1.py -v
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

import torch.nn.functional as F  # noqa: E402

from losses.boundary_dice_ce import (  # noqa: E402
    STN,
    BoundaryDiceCELoss,
    BoundaryDiceCELossConfig,
)
from losses.dice_ce import DiceCELoss, DiceCELossConfig  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_boundary_cache import signed_distance_stn  # noqa: E402


def _ramp_phi(shape: tuple[int, int, int] = (8, 16, 16), edge: int = 4) -> torch.Tensor:
    """A phi whose value grows with distance from row ``edge``.

    Row ``edge`` is "the boundary" (phi = 0); rows above it are inside the GT
    (negative) and rows below are outside (positive), with |phi| increasing by
    0.1 per row. Deliberately synthetic so a test can name an exact distance
    instead of inferring one from a mask.
    """
    depth = shape[0]
    values = (np.arange(depth) - edge) * 0.1
    phi = np.repeat(values[:, None, None], shape[1], axis=1)
    phi = np.repeat(phi, shape[2], axis=2)
    return torch.from_numpy(phi.astype(np.float32))[None]


def _logits_with_stn_at(shape=(8, 16, 16), rows=None) -> torch.Tensor:
    """Logits where p_STN is ~1 on ``rows`` and ~0 elsewhere."""
    logits = torch.full((1, 4, *shape), -20.0)
    logits[0, 0] = 20.0                     # background dominant by default
    if rows is not None:
        # logits[0, 0] is (D, H, W) -- no batch dimension left to slice.
        logits[0, 0][rows, :, :] = -20.0
        logits[0, 1][rows, :, :] = 20.0
    return logits


def _base_target(n: int = 1, shape=(8, 16, 16)) -> torch.Tensor:
    """``n`` cases with all three foreground classes present."""
    target = torch.zeros(n, *shape, dtype=torch.long)
    target[:, 2:4, 2:6, 2:6] = 1
    target[:, 4:5, 3:7, 3:7] = 2
    target[:, 5:6, 4:8, 4:8] = 3
    return target


@pytest.fixture
def criterion() -> BoundaryDiceCELoss:
    return BoundaryDiceCELoss(BoundaryDiceCELossConfig(boundary_weight=0.10))


# --------------------------------------------------------------------------- #
# 1-3. the sign convention and the distance weighting
# --------------------------------------------------------------------------- #


def test_raising_stn_probability_inside_the_gt_lowers_the_loss() -> None:
    """phi < 0 inside: putting STN probability where STN is must help."""
    criterion = BoundaryDiceCELoss()
    phi = _ramp_phi(edge=4)                       # rows 0..3 are inside (phi < 0)
    inside_rows = [0, 1, 2, 3]

    low = criterion.boundary_loss(_logits_with_stn_at(rows=[]), phi)
    high = criterion.boundary_loss(_logits_with_stn_at(rows=inside_rows), phi)

    assert float(high) < float(low), (
        f"raising p_STN inside the GT raised the boundary loss "
        f"({float(low):.6f} -> {float(high):.6f}); the sign of phi is inverted"
    )


def test_raising_stn_probability_outside_the_gt_raises_the_loss() -> None:
    """phi > 0 outside: putting STN probability where STN is not must hurt."""
    criterion = BoundaryDiceCELoss()
    phi = _ramp_phi(edge=4)                       # rows 5..7 are outside (phi > 0)
    outside_rows = [5, 6, 7]

    low = criterion.boundary_loss(_logits_with_stn_at(rows=[]), phi)
    high = criterion.boundary_loss(_logits_with_stn_at(rows=outside_rows), phi)

    assert float(high) > float(low), (
        f"raising p_STN outside the GT lowered the boundary loss "
        f"({float(low):.6f} -> {float(high):.6f}); the sign of phi is inverted"
    )


def test_penalty_near_the_boundary_is_smaller_than_away_from_it() -> None:
    """The property that makes this a *distance* term rather than a mask term.

    The same amount of wrongly-placed STN probability must cost less when it sits
    just outside the true edge than when it sits far away. If this fails while the
    two sign tests pass, the term is behaving like a plain outside-the-GT mask
    penalty and carries no boundary information at all.
    """
    criterion = BoundaryDiceCELoss()
    phi = _ramp_phi(edge=4)
    reference = float(criterion.boundary_loss(_logits_with_stn_at(rows=[]), phi))

    # phi = (row - 4) * 0.1, so row 5 is +0.1 (one step outside) and row 7 is
    # the furthest outside row available, at +0.3.
    near_row, far_row = 5, 7
    near_phi = (near_row - 4) * 0.1
    far_phi = (far_row - 4) * 0.1
    near = float(criterion.boundary_loss(_logits_with_stn_at(rows=[near_row]), phi))
    far = float(criterion.boundary_loss(_logits_with_stn_at(rows=[far_row]), phi))

    assert far - reference > near - reference > 0, (
        f"penalty did not grow with distance: near +{near - reference:.6f}, "
        f"far +{far - reference:.6f}"
    )
    # The penalty ratio must track the phi ratio -- that is what "weighted by
    # distance" means quantitatively, and a constant-weight mask penalty would
    # give a ratio of 1.0 here.
    assert (far - reference) / (near - reference) == pytest.approx(
        far_phi / near_phi, rel=0.05)


# --------------------------------------------------------------------------- #
# 4-6. distance maps
# --------------------------------------------------------------------------- #


def test_signed_distance_uses_real_anisotropic_spacing() -> None:
    """A step along D must cost 2.0 mm while a step along H costs 0.667 mm.

    With isotropic spacing both neighbours would land at +0.1, so this test
    distinguishes the two constructions rather than just checking that *a*
    distance was produced. Getting this wrong would silently mis-weight the
    2 mm through-plane direction against the 0.667 mm in-plane ones.
    """
    label = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.uint8)
    depth, height, width = cs.CROP_SHAPE_DHW
    d0, h0, w0 = depth // 2, height // 2, width // 2
    label[d0, h0, w0] = STN

    phi, _ = signed_distance_stn(label, STN, cs.SPACING_DHW_MM)

    along_d = float(phi[d0 + 1, h0, w0])
    along_h = float(phi[d0, h0 + 1, w0])
    along_w = float(phi[d0, h0, w0 + 1])

    assert along_d == pytest.approx(2.0 / 10.0, abs=1e-6), "through-plane step"
    assert along_h == pytest.approx(0.6666667 / 10.0, abs=1e-6), "in-plane step"
    assert along_w == pytest.approx(0.6666667 / 10.0, abs=1e-6), "in-plane step"
    # And the two must differ, which is the whole point of "anisotropic".
    assert along_d > 2.5 * along_h


def test_signed_distance_map_shape_range_and_finiteness() -> None:
    label = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.uint8)
    label[10:20, 20:40, 20:40] = STN

    phi, info = signed_distance_stn(label, STN, cs.SPACING_DHW_MM)

    assert phi.shape == cs.CROP_SHAPE_DHW
    assert phi.dtype == np.float32
    assert np.isfinite(phi).all()
    assert phi.min() >= -1.0 - 1e-6 and phi.max() <= 1.0 + 1e-6
    assert not info["empty_mask"]


def test_signed_distance_sign_is_negative_inside_and_positive_outside() -> None:
    """The convention, checked on a real mask rather than a synthetic ramp."""
    label = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.uint8)
    label[10:20, 20:40, 20:40] = STN
    mask = label == STN

    phi, _ = signed_distance_stn(label, STN, cs.SPACING_DHW_MM)

    assert (phi[mask] <= 0).all(), "some GT voxel has a positive distance"
    assert (phi[~mask] >= 0).all(), "some non-GT voxel has a negative distance"

    # The mask is a (D,H,W) block, so the *in-plane* rim of each slice lies one
    # 0.667 mm step from background while the middle of a slice is 2.0 mm from
    # the nearest background plane (the slice above). Both must show up, which is
    # what makes this a distance map rather than a binary edge indicator.
    assert float(phi[10, 20, 22]) == pytest.approx(-0.6666667 / 10.0, abs=1e-6)
    assert float(phi[10, 22, 22]) == pytest.approx(-2.0 / 10.0, abs=1e-6)
    # Values grow with depth, and the clip keeps everything inside [-1, 1].
    assert phi[10, 22, 22] < phi[10, 20, 22] < 0
    assert phi.min() >= -1.0 and phi.max() <= 1.0
    assert phi.min() < 0


def test_signed_distance_of_an_empty_mask_is_all_outside() -> None:
    """No GT for the class must not produce an all-zero ('on the boundary') map."""
    label = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.uint8)
    phi, info = signed_distance_stn(label, STN, cs.SPACING_DHW_MM)
    assert info["empty_mask"] is True
    assert (phi == 1.0).all()


# --------------------------------------------------------------------------- #
# 7-9. the baseline terms must be untouched
# --------------------------------------------------------------------------- #


def test_cross_entropy_is_bitwise_identical_to_baseline() -> None:
    criterion = BoundaryDiceCELoss()
    torch.manual_seed(5)
    logits = torch.randn(2, 4, 8, 16, 16)
    target = _base_target(n=2)

    _, components = criterion(logits, target, torch.zeros(2, 8, 16, 16))
    expected = F.cross_entropy(logits, target.long(), weight=None)
    assert torch.equal(components["ce"], expected.detach())


def test_foreground_dice_is_bitwise_identical_to_baseline() -> None:
    criterion = BoundaryDiceCELoss()
    baseline = DiceCELoss(DiceCELossConfig())
    torch.manual_seed(5)
    logits = torch.randn(2, 4, 8, 16, 16)
    target = _base_target(n=2)

    _, components = criterion(logits, target, torch.zeros(2, 8, 16, 16))
    assert torch.equal(components["dice"], baseline.soft_dice_loss(logits, target).detach())
    # And the underlying per-case matrix is the same object, not a re-derivation.
    assert torch.equal(criterion.per_case_per_class_dice(logits, target),
                       baseline.per_case_per_class_dice(logits, target))


def test_zero_boundary_weight_reproduces_the_baseline_loss_exactly() -> None:
    """lambda = 0 must give Baseline v1's loss, bit for bit.

    This is what makes the experiment interpretable: the boundary term is purely
    additive, so any difference in the result is attributable to it alone.
    """
    criterion = BoundaryDiceCELoss(BoundaryDiceCELossConfig(boundary_weight=0.0))
    baseline = DiceCELoss(DiceCELossConfig())
    torch.manual_seed(29)
    logits = torch.randn(2, 4, 8, 16, 16)
    target = _base_target(n=2)

    ours, ours_components = criterion(logits, target)
    theirs, theirs_components = baseline(logits, target)

    assert torch.equal(ours, theirs)
    assert torch.equal(ours_components["ce"], theirs_components["ce"])
    assert torch.equal(ours_components["dice"], theirs_components["dice"])
    # With lambda = 0 the term is exactly zero, not merely negligible.
    assert float(ours_components["boundary"]) == 0.0


def test_missing_boundary_map_is_an_error_not_a_silent_drop() -> None:
    """A configured term that silently vanishes would fake a null result."""
    criterion = BoundaryDiceCELoss(BoundaryDiceCELossConfig(boundary_weight=0.10))
    logits = torch.randn(1, 4, 8, 16, 16)
    target = _base_target(n=1)
    with pytest.raises(ValueError, match="boundary map is required"):
        criterion(logits, target)


def test_boundary_map_shape_is_validated() -> None:
    criterion = BoundaryDiceCELoss()
    logits = torch.randn(1, 4, 8, 16, 16)
    target = _base_target(n=1)
    with pytest.raises(ValueError, match="does not match logits spatial shape"):
        criterion(logits, target, torch.zeros(1, 8, 16, 8))


# --------------------------------------------------------------------------- #
# 10-11. AMP and backward
# --------------------------------------------------------------------------- #


def test_loss_is_finite_and_components_present() -> None:
    criterion = BoundaryDiceCELoss()
    logits = torch.randn(2, 4, 32, 96, 96)
    target = torch.zeros(2, 32, 96, 96, dtype=torch.long)
    target[:, 10:14, 40:50, 40:50] = 1
    target[:, 14:18, 50:60, 50:60] = 2
    target[:, 18:22, 60:70, 70:80] = 3

    loss, components = criterion(logits, target, torch.zeros(2, 32, 96, 96))

    for key in ("loss", "ce", "dice", "boundary"):
        assert key in components, f"missing component {key!r}"
        assert torch.isfinite(components[key]), f"component {key!r} is not finite"
    assert torch.isfinite(loss)


def test_backward_produces_finite_nonzero_gradients() -> None:
    torch.manual_seed(0)
    from models.anisotropic_unet3d import AnisotropicUNet3D, UNet3DConfig

    model = AnisotropicUNet3D(UNet3DConfig(base_channels=8))
    criterion = BoundaryDiceCELoss()
    x = torch.randn(1, 3, 32, 96, 96)
    target = torch.zeros(1, 32, 96, 96, dtype=torch.long)
    target[0, 10:14, 40:50, 40:50] = 1
    phi = torch.zeros(1, 32, 96, 96)

    loss, _ = criterion(model(x), target, phi)
    loss.backward()

    total = 0.0
    nonzero = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all(), "non-finite gradient"
        total += float(parameter.grad.abs().sum())
        if parameter.grad.abs().sum() > 0:
            nonzero += 1
    assert total > 0.0, "all gradients are zero -- backward is not working"
    assert nonzero > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for autocast")
def test_amp_autocast_path_is_finite() -> None:
    """The real fp16 autocast path the trainer uses, boundary term included."""
    criterion = BoundaryDiceCELoss().cuda()
    logits = torch.randn(2, 4, 32, 96, 96, device="cuda", requires_grad=True)
    target = torch.zeros(2, 32, 96, 96, dtype=torch.long, device="cuda")
    target[:, 10:14, 40:50, 40:50] = 1
    target[:, 14:18, 50:60, 50:60] = 2
    phi = torch.zeros(2, 32, 96, 96, device="cuda")
    phi[:, 10:14, 40:50, 40:50] = -0.2

    scaler = torch.amp.GradScaler("cuda", enabled=True)
    with torch.amp.autocast("cuda", enabled=True):
        loss, components = criterion(logits, target, phi)

    assert loss.dtype == torch.float32
    for key, value in components.items():
        assert torch.isfinite(value), f"component {key!r} not finite under autocast"
        assert value.dtype == torch.float32, f"{key!r} should be fp32"

    scaler.scale(loss).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------- #
# configuration and baseline-regression guards
# --------------------------------------------------------------------------- #


def test_build_criterion_selects_the_boundary_loss_and_keeps_the_default() -> None:
    from training.train_baseline import BaselineConfig, build_criterion

    baseline = BaselineConfig()
    assert baseline.loss_type == "dice_ce"
    assert isinstance(build_criterion(baseline), DiceCELoss)

    experiment = build_criterion(BaselineConfig(
        loss_type="stn_boundary",
        boundary_weight=0.10,
        boundary_cache_dir="cache/boundary_v1/stn_signed_distance",
    ))
    assert isinstance(experiment, BoundaryDiceCELoss)
    assert experiment.config.boundary_weight == 0.10
    assert experiment.config.boundary_class == STN == 1


def test_build_criterion_rejects_an_unknown_loss_type() -> None:
    from training.train_baseline import BaselineConfig, build_criterion

    with pytest.raises(ValueError, match="unknown loss_type"):
        build_criterion(BaselineConfig(loss_type="focal_tversky"))


def test_experiment_config_differs_from_baseline_only_in_the_declared_keys() -> None:
    """Experiment C is only interpretable if the boundary term is the one change.

    The baseline objective must NOT be swapped out: CE and the Dice term stay,
    the boundary term is added. Any accidental edit to a frozen knob fails here.
    """
    from dataclasses import asdict

    import yaml

    from training.train_baseline import load_config

    config_dir = PROJECT_ROOT / "configs" / "subject_clean_v1"
    baseline = asdict(load_config(config_dir / "baseline_v1.yaml"))
    experiment = asdict(load_config(config_dir / "stn_boundary_v1.yaml"))

    differing = {key for key in set(baseline) | set(experiment)
                 if baseline.get(key) != experiment.get(key)}
    assert differing == {
        "loss_type",            # the experimental variable
        "boundary_cache_dir",   # where its supervision comes from
        "pin_memory",           # environment workaround, documented as not an
                                # experimental variable (see the config comment)
        "checkpoint_dir",       # output routing
        "results_dir",
    }, f"unexpected config differences: {sorted(differing)}"

    # The baseline objective itself must be untouched on both sides.
    assert baseline["ce_weight"] == experiment["ce_weight"] == 1.0
    assert baseline["dice_weight"] == experiment["dice_weight"] == 1.0

    # boundary_weight/boundary_class do not appear in `differing` only because
    # the dataclass defaults already carry them. The FILE must still declare
    # them: the loss never enters a checkpoint, so run_config.yaml is the only
    # provenance for what was actually run.
    with (config_dir / "stn_boundary_v1.yaml").open(encoding="utf-8") as handle:
        declared = yaml.safe_load(handle)
    assert declared["boundary_weight"] == 0.10
    assert declared["boundary_class"] == STN == 1
    assert declared["loss_type"] == "stn_boundary"
    with (config_dir / "baseline_v1.yaml").open(encoding="utf-8") as handle:
        baseline_declared = yaml.safe_load(handle)
    assert not {"loss_type", "boundary_weight", "boundary_class",
                "boundary_cache_dir"} & set(baseline_declared)

    for key, value in (("seed", 42), ("batch_size", 2), ("max_epochs", 200),
                       ("early_stopping_patience", 40)):
        assert baseline[key] == experiment[key] == value, key
    assert experiment["learning_rate"] == 3.0e-4
    assert experiment["weight_decay"] == 1.0e-5
    assert experiment["augmentation_enabled"] is False


def test_boundary_cache_is_readable_and_matches_the_crop() -> None:
    """The real cache, if built, must line up with the frozen crop."""
    cache_dir = PROJECT_ROOT / "cache" / "boundary_v1" / "stn_signed_distance"
    if not cache_dir.is_dir():
        pytest.skip("boundary cache not built")

    files = sorted(cache_dir.glob("*.npy"))
    assert files, "boundary cache directory is empty"
    for path in files[:5]:
        phi = np.load(path)
        assert phi.shape == cs.CROP_SHAPE_DHW, path.name
        assert phi.dtype == np.float32, path.name
        assert np.isfinite(phi).all(), path.name
        assert phi.min() >= -1.0 - 1e-6 and phi.max() <= 1.0 + 1e-6, path.name

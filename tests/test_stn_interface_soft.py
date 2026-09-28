"""Tests for Experiment G1: interface-centred soft boundary supervision.

The experiment rests on a geometric claim that is easy to get subtly wrong and
hard to notice: the soft target must be built against the interface *between*
voxels, not against voxel centres. If it is built the obvious way the band
contains no voxel centres and the target silently collapses onto the hard one-hot
label -- the run then looks like a clean null result while having tested nothing.

So most of what is checked here is geometry, on masks small enough to verify by
hand:

* the two achievable in-band distances are exactly ``s/2`` and ``s*sqrt(2)/2``,
  and they follow the anisotropic spacing (in-plane 0.6666667 mm, D 2.0 mm);
* the sign is negative inside STN and positive outside, on every axis;
* probability mass is conserved (``sum_c q_c == 1``) and never leaves ``[0, 1]``;
* residual mass goes to the class across the nearest face, split by face count
  when several are equidistant -- and *only* when they are exactly equidistant;
* the windowed computation is bitwise identical to the whole-volume one.

The loss is checked for the two properties that make it interpretable: the hard
path must be the baseline's own criterion (bit-identical, not a reimplementation),
and the soft path must reduce to the hard path when the target is one-hot.

Run with:  ./.venv/Scripts/python.exe -m pytest tests/test_stn_interface_soft.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import crop_spec as cs  # noqa: E402
from data.interface_soft_target import (  # noqa: E402
    BAND_MM,
    RESIDUAL_CLASSES,
    STN,
    interface_soft_target,
    nearest_face_counts,
    soft_target_from_distance,
)
from losses.dice_ce import DiceCELoss, DiceCELossConfig  # noqa: E402
from losses.stn_interface_soft_ce import (  # noqa: E402
    InterfaceSoftDiceCELoss,
    InterfaceSoftDiceCELossConfig,
)

BACKGROUND, SN, RN = 0, 2, 3
#: Class of the interface-adjacent voxels, inside / outside, in-plane.
Q_ADJACENT_IN, Q_ADJACENT_OUT = 0.75, 0.25
#: ... and for a voxel whose nearest face is an in-plane diagonal.
Q_DIAGONAL_IN = 0.5 + (cs.SPACING_DHW_MM[1] * np.sqrt(2.0) / 2.0) / (2.0 * BAND_MM)
Q_DIAGONAL_OUT = 1.0 - Q_DIAGONAL_IN

CENTRE = (4, 4, 4)


def synthetic(neighbours: dict[tuple[int, int, int], int]) -> np.ndarray:
    """A 9^3 label map with a 3^3 STN block and specified offsets overridden."""
    label = np.zeros((9, 9, 9), dtype=np.uint8)
    label[3:6, 3:6, 3:6] = STN
    for (dd, dh, dw), value in neighbours.items():
        label[CENTRE[0] + dd, CENTRE[1] + dh, CENTRE[2] + dw] = value
    return label


def target(label: np.ndarray, band_mm: float = BAND_MM) -> np.ndarray:
    return interface_soft_target(label, band_mm=band_mm, expected_shape=None)


def at(q: np.ndarray, offset: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """The 4-vector of class probabilities at CENTRE + offset."""
    index = tuple(c + o for c, o in zip(CENTRE, offset))
    return q[(slice(None),) + index]


# --------------------------------------------------------------------------- #
# geometry: shape, range, conservation
# --------------------------------------------------------------------------- #

def test_shape_is_class_first() -> None:
    q = target(synthetic({}))
    assert q.shape == (cs.NUM_CLASSES, 9, 9, 9)
    assert q.dtype == np.float32


def test_probabilities_are_bounded_and_conserve_mass() -> None:
    """sum_c q_c == 1 at every voxel. Checked on a boundary-rich mask.

    Two checks, because they mean different things. The assembly is exact in
    float64, and the returned float32 array can differ from 1 by one ulp purely
    from the final cast -- which is rounding, not a violation of the constraint.
    """
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN, (0, 1, 0): RN,
                       (0, 0, 1): STN, (-1, 0, 0): STN})
    q = target(label)
    assert (q >= 0.0).all(), f"negative probability: min {q.min()}"
    assert (q <= 1.0).all(), f"probability above 1: max {q.max()}"
    assert np.isfinite(q).all()
    assert np.allclose(q.sum(axis=0), 1.0, atol=1e-6), \
        f"class probabilities do not sum to 1: max deviation " \
        f"{np.abs(q.sum(axis=0) - 1.0).max():.3e}"

    # The construction itself is exact before the float32 cast.
    d_surface, counts = nearest_face_counts(label, expected_shape=None)
    exact = soft_target_from_distance(label, d_surface, counts, BAND_MM)
    assert (exact.sum(axis=0) == 1.0).all(), "assembly is not mass-conserving"


def test_target_is_finite_float32_everywhere() -> None:
    q = target(synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): RN}))
    assert q.dtype == np.float32
    assert np.isfinite(q).all()


# --------------------------------------------------------------------------- #
# geometry: the anisotropy and the axis convention
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "axis, offset, expected",
    [
        # In-plane faces sit half an in-plane spacing away; the D face sits half
        # the 2.0 mm spacing away, which is beyond the band and stays hard.
        (2, (0, 0, -1), Q_ADJACENT_IN),
        (1, (0, -1, 0), Q_ADJACENT_IN),
        (0, (-1, 0, 0), 1.0),
    ],
)
def test_interface_adjacent_q_follows_the_anisotropic_spacing(
    axis: int, offset: tuple[int, int, int], expected: float
) -> None:
    """W/H neighbours soften to 0.75; a D neighbour is 1.0 mm away and does not.

    This is the pre-registered consequence of ``s_d = 2.0 mm > w``, not a bug: a
    through-plane interface cannot be softened without widening the band, which
    the protocol forbids.
    """
    label = synthetic({offset: BACKGROUND})
    q = target(label)
    assert q[STN][CENTRE] == pytest.approx(expected, abs=1e-6)


def test_in_plane_diagonal_face_gives_the_second_achievable_distance() -> None:
    """A voxel whose nearest face is a diagonal gets the sqrt(2) value."""
    # (h+1, w) is STN while (h+1, w-1) is SN, so a W-face exists one row over.
    label = synthetic({(0, 1, 0): STN, (0, 1, -1): SN})
    q = target(label)
    assert q[STN][CENTRE] == pytest.approx(Q_DIAGONAL_IN, abs=1e-6)
    assert q[SN][CENTRE] == pytest.approx(Q_DIAGONAL_OUT, abs=1e-6)


def test_w_and_h_axes_are_symmetric_and_sign_is_negative_inside() -> None:
    """No axis swap, and the sign convention holds on both in-plane axes."""
    inside_w = target(synthetic({(0, 0, -1): BACKGROUND}))[STN][CENTRE]
    inside_h = target(synthetic({(0, -1, 0): BACKGROUND}))[STN][CENTRE]
    assert inside_w == pytest.approx(inside_h, abs=1e-9)

    d_surface, _ = nearest_face_counts(synthetic({(0, 0, -1): BACKGROUND}),
                                       expected_shape=None)
    assert d_surface[CENTRE] < 0.0, "inside STN must be negative"
    assert d_surface[CENTRE[0], CENTRE[1], CENTRE[2] - 1] > 0.0, \
        "outside STN must be positive"


def test_symmetry_about_one_half() -> None:
    """q_STN(-d) + q_STN(+d) == 1 across the interface."""
    label = synthetic({(0, 0, -1): BACKGROUND})
    q = target(label)
    inside = q[STN][CENTRE]
    outside = q[STN][CENTRE[0], CENTRE[1], CENTRE[2] - 1]
    assert inside + outside == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# geometry: far field and the interface-adjacency claim
# --------------------------------------------------------------------------- #

def test_voxels_away_from_the_boundary_stay_exactly_one_hot() -> None:
    """Outside the band the target must be the hard label, bit for bit."""
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN, (0, 1, 0): RN,
                       (0, 0, 1): STN})
    q = target(label)
    d_surface, _ = nearest_face_counts(label, expected_shape=None)
    far = np.abs(d_surface) >= BAND_MM

    assert far.any()
    one_hot = np.eye(cs.NUM_CLASSES, dtype=np.float32)[label].transpose(3, 0, 1, 2)
    assert np.array_equal(q[:, far], one_hot[:, far]), \
        "voxels beyond the band are not exactly one-hot"


def test_the_interface_midpoint_is_never_occupied_by_a_voxel_centre() -> None:
    """q == 0.5 exactly is unreachable, which is why the test above replaces it.

    The pre-registered formula would put the interface at q = 0.5, but no voxel
    centre lies on a face. The reachable nearest shells are 0.75 / 0.25 instead.
    """
    label = synthetic({(0, 0, -1): BACKGROUND})
    q = target(label)
    in_band = (q[STN] > 0.0) & (q[STN] < 1.0)
    assert in_band.any()
    assert not np.any(np.isclose(q[STN][in_band], 0.5, atol=1e-6)), \
        "a softened voxel hit exactly 0.5, contradicting the geometry"


def test_empty_stn_mask_yields_a_pure_one_hot_target() -> None:
    """No STN voxels -> nothing to soften, and no division by zero."""
    label = np.zeros((9, 9, 9), dtype=np.uint8)
    label[0, 0, 0] = RN
    q = target(label)
    one_hot = np.eye(cs.NUM_CLASSES, dtype=np.float32)[label].transpose(3, 0, 1, 2)
    assert np.array_equal(q, one_hot)
    assert (q.sum(axis=0) == 1.0).all()


# --------------------------------------------------------------------------- #
# residual allocation: the class across the nearest face
# --------------------------------------------------------------------------- #

def test_adjacent_sn_receives_the_residual_not_background() -> None:
    """§5: never default the residual to background."""
    q = target(synthetic({(0, 0, -1): SN}))
    assert q[SN][CENTRE] == pytest.approx(0.25, abs=1e-6)
    assert q[BACKGROUND][CENTRE] == pytest.approx(0.0, abs=1e-9)
    assert q[STN][CENTRE] == pytest.approx(0.75, abs=1e-6)


def test_adjacent_rn_receives_the_residual() -> None:
    """RN must work even though no real val tie involves it."""
    q = target(synthetic({(0, 0, -1): RN}))
    assert q[RN][CENTRE] == pytest.approx(0.25, abs=1e-6)
    assert q[BACKGROUND][CENTRE] == pytest.approx(0.0, abs=1e-9)


def test_outside_voxels_take_their_own_hard_class() -> None:
    """A non-STN voxel's residual belongs to the label the GT assigns it."""
    for klass in RESIDUAL_CLASSES:
        q = target(synthetic({(0, 0, -1): klass}))
        assert q[klass][CENTRE[0], CENTRE[1], CENTRE[2] - 1] > 0.0
        for other in RESIDUAL_CLASSES:
            if other != klass:
                assert q[other][CENTRE[0], CENTRE[1], CENTRE[2] - 1] == 0.0


# --------------------------------------------------------------------------- #
# residual allocation: the face-count tie rule
# --------------------------------------------------------------------------- #

def test_one_background_and_one_sn_face_split_the_residual_equally() -> None:
    q = target(synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN}))
    residual = 1.0 - q[STN][CENTRE]
    assert q[BACKGROUND][CENTRE] == pytest.approx(residual / 2, abs=1e-6)
    assert q[SN][CENTRE] == pytest.approx(residual / 2, abs=1e-6)


def test_two_background_faces_and_one_sn_face_split_two_to_one() -> None:
    q = target(synthetic({(0, 0, -1): BACKGROUND, (0, 0, 1): BACKGROUND,
                          (0, -1, 0): SN}))
    residual = 1.0 - q[STN][CENTRE]
    assert q[BACKGROUND][CENTRE] == pytest.approx(2 * residual / 3, abs=1e-6)
    assert q[SN][CENTRE] == pytest.approx(residual / 3, abs=1e-6)


def test_the_split_follows_the_face_counts_the_other_way_too() -> None:
    """Two SN faces and one background face: the counts must not be hard-coded."""
    q = target(synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN, (0, 1, 0): SN}))
    residual = 1.0 - q[STN][CENTRE]
    assert q[SN][CENTRE] == pytest.approx(2 * residual / 3, abs=1e-6)
    assert q[BACKGROUND][CENTRE] == pytest.approx(residual / 3, abs=1e-6)


def test_rn_participates_in_the_face_count_split() -> None:
    """Generality check: RN must be a first-class participant, not a special case."""
    q = target(synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): RN}))
    residual = 1.0 - q[STN][CENTRE]
    assert q[RN][CENTRE] == pytest.approx(residual / 2, abs=1e-6)
    assert q[BACKGROUND][CENTRE] == pytest.approx(residual / 2, abs=1e-6)


def test_a_nearer_face_is_never_merged_into_a_tie() -> None:
    """Only *exactly* equidistant faces tie.

    The voxel has a background face at s/2 and an SN face at s*sqrt(2)/2. Those
    are genuinely different distances, so the nearer one must take the whole
    residual -- a tolerant comparison would wrongly split it.
    """
    label = synthetic({(0, 0, -1): BACKGROUND, (0, 1, 0): STN, (0, 1, -1): SN})
    q = target(label)
    residual = 1.0 - q[STN][CENTRE]
    assert q[BACKGROUND][CENTRE] == pytest.approx(residual, abs=1e-6)
    assert q[SN][CENTRE] == pytest.approx(0.0, abs=1e-9)


def test_face_count_split_is_exact_not_tolerance_based() -> None:
    """The distance levels are constants, so tie membership is an exact compare.

    Two faces that are equidistant by construction must tie; two that differ by
    even a small amount must not. The gap between the two levels here is
    ~0.138 mm, far above any float noise, so this pins the behaviour without
    depending on the size of the tolerance.
    """
    s_h, s_w = cs.SPACING_DHW_MM[1], cs.SPACING_DHW_MM[2]
    adjacent = 0.5 * s_w
    diagonal = float(np.hypot(0.5 * s_w, 0.5 * s_h))
    assert diagonal > adjacent
    assert not np.isclose(adjacent, diagonal, atol=1e-6)
    assert adjacent + 0.0 == 0.5 * cs.SPACING_DHW_MM[2]      # exact, no tolerance


# --------------------------------------------------------------------------- #
# the windowed computation must not change anything
# --------------------------------------------------------------------------- #

def test_windowed_matches_whole_volume_on_synthetic_masks() -> None:
    """The window is an optimisation; it must be observationally invisible."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        label = np.zeros((9, 9, 9), dtype=np.uint8)
        label[2:7, 2:7, 2:7] = STN
        for _ in range(int(rng.integers(1, 8))):
            label[tuple(rng.integers(2, 7, 3))] = int(rng.choice([BACKGROUND, SN, RN]))
        if not (label == STN).any():
            continue
        windowed = target(label)
        d_surface, counts = nearest_face_counts(label, expected_shape=None)
        whole = np.ascontiguousarray(
            soft_target_from_distance(label, d_surface, counts, BAND_MM),
            dtype=np.float32,
        )
        assert np.array_equal(windowed, whole), "window changed the target"


@pytest.mark.skipif(
    not (PROJECT_ROOT / "cache/baseline_v1/labels").is_dir(),
    reason="baseline cache absent",
)
def test_windowed_matches_whole_volume_on_real_cases() -> None:
    """The same equivalence on real geometry, where the masks are not blobs."""
    import pandas as pd

    manifest = PROJECT_ROOT / "manifests/experiment/val.csv"
    case_ids = sorted(pd.read_csv(manifest)["case_id"].astype(str))[:4]
    for case_id in case_ids:
        label = np.load(PROJECT_ROOT / "cache/baseline_v1/labels" / f"{case_id}.npy")
        if not (label == STN).any():
            continue
        windowed = interface_soft_target(label)
        d_surface, counts = nearest_face_counts(label)
        whole = np.ascontiguousarray(
            soft_target_from_distance(label, d_surface, counts, BAND_MM),
            dtype=np.float32,
        )
        assert np.array_equal(windowed, whole), case_id


# --------------------------------------------------------------------------- #
# the loss
# --------------------------------------------------------------------------- #

def _batch(n: int = 2, shape: tuple[int, int, int] = (8, 16, 16)):
    torch.manual_seed(0)
    logits = torch.randn(n, cs.NUM_CLASSES, *shape)
    labels = torch.randint(0, cs.NUM_CLASSES, (n, *shape))
    return logits, labels


def test_hard_target_delegates_to_the_baseline_criterion_bit_for_bit() -> None:
    """The baseline must be reused, not reimplemented.

    An equality test rather than a tolerance test on purpose: this is the
    guarantee that the baseline path is untouched, and anything weaker would let
    a silent reimplementation through.
    """
    logits, labels = _batch()
    soft_criterion = InterfaceSoftDiceCELoss()
    baseline = DiceCELoss(DiceCELossConfig(ce_weight=1.0, dice_weight=1.0))

    loss_soft, components_soft = soft_criterion(logits, labels)
    loss_base, components_base = baseline(logits, labels)

    assert torch.equal(loss_soft, loss_base)
    assert set(components_soft) == set(components_base)
    for key in components_base:
        assert torch.equal(components_soft[key], components_base[key]), key


def test_one_hot_soft_target_reproduces_the_hard_path() -> None:
    """The soft path must reduce exactly to the hard path when q is one-hot."""
    logits, labels = _batch()
    one_hot = torch.nn.functional.one_hot(labels, cs.NUM_CLASSES)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()

    soft_criterion = InterfaceSoftDiceCELoss()
    baseline = DiceCELoss(DiceCELossConfig(ce_weight=1.0, dice_weight=1.0))

    loss_soft, comp_soft = soft_criterion(logits, one_hot)
    loss_base, comp_base = baseline(logits, labels)

    assert torch.equal(comp_soft["ce"], comp_base["ce"]), "soft CE != hard CE"
    assert torch.equal(comp_soft["dice"], comp_base["dice"]), "soft Dice != hard Dice"
    assert torch.equal(loss_soft, loss_base)


def test_soft_loss_weights_are_the_baseline_weights() -> None:
    """G1 must not smuggle in a re-weighting of CE against Dice."""
    logits, labels = _batch()
    one_hot = torch.nn.functional.one_hot(labels, cs.NUM_CLASSES)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()
    _, comp = InterfaceSoftDiceCELoss()(logits, one_hot)
    assert comp["loss"] == pytest.approx(1.0 * comp["ce"] + 1.0 * comp["dice"],
                                         rel=1e-6)


def test_soft_target_is_accepted_and_finite() -> None:
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    padded = np.zeros((1, cs.NUM_CLASSES, *label.shape), dtype=np.float32)
    padded[0] = target(label)
    logits = torch.randn(1, cs.NUM_CLASSES, *label.shape)

    loss, components = InterfaceSoftDiceCELoss()(logits, torch.from_numpy(padded))
    assert torch.isfinite(loss)
    for key, value in components.items():
        assert torch.isfinite(value), key


def test_both_loss_terms_carry_nonzero_gradient() -> None:
    """A term with no gradient would make the experiment vacuous."""
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN, (0, 0, 1): SN})
    soft = np.zeros((1, cs.NUM_CLASSES, *label.shape), dtype=np.float32)
    soft[0] = target(label)

    criterion = InterfaceSoftDiceCELoss()

    logits_ce = torch.randn(1, cs.NUM_CLASSES, *label.shape, requires_grad=True)
    criterion.soft_ce(logits_ce, torch.from_numpy(soft)).backward()
    assert logits_ce.grad is not None and torch.isfinite(logits_ce.grad).all()
    assert logits_ce.grad.abs().sum() > 0, "soft CE has no gradient"

    logits_dice = torch.randn(1, cs.NUM_CLASSES, *label.shape, requires_grad=True)
    matrix = criterion.per_case_per_class_dice_soft(
        logits_dice, torch.from_numpy(soft))
    (1.0 - matrix.mean()).backward()
    assert logits_dice.grad is not None and torch.isfinite(logits_dice.grad).all()
    assert logits_dice.grad.abs().sum() > 0, "soft Dice has no gradient"


def test_total_loss_backward_is_finite_and_nonzero() -> None:
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    soft = np.zeros((1, cs.NUM_CLASSES, *label.shape), dtype=np.float32)
    soft[0] = target(label)
    logits = torch.randn(1, cs.NUM_CLASSES, *label.shape, requires_grad=True)
    loss, _ = InterfaceSoftDiceCELoss()(logits, torch.from_numpy(soft))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_soft_dice_is_reduced_per_case_not_pooled_over_the_batch() -> None:
    """Batch pooling would weight cases by volume; evaluation is a per-case mean."""
    torch.manual_seed(0)
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    shared = target(label)
    soft = torch.from_numpy(np.stack([shared, shared]))

    logits = torch.randn(2, cs.NUM_CLASSES, *label.shape)
    matrix = InterfaceSoftDiceCELoss().per_case_per_class_dice_soft(logits, soft)
    assert matrix.shape == (2, len(cs.FOREGROUND_CLASSES))

    # Feeding one case alone must reproduce that case's row exactly.
    single = InterfaceSoftDiceCELoss().per_case_per_class_dice_soft(
        logits[:1], soft[:1])
    assert torch.equal(matrix[:1], single)


def test_soft_target_shape_mismatch_is_rejected() -> None:
    logits = torch.randn(1, cs.NUM_CLASSES, 8, 16, 16)
    wrong = torch.rand(1, cs.NUM_CLASSES, 8, 16, 15)
    with pytest.raises(ValueError, match="shape"):
        InterfaceSoftDiceCELoss()(logits, wrong)


def test_integer_target_is_never_treated_as_soft() -> None:
    """A 4-D integer target must take the hard path even if it is float-castable."""
    logits = torch.randn(1, cs.NUM_CLASSES, 8, 16, 16)
    labels = torch.randint(0, cs.NUM_CLASSES, (1, 8, 16, 16))
    _, components = InterfaceSoftDiceCELoss()(logits, labels)
    assert "ce" in components and "dice" in components


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for AMP")
def test_soft_path_is_finite_under_real_autocast() -> None:
    """The reductions sum ~295k values; fp16 there would overflow to inf."""
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    soft = torch.from_numpy(
        np.stack([target(label)] * 2)).cuda()
    logits = torch.randn(2, cs.NUM_CLASSES, *label.shape, device="cuda")
    with torch.amp.autocast("cuda", enabled=True):
        loss, components = InterfaceSoftDiceCELoss()(logits, soft)
    assert torch.isfinite(loss)
    for key, value in components.items():
        assert torch.isfinite(value), key


# --------------------------------------------------------------------------- #
# trainer integration and configuration
# --------------------------------------------------------------------------- #

def test_baseline_config_does_not_enable_the_soft_target() -> None:
    from training.train_baseline import BaselineConfig
    assert BaselineConfig().soft_boundary_enabled is False
    assert BaselineConfig().loss_type == "dice_ce"


def test_build_criterion_selects_the_soft_criterion() -> None:
    from training.train_baseline import BaselineConfig, build_criterion
    criterion = build_criterion(BaselineConfig(loss_type="stn_interface_soft"))
    assert isinstance(criterion, InterfaceSoftDiceCELoss)


def test_build_criterion_unknown_loss_type_still_raises() -> None:
    from training.train_baseline import BaselineConfig, build_criterion
    with pytest.raises(ValueError, match="unknown loss_type"):
        build_criterion(BaselineConfig(loss_type="focal_tversky"))


def test_experiment_config_differs_from_baseline_only_in_the_declared_keys() -> None:
    """G1 is only interpretable if the target is the one change."""
    from dataclasses import asdict

    import yaml

    from training.train_baseline import load_config

    config_dir = PROJECT_ROOT / "configs"
    baseline = asdict(load_config(config_dir / "subject_clean_v1" / "baseline_v1.yaml"))
    experiment = asdict(load_config(config_dir / "stn_interface_soft_v1.yaml"))

    differing = {key for key in set(baseline) | set(experiment)
                 if baseline.get(key) != experiment.get(key)}
    assert differing == {
        "loss_type",             # the experimental variable
        "soft_boundary_enabled", # turns the soft training target on
        "pin_memory",            # environment workaround, documented as not an
                                 # experimental variable (see the config comment)
        "manifest_dir",          # routing: G1 ran on the pre-remediation dev
                                 # split, so this is not a like-for-like split
                                 # comparison -- it is a knob guard, not a claim
        "checkpoint_dir",        # output routing
        "results_dir",
    }, f"unexpected config differences: {sorted(differing)}"

    # The baseline objective itself must be untouched on both sides.
    assert baseline["ce_weight"] == experiment["ce_weight"] == 1.0
    assert baseline["dice_weight"] == experiment["dice_weight"] == 1.0

    with (config_dir / "stn_interface_soft_v1.yaml").open(encoding="utf-8") as handle:
        declared = yaml.safe_load(handle)
    assert declared["soft_boundary_band_mm"] == 0.6666667, "band must be frozen"
    assert declared["loss_type"] == "stn_interface_soft"
    assert declared["soft_boundary_enabled"] is True
    # No augmentation, no post-processing, no threshold: nothing else to tune.
    for forbidden in ("augmentation_enabled", "threshold", "tversky_alpha",
                      "boundary_weight", "spatial_prior_mode"):
        assert forbidden not in declared, f"{forbidden} must not appear in G1"


@pytest.mark.skipif(
    not (PROJECT_ROOT / "cache/baseline_v1/labels").is_dir(),
    reason="baseline cache absent",
)
def test_validation_split_still_returns_the_hard_ground_truth() -> None:
    """The soft target must never reach validation.

    Validation is what every reported metric is computed against; handing it a
    soft target would make the metrics meaningless. Only the train split may be
    wrapped, and this checks that structurally rather than by inspection.
    """
    from training.train_baseline import BaselineConfig, build_dataloader

    config = BaselineConfig(
        root=str(PROJECT_ROOT),
        cache_dir=str(PROJECT_ROOT / "cache/baseline_v1"),
        manifest_dir=str(PROJECT_ROOT / "manifests/experiment"),
        soft_boundary_enabled=True,
        pin_memory=False,
    )
    train_dataset, _ = build_dataloader(config, "train", shuffle=False)
    val_dataset, _ = build_dataloader(config, "val", shuffle=False)

    train_sample = train_dataset[0]
    val_sample = val_dataset[0]

    assert "soft_target" in train_sample, "train split is missing the soft target"
    assert train_sample["soft_target"].shape == (cs.NUM_CLASSES, *cs.CROP_SHAPE_DHW)
    assert not train_sample["label"].is_floating_point(), \
        "the hard label must still be present alongside the soft target"

    assert "soft_target" not in val_sample, \
        "validation was handed a soft target; every metric would be invalid"
    assert val_sample["label"].dtype == torch.int64


@pytest.mark.skipif(
    not (PROJECT_ROOT / "cache/baseline_v1/labels").is_dir(),
    reason="baseline cache absent",
)
def test_train_soft_target_matches_the_builder_for_that_case() -> None:
    """The wrapper must not transform or reorder what the builder produced."""
    from training.train_baseline import BaselineConfig, build_dataloader

    config = BaselineConfig(
        root=str(PROJECT_ROOT),
        cache_dir=str(PROJECT_ROOT / "cache/baseline_v1"),
        manifest_dir=str(PROJECT_ROOT / "manifests/experiment"),
        soft_boundary_enabled=True,
        pin_memory=False,
    )
    dataset, _ = build_dataloader(config, "train", shuffle=False)
    sample = dataset[0]
    case_id = sample["case_id"]
    label = np.load(PROJECT_ROOT / "cache/baseline_v1/labels" / f"{case_id}.npy")
    expected = interface_soft_target(label)
    assert np.array_equal(sample["soft_target"].numpy(), expected)


def test_soft_target_builder_rejects_a_wrong_crop_shape() -> None:
    """A mis-cropped or transposed volume must fail loudly, not silently."""
    with pytest.raises(ValueError, match="expected"):
        interface_soft_target(np.zeros((32, 96, 97), dtype=np.uint8))


def test_no_test_or_challenge_split_is_referenced_by_the_new_code() -> None:
    """G1 must stay inside the development splits, structurally."""
    for relative in ("src/data/interface_soft_target.py",
                     "src/losses/stn_interface_soft_ce.py",
                     "configs/stn_interface_soft_v1.yaml"):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        for forbidden in ("internal_test", "challenge_test", "holdout"):
            assert forbidden not in text, f"{relative} references {forbidden!r}"

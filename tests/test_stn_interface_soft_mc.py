"""Tests for Experiment G1-MC: the mass-conserving interface soft target.

G1-MC exists to remove one specific defect, so the tests are mostly about that
defect being gone while nothing else moved:

* per-case STN target mass must equal the hard count **exactly**, not
  approximately, and for every case;
* the soft support, the inside shell, the out-of-band voxels, the band, the tie
  rule and the class semantics must be **identical to G1** -- the correction is
  supposed to touch only the outside shell's magnitude;
* ``gamma`` must come from each case's own balance. A global ``gamma`` would look
  right on aggregate and be wrong per case, so that is checked directly.

The baseline path is checked too: G1-MC shares G1's loss, and the hard-label path
must still be the baseline's own ``DiceCELoss`` bit for bit.

Run with:  ./.venv/Scripts/python.exe -m pytest tests/test_stn_interface_soft_mc.py -v
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
)
from data.interface_soft_target_mass_conserving import (  # noqa: E402
    conservation_terms,
    mass_conserving_soft_target,
)
from losses.dice_ce import DiceCELoss, DiceCELossConfig  # noqa: E402
from losses.stn_interface_soft_ce import InterfaceSoftDiceCELoss  # noqa: E402

BACKGROUND, SN, RN = 0, 2, 3
CENTRE = (4, 4, 4)

needs_cache = pytest.mark.skipif(
    not (PROJECT_ROOT / "cache/baseline_v1/labels").is_dir(),
    reason="baseline cache absent",
)


def synthetic(neighbours: dict[tuple[int, int, int], int] | None = None,
              core: tuple[int, int, int] = (3, 6)) -> np.ndarray:
    label = np.zeros((9, 9, 9), dtype=np.uint8)
    label[core[0]:core[1], core[0]:core[1], core[0]:core[1]] = STN
    for (dd, dh, dw), value in (neighbours or {}).items():
        label[CENTRE[0] + dd, CENTRE[1] + dh, CENTRE[2] + dw] = value
    return label


def mc(label: np.ndarray) -> np.ndarray:
    return mass_conserving_soft_target(label, expected_shape=None)


def g1(label: np.ndarray) -> np.ndarray:
    return interface_soft_target(label, expected_shape=None)


def _real_cases(limit: int = 8) -> list[str]:
    import pandas as pd
    manifest = PROJECT_ROOT / "manifests/experiment/train.csv"
    return sorted(pd.read_csv(manifest)["case_id"].astype(str))[:limit]


# --------------------------------------------------------------------------- #
# 1-2. conservation and the simplex
# --------------------------------------------------------------------------- #

#: The construction is exact in float64; the returned float32 target carries only
#: the rounding of casting and of summing ~250 float32 values. Measured across
#: development train this is <= 6e-9 relative, so this bound has ~15x of margin.
FLOAT32_CONSERVATION_RTOL = 1e-7


def test_per_case_stn_mass_is_conserved_to_float32_precision() -> None:
    """sum_v q'_STN == hard STN count, up to float32 accumulation only."""
    for neighbours in ({(0, 0, -1): BACKGROUND},
                       {(0, 0, -1): BACKGROUND, (0, -1, 0): SN},
                       {(0, 0, -1): BACKGROUND, (0, -1, 0): RN, (0, 1, 0): SN}):
        label = synthetic(neighbours)
        q = mc(label)
        hard = float((label == STN).sum())
        assert q[STN].sum(dtype=np.float64) == pytest.approx(
            hard, rel=FLOAT32_CONSERVATION_RTOL)


def test_conservation_holds_where_g1_does_not() -> None:
    """The property that distinguishes G1-MC: G1 over-delivers, G1-MC does not."""
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    hard = float((label == STN).sum())
    g1_mass = g1(label)[STN].sum(dtype=np.float64)
    assert g1_mass > hard * 1.05, "G1 should over-deliver clearly, not marginally"
    assert mc(label)[STN].sum(dtype=np.float64) == pytest.approx(
        hard, rel=FLOAT32_CONSERVATION_RTOL)


def test_probability_simplex_holds_everywhere() -> None:
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN, (0, 1, 0): RN})
    q = mc(label)
    assert (q >= 0.0).all() and (q <= 1.0).all()
    assert np.allclose(q.sum(axis=0), 1.0, atol=1e-6)
    assert np.isfinite(q).all()


def test_outside_added_equals_inside_removed() -> None:
    """The identity the whole design rests on: added == removed, per case.

    Measured on the returned float32 target, so the tolerance is that of float32
    accumulation. The float64 assembly is exact by construction -- ``gamma`` is
    ``R/A``, so ``gamma * A == R`` -- and this is what that looks like after the
    cast.
    """
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    q = mc(label).astype(np.float64)
    mask = label == STN
    removed, added, _ = conservation_terms(q[STN], mask)
    assert added == pytest.approx(removed, rel=FLOAT32_CONSERVATION_RTOL)


# --------------------------------------------------------------------------- #
# 3-5. what must NOT have changed relative to G1
# --------------------------------------------------------------------------- #

def test_soft_support_is_identical_to_g1() -> None:
    """Only the outside shell's magnitude moves; which voxels soften must not."""
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN, (0, 1, 0): RN})
    mask = label == STN
    a, b = g1(label), mc(label)
    assert np.array_equal(np.abs(a[STN] - mask) > 1e-9,
                          np.abs(b[STN] - mask) > 1e-9)


def test_inside_q_stn_is_bit_identical_to_g1() -> None:
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    mask = label == STN
    assert np.array_equal(g1(label)[STN][mask], mc(label)[STN][mask])


def test_out_of_band_voxels_stay_bit_identical_one_hot() -> None:
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    a, b = g1(label), mc(label)
    one_hot = np.eye(cs.NUM_CLASSES, dtype=np.float32)[label].transpose(3, 0, 1, 2)
    d, _ = __import__("data.interface_soft_target", fromlist=["x"]).nearest_face_counts(
        label, expected_shape=None)
    far = np.abs(d) >= BAND_MM
    assert far.any()
    assert np.array_equal(a[:, far], b[:, far])
    assert np.array_equal(b[:, far], one_hot[:, far])


def test_outside_shell_is_scaled_uniformly_within_a_case() -> None:
    """A single per-case factor, so the two outside levels keep their ratio."""
    label = synthetic({(0, 0, -1): BACKGROUND, (0, 1, 0): STN, (0, 1, -1): SN})
    a, b = g1(label), mc(label)
    mask = label == STN
    _, _, gamma = conservation_terms(a[STN].astype(np.float64), mask)
    outside = ~mask
    assert np.allclose(b[STN][outside], a[STN][outside] * gamma, atol=1e-6)
    # ... and strictly lower than G1 everywhere it is non-zero.
    assert (b[STN][outside] <= a[STN][outside] + 1e-9).all()


def test_class_semantics_are_preserved() -> None:
    """Reduced STN mass goes to the voxel's own hard class and nowhere else."""
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    q = mc(label)
    mask = label == STN
    for klass in RESIDUAL_CLASSES:
        wrong = (~mask) & (label != klass)
        assert np.array_equal(q[klass][wrong],
                              np.zeros(int(wrong.sum()), dtype=q.dtype))


def test_band_membership_is_unchanged() -> None:
    from data.interface_soft_target import nearest_face_counts
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    d, _ = nearest_face_counts(label, expected_shape=None)
    in_band = np.abs(d) < BAND_MM
    a, b = g1(label), mc(label)
    assert (np.abs(b[STN] - (label == STN)) > 1e-9).sum() == int(
        (in_band).sum())


# --------------------------------------------------------------------------- #
# 6. gamma is per case and comes only from the mass balance
# --------------------------------------------------------------------------- #

def test_gamma_is_per_case_not_global() -> None:
    """Two different cases must get different gammas, each conserving exactly."""
    small = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    large = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN,
                       (0, 0, 1): BACKGROUND, (0, 1, 0): BACKGROUND})

    gammas = []
    for label in (small, large):
        mask = label == STN
        _, _, gamma = conservation_terms(
            g1(label)[STN].astype(np.float64), mask)
        gammas.append(gamma)
        assert float(mc(label)[STN].sum(dtype=np.float64)) == pytest.approx(
            float(mask.sum()), rel=FLOAT32_CONSERVATION_RTOL)

    assert gammas[0] != gammas[1], "gamma must vary with the case, not be global"


def test_a_global_gamma_would_break_per_case_conservation() -> None:
    """Shows the per-case requirement is load-bearing, not cosmetic."""
    a = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    b = synthetic({(0, 0, -1): BACKGROUND, (0, 0, 1): BACKGROUND,
                   (0, 1, 0): BACKGROUND})
    ga = conservation_terms(g1(a)[STN].astype(np.float64), a == STN)[2]
    gb = conservation_terms(g1(b)[STN].astype(np.float64), b == STN)[2]
    mean_gamma = (ga + gb) / 2.0

    # Apply the mean gamma to case a directly; it must miss the hard count.
    mask = a == STN
    q = g1(a).astype(np.float64)
    applied = float(q[STN][mask].sum() + mean_gamma * q[STN][~mask].sum())
    assert abs(applied - float(mask.sum())) > 1e-4, \
        "a global gamma should visibly miss the hard count on this case"


def test_builder_is_stateless_across_cases() -> None:
    """No cross-case state: a case's target is the same alone or in sequence."""
    a = synthetic({(0, 0, -1): BACKGROUND})
    b = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    alone = mc(a)
    _ = mc(b)
    assert np.array_equal(alone, mc(a))


def test_gamma_requires_some_outside_mass() -> None:
    """Guarded explicitly rather than dividing by zero."""
    from data.interface_soft_target_mass_conserving import conservation_terms
    q = np.zeros(10)
    mask = np.zeros(10, dtype=bool)
    with pytest.raises(ValueError, match="undefined"):
        conservation_terms(q, mask)


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #

@needs_cache
def test_conservation_on_real_cases() -> None:
    """Every sampled real case must conserve exactly."""
    for case_id in _real_cases():
        label = np.load(PROJECT_ROOT / "cache/baseline_v1/labels" / f"{case_id}.npy")
        if not (label == STN).any():
            continue
        q = mass_conserving_soft_target(label)
        hard = float((label == STN).sum())
        assert float(q[STN].sum(dtype=np.float64)) == pytest.approx(hard, rel=1e-7), case_id
        assert np.allclose(q.sum(axis=0), 1.0, atol=1e-6), case_id


@needs_cache
def test_g1_mass_bias_is_removed_and_support_preserved_on_real_data() -> None:
    for case_id in _real_cases(limit=4):
        label = np.load(PROJECT_ROOT / "cache/baseline_v1/labels" / f"{case_id}.npy")
        if not (label == STN).any():
            continue
        mask = label == STN
        a, b = interface_soft_target(label), mass_conserving_soft_target(label)
        assert a[STN].sum(dtype=np.float64) > mask.sum(), "G1 should over-deliver"
        assert b[STN].sum(dtype=np.float64) == pytest.approx(float(mask.sum()), rel=1e-7)
        assert np.array_equal(np.abs(a[STN] - mask) > 1e-9,
                              np.abs(b[STN] - mask) > 1e-9), case_id
        assert np.array_equal(a[STN][mask], b[STN][mask]), case_id


# --------------------------------------------------------------------------- #
# 7-9. wiring, isolation, and the baseline
# --------------------------------------------------------------------------- #

def test_mode_resolution_and_config_default() -> None:
    from training.train_baseline import (
        BaselineConfig, SOFT_BOUNDARY_MODES, resolve_soft_boundary_mode)
    assert set(SOFT_BOUNDARY_MODES) == {"interface", "mass_conserving"}
    assert BaselineConfig().soft_boundary_mode == "interface"
    assert BaselineConfig().soft_boundary_enabled is False
    with pytest.raises(ValueError, match="unknown soft_boundary_mode"):
        resolve_soft_boundary_mode("global_gamma")


def test_mass_conserving_config_differs_from_g1_only_in_declared_keys() -> None:
    from dataclasses import asdict
    import yaml
    from training.train_baseline import load_config

    config_dir = PROJECT_ROOT / "configs"
    g1 = asdict(load_config(config_dir / "stn_interface_soft_v1.yaml"))
    mc = asdict(load_config(config_dir / "stn_interface_soft_mc_v1.yaml"))
    differing = {k for k in set(g1) | set(mc) if g1.get(k) != mc.get(k)}
    assert differing == {"soft_boundary_mode", "checkpoint_dir", "results_dir"}, \
        f"unexpected differences: {sorted(differing)}"

    with (config_dir / "stn_interface_soft_mc_v1.yaml").open(encoding="utf-8") as fh:
        declared = yaml.safe_load(fh)
    assert declared["soft_boundary_mode"] == "mass_conserving"
    assert declared["soft_boundary_band_mm"] == 0.6666667, "band must stay frozen"
    assert declared["ce_weight"] == 1.0 and declared["dice_weight"] == 1.0
    for forbidden in ("augmentation_enabled", "threshold", "tversky_alpha",
                      "boundary_weight", "spatial_prior_mode", "gamma"):
        assert forbidden not in declared, f"{forbidden} must not appear"


@needs_cache
def test_validation_split_never_receives_a_soft_target() -> None:
    from training.train_baseline import BaselineConfig, build_dataloader
    config = BaselineConfig(
        root=str(PROJECT_ROOT),
        cache_dir=str(PROJECT_ROOT / "cache/baseline_v1"),
        manifest_dir=str(PROJECT_ROOT / "manifests/experiment"),
        soft_boundary_enabled=True,
        soft_boundary_mode="mass_conserving",
        pin_memory=False,
    )
    train_ds, _ = build_dataloader(config, "train", shuffle=False)
    val_ds, _ = build_dataloader(config, "val", shuffle=False)

    train_sample = train_ds[0]
    val_sample = val_ds[0]
    assert "soft_target" in train_sample
    assert train_sample["label"].dtype == torch.int64
    assert "soft_target" not in val_sample, \
        "validation was handed a soft target; every metric would be invalid"
    assert val_sample["label"].dtype == torch.int64


@needs_cache
def test_train_soft_target_is_the_mass_conserving_one() -> None:
    from training.train_baseline import BaselineConfig, build_dataloader
    config = BaselineConfig(
        root=str(PROJECT_ROOT),
        cache_dir=str(PROJECT_ROOT / "cache/baseline_v1"),
        manifest_dir=str(PROJECT_ROOT / "manifests/experiment"),
        soft_boundary_enabled=True,
        soft_boundary_mode="mass_conserving",
        pin_memory=False,
    )
    dataset, _ = build_dataloader(config, "train", shuffle=False)
    sample = dataset[0]
    label = np.load(PROJECT_ROOT / "cache/baseline_v1/labels"
                    / f"{sample['case_id']}.npy")
    expected = mass_conserving_soft_target(label)
    assert np.array_equal(sample["soft_target"].numpy(), expected)
    hard = float((label == STN).sum())
    delivered = float(sample["soft_target"][STN].sum().item())
    assert delivered == pytest.approx(hard, rel=FLOAT32_CONSERVATION_RTOL)


# --------------------------------------------------------------------------- #
# 8. baseline hard path
# --------------------------------------------------------------------------- #

def test_baseline_hard_path_is_still_the_baseline_criterion() -> None:
    """Delegation is bit-identical, so the baseline cannot have drifted."""
    torch.manual_seed(0)
    logits = torch.randn(2, cs.NUM_CLASSES, 8, 16, 16)
    labels = torch.randint(0, cs.NUM_CLASSES, (2, 8, 16, 16))
    soft_loss, soft_comp = InterfaceSoftDiceCELoss()(logits, labels)
    base_loss, base_comp = DiceCELoss(
        DiceCELossConfig(ce_weight=1.0, dice_weight=1.0))(logits, labels)
    assert torch.equal(soft_loss, base_loss)
    for key in base_comp:
        assert torch.equal(soft_comp[key], base_comp[key]), key


def test_build_criterion_defaults_are_untouched() -> None:
    from training.train_baseline import BaselineConfig, build_criterion
    assert isinstance(build_criterion(BaselineConfig()),
                      DiceCELoss)
    assert isinstance(build_criterion(BaselineConfig(loss_type="stn_interface_soft")),
                      InterfaceSoftDiceCELoss)


# --------------------------------------------------------------------------- #
# 10-12. numerical health of the soft path
# --------------------------------------------------------------------------- #

def test_soft_paths_are_finite_and_backward_succeeds() -> None:
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    q = mass_conserving_soft_target(label, expected_shape=None)
    soft = torch.from_numpy(np.stack([q])).float()
    criterion = InterfaceSoftDiceCELoss()

    logits_ce = torch.randn(1, cs.NUM_CLASSES, *label.shape, requires_grad=True)
    ce = criterion.soft_ce(logits_ce, soft)
    assert torch.isfinite(ce)
    ce.backward()
    assert torch.isfinite(logits_ce.grad).all()

    logits_dice = torch.randn(1, cs.NUM_CLASSES, *label.shape, requires_grad=True)
    matrix = criterion.per_case_per_class_dice_soft(logits_dice, soft)
    assert torch.isfinite(matrix).all()
    dice = 1.0 - matrix.mean()
    dice.backward()
    assert torch.isfinite(logits_dice.grad).all()

    logits = torch.randn(1, cs.NUM_CLASSES, *label.shape, requires_grad=True)
    loss, components = criterion(logits, soft)
    assert torch.isfinite(loss)
    for key, value in components.items():
        assert torch.isfinite(value), key
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for AMP")
def test_mass_conserving_target_is_finite_under_real_autocast() -> None:
    label = synthetic({(0, 0, -1): BACKGROUND, (0, -1, 0): SN})
    q = mass_conserving_soft_target(label, expected_shape=None)
    soft = torch.from_numpy(np.stack([q] * 2)).cuda()
    logits = torch.randn(2, cs.NUM_CLASSES, *label.shape, device="cuda")
    with torch.amp.autocast("cuda", enabled=True):
        loss, _ = InterfaceSoftDiceCELoss()(logits, soft)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# 9. isolation
# --------------------------------------------------------------------------- #

def test_no_forbidden_split_is_accessed() -> None:
    """No *access* to a reserved split.

    What matters is whether a reserved split is reachable as data, so the check
    is for path-like references rather than for any mention of a split name.
    """
    forbidden = ("internal_test.csv", "challenge_test.csv", "internal_test/",
                 "challenge_test/", "internal_test\"", "secondary_frozen_holdout")
    for relative in ("src/data/interface_soft_target_mass_conserving.py",
                     "src/losses/stn_interface_soft_ce.py",
                     "configs/stn_interface_soft_mc_v1.yaml"):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text, f"{relative} references {pattern!r}"

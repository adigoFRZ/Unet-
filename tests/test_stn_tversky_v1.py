"""Unit tests for Experiment B -- the STN-specific asymmetric overlap loss.

The tests that matter most here are the ones pinning the **parameter
convention**. The Tversky index is written both ways in the literature: this
experiment uses ``alpha`` for FALSE POSITIVES and ``beta`` for FALSE NEGATIVES,
which is the opposite of Salehi et al.'s focal-Tversky paper. A mix-up does not
raise, does not produce NaN and does not even look wrong -- it silently moves the
loss in the direction the experiment was meant to correct, and the run would
report a plausible-looking number. So the convention is asserted three ways:

  * at the counts level, where nothing else is in the way (``..._from_counts``);
  * by swapping alpha and beta and requiring the ordering to REVERSE, which no
    accidental convention can satisfy;
  * end to end through the real tensor path, and by gradient sign.

The remaining tests guard the two parts of the loss that must NOT have changed:
the SN/RN soft Dice and the CE term, both compared bitwise against the frozen
Baseline v1 implementation.

Run with:  python -m pytest tests/test_stn_tversky_v1.py -v
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

from losses.dice_ce import DiceCELoss, DiceCELossConfig  # noqa: E402
from losses.tversky_dice_ce import (  # noqa: E402
    STN,
    HybridOverlapCELoss,
    HybridOverlapCELossConfig,
    tversky_index_from_counts,
)

ALPHA_FP = 0.6
BETA_FN = 0.4


@pytest.fixture
def criterion() -> HybridOverlapCELoss:
    return HybridOverlapCELoss(HybridOverlapCELossConfig(
        tversky_alpha=ALPHA_FP, tversky_beta=BETA_FN))


def _three_class_target(n: int = 2, size: int = 16) -> torch.Tensor:
    """``n`` cases with STN, SN and RN all present.

    All three classes must be present for the identity test against soft Dice:
    where a class is absent the epsilon dominates and the two formulas separate
    by design (see ``test_eps_convention_*``).
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


def _exact_count_case(tp: int = 100, fp: int = 80, fn: int = 20):
    """A tensor pair whose STN counts are EXACTLY (tp, fp, fn).

    Built by saturating the softmax: inside the "predicted STN" region the STN
    logit dominates and inside it p_STN is 1.0 to within float32, so ``sum(p*g)``
    is an exact voxel count rather than the usual soft approximation. That is what
    makes a counts-level assertion possible through the real tensor path.
    """
    shape = (8, 16, 16)
    target = np.zeros(shape, dtype=np.int64)
    predicted = np.zeros(shape, dtype=bool)
    target.flat[0:tp] = 1                      # true positives
    target.flat[tp + fp:tp + fp + fn] = 1      # false negatives
    predicted.flat[0:tp + fp] = True           # true positives + false positives

    logits = torch.full((1, 4, *shape), -20.0)
    # Background dominates everywhere (so p_STN is ~0 outside the prediction),
    # except inside the predicted region, where STN dominates instead.
    logits[0, 0] = 20.0
    logits[0, 0][torch.from_numpy(predicted)] = -20.0
    logits[0, 1][torch.from_numpy(predicted)] = 20.0
    return logits, torch.from_numpy(target)[None]


# --------------------------------------------------------------------------- #
# 1-3. the alpha/beta convention
# --------------------------------------------------------------------------- #


def test_more_false_positives_increases_stn_tversky_loss() -> None:
    """Fixed TP and FN: adding false positives must make the loss worse."""
    base = 1.0 - tversky_index_from_counts(100, 80, 20, ALPHA_FP, BETA_FN)
    with_more_fp = 1.0 - tversky_index_from_counts(100, 80 + 10, 20, ALPHA_FP, BETA_FN)
    assert with_more_fp > base


def test_fn_increase_costs_less_than_the_same_fp_increase() -> None:
    """The property the whole experiment rests on: alpha=0.6 > beta=0.4.

    Same TP, same starting FP/FN, same increment -- the FP increment must raise
    the loss by MORE than the FN increment, otherwise the loss does not fight
    over-segmentation at all.
    """
    delta = 10
    base = 1.0 - tversky_index_from_counts(100, 80, 20, ALPHA_FP, BETA_FN)
    d_fp = (1.0 - tversky_index_from_counts(100, 80 + delta, 20, ALPHA_FP, BETA_FN)) - base
    d_fn = (1.0 - tversky_index_from_counts(100, 80, 20 + delta, ALPHA_FP, BETA_FN)) - base

    assert d_fp > d_fn > 0, (
        f"alpha={ALPHA_FP} must weight FALSE POSITIVES and beta={BETA_FN} FALSE "
        f"NEGATIVES; got dLoss(FP+{delta})={d_fp:.8f} <= dLoss(FN+{delta})={d_fn:.8f}"
    )


def test_swapping_alpha_and_beta_reverses_which_error_is_penalised() -> None:
    """Swapping the coefficients must flip the ordering, not just shrink it.

    A test that only checked "FP costs more" could pass by accident if some other
    quantity were doing the work. Requiring the ordering to REVERSE when the
    coefficients swap leaves no room for that: the sign of the effect can only
    come from alpha and beta themselves.
    """
    delta = 10
    fn_weighted = dict(alpha=0.4, beta=0.6)   # FN now the expensive one

    base = 1.0 - tversky_index_from_counts(100, 80, 20, **fn_weighted)
    d_fp = (1.0 - tversky_index_from_counts(100, 80 + delta, 20, **fn_weighted)) - base
    d_fn = (1.0 - tversky_index_from_counts(100, 80, 20 + delta, **fn_weighted)) - base

    assert d_fn > d_fp > 0, (
        "with alpha=0.4 < beta=0.6 the FN increment must now cost more; the "
        "coefficients are not reaching the formula the way the names claim"
    )


def test_end_to_end_counts_match_the_counts_formula() -> None:
    """The tensor path must compute the same TP/FP/FN the formula assumes.

    Guards the definitions (``fp = sum(p*(1-g))``, ``fn = sum((1-p)*g)``), which
    is where an off-by-one-class or a swapped term would hide while every
    monotonicity test above still passed.
    """
    criterion = HybridOverlapCELoss()
    logits, target = _exact_count_case(tp=100, fp=80, fn=20)

    probability = torch.softmax(logits, dim=1)[0, STN]
    truth = (target[0] == STN).to(probability.dtype)
    tp = float((probability * truth).sum())
    fp = float((probability * (1.0 - truth)).sum())
    fn = float(((1.0 - probability) * truth).sum())
    assert (tp, fp, fn) == pytest.approx((100.0, 80.0, 20.0), abs=1e-4), (
        f"the test fixture does not produce the intended counts: {(tp, fp, fn)}"
    )

    actual = float(criterion.per_case_per_class_tversky(logits, target)[0, 0])
    expected = float(tversky_index_from_counts(100, 80, 20, ALPHA_FP, BETA_FN))
    assert actual == pytest.approx(expected, abs=1e-6)


def test_alpha_beta_swap_changes_the_loss_on_a_real_tensor() -> None:
    """End-to-end counterpart of the ordering test, through softmax and reduction."""
    logits, target = _exact_count_case(tp=100, fp=80, fn=20)
    fp_expensive = HybridOverlapCELoss(HybridOverlapCELossConfig(
        tversky_alpha=0.6, tversky_beta=0.4))
    fn_expensive = HybridOverlapCELoss(HybridOverlapCELossConfig(
        tversky_alpha=0.4, tversky_beta=0.6))

    loss_fp = float(fp_expensive.per_case_per_class_tversky(logits, target)[0, 0])
    loss_fn = float(fn_expensive.per_case_per_class_tversky(logits, target)[0, 0])
    # This case has FP=80 > FN=20, so weighting FP more must give the LOWER index
    # (a higher loss).
    assert loss_fp < loss_fn


def test_gradient_pushes_stn_down_where_it_is_wrong_and_up_where_it_is_right() -> None:
    """The direction that actually matters at training time.

    On the overlap term alone: the mean gradient of the STN logit must be
    POSITIVE on background voxels (gradient descent lowers p_STN there, shrinking
    the prediction) and NEGATIVE on STN-target voxels (raising p_STN there). This
    is "the loss fights over-segmentation" stated directly, without relying on
    any monotonicity argument.
    """
    criterion = HybridOverlapCELoss()
    torch.manual_seed(3)
    logits = torch.randn(1, 4, 8, 16, 16) * 0.8
    target = torch.zeros(1, 8, 16, 16, dtype=torch.long)
    target[0, 2:4, 2:6, 2:6] = STN
    target[0, 4:6, 6:12, 6:12] = 2
    logits[0, STN][target[0] == STN] += 1.5      # a partly over-segmented prediction
    logits.requires_grad_(True)

    criterion.overlap_loss(logits, target).backward()
    grad = logits.grad[0, STN]

    background = (target[0] != STN)
    foreground = (target[0] == STN)
    assert float(grad[background].mean()) > 0, (
        "background voxels must be pushed DOWN; the loss is not fighting "
        "over-segmentation"
    )
    assert float(grad[foreground].mean()) < 0, (
        "STN voxels must be pushed UP; the loss is not recovering missed STN"
    )


# --------------------------------------------------------------------------- #
# 4-6. the terms that must NOT have changed
# --------------------------------------------------------------------------- #


def test_sn_and_rn_dice_are_bitwise_identical_to_baseline() -> None:
    """SN and RN must be the Baseline's own Dice, not a re-derivation of it.

    Compared with ``torch.equal`` rather than ``allclose``: the requirement is
    that these two columns are the same computation, and a tolerance would hide a
    changed ``smooth`` or a different reduction order.
    """
    criterion = HybridOverlapCELoss()
    baseline = DiceCELoss(DiceCELossConfig())
    torch.manual_seed(11)
    logits = torch.randn(2, 4, 8, 16, 16)
    target = _three_class_target(n=2)

    ours = criterion.per_case_per_class_dice(logits, target)
    theirs = baseline.per_case_per_class_dice(logits, target)[:, [1, 2]]

    assert ours.shape == theirs.shape == (2, 2)
    assert torch.equal(ours, theirs), (
        "the SN/RN Dice columns differ from the frozen baseline implementation"
    )


def test_cross_entropy_matches_unweighted_baseline() -> None:
    """CE must stay the plain unweighted cross-entropy Baseline v1 uses."""
    criterion = HybridOverlapCELoss()
    torch.manual_seed(5)
    logits = torch.randn(2, 4, 8, 16, 16)
    target = _three_class_target(n=2)

    _, components = criterion(logits, target)
    expected = F.cross_entropy(logits, target.long(), weight=None)
    assert torch.equal(components["ce"], expected.detach())


def test_overlap_is_one_minus_the_mean_over_cases_and_classes() -> None:
    """``1 - mean(matrix)``, over the (case, class) matrix -- STN via Tversky."""
    criterion = HybridOverlapCELoss()
    torch.manual_seed(13)
    logits = torch.randn(2, 4, 8, 16, 16)
    target = _three_class_target(n=2)

    matrix = criterion.per_case_per_class_overlap(logits, target)
    assert matrix.shape == (2, 3), "expected one column per foreground class"
    assert float(criterion.overlap_loss(logits, target)) == pytest.approx(
        1.0 - float(matrix.mean()), abs=1e-6)

    stn = criterion.per_case_per_class_tversky(logits, target)[:, 0]
    assert torch.equal(matrix[:, 0], stn), "column 0 must be the STN Tversky index"


def test_tversky_half_half_reproduces_soft_dice() -> None:
    """alpha = beta = 0.5 is the symmetric case, which must reduce to soft Dice.

    Compared with ``allclose`` and not ``torch.equal``: the baseline writes its
    smoothing as ``(2*TP + eps) / (...)`` whereas the Tversky form with
    alpha = beta = 0.5 is ``(TP + eps) / (...)``, which is the same function only
    up to ``smooth = eps/2``. On volumes this size that is ~1e-8. The
    eps-dominated regime, where the two genuinely separate, is pinned by
    ``test_eps_convention_*`` below.
    """
    torch.manual_seed(17)
    logits = torch.randn(2, 4, 8, 16, 16)
    target = _three_class_target(n=2)

    symmetric = HybridOverlapCELoss(HybridOverlapCELossConfig(
        tversky_alpha=0.5, tversky_beta=0.5))
    baseline = DiceCELoss(DiceCELossConfig())

    tv = symmetric.per_case_per_class_tversky(logits, target)[:, 0]
    dice = baseline.per_case_per_class_dice(logits, target)[:, 0]
    assert torch.allclose(tv, dice, atol=1e-6, rtol=0), (
        f"Tversky(0.5, 0.5) should equal soft Dice; max diff "
        f"{float((tv - dice).abs().max()):.3e}"
    )


def test_eps_convention_separates_only_where_the_class_is_absent() -> None:
    """Records the smoothing convention so it cannot drift unnoticed.

    With a class absent from the ground truth the epsilon stops being negligible
    and the two formulas stop agreeing -- Tversky(0.5, 0.5) comes out exactly
    twice the baseline Dice value there. That is a property of writing the
    smoothing as ``(TP + eps) / (...)`` (the protocol's form) rather than as
    ``(2*TP + eps) / (...)`` (the baseline's), and it is harmless here because STN
    is present in every case, but it should fail loudly if someone "fixes" the
    smoothing one way in one place and the other way elsewhere.
    """
    criterion = HybridOverlapCELoss(HybridOverlapCELossConfig(
        tversky_alpha=0.5, tversky_beta=0.5))
    baseline = DiceCELoss(DiceCELossConfig())
    uniform = torch.zeros(1, 4, 8, 16, 16)                  # p = 0.25 everywhere
    empty = torch.zeros(1, 8, 16, 16, dtype=torch.long)     # no foreground at all

    tv = float(criterion.per_case_per_class_tversky(uniform, empty)[0, 0])
    dice = float(baseline.per_case_per_class_dice(uniform, empty)[0, 0])

    assert dice > 0 and tv > 0
    assert tv / dice == pytest.approx(2.0, rel=1e-3), (
        f"eps convention changed: tversky/dice = {tv / dice:.6f}, expected 2.0"
    )


# --------------------------------------------------------------------------- #
# 7. per-case reduction
# --------------------------------------------------------------------------- #


def test_per_case_reduction_is_not_pooled_over_the_batch() -> None:
    """A case's own overlap must not depend on the other cases in its batch.

    Batch pooling would weight every case by its target volume, so a case with
    big structures would dominate the gradient while a case with small ones
    barely contributed -- yet evaluation is an unweighted per-case mean. The two
    reductions must agree.
    """
    criterion = HybridOverlapCELoss()
    torch.manual_seed(7)

    target_a = _three_class_target(n=1)
    logits_a = torch.randn(1, 4, 8, 16, 16)
    alone = criterion.per_case_per_class_overlap(logits_a, target_a)

    target_b = _three_class_target(n=2)[1:2]
    logits_b = torch.randn(1, 4, 8, 16, 16)
    batched = criterion.per_case_per_class_overlap(
        torch.cat([logits_a, logits_b], dim=0),
        torch.cat([target_a, target_b], dim=0),
    )

    assert torch.allclose(alone[0], batched[0], atol=1e-6), (
        "case 0's overlap changed when another case joined the batch -- the "
        "reduction is still pooled over the batch"
    )


def test_reduction_is_macro_over_cases_not_volume_weighted() -> None:
    """The macro mean must differ from the volume-weighted pooling it forbids."""
    criterion = HybridOverlapCELoss()
    torch.manual_seed(19)
    logits = torch.randn(2, 4, 8, 16, 16)
    target = _three_class_target(n=2)

    matrix = criterion.per_case_per_class_overlap(logits, target)
    volumes = torch.tensor([
        [float((target[i] == c).sum()) for c in cs.FOREGROUND_CLASSES]
        for i in range(2)
    ])
    pooled = float((matrix * volumes).sum() / volumes.sum())
    macro = float(matrix.mean())

    assert abs(macro - pooled) > 1e-4, (
        "the two reductions coincide here, so this test cannot detect pooling; "
        "make the per-case target volumes more different"
    )
    assert float(criterion.overlap_loss(logits, target)) == pytest.approx(
        1.0 - macro, abs=1e-6)


# --------------------------------------------------------------------------- #
# 8-11. finiteness, backward, AMP
# --------------------------------------------------------------------------- #


def test_loss_is_finite_and_components_are_present() -> None:
    criterion = HybridOverlapCELoss()
    logits = torch.randn(2, 4, 32, 96, 96)
    target = torch.zeros(2, 32, 96, 96, dtype=torch.long)
    target[:, 10:14, 40:50, 40:50] = 1
    target[:, 14:18, 50:60, 50:60] = 2
    target[:, 18:22, 60:70, 60:70] = 3

    loss, components = criterion(logits, target)

    for key in ("loss", "ce", "overlap", "stn_tversky_loss",
                "sn_dice_loss", "rn_dice_loss"):
        assert key in components, f"missing component {key!r}"
        assert torch.isfinite(components[key]), f"component {key!r} is not finite"
    assert 0.0 <= float(components["overlap"]) <= 1.0
    assert torch.isfinite(loss)


def test_stn_term_is_named_tversky_not_dice() -> None:
    """The STN term is not a Dice and must not be labelled as one."""
    criterion = HybridOverlapCELoss()
    logits = torch.randn(1, 4, 8, 16, 16)
    target = _three_class_target(n=1)
    _, components = criterion(logits, target)

    assert "stn_tversky_loss" in components
    assert "stn_dice_loss" not in components


def test_component_keys_map_to_baseline_safe_column_names() -> None:
    """The history column map must not rename the Baseline's own columns."""
    from training.train_baseline import COMPONENT_COLUMNS

    assert COMPONENT_COLUMNS["loss"] == "train_total_loss"
    assert COMPONENT_COLUMNS["ce"] == "train_ce_loss"
    assert COMPONENT_COLUMNS["dice"] == "train_dice_loss"
    assert COMPONENT_COLUMNS["overlap"] == "train_overlap_loss"
    # `loss` must NOT be routed to f"train_{key}", which would be train_loss_loss.
    assert COMPONENT_COLUMNS["loss"] != "train_loss"


def test_backward_produces_finite_nonzero_gradients() -> None:
    torch.manual_seed(0)
    from models.anisotropic_unet3d import AnisotropicUNet3D, UNet3DConfig

    model = AnisotropicUNet3D(UNet3DConfig(base_channels=8))
    criterion = HybridOverlapCELoss()
    x = torch.randn(1, 3, 32, 96, 96)
    target = torch.zeros(1, 32, 96, 96, dtype=torch.long)
    target[0, 10:14, 40:50, 40:50] = 1

    loss, _ = criterion(model(x), target)
    assert torch.isfinite(loss)
    loss.backward()

    total = 0.0
    nonzero_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all(), "non-finite gradient"
        total += float(parameter.grad.abs().sum())
        if parameter.grad.abs().sum() > 0:
            nonzero_tensors += 1
    assert total > 0.0, "all gradients are zero -- backward is not working"
    assert nonzero_tensors > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for autocast")
def test_amp_autocast_path_is_finite() -> None:
    """Exercise the real fp16 autocast path the training loop uses.

    This is not a formality. The crop has 32*96*96 = 294912 voxels, and an fp16
    sum over that many values overflows to inf (fp16 tops out at 65504). The
    accumulators stay safe only because autocast promotes ``softmax`` and ``sum``
    to fp32 -- so the check has to run on the same ops the trainer runs, on a
    CUDA device, under autocast. A CPU-only test cannot see this failure at all.
    """
    criterion = HybridOverlapCELoss().cuda()
    logits = torch.randn(2, 4, 32, 96, 96, device="cuda", requires_grad=True)
    target = torch.zeros(2, 32, 96, 96, dtype=torch.long, device="cuda")
    target[:, 10:14, 40:50, 40:50] = 1
    target[:, 14:18, 50:60, 50:60] = 2
    target[:, 18:22, 60:70, 60:70] = 3

    scaler = torch.amp.GradScaler("cuda", enabled=True)
    with torch.amp.autocast("cuda", enabled=True):
        loss, components = criterion(logits, target)

    assert loss.dtype == torch.float32, (
        f"the loss must come out of autocast as fp32, got {loss.dtype}")
    for key, value in components.items():
        assert torch.isfinite(value), f"component {key!r} is not finite under autocast"
        assert value.dtype == torch.float32, f"{key!r} should be fp32, got {value.dtype}"

    scaler.scale(loss).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), "non-finite input gradient under autocast"


# --------------------------------------------------------------------------- #
# configuration guards
# --------------------------------------------------------------------------- #


def test_rejects_a_class_partition_that_is_not_the_foreground_classes() -> None:
    """A wrong column order would silently scramble the per-class diagnostics."""
    with pytest.raises(ValueError):
        HybridOverlapCELoss(HybridOverlapCELossConfig(
            tversky_class=2, dice_classes=(1, 3)))       # (2, 1, 3) != (1, 2, 3)
    with pytest.raises(ValueError):
        HybridOverlapCELoss(HybridOverlapCELossConfig(tversky_class=99))


def test_build_criterion_default_is_bitwise_the_baseline_loss() -> None:
    """``loss_type`` defaults to the baseline, and yields exactly its criterion."""
    from training.train_baseline import BaselineConfig, build_criterion

    config = BaselineConfig()
    assert config.loss_type == "dice_ce"
    criterion = build_criterion(config)
    assert isinstance(criterion, DiceCELoss)
    assert not isinstance(criterion, HybridOverlapCELoss)

    reference = DiceCELoss(DiceCELossConfig(
        ce_weight=config.ce_weight, dice_weight=config.dice_weight))
    torch.manual_seed(23)
    logits = torch.randn(2, 4, 8, 16, 16)
    target = _three_class_target(n=2)

    ours, ours_components = criterion(logits, target)
    theirs, theirs_components = reference(logits, target)
    assert torch.equal(ours, theirs)
    assert set(ours_components) == set(theirs_components) == {"loss", "ce", "dice"}
    for key in ours_components:
        assert torch.equal(ours_components[key], theirs_components[key])


def test_build_criterion_rejects_an_unknown_loss_type() -> None:
    from training.train_baseline import BaselineConfig, build_criterion

    with pytest.raises(ValueError, match="unknown loss_type"):
        build_criterion(BaselineConfig(loss_type="focal_tversky"))


def test_experiment_config_differs_from_baseline_only_in_the_declared_keys() -> None:
    """Experiment B is only interpretable if the loss is the single change.

    ``configs/subject_clean_v1/stn_tversky_v1.yaml`` claims to differ from the
    baseline in a small set of declared places and that claim lives only in a
    comment. Here it is enforced: any accidental edit to a frozen knob fails
    this test.
    """
    from dataclasses import asdict

    import yaml

    from training.train_baseline import load_config

    config_dir = PROJECT_ROOT / "configs" / "subject_clean_v1"
    baseline = asdict(load_config(config_dir / "baseline_v1.yaml"))
    experiment = asdict(load_config(config_dir / "stn_tversky_v1.yaml"))

    differing = {key for key in set(baseline) | set(experiment)
                 if baseline.get(key) != experiment.get(key)}
    assert differing == {
        "loss_type",           # the experimental variable
        "pin_memory",          # environment workaround, documented as not an
                               # experimental variable (see the config comment)
        "checkpoint_dir",      # output routing
        "results_dir",
    }, f"unexpected config differences: {sorted(differing)}"

    # tversky_alpha/beta are absent from `differing` only because the dataclass
    # defaults already carry the experiment's values, so the parsed configs agree.
    # What actually matters for reproducibility is that the FILE declares them:
    # the run is not self-describing otherwise, and the loss never enters a
    # checkpoint's state_dict -- run_config.yaml is the only provenance.
    with (config_dir / "stn_tversky_v1.yaml").open(encoding="utf-8") as handle:
        declared = yaml.safe_load(handle)
    assert declared["tversky_alpha"] == ALPHA_FP == 0.6
    assert declared["tversky_beta"] == BETA_FN == 0.4
    assert declared["tversky_class"] == STN == 1
    assert declared["loss_type"] == "stn_tversky"
    # The baseline config must NOT declare any of them: it takes the defaults and
    # never reaches the Tversky code path.
    with (config_dir / "baseline_v1.yaml").open(encoding="utf-8") as handle:
        baseline_declared = yaml.safe_load(handle)
    assert not {"loss_type", "tversky_alpha", "tversky_beta",
                "tversky_class"} & set(baseline_declared)

    # And the knobs the protocol freezes must be identical on both sides.
    for key, value in (("seed", 42), ("batch_size", 2), ("max_epochs", 200),
                       ("early_stopping_patience", 40), ("ce_weight", 1.0),
                       ("dice_weight", 1.0)):
        assert baseline[key] == experiment[key] == value, key
    assert baseline["learning_rate"] == experiment["learning_rate"] == 3.0e-4
    assert baseline["weight_decay"] == experiment["weight_decay"] == 1.0e-5
    assert experiment["augmentation_enabled"] is False


def test_experiment_config_keeps_the_frozen_crop_and_split() -> None:
    """The crop and the split the protocol allows, restated as a guard."""
    from training.train_baseline import load_config

    config = load_config(PROJECT_ROOT / "configs" / "subject_clean_v1"
                         / "stn_tversky_v1.yaml")
    assert cs.CROP_SHAPE_XYZ == (96, 96, 32)
    assert cs.CROP_SHAPE_DHW == (32, 96, 96)
    assert cs.IN_CHANNELS == config.in_channels == 3
    assert cs.NUM_CLASSES == config.num_classes == 4
    assert config.base_channels == 16
    # The subject-clean manifests are the split the reported results use.
    assert config.manifest_dir.endswith("manifests/subject_clean_v1")

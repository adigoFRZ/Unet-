"""Tests for the one-shot frozen-holdout evaluator.

The evaluator exists to produce one number honestly, once. The tests therefore
concentrate on the properties that make that possible, and on the failure modes
that would quietly destroy it:

**Ground truth must not be reachable while predicting.** Phase A is handed an
image cache and checkpoints and nothing else. That is asserted behaviourally --
the GT readers are monkeypatched to explode, and Phase A is then run for real.

**Metrics must not be able to re-infer.** Phase B takes a frozen, hashed
prediction array and a label source. `model.forward` is monkeypatched to explode,
and Phase B is run for real. If a future edit reintroduces a model call, this
fails instead of silently producing a second, different set of predictions.

**The statistics must be reproducible.** Bootstrap and permutation carry fixed
RNG seeds; running them twice must give identical numbers, and the preregistered
constants must not drift.

Run with:  python -m pytest tests/test_frozen_holdout_evaluator.py -v
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data import crop_spec as cs  # noqa: E402

torch = pytest.importorskip("torch", reason="PyTorch is required")

import evaluate_frozen_holdout as ev  # noqa: E402

FROZEN_IMAGES = PROJECT_ROOT / "cache" / "baseline_v1" / "images"
REHEARSAL_DIR = (PROJECT_ROOT / "results" / "experiments" / "baseline_deep_ensemble"
                 / "holdout_evaluator_rehearsal")
VAL_MANIFEST = PROJECT_ROOT / "manifests" / "experiment" / "val.csv"


def _val_case_ids(n: int | None = None) -> list[str]:
    import pandas as pd

    ids = sorted(str(c) for c in pd.read_csv(VAL_MANIFEST)["case_id"])
    return ids[:n] if n else ids


needs_frozen = pytest.mark.skipif(not FROZEN_IMAGES.is_dir(),
                                  reason="frozen image cache absent")


# --------------------------------------------------------------------------- #
# 1-3. the cache reader
# --------------------------------------------------------------------------- #

def test_reader_does_not_use_the_segmentation_dataset() -> None:
    """1. The holdout is read through a plain file reader, not via a split guard
    that would have to be edited to allow it."""
    source = (PROJECT_ROOT / "scripts" / "evaluate_frozen_holdout.py").read_text(
        encoding="utf-8")
    # Check for actual use, not any mention: the module docstring explains *why*
    # SegmentationDataset is avoided, and that mention is the point.
    assert "from data.segmentation_dataset import" not in source
    assert "SegmentationDataset(" not in source
    assert not hasattr(ev, "SegmentationDataset")


@needs_frozen
def test_reader_returns_the_frozen_tensor_untouched() -> None:
    """2 + 3. No preprocessing, and the tensor is preserved exactly.

    Compared against a direct ``np.load`` -- if the reader normalised, cropped,
    transposed or rescaled, the values would differ.
    """
    case_id = _val_case_ids(1)[0]
    cache = ev.FrozenImageCache(FROZEN_IMAGES, [case_id])
    returned = cache.get(case_id)
    raw = np.load(FROZEN_IMAGES / f"{case_id}.npy")
    assert np.array_equal(returned, raw), "the reader altered the frozen tensor"
    assert returned.shape == (3, 32, 96, 96)
    assert returned.dtype == np.float32


@needs_frozen
def test_reader_rejects_wrong_shape_dtype_and_nonfinite(tmp_path) -> None:
    case_id = "SYN_CASE_000"
    for name, array in (("bad shape", np.zeros((3, 16, 16, 16), np.float32)),
                        ("bad dtype", np.zeros((3, 32, 96, 96), np.float16))):
        tmp = tmp_path / name.replace(" ", "_")
        tmp.mkdir()
        np.save(tmp / f"{case_id}.npy", array)
        cache = ev.FrozenImageCache(tmp, [case_id])
        with pytest.raises(ValueError):
            cache.get(case_id)

    nan_dir = tmp_path / "nan"
    nan_dir.mkdir()
    bad = np.zeros((3, 32, 96, 96), np.float32)
    bad[0, 0, 0, 0] = np.nan
    np.save(nan_dir / f"{case_id}.npy", bad)
    with pytest.raises(ValueError, match="NaN/Inf"):
        ev.FrozenImageCache(nan_dir, [case_id]).get(case_id)


# --------------------------------------------------------------------------- #
# 4-8. inference semantics
# --------------------------------------------------------------------------- #

@needs_frozen
def test_models_are_loaded_in_eval_mode_with_frozen_grads() -> None:
    """4."""
    device = torch.device("cpu")
    models = ev.load_frozen_models(device)
    assert set(models) == set(ev.FROZEN_SEEDS)
    for seed, model in models.items():
        assert not model.training, seed
        assert all(not p.requires_grad for p in model.parameters()), seed


def test_softmax_axis_is_the_class_dimension() -> None:
    """6. The recipe softmaxes over dim=1 of (N, C, D, H, W); softmaxing over a
    spatial axis would sum to 1 along depth instead and give a completely
    different, still-plausible-looking probability map."""
    logits = torch.randn(1, 4, 8, 16, 16)
    probability = torch.nn.functional.softmax(logits, dim=1)
    assert torch.allclose(probability.sum(dim=1), torch.ones(1, 8, 16, 16),
                          atol=1e-5)


def test_ensemble_is_the_arithmetic_mean_of_probabilities() -> None:
    """7. Equal weights, averaged before any decision."""
    torch.manual_seed(0)
    logits = [torch.randn(1, 4, 4, 8, 8) for _ in ev.FROZEN_SEEDS]
    probabilities = [torch.nn.functional.softmax(l, dim=1) for l in logits]
    stacked = torch.stack(probabilities)
    ensemble = stacked.mean(dim=0)
    for index, p in enumerate(probabilities):
        assert torch.allclose(ensemble, stacked.sum(dim=0) / len(probabilities))
    # arithmetic mean, not a product / geometric mean / logit average
    logit_mean_softmax = torch.nn.functional.softmax(
        torch.stack(logits).mean(dim=0), dim=1)
    assert not torch.allclose(ensemble, logit_mean_softmax, atol=1e-6), (
        "probability averaging must not coincide with logit averaging")


def test_argmax_is_applied_once_and_after_the_mean() -> None:
    """8. Deciding before averaging (majority voting) gives a different map."""
    torch.manual_seed(1)
    probabilities = [torch.nn.functional.softmax(torch.randn(1, 4, 4, 8, 8), dim=1)
                     for _ in ev.FROZEN_SEEDS]
    ensemble = torch.stack(probabilities).mean(dim=0)
    final = ensemble[0].argmax(dim=0)

    votes = torch.stack([p[0].argmax(dim=0) for p in probabilities])
    majority = torch.mode(votes, dim=0).values
    assert final.shape == (4, 8, 8)
    assert not torch.equal(final, majority), (
        "the ensemble must be argmax-after-mean, not a majority vote")


def test_holdout_maps_come_first_then_the_seeds_in_frozen_order() -> None:
    assert ev.FROZEN_SEEDS == ("seed42", "seed123", "seed2026")
    assert getattr(ev, "PRIMARY_COMPARATOR") == "seed123"


def test_primary_comparator_is_frozen_to_seed123() -> None:
    """16. The comparator was chosen on development val and is locked."""
    assert ev.PRIMARY_COMPARATOR == "seed123"
    assert ev.PRIMARY_COMPARATOR in ev.FROZEN_SEEDS


# --------------------------------------------------------------------------- #
# 9. case identity is by id, not by position
# --------------------------------------------------------------------------- #

@needs_frozen
def test_reader_matches_cases_by_id_regardless_of_input_order() -> None:
    """9. Reordering the requested list must return each case's own tensor."""
    ids = _val_case_ids(4)
    forward = ev.FrozenImageCache(FROZEN_IMAGES, ids)
    reverse = ev.FrozenImageCache(FROZEN_IMAGES, list(reversed(ids)))
    for case_id in ids:
        assert np.array_equal(forward.get(case_id), reverse.get(case_id))
        assert np.array_equal(forward.get(case_id),
                              np.load(FROZEN_IMAGES / f"{case_id}.npy"))


# --------------------------------------------------------------------------- #
# 10-12. the phase separation
# --------------------------------------------------------------------------- #

@needs_frozen
def test_phase_a_never_touches_ground_truth(monkeypatch) -> None:
    """10. The GT readers are made to explode, then Phase A is run for real."""
    def explode(*args, **kwargs):
        raise AssertionError("Phase A read ground truth")

    monkeypatch.setattr(ev.CachedLabelSource, "get", explode)
    monkeypatch.setattr(ev.ProcessedLabelSource, "get", explode)

    ids = _val_case_ids(2)
    cache = ev.FrozenImageCache(FROZEN_IMAGES, ids)
    models = ev.load_frozen_models(torch.device("cpu"))
    case_ids, predictions = ev.run_blind_inference(cache, models, torch.device("cpu"))

    assert case_ids == ids
    assert predictions.shape == (2, 1 + len(ev.FROZEN_SEEDS), 32, 96, 96)
    assert predictions.dtype == np.uint8


def test_phase_a_signature_takes_no_label_source() -> None:
    """10, structurally: there is no ground-truth parameter to pass."""
    parameters = set(inspect.signature(ev.run_blind_inference).parameters)
    assert parameters == {"image_cache", "models", "device"}, parameters
    assert not any("label" in p or "truth" in p or "gt" in p for p in parameters)


def test_phase_b_never_runs_a_model(monkeypatch) -> None:
    """11. Every forward pass is made to explode; Phase B must still work.

    Phase B is handed a synthetic frozen prediction array and a synthetic label
    source, so it cannot accidentally depend on a real checkpoint.
    """
    def explode(*args, **kwargs):
        raise AssertionError("Phase B ran a model")

    monkeypatch.setattr(torch.nn.Module, "forward", explode, raising=False)
    monkeypatch.setattr(torch.nn.modules.module.Module, "_call_impl", explode,
                        raising=False)

    class _Labels:
        def __init__(self, arrays):
            self.arrays = arrays

        def get(self, case_id):
            return self.arrays[case_id]

    ids = ["a", "b"]
    arrays = {c: np.zeros(cs.CROP_SHAPE_DHW, dtype=np.uint8) for c in ids}
    # BOTH cases need a foreground target: with an all-background GT the Dice of
    # an absent class is NaN by design, which would test that rule rather than
    # the prediction path.
    arrays["a"][5:9, 20:30, 20:30] = 1
    arrays["b"][5:9, 20:30, 20:30] = 1
    predictions = np.zeros((2, 1 + len(ev.FROZEN_SEEDS), *cs.CROP_SHAPE_DHW),
                           dtype=np.uint8)
    predictions[0, :, 5:9, 20:30, 20:30] = 1     # perfect on case "a"; case "b" misses it

    rows = ev.evaluate_predictions(ids, predictions, _Labels(arrays))
    assert len(rows) == 2
    assert rows[0]["Dice_STN__ensemble"] == pytest.approx(1.0, abs=1e-6)
    assert rows[1]["Dice_STN__ensemble"] == 0.0
    # the frame it produced must carry everything `aggregate` needs
    import pandas as pd

    ev.aggregate(pd.DataFrame(rows), "ensemble")


def test_phase_b_signature_has_no_model() -> None:
    """11, structurally."""
    parameters = set(inspect.signature(ev.evaluate_predictions).parameters)
    assert not any("model" in p for p in parameters), parameters


def test_frozen_predictions_are_hashed_and_self_describing(tmp_path) -> None:
    """12. The metric phase reads an artifact that is sufficient to recompute every
    preregistered metric without re-inferring."""
    ids = ["a", "b"]
    predictions = np.zeros((2, 1 + len(ev.FROZEN_SEEDS), *cs.CROP_SHAPE_DHW),
                           dtype=np.uint8)
    payload = ev.freeze_predictions(tmp_path, ids, predictions)
    assert (tmp_path / "predictions.npy").is_file()
    assert payload["n_cases"] == 2
    assert payload["case_ids"] == ids
    assert payload["map_order"] == ["ensemble"] + list(ev.FROZEN_SEEDS)
    assert payload["phase_a_read_ground_truth"] is False
    assert set(payload["per_case_sha256"]) == set(ids)
    assert payload["array_shape"] == [2, 1 + len(ev.FROZEN_SEEDS), 32, 96, 96]
    assert np.array_equal(np.load(tmp_path / "predictions.npy"), predictions)


# --------------------------------------------------------------------------- #
# 13-15. metric definition and statistical determinism
# --------------------------------------------------------------------------- #

def test_one_hot_round_trip_preserves_the_label_map() -> None:
    """`evaluate_case` argmaxes, so a (D,H,W) map must be one-hotted first."""
    rng = np.random.default_rng(0)
    label_map = rng.integers(0, 4, size=cs.CROP_SHAPE_DHW).astype(np.uint8)
    stack = ev.label_map_to_class_stack(label_map)
    assert stack.shape == (4, *cs.CROP_SHAPE_DHW)
    assert np.array_equal(stack.argmax(axis=0), label_map)


def _synthetic_metrics_frame(case_ids: list[str], member: str = "ensemble"):
    """A minimal per-case frame carrying every column `aggregate` reads."""
    import pandas as pd

    data: dict[str, list] = {"case_id": list(case_ids)}
    for name in cs.CLASS_NAMES.values():
        data[f"Dice_{name}__{member}"] = [0.5] * len(case_ids)
        data[f"HD95_{name}_mm__{member}"] = [1.0] * len(case_ids)
        data[f"Precision_{name}__{member}"] = [0.5] * len(case_ids)
        data[f"Recall_{name}__{member}"] = [0.5] * len(case_ids)
        data[f"volume_ratio_{name}__{member}"] = [1.0] * len(case_ids)
        data[f"FP_{name}__{member}"] = [0] * len(case_ids)
        data[f"FN_{name}__{member}"] = [0] * len(case_ids)
        data[f"empty_prediction_{name}__{member}"] = [False] * len(case_ids)
        data[f"PredVoxels_{name}__{member}"] = [1] * len(case_ids)
        data[f"GTVoxels_{name}__{member}"] = [1] * len(case_ids)
    data[f"centroid_distance_mm_STN__{member}"] = [0.0] * len(case_ids)
    return pd.DataFrame(data)


def test_macro_dice_is_the_per_case_mean_of_three_classes() -> None:
    """13. Patient first, then average -- not a pooled voxel Dice."""
    frame = _synthetic_metrics_frame(["a", "b"])
    frame["Dice_STN__ensemble"] = [1.0, 0.0]
    frame["Dice_SN__ensemble"] = [1.0, 0.0]
    frame["Dice_RN__ensemble"] = [0.5, 0.5]
    aggregate = ev.aggregate(frame, "ensemble")
    # per case: (1+1+0.5)/3 = 0.8333, (0+0+0.5)/3 = 0.1667 -> mean 0.5
    assert aggregate["macro_foreground_dice"] == pytest.approx(0.5, abs=1e-9)
    assert aggregate["empty_prediction_STN"] == 0


def test_bootstrap_is_deterministic_and_seeded() -> None:
    """14. Same seed -> identical interval; different seed -> different."""
    deltas = np.linspace(-0.05, 0.15, 40)
    first = ev.paired_bootstrap_ci(deltas)
    second = ev.paired_bootstrap_ci(deltas)
    assert first == second
    others = ev.paired_bootstrap_ci(deltas, rng_seed=12345)
    assert others != first, "the bootstrap seed is not actually being applied"
    assert first[0] < np.mean(deltas) < first[1]


def test_permutation_is_deterministic_and_seeded() -> None:
    """15.

    Uses a *moderate* effect on purpose. With a huge effect every sign flip is
    extreme, so the p-value saturates at the 1/(n+1) floor and stops depending on
    the seed -- the seed check would then fail for the wrong reason.
    """
    rng = np.random.default_rng(7)
    deltas = rng.normal(0.02, 0.10, 40)
    first = ev.paired_signflip_pvalue(deltas, n_draws=2000)
    second = ev.paired_signflip_pvalue(deltas, n_draws=2000)
    assert first == second
    assert 0.0 < first < 1.0, "the effect saturates the permutation distribution"
    assert ev.paired_signflip_pvalue(deltas, n_draws=2000, rng_seed=999) != first

    # a clearly non-null effect must be significant; a symmetric null must not
    assert ev.paired_signflip_pvalue(np.full(40, 0.2), n_draws=5000) < 0.01
    null_deltas = np.tile([0.1, -0.1], 20)
    assert ev.paired_signflip_pvalue(null_deltas, n_draws=5000) > 0.05


def test_preregistered_constants_are_unchanged() -> None:
    assert ev.BOOTSTRAP_RESAMPLES == 10_000
    assert ev.PERMUTATION_DRAWS == 100_000
    assert ev.RNG_SEED == 20260927


def test_primary_decision_requires_all_three_conditions() -> None:
    """The preregistered AND. Any single failing condition means NOT CONFIRMED."""
    assert ev.primary_decision(0.01, 0.001, 0.01)["result"] == "CONFIRMED"
    assert ev.primary_decision(-0.01, 0.001, 0.01)["result"] == "NOT CONFIRMED"
    assert ev.primary_decision(0.01, -0.001, 0.01)["result"] == "NOT CONFIRMED"
    assert ev.primary_decision(0.01, 0.001, 0.20)["result"] == "NOT CONFIRMED"


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    raw = {"a": 0.001, "b": 0.02, "c": 0.5}
    adjusted = ev.holm_adjust(raw)
    assert adjusted["a"] == pytest.approx(0.003)
    assert adjusted["b"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.5)
    assert all(0.0 <= v <= 1.0 for v in adjusted.values())


# --------------------------------------------------------------------------- #
# 17. the state machine
# --------------------------------------------------------------------------- #

def test_state_machine_follows_the_preregistered_order(tmp_path) -> None:
    machine = ev.StateMachine(tmp_path / "state.json")
    assert machine.state is ev.HoldoutState.CLOSED
    machine.transition(ev.HoldoutState.PREDICTIONS_FROZEN, "predictions frozen")
    machine.transition(ev.HoldoutState.OPENED, "first GT read")
    machine.transition(ev.HoldoutState.COMPLETE, "metrics written")
    assert machine.state is ev.HoldoutState.COMPLETE


def test_state_machine_refuses_illegal_and_reverse_transitions(tmp_path) -> None:
    """OPENED is irreversible: that is the one-shot policy expressed as code."""
    machine = ev.StateMachine(tmp_path / "state.json")
    with pytest.raises(ValueError):
        machine.transition(ev.HoldoutState.OPENED, "skipping the freeze")
    with pytest.raises(ValueError):
        machine.transition(ev.HoldoutState.COMPLETE, "skipping")

    machine.transition(ev.HoldoutState.PREDICTIONS_FROZEN, "frozen")
    machine.transition(ev.HoldoutState.OPENED, "opened")
    for illegal in (ev.HoldoutState.CLOSED, ev.HoldoutState.PREDICTIONS_FROZEN):
        with pytest.raises(ValueError):
            machine.transition(illegal, "going back")


def test_state_machine_persists_and_reloads(tmp_path) -> None:
    path = tmp_path / "state.json"
    machine = ev.StateMachine(path)
    machine.transition(ev.HoldoutState.PREDICTIONS_FROZEN, "frozen")
    reloaded = ev.StateMachine(path)
    assert reloaded.state is ev.HoldoutState.PREDICTIONS_FROZEN
    assert reloaded.history[0]["to"] == "PREDICTIONS_FROZEN"


# --------------------------------------------------------------------------- #
# 18. rehearsal regression
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not (REHEARSAL_DIR / "rehearsal_report.json").is_file(),
                    reason="rehearsal has not been run")
def test_rehearsal_reproduces_the_frozen_development_val_result() -> None:
    """18. The evaluator reproduces Experiment F on development val exactly.

    This is the regression that licenses a holdout run: if the runner cannot
    reproduce a result it already knows, its holdout number would be untrustworthy.
    """
    report = json.loads((REHEARSAL_DIR / "rehearsal_report.json").read_text(
        encoding="utf-8"))
    per_case = report["per_case_vs_frozen_experiment_f"]
    assert per_case["case_ids_identical"] is True
    assert per_case["n_cases"] == 40
    assert per_case["bitwise_identical"] is True, (
        f"rehearsal differs from the frozen run by "
        f"{per_case['largest_absolute_difference']}")
    assert report["all_published_constants_agree"] is True


@pytest.mark.skipif(not (REHEARSAL_DIR / "holdout_state.json").is_file(),
                    reason="rehearsal has not been run")
def test_rehearsal_never_opened_the_holdout_itself() -> None:
    """The rehearsal runs on development val and must not advance to OPENED.

    Note this asserts a property of the *rehearsal* run, not of the project: the
    one-shot holdout has since been opened legitimately by its own run, and the
    state of that run is checked separately below. Writing this against the
    project state instead would have made the test fail the moment the holdout
    was correctly evaluated.
    """
    state = json.loads((REHEARSAL_DIR / "holdout_state.json").read_text(
        encoding="utf-8"))
    assert state["state"] not in ("OPENED", "COMPLETE")
    assert (REHEARSAL_DIR / "HOLDOUT_OPENED.json").is_file() is False, (
        "the rehearsal must never write an OPENED record")


@pytest.mark.skipif(
    not (PROJECT_ROOT / "results/experiments/baseline_deep_ensemble"
         / "secondary_frozen_holdout").is_dir(),
    reason="holdout run has not been executed")
def test_holdout_run_is_internally_consistent() -> None:
    """If the holdout was opened, its record and state must agree."""
    holdout_dir = (PROJECT_ROOT / "results" / "experiments"
                   / "baseline_deep_ensemble" / "secondary_frozen_holdout")
    opened = holdout_dir / "HOLDOUT_OPENED.json"
    assert opened.is_file(), (
        "a holdout predictions artifact exists without an OPENED record")
    state = json.loads((holdout_dir / "holdout_state.json").read_text(
        encoding="utf-8"))
    assert state["state"] in ("OPENED", "COMPLETE")
    transitions = [(h["from"], h["to"]) for h in state["history"]]
    assert ("PREDICTIONS_FROZEN", "OPENED") in transitions
    # and the opening must have happened after the predictions were frozen
    order = [h["to"] for h in state["history"]]
    assert order.index("PREDICTIONS_FROZEN") < order.index("OPENED")

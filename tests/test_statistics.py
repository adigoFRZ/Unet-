"""Tests for the paired statistics behind the reported comparisons.

Two things are being protected.

**The definitions.** These functions produced the numbers in the paper. A
"harmless" refactor -- swapping the percentile method, dropping the ``+1`` in the
sign-flip count, averaging in the wrong order -- would change a reported value
while every test that only checks "p < 0.05" still passed. So the tests below
pin exact values and cross-check the t-test against ``scipy`` directly.

**The preregistered constants.** The resampling settings are part of the frozen
analysis, not tuning knobs. If a default drifts, every future run silently
becomes a different experiment.

Run with:  ./.venv/Scripts/python.exe -m pytest tests/test_statistics.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation import statistics as st  # noqa: E402


# --------------------------------------------------------------------------- #
# preregistered constants
# --------------------------------------------------------------------------- #

def test_preregistered_constants_are_unchanged() -> None:
    """The frozen analysis settings. Changing these changes the reported result."""
    assert st.DEFAULT_BOOTSTRAP_RESAMPLES == 10_000
    assert st.DEFAULT_PERMUTATION_DRAWS == 100_000
    assert st.DEFAULT_RNG_SEED == 20260927
    assert st.UNCHANGED_EPSILON == 0.01


# --------------------------------------------------------------------------- #
# paired_stats
# --------------------------------------------------------------------------- #

def test_paired_stats_on_a_hand_computable_case() -> None:
    base = np.array([0.5, 0.6, 0.7, 0.8])
    exp = np.array([0.6, 0.55, 0.75, 0.9])

    out = st.paired_stats(base, exp)

    assert out["n_paired_cases"] == 4
    assert out["baseline_mean"] == pytest.approx(0.65)
    assert out["experiment_mean"] == pytest.approx(0.7)
    assert out["mean_difference"] == pytest.approx(0.05)
    # sorted deltas: -0.05, 0.05, 0.10, 0.10
    assert out["median_difference"] == pytest.approx(0.075)
    assert out["min_difference"] == pytest.approx(-0.05)
    assert out["max_difference"] == pytest.approx(0.10)
    assert out["std_difference"] == pytest.approx(np.std(exp - base, ddof=1))
    # dead band: 3 cases up, 1 down, 0 within epsilon
    assert (out["improved_cases"], out["degraded_cases"], out["unchanged_cases"]) == (3, 1, 0)
    assert out["unchanged_epsilon"] == 0.01


def test_paired_stats_t_test_is_the_standard_paired_t_test() -> None:
    """The reported t-test must stay ``scipy.stats.ttest_rel(experiment, baseline)``.

    The argument order is part of the definition: it sets the sign of the
    statistic, and a flipped sign in a per-case table is easy to miss.
    """
    from scipy import stats

    rng = np.random.default_rng(0)
    base = rng.normal(0.70, 0.08, size=40)
    exp = base + rng.normal(0.01, 0.02, size=40)

    out = st.paired_stats(base, exp)
    reference = stats.ttest_rel(exp, base)

    assert out["paired_t_statistic"] == pytest.approx(float(reference.statistic))
    assert out["paired_t_pvalue"] == pytest.approx(float(reference.pvalue))
    assert out["paired_t_statistic"] == pytest.approx(float(stats.ttest_rel(exp, base).statistic))


def test_paired_stats_drops_non_finite_pairs_pairwise() -> None:
    """One bad case must not turn every statistic into NaN, and alignment stays."""
    base = np.array([0.5, 0.6, np.nan, 0.8, 0.9])
    exp = np.array([0.6, 0.7, 0.8, np.inf, 0.9])

    out = st.paired_stats(base, exp)

    # cases 2 and 3 are dropped, the other three survive
    assert out["n_paired_cases"] == 3
    assert out["mean_difference"] == pytest.approx((0.1 + 0.1 + 0.0) / 3)
    assert math.isfinite(out["paired_t_pvalue"]) or out["n_paired_cases"] < 3


def test_paired_stats_degenerate_inputs_return_nan_not_an_exception() -> None:
    """n < 3 has no test to report, and must say so as NaN rather than raise."""
    for base, exp in ((np.array([]), np.array([])),
                      (np.array([0.5]), np.array([0.6])),
                      (np.array([0.5, 0.6]), np.array([0.6, 0.7]))):
        out = st.paired_stats(base, exp)
        assert out["n_paired_cases"] == base.size
        assert math.isnan(out["paired_t_statistic"])
        assert math.isnan(out["paired_t_pvalue"])
        assert math.isnan(out["wilcoxon_pvalue"])


def test_paired_stats_with_zero_variance_is_not_reported_as_evidence() -> None:
    """Cases that all moved by the same amount have no variance to test.

    The guard is exact -- ``np.ptp(delta) > 0`` -- so this pins that a
    bit-identical constant difference yields NaN, not a tiny p-value.
    """
    base = np.array([0.5, 0.5, 0.5])
    exp = np.array([0.6, 0.6, 0.6])

    out = st.paired_stats(base, exp)

    assert out["mean_difference"] == pytest.approx(0.1)
    assert math.isnan(out["paired_t_pvalue"])
    assert math.isnan(out["wilcoxon_pvalue"])


def test_paired_stats_records_why_a_refused_test_did_not_run(monkeypatch) -> None:
    """scipy refuses some inputs. The reason must be recorded, not swallowed.

    The paired t-test still runs in that situation, so the summary stays usable;
    only the Wilcoxon entry becomes NaN plus a note.
    """
    def refuse(*args, **kwargs):
        raise ValueError("x - y is zero for all elements")

    monkeypatch.setattr(st.stats, "wilcoxon", refuse)
    rng = np.random.default_rng(5)
    base = rng.normal(0.7, 0.05, size=10)

    out = st.paired_stats(base, base + rng.normal(0.01, 0.01, size=10))

    assert math.isnan(out["wilcoxon_pvalue"])
    assert out["wilcoxon_note"] == "x - y is zero for all elements"
    assert math.isfinite(out["paired_t_pvalue"]), "the t-test is independent of it"


# --------------------------------------------------------------------------- #
# bootstrap CI
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_is_reproducible_and_brackets_the_mean() -> None:
    rng = np.random.default_rng(1)
    deltas = rng.normal(0.01, 0.03, size=60)

    first = st.paired_bootstrap_ci(deltas, n_resamples=400, rng_seed=7)
    second = st.paired_bootstrap_ci(deltas, n_resamples=400, rng_seed=7)
    other = st.paired_bootstrap_ci(deltas, n_resamples=400, rng_seed=8)

    assert first == second, "the same seed must give the same interval"
    assert first != other, "a different seed should give a different interval"
    low, high = first
    assert low < high
    assert low <= float(np.mean(deltas)) <= high


def test_bootstrap_ci_rejects_an_empty_input() -> None:
    with pytest.raises(ValueError, match="no paired differences"):
        st.paired_bootstrap_ci(np.array([]))


# --------------------------------------------------------------------------- #
# sign-flip permutation
# --------------------------------------------------------------------------- #

def test_signflip_is_reproducible_and_seed_sensitive() -> None:
    rng = np.random.default_rng(2)
    deltas = rng.normal(0.01, 0.03, size=25)

    first = st.paired_signflip_pvalue(deltas, n_draws=500, rng_seed=11)
    assert first == st.paired_signflip_pvalue(deltas, n_draws=500, rng_seed=11)
    assert first != st.paired_signflip_pvalue(deltas, n_draws=500, rng_seed=12)


def test_signflip_separates_a_consistent_effect_from_noise() -> None:
    """A one-directional shift is evidence; symmetric noise is not."""
    consistent = st.paired_signflip_pvalue(np.full(40, 0.2), n_draws=5000)
    assert consistent < 0.01

    rng = np.random.default_rng(3)
    null = rng.normal(0.0, 0.05, size=40)
    assert st.paired_signflip_pvalue(null, n_draws=5000) > 0.05


def test_signflip_pvalue_stays_in_range_and_never_reaches_zero() -> None:
    """The observed statistic is counted, which keeps p > 0 rather than anti-conservative."""
    rng = np.random.default_rng(4)
    for size in (5, 17, 40):
        deltas = rng.normal(0.05, 0.01, size=size)
        p = st.paired_signflip_pvalue(deltas, n_draws=1000)
        assert 0.0 < p <= 1.0


def test_signflip_rejects_an_empty_input() -> None:
    with pytest.raises(ValueError, match="no paired differences"):
        st.paired_signflip_pvalue(np.array([]))


# --------------------------------------------------------------------------- #
# decision rule and multiplicity
# --------------------------------------------------------------------------- #

def test_primary_decision_requires_all_three_conditions() -> None:
    assert st.primary_decision(0.01, 0.005, 0.001)["result"] == "CONFIRMED"
    # each single failure flips the decision
    for args in ((0.01, -0.001, 0.001), (0.01, 0.005, 0.5), (-0.01, 0.005, 0.001)):
        decision = st.primary_decision(*args)
        assert decision["result"] == "NOT CONFIRMED"
        assert not decision["all_met"]


def test_holm_adjust_matches_a_worked_example() -> None:
    """Holm step-down, with the running maximum making the result monotone."""
    adjusted = st.holm_adjust({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.2})
    assert adjusted["a"] == pytest.approx(0.004)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["d"] == pytest.approx(0.2)
    # monotone in the sorted order
    ordered = [adjusted[k] for k in ("a", "b", "c", "d")]
    assert ordered == sorted(ordered)


def test_holm_adjust_never_exceeds_one() -> None:
    adjusted = st.holm_adjust({"a": 0.5, "b": 0.9})
    assert all(value <= 1.0 for value in adjusted.values())

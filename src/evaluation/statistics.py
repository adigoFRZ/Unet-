"""Paired statistics for the reported comparisons.

Every function here works on **paired per-case values**: element ``i`` of the
baseline array and element ``i`` of the experiment array must be the same case.
The whole evaluation is paired by case id, so an unpaired test would be the wrong
measurement -- cases differ from each other far more than the two arms differ.

Three quantities are reported, and they answer different questions:

* ``paired_stats`` -- descriptive summary plus a paired t-test and a Wilcoxon
  signed-rank test. The t-test is reported for continuity; the Wilcoxon test does
  not assume normality and is the one that survives a skewed per-case
  distribution.
* ``paired_bootstrap_ci`` -- resamples **cases**, not values: the interval it
  produces is "how much would the mean difference move if these 100 cases were a
  different draw of 100 cases", which is the question a reader actually has.
* ``paired_signflip_pvalue`` -- the preregistered test. Under the null the sign
  of each paired difference is exchangeable, so the reference distribution is the
  mean of ``deltas * random ±1``.

Both resampling procedures are Monte Carlo with a **fixed seed**, so the reported
interval and p-value are reproducible rather than merely asserted.

The defaults below are the preregistered constants of the frozen evaluation. They
are defaults, not ambient state: a caller that has its own frozen constants
should pass them explicitly, and the experiment scripts do.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

#: |difference| at or below this counts as "unchanged" rather than improved or
#: degraded -- a Dice difference of 0.01 is not a claim about a case.
UNCHANGED_EPSILON = 0.01

#: Preregistered resampling settings for the frozen holdout evaluation.
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_PERMUTATION_DRAWS = 100_000
DEFAULT_RNG_SEED = 20260927


def paired_stats(baseline: np.ndarray, experiment: np.ndarray) -> dict[str, Any]:
    """Paired t-test and Wilcoxon signed-rank on per-case differences.

    Non-finite pairs are dropped before anything is computed, so a single NaN
    case does not silently turn every statistic into NaN. ``improved_cases`` /
    ``degraded_cases`` use ``UNCHANGED_EPSILON`` as the dead band.
    """
    mask = np.isfinite(baseline) & np.isfinite(experiment)
    base, exp = baseline[mask], experiment[mask]
    delta = exp - base
    n = int(delta.size)

    out: dict[str, Any] = {
        "n_paired_cases": n,
        "baseline_mean": float(base.mean()) if n else math.nan,
        "experiment_mean": float(exp.mean()) if n else math.nan,
        "mean_difference": float(delta.mean()) if n else math.nan,
        "median_difference": float(np.median(delta)) if n else math.nan,
        "std_difference": float(delta.std(ddof=1)) if n > 1 else math.nan,
        "min_difference": float(delta.min()) if n else math.nan,
        "max_difference": float(delta.max()) if n else math.nan,
    }

    # n < 3 has no meaningful test, and a constant delta has zero variance, which
    # makes the t statistic undefined rather than infinite.
    if n >= 3 and np.ptp(delta) > 0:
        ttest = stats.ttest_rel(exp, base)
        out["paired_t_statistic"] = float(ttest.statistic)
        out["paired_t_pvalue"] = float(ttest.pvalue)
        try:
            wilcoxon = stats.wilcoxon(exp, base)
            out["wilcoxon_statistic"] = float(wilcoxon.statistic)
            out["wilcoxon_pvalue"] = float(wilcoxon.pvalue)
        except ValueError as exc:
            # scipy raises for an all-zero difference vector; that is "no
            # evidence", which is a NaN here rather than an exception.
            out["wilcoxon_statistic"] = math.nan
            out["wilcoxon_pvalue"] = math.nan
            out["wilcoxon_note"] = str(exc)
    else:
        out.update({
            "paired_t_statistic": math.nan,
            "paired_t_pvalue": math.nan,
            "wilcoxon_statistic": math.nan,
            "wilcoxon_pvalue": math.nan,
        })

    improved = int((delta > UNCHANGED_EPSILON).sum())
    degraded = int((delta < -UNCHANGED_EPSILON).sum())
    out.update({
        "improved_cases": improved,
        "degraded_cases": degraded,
        "unchanged_cases": int(n - improved - degraded),
        "unchanged_epsilon": UNCHANGED_EPSILON,
    })
    return out


def paired_bootstrap_ci(deltas: np.ndarray,
                        n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
                        rng_seed: int = DEFAULT_RNG_SEED) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean of paired differences.

    Cases are resampled with replacement (``deltas[draws]``), which is the paired
    procedure: each draw keeps a case's two arms together.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        raise ValueError("no paired differences supplied")
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, deltas.size, size=(n_resamples, deltas.size))
    means = deltas[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_signflip_pvalue(deltas: np.ndarray,
                           n_draws: int = DEFAULT_PERMUTATION_DRAWS,
                           rng_seed: int = DEFAULT_RNG_SEED) -> float:
    """Two-sided paired sign-flip permutation p-value.

    Under the null the sign of each paired difference is exchangeable, so the
    reference distribution is the mean of ``deltas * random ±1``. 2^n sign
    patterns is not exhaustible for n = 100, so this is Monte Carlo. The observed
    statistic is included in the count, which keeps the p-value valid rather than
    anti-conservative.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        raise ValueError("no paired differences supplied")
    observed = float(np.mean(deltas))
    rng = np.random.default_rng(rng_seed)
    extremes = 0
    remaining = n_draws
    batch = 10_000
    while remaining > 0:
        size = min(batch, remaining)
        signs = rng.integers(0, 2, size=(size, deltas.size)) * 2 - 1
        means = (signs * deltas).mean(axis=1)
        extremes += int((np.abs(means) >= abs(observed)).sum())
        remaining -= size
    return float((extremes + 1) / (n_draws + 1))


def primary_decision(mean_delta: float, ci_low: float, p_value: float) -> dict[str, Any]:
    """The preregistered AND of three conditions. Ordering is fixed here."""
    conditions = {
        "A_mean_delta_gt_0": bool(mean_delta > 0),
        "B_bootstrap_ci_low_gt_0": bool(ci_low > 0),
        "C_permutation_p_lt_0.05": bool(p_value < 0.05),
    }
    return {
        "conditions": conditions,
        "all_met": all(conditions.values()),
        "result": "CONFIRMED" if all(conditions.values()) else "NOT CONFIRMED",
        "note": ("a single secondary or per-class result may never overturn this "
                 "decision"),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down correction. Secondary analyses only; never feeds the primary."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(items):
        candidate = min(1.0, (m - index) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted

#!/usr/bin/env python
"""Re-derive the reported holdout numbers from the local per-case table.

The inputs are the frozen evaluation records kept on this machine. They name
real participant cases, so they are excluded from the public repository
(``results/`` is in ``.gitignore``) and this script only runs where those files
still exist:

* ``results/subject_clean_holdout/HOLDOUT_PER_CASE.csv`` -- the frozen per-case
  metrics, 100 cases x 153 columns;
* ``results/subject_clean_holdout/HOLDOUT_PRIMARY_ENDPOINT.json`` -- the
  numbers as reported;
* the statistics functions in ``scripts/evaluate_frozen_holdout.py`` -- the
  same code that produced them.

No model runs, no inference, no ground-truth access: this only re-aggregates a
table that is already on disk. It therefore says nothing new about the model;
it checks that the reported numbers are still what the frozen data implies.

    python scripts/reproduce_headline_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_frozen_holdout as efh  # noqa: E402

CLASSES = ("STN", "SN", "RN")
MEMBERS = ("seed42", "seed123", "seed2026", "ensemble")
PRIMARY = ROOT / "results/subject_clean_holdout/HOLDOUT_PRIMARY_ENDPOINT.json"
PER_CASE = ROOT / "results/subject_clean_holdout/HOLDOUT_PER_CASE.csv"

#: The two macro values are stored rounded to 6 decimals, so they are compared
#: with half a unit in the last stored place. The interval bounds and the
#: p-value are stored at full precision.
ROUNDED_TOLERANCE = 5e-7
EXACT_TOLERANCE = 1e-9


def main() -> int:
    expected = json.loads(PRIMARY.read_text(encoding="utf-8"))
    frame = pd.read_csv(PER_CASE, encoding="utf-8-sig")

    def macro(member: str) -> np.ndarray:
        """Case-wise macro foreground Dice = mean of the three class Dice."""
        columns = [f"Dice_{name}__{member}" for name in CLASSES]
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"{PER_CASE.name} is missing {missing}")
        return frame[columns].to_numpy(dtype=float).mean(axis=1)

    ensemble = macro("ensemble")
    comparator = macro(efh.PRIMARY_COMPARATOR)
    deltas = ensemble - comparator

    mean_delta = float(deltas.mean())
    ci_low, ci_high = efh.paired_bootstrap_ci(deltas)
    p_value = efh.paired_signflip_pvalue(deltas)
    decision = efh.primary_decision(mean_delta, ci_low, p_value)

    rows = [
        ("ensemble macro Dice", expected["ensemble_macro_Dice"],
         float(ensemble.mean()), ROUNDED_TOLERANCE),
        ("comparator macro Dice", expected["comparator_macro_Dice"],
         float(comparator.mean()), ROUNDED_TOLERANCE),
        ("mean paired delta", expected["mean_delta"], mean_delta, EXACT_TOLERANCE),
        ("median paired delta", expected["median_delta"],
         float(np.median(deltas)), EXACT_TOLERANCE),
        ("bootstrap CI low", expected["bootstrap_ci_low"], ci_low, EXACT_TOLERANCE),
        ("bootstrap CI high", expected["bootstrap_ci_high"], ci_high, EXACT_TOLERANCE),
        ("permutation p", expected["permutation_p"], p_value, 1e-12),
    ]

    print(f"per-case table : {PER_CASE.relative_to(ROOT)} "
          f"({frame.shape[0]} rows x {frame.shape[1]} columns)")
    print(f"comparison     : ensemble vs {efh.PRIMARY_COMPARATOR}\n")
    print(f"{'quantity':<24}{'reported':>12}{'recomputed':>13}")
    print("-" * 49)

    ok = True
    for label, reported, actual, tolerance in rows:
        agrees = abs(reported - actual) <= tolerance
        ok &= agrees
        print(f"{label:<24}{reported:>12.6f}{actual:>13.6f}  "
              f"{'OK' if agrees else 'MISMATCH'}")

    same_decision = decision["result"] == expected["decision"]["result"]
    ok &= same_decision
    print(f"\npreregistered decision: {decision['result']} "
          f"(reported {expected['decision']['result']})")

    print("\ndescriptive per-member macro Dice over the same 100 cases:")
    for member in MEMBERS:
        print(f"   {member:<10}{float(macro(member).mean()):.6f}")

    if not same_decision:
        ok = False
    print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

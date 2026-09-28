"""Tests for the Experiment G0 STN failure-mechanism audit.

The audit is exploratory, so the tests cannot check that its conclusions are
"right" -- that is the report's job. What they can check is that the analysis is
internally consistent and stayed inside its data boundary:

* the per-case table is the development validation split, all 40 of it, and
  contains **no** holdout case;
* the voxel accounting closes (tp + fp == predicted volume, and the class
  confusion breakdown sums to the FP/FN totals) -- an audit that silently
  double-counts or drops voxels would produce a mechanism story out of arithmetic;
* provenance claims in the summary are true, and the frozen Experiment F
  artifacts are untouched by the analysis.

Run with:  python -m pytest tests/test_stn_failure_analysis.py -v
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data import crop_spec as cs  # noqa: E402

OUT = PROJECT_ROOT / "results" / "experiments" / "stn_failure_audit"
SUMMARY = OUT / "stn_failure_audit_summary.json"
PER_CASE = OUT / "stn_failure_per_case.csv"
CONFUSION = OUT / "stn_confusion_summary.csv"
DELTAS = OUT / "ensemble_delta_analysis.csv"

REQUIRED_ARTIFACTS = (
    "stn_failure_audit_summary.json", "stn_failure_per_case.csv",
    "stn_volume_bias.png", "stn_ratio_vs_dice.png", "stn_ratio_vs_hd95.png",
    "stn_dice_vs_centroid.png", "stn_fp_frequency_map.png",
    "stn_fn_frequency_map.png", "stn_confusion_summary.csv",
    "stn_probability_diagnostics.png", "stn_crop_margin_analysis.png",
    "ensemble_delta_analysis.csv",
)

needs_audit = pytest.mark.skipif(not SUMMARY.is_file(), reason="audit not run")


def _val_case_ids() -> list[str]:
    return sorted(str(c) for c in pd.read_csv(
        PROJECT_ROOT / "manifests/experiment/val.csv")["case_id"])


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #

@needs_audit
def test_all_required_artifacts_exist() -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (OUT / name).is_file()]
    assert not missing, f"missing audit artifacts: {missing}"


@needs_audit
def test_no_audit_output_overwrote_an_existing_experiment() -> None:
    """The audit writes only into its own directory.

    Enumerated from the filesystem rather than from a fixed list: a hard-coded
    list stops covering directories added later, and one naming a directory that
    has since been removed fails for the wrong reason.
    """
    assert OUT.is_dir()
    results_root = PROJECT_ROOT / "results"
    candidates = [results_root]
    for top in results_root.iterdir():
        if top.is_dir():
            candidates.append(top)
            candidates.extend(p for p in top.iterdir() if p.is_dir())

    checked = 0
    for directory in candidates:
        if directory.resolve() == OUT.resolve():
            continue
        checked += 1
        assert not (directory / "stn_failure_audit_summary.json").exists(), directory
    assert checked > 0, "no sibling results directory found to check against"


# --------------------------------------------------------------------------- #
# data boundary
# --------------------------------------------------------------------------- #

@needs_audit
def test_per_case_table_is_the_development_validation_split() -> None:
    frame = pd.read_csv(PER_CASE)
    assert len(frame) == 40
    assert sorted(frame["case_id"]) == _val_case_ids()


@needs_audit
def test_no_holdout_case_appears_in_the_audit() -> None:
    """G0 is development-only. A holdout case here would mean the boundary broke."""
    frame = pd.read_csv(PER_CASE)
    for split in ("internal_test", "challenge_test"):
        other = set(pd.read_csv(
            PROJECT_ROOT / f"manifests/experiment/{split}.csv")["case_id"].astype(str))
        overlap = set(frame["case_id"]) & other
        assert not overlap, f"audit contains {split} cases: {sorted(overlap)[:5]}"


@needs_audit
def test_summary_declares_its_data_boundary() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["data_used"] == "development val only (n=40)"
    assert summary["holdout_accessed"] is False
    assert summary["n_cases"] == 40


# --------------------------------------------------------------------------- #
# internal consistency of the voxel accounting
# --------------------------------------------------------------------------- #

@needs_audit
def test_voxel_accounting_closes() -> None:
    """tp + fp == predicted volume and tp + fn == GT volume, per member."""
    frame = pd.read_csv(PER_CASE)
    for member in ("ensemble", "seed42", "seed123", "seed2026"):
        deficit = frame[f"tp__{member}"] + frame[f"fp__{member}"]
        assert (deficit == frame[f"predvox__{member}"]).all(), member
        assert (frame[f"tp__{member}"] + frame[f"fn__{member}"]
                == frame[f"gtvox__{member}"]).all(), member
        # and the Dice implied by the counts must match the stored Dice
        implied = 2 * frame[f"tp__{member}"] / (
            2 * frame[f"tp__{member}"] + frame[f"fp__{member}"]
            + frame[f"fn__{member}"])
        assert np.allclose(implied, frame[f"dice__{member}"], atol=1e-6), member


@needs_audit
def test_gt_volume_is_member_independent() -> None:
    """The GT is the same array for every member; only predictions differ."""
    frame = pd.read_csv(PER_CASE)
    reference = frame["gtvox__ensemble"].to_numpy()
    for member in ("seed42", "seed123", "seed2026"):
        assert np.array_equal(frame[f"gtvox__{member}"].to_numpy(), reference), member


@needs_audit
def test_volume_ratio_matches_the_voxel_counts() -> None:
    frame = pd.read_csv(PER_CASE)
    for member in ("ensemble", "seed42"):
        expected = frame[f"predvox__{member}"] / frame[f"gtvox__{member}"]
        assert np.allclose(frame[f"ratio__{member}"], expected, atol=1e-9), member


@needs_audit
def test_class_confusion_sums_to_the_fp_and_fn_totals() -> None:
    """Every FP voxel is one of {background, SN, RN} in the GT -- no other class
    exists in this label scheme, so the three must account for the total."""
    confusion = pd.read_csv(CONFUSION)
    frame = pd.read_csv(PER_CASE)
    for _, row in confusion.iterrows():
        member = row["member"]
        assert (row["fp_from_background"] + row["fp_from_sn"] + row["fp_from_rn"]
                == row["fp_total"]), member
        assert (row["fn_as_background"] + row["fn_as_sn"] + row["fn_as_rn"]
                == row["fn_total"]), member
        assert row["fp_total"] == int(frame[f"fp__{member}"].sum()), member
        assert row["fn_total"] == int(frame[f"fn__{member}"].sum()), member


@needs_audit
def test_rn_contributes_no_stn_confusion() -> None:
    """A structural check on the label geometry: RN is distant from STN, so a
    non-zero RN term would suggest the masks had been mixed up."""
    confusion = pd.read_csv(CONFUSION)
    for _, row in confusion.iterrows():
        assert row["fp_from_rn"] == 0
        assert row["fn_as_rn"] == 0


# --------------------------------------------------------------------------- #
# the reported mechanism, as far as it can be pinned
# --------------------------------------------------------------------------- #

@needs_audit
def test_dice_and_hd95_are_negatively_associated() -> None:
    """The headline association must have the sign the report claims.

    Note this is close to tautological -- Dice and HD95 are both functions of the
    same two masks -- so it is a consistency check, not independent evidence.
    """
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    entry = summary["G0_B_correlations"]["dice_vs_hd95__ensemble"]
    assert entry["rho"] < 0
    assert entry["bh_fdr"] < 0.05


@needs_audit
def test_crop_margin_shows_no_association_with_stn_error() -> None:
    """The audit reports no crop-boundary effect; the numbers must back that."""
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for key in ("min_margin_vs_dice__ensemble", "min_margin_vs_hd95__ensemble"):
        entry = summary["G0_B_correlations"][key]
        assert entry["p"] > 0.05, key
        assert abs(entry["rho"]) < 0.2, key


@needs_audit
def test_fp_shell_is_in_plane_dominant() -> None:
    """The resolution-limit claim rests on this: the one-voxel FP rate must be
    higher in the 0.667 mm in-plane directions than in the 2.0 mm D direction."""
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    shell = summary["G0_H_directional_shell"]["per_direction"]
    in_plane = np.mean([shell[f"{a}{s}"]["fp_rate"] for a in "hw"
                        for s in ("_plus", "_minus")])
    through = np.mean([shell[f"d{s}"]["fp_rate"] for s in ("_plus", "_minus")])
    assert in_plane > through, (
        f"in-plane FP rate {in_plane:.3f} should exceed through-plane "
        f"{through:.3f} for the resolution-limit reading to hold")


@needs_audit
def test_delta_table_is_paired_on_the_same_cases() -> None:
    frame = pd.read_csv(DELTAS)
    assert len(frame) == 40
    assert sorted(frame["case_id"]) == _val_case_ids()
    recomputed = frame["ensemble_dice"] - frame["baseline_dice"]
    assert np.allclose(frame["delta_dice"], recomputed, atol=1e-9)


# --------------------------------------------------------------------------- #
# nothing frozen was disturbed
# --------------------------------------------------------------------------- #

@needs_audit
def test_frozen_experiment_f_artifacts_are_unchanged() -> None:
    record = (PROJECT_ROOT / "results/experiments/baseline_deep_ensemble"
              / "FREEZE_RECORD.json")
    if not record.is_file():
        pytest.skip("freeze record absent")
    payload = json.loads(record.read_text(encoding="utf-8"))
    for artifact in payload["frozen_artifacts"]:
        path = Path(artifact["abs_path"])
        if not path.is_file():
            pytest.skip(f"{artifact['rel_path']} unavailable")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        assert digest.hexdigest() == artifact["sha256"], artifact["rel_path"]


@needs_audit
def test_holdout_predictions_were_not_reopened() -> None:
    """G0 must not touch the frozen holdout prediction artifact."""
    holdout = (PROJECT_ROOT / "results/experiments/baseline_deep_ensemble"
               / "secondary_frozen_holdout")
    recorded_path = holdout / "prediction_hashes.json"
    predictions = holdout / "predictions.npy"
    if not recorded_path.is_file():
        pytest.skip("holdout run absent")
    if not predictions.is_file():
        # The arrays are git-ignored bulk artifacts and are not shipped with the
        # repository, so a fresh checkout has nothing to hash. Same guard as the
        # freeze-record test above: verify if present, skip if not.
        pytest.skip("holdout prediction array not present in this checkout")
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    with predictions.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == recorded["file_sha256"]


def test_audit_script_does_not_read_holdout_ground_truth() -> None:
    """Structural: no *executable* reference to holdout data.

    The module docstring legitimately discusses what is avoided, so docstrings
    are stripped first -- otherwise the test would fail on the very text that
    documents the restriction.
    """
    import ast

    source = (PROJECT_ROOT / "scripts" / "analyze_stn_failure_mechanism.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body[0].value.value = ""          # drop the docstring
    code = ast.unparse(tree)

    for forbidden in ("processed/labels", "holdout_frozen_v1",
                      "internal_test", "secondary_frozen_holdout"):
        assert forbidden not in code, (
            f"the audit script references {forbidden!r} in executable code")

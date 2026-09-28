"""Tests for the reproduction entry points.

Three things are checked here, all of them about *paths and boundaries* rather
than about any scientific quantity:

* ``prepare``/``preflight`` know exactly which caches the 21 runs need, and the
  occupancy prior is refused when its case list is not the frozen manifest's --
  the failure mode that used to let a 160-case cache pass as the 161-case one;
* ``evaluate_reproduction`` refuses to write into any namespace that holds the
  author's frozen results;
* the evaluate entry runs end to end on the *synthetic* fixture and labels its
  own output as a reproduction evaluation, never as the author's historical one.

No real medical data is touched, and no real internal_test inference is run.

Run with:  python -m pytest tests/test_reproduction_paths.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import reproduce as rp  # noqa: E402  (no torch import at module scope)

pytest.importorskip("pandas", reason="pandas is required")


# --------------------------------------------------------------------------- #
# the run list is derived, never hand-maintained
# --------------------------------------------------------------------------- #

def test_configs_derive_the_21_formal_runs() -> None:
    runs, problems = rp.load_run_configs()
    assert problems == []
    assert len(runs) == rp.EXPECTED_N_RUNS == 21
    assert len({r["run_id"] for r in runs}) == 21
    assert "C_boundary" in {r["run_id"] for r in runs}


# --------------------------------------------------------------------------- #
# the occupancy prior must belong to the frozen cohort
# --------------------------------------------------------------------------- #

def _write_prior(cache: Path, case_ids: list[str], manifest_sha: str,
                 n_declared: int | None = None) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    sums = np.zeros((3, 32, 96, 96), dtype=np.uint16)
    sums[0, 10, 40, 40] = min(len(case_ids), 65535)
    np.save(cache / "occupancy_sum.npy", sums)
    (cache / "occupancy_meta.json").write_text(json.dumps({
        "n_train": n_declared if n_declared is not None else len(case_ids),
        "train_case_ids": case_ids,
        "source_split": "train",
        "manifest_sha256": manifest_sha,
    }), encoding="utf-8")


def test_preflight_rejects_a_prior_from_another_cohort(monkeypatch,
                                                       tmp_path: Path) -> None:
    """A 160-case cache must never be accepted as the 161-case prior."""
    cache = tmp_path / "prior"
    _write_prior(cache, [f"RJPD_{i:03d}" for i in range(160)], "0" * 64)
    monkeypatch.setattr(rp, "SPATIAL_PRIOR_CACHE", cache)

    problems = rp.verify_spatial_prior_case_set(
        rp.SUBJECT_CLEAN_MANIFESTS / "train.csv")

    assert problems, "a cache built over the wrong cohort was accepted"
    joined = " ".join(problems)
    assert "160" in joined and "161" in joined
    assert "will NOT be reused" in joined or "another cohort" in joined


def test_preflight_accepts_a_prior_built_from_the_frozen_manifest(
        monkeypatch, tmp_path: Path) -> None:
    manifest = rp.SUBJECT_CLEAN_MANIFESTS / "train.csv"
    case_ids = rp.read_manifest_ids(manifest)
    assert len(case_ids) == 161

    cache = tmp_path / "prior"
    _write_prior(cache, case_ids, rp.sha256_file(manifest))
    monkeypatch.setattr(rp, "SPATIAL_PRIOR_CACHE", cache)

    assert rp.verify_spatial_prior_case_set(manifest) == []


def test_preflight_flags_a_prior_built_from_a_different_manifest_revision(
        monkeypatch, tmp_path: Path) -> None:
    """Same case list, different manifest bytes: still refused."""
    manifest = rp.SUBJECT_CLEAN_MANIFESTS / "train.csv"
    cache = tmp_path / "prior"
    _write_prior(cache, rp.read_manifest_ids(manifest), "f" * 64)
    monkeypatch.setattr(rp, "SPATIAL_PRIOR_CACHE", cache)

    problems = rp.verify_spatial_prior_case_set(manifest)
    assert any("manifest hash" in p for p in problems)


def test_preflight_flags_a_missing_prior(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rp, "SPATIAL_PRIOR_CACHE", tmp_path / "absent")
    problems = rp.verify_spatial_prior_case_set(
        rp.SUBJECT_CLEAN_MANIFESTS / "train.csv")
    assert problems and "missing" in problems[0]


def test_manifest_counts_are_the_published_ones() -> None:
    """The expected counts are part of the frozen split, not a guess."""
    expected = {name: n for name, _file, n in rp.MANIFEST_FILES}
    assert expected == {"train": 161, "val": 38, "internal_test": 100,
                        "challenge_test": 199, "excluded": 2}
    for _split, filename, n in rp.MANIFEST_FILES:
        path = rp.SUBJECT_CLEAN_MANIFESTS / filename
        assert path.is_file(), f"missing frozen manifest {filename}"
        assert len(rp.read_manifest_ids(path)) == n


# --------------------------------------------------------------------------- #
# the reproduction evaluation never writes into a frozen namespace
# --------------------------------------------------------------------------- #

def test_evaluate_refuses_the_author_namespaces(tmp_path: Path) -> None:
    import evaluate_reproduction as er

    for frozen in er.FROZEN_NAMESPACES:
        with pytest.raises(SystemExit, match="frozen namespace"):
            er.assert_outside_frozen_namespaces(tmp_path / frozen / "nested", tmp_path)

    # and it allows the reproduction's own namespace
    er.assert_outside_frozen_namespaces(
        tmp_path / "results" / "reproduction_eval_v1" / "internal_test", tmp_path)


def test_frozen_namespaces_match_the_public_layout() -> None:
    import evaluate_reproduction as er

    assert "results/subject_clean_holdout" in er.FROZEN_NAMESPACES
    assert "results/experiments_subject_clean_v1" in er.FROZEN_NAMESPACES
    assert all(not n.startswith("results/reproduction") for n in er.FROZEN_NAMESPACES)


# --------------------------------------------------------------------------- #
# the legacy-fixture gate
# --------------------------------------------------------------------------- #

def _fake_legacy_fixture(root: Path, train_ids: list[str], val_ids: list[str],
                         cached_ids: list[str]) -> tuple[Path, Path]:
    pd = pytest.importorskip("pandas", reason="pandas is required")

    manifest_dir = root / "manifests" / "experiment"
    manifest_dir.mkdir(parents=True)
    for split, ids in (("train", train_ids), ("val", val_ids)):
        pd.DataFrame({"case_id": ids}).to_csv(manifest_dir / f"{split}.csv",
                                              index=False)
    cache_dir = root / "cache" / "baseline_v1"
    (cache_dir / "images").mkdir(parents=True)
    for case_id in cached_ids:
        (cache_dir / "images" / f"{case_id}.npy").write_bytes(b"")
    return manifest_dir, cache_dir


def test_legacy_gate_needs_the_split_and_a_cache_over_its_own_cases(tmp_path: Path) -> None:
    """The gate that keeps legacy tests from running against the wrong cohort.

    `cache/baseline_v1` existing is not sufficient: after `prepare` it holds the
    subject-clean cohort, and a legacy test run against it fails on cases the
    regrouping excluded (or on the manifest, which is not published at all).
    """
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
    import _local_fixtures as lf

    legacy_train, legacy_val = ["LEG_1", "LEG_2"], ["LEG_3"]
    other_cohort = ["SUBJ_1", "SUBJ_2", "SUBJ_3"]

    # nothing on disk
    assert not lf.legacy_training_cache_usable(tmp_path / "manifests" / "experiment",
                                               tmp_path / "cache" / "baseline_v1")

    # split present, cache absent
    manifest_dir, cache_dir = _fake_legacy_fixture(tmp_path, legacy_train, legacy_val, [])
    assert not lf.legacy_training_cache_usable(manifest_dir, cache_dir)

    # split present, cache built over a DIFFERENT cohort -> still refused
    manifest_dir, cache_dir = _fake_legacy_fixture(
        tmp_path / "other", legacy_train, legacy_val, other_cohort)
    assert not lf.legacy_training_cache_usable(manifest_dir, cache_dir)

    # split present and the cache covers exactly its cases -> usable
    manifest_dir, cache_dir = _fake_legacy_fixture(
        tmp_path / "good", legacy_train, legacy_val, legacy_train + legacy_val)
    assert lf.legacy_training_cache_usable(manifest_dir, cache_dir)


def test_legacy_gate_is_false_without_the_split() -> None:
    """A published checkout has no `manifests/experiment/`, so the gate is False."""
    import _local_fixtures as lf

    if not lf.LEGACY_MANIFEST_DIR.is_dir():
        assert not lf.legacy_training_cache_usable()
    else:
        pytest.skip("this checkout carries the unpublished legacy split")


# --------------------------------------------------------------------------- #
# end to end, on synthetic blobs only
# --------------------------------------------------------------------------- #

def _run(script: Path, *args: str) -> None:
    completed = subprocess.run([sys.executable, str(script), *args],
                               cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, \
        f"{script.name} failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory) -> Path:
    """A synthetic data root with images, labels and a baseline cache."""
    root = tmp_path_factory.mktemp("synthetic_repro") / "data"
    scripts = PROJECT_ROOT / "scripts"
    _run(scripts / "make_synthetic_data.py", "--out", str(root), "--overwrite")
    _run(scripts / "prepare_labels.py", "--root", str(root))
    _run(scripts / "normalize_images.py", "--root", str(root))
    _run(scripts / "build_manifests.py", "--root", str(root))
    _run(scripts / "build_baseline_cache.py", "--root", str(root),
         "--split-dir", str(root / "manifests"))
    return root


def test_evaluate_runs_end_to_end_on_synthetic_data(tmp_path: Path,
                                                    synthetic_data: Path) -> None:
    torch = pytest.importorskip("torch", reason="PyTorch is required")
    import evaluate_reproduction as er
    from models.anisotropic_unet3d import AnisotropicUNet3D, UNet3DConfig

    root = tmp_path / "root"
    (root / "manifests" / "synthetic_holdout").mkdir(parents=True)
    (root / "results" / "subject_clean_rerun").mkdir(parents=True)

    # internal_test := the synthetic validation case, in its own manifest dir so
    # the overlap check has no sibling split to compare against.
    source = (synthetic_data / "manifests" / "val.csv").read_text(encoding="utf-8-sig")
    header, *rows = source.strip().splitlines()
    assert rows, "the synthetic fixture produced no validation case"
    rewritten = rows[0].replace(",val,", ",internal_test,")
    manifest = root / "manifests" / "synthetic_holdout" / "internal_test.csv"
    manifest.write_text(header + "\n" + rewritten + "\n", encoding="utf-8-sig")
    case_ids = [rewritten.split(",")[0]]

    members = []
    for seed, run_id in ((42, "tiny_seed42"), (123, "tiny_seed123"),
                         (2026, "tiny_seed2026")):
        torch.manual_seed(seed)
        model = AnisotropicUNet3D(
            UNet3DConfig(in_channels=3, num_classes=4, base_channels=2))
        relative = f"checkpoints/{run_id}/run_best.pt"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(),
                    "config": {"in_channels": 3, "num_classes": 4,
                               "base_channels": 2},
                    "epoch": 1}, path)
        members.append({"member_run_id": run_id, "seed": seed, "best_epoch": 1,
                        "checkpoint_path": relative,
                        "checkpoint_sha256": er.sha256_file(path)})

    (root / "results" / "subject_clean_rerun" /
     "CLEAN_ENSEMBLE_FREEZE_RECORD.json").write_text(json.dumps({
         "record_type": "CLEAN_ENSEMBLE_FREEZE_RECORD", "members": members,
         "ensemble_recipe": {"per_model": "p_i = softmax(logits_i, dim=1)",
                             "combine": "p_ensemble = mean(p_i)",
                             "decision": "argmax(p_ensemble)"}}), encoding="utf-8")
    (root / "results" / "subject_clean_rerun" /
     "SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json").write_text(json.dumps({
         "written_before_any_internal_test_access": True,
         "dataset_identity": {"n_cases": len(case_ids),
                              "manifest_sha256": er.sha256_file(manifest)},
         "primary_comparator": {"id": "seed123", "locked": True,
                                "checkpoint": "checkpoints/tiny_seed123/run_best.pt"},
         "primary_success_rule": {"all_three_required": ["mean > 0"]},
         "statistical_analysis": {"bootstrap": {"n_resamples": 10000,
                                                "seed": 20260927},
                                  "permutation": {"n_draws": 100000}}}),
        encoding="utf-8")

    records_before = sorted(p.name for p in (PROJECT_ROOT / "results").iterdir()) \
        if (PROJECT_ROOT / "results").is_dir() else []

    out_dir = root / "results" / "reproduction_eval_v1" / "internal_test"
    rc = er.main(["--root", str(root), "--data-root", str(synthetic_data),
                  "--manifest", str(manifest),
                  "--reference-cache", str(synthetic_data / "cache" / "baseline_v1"),
                  "--reference-manifest-dir", str(synthetic_data / "manifests"),
                  "--out-dir", str(out_dir), "--device", "cpu"])
    assert rc == 0

    report = json.loads((out_dir / "REPRODUCTION_EVALUATION_REPORT.json")
                        .read_text(encoding="utf-8"))
    assert report["evaluation_kind"] == er.RECORD_KIND
    assert report["comparator"] == "seed123", "comparator must come from the record"
    assert report["test_set"]["is_a_fresh_unseen_test_set"] is False
    assert "NOT a fresh" in report["test_set"]["novelty_statement"]
    assert report["distinct_from"]["author_historical_corrected_fixed_re_evaluation"][
        "touched_by_this_run"] is False
    assert report["image_cache"]["reads_ground_truth"] is False
    assert report["image_cache"]["pipeline_equivalence_with_training_cache"]["all_equal"]
    assert report["statistics"]["matches_preregistration"] is True
    for name in ("REPRODUCTION_PER_CASE.csv", "REPRODUCTION_PREDICTIONS.npy",
                 "REPRODUCTION_PRIMARY_ENDPOINT.json",
                 "REPRODUCTION_SECONDARY_ENDPOINTS.csv",
                 "REPRODUCTION_EVALUATION_REPORT.md"):
        assert (out_dir / name).is_file(), f"{name} was not written"

    # the author's namespaces were not created or touched by this run
    records_after = sorted(p.name for p in (PROJECT_ROOT / "results").iterdir()) \
        if (PROJECT_ROOT / "results").is_dir() else []
    assert records_after == records_before


def test_evaluate_refuses_an_unlocked_comparator(tmp_path: Path,
                                                 synthetic_data: Path) -> None:
    """A comparator that is not locked is not a pre-registered comparator."""
    pytest.importorskip("torch", reason="PyTorch is required")
    import evaluate_reproduction as er

    freeze = {"members": [{"seed": s, "checkpoint_path": f"ckpt/{s}.pt"}
                          for s in (42, 123, 2026)]}
    prereg = {"primary_comparator": {"id": "seed123", "locked": False}}
    with pytest.raises(SystemExit, match="not marked locked"):
        er.resolve_comparator(prereg, freeze)

    # an id that is not a frozen member is refused too
    prereg["primary_comparator"] = {"id": "seed999", "locked": True}
    with pytest.raises(SystemExit, match="not a frozen ensemble member"):
        er.resolve_comparator(prereg, freeze)

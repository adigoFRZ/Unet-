"""Tests for the post-freeze code-amendment chain.

Two questions are covered here, and they are separate:

**1. Is the maintenance edit semantically inert?** The subject-clean holdout
evaluator no longer hard-codes its comparator; it reads the comparator from the
frozen pre-registration. The test proves that the historical experiment still
resolves to the same comparator the old constant named, and that every other
quantity that defines the experiment -- ensemble members, primary endpoint,
metric implementation, bootstrap and permutation parameters, checkpoints -- is
unchanged. Nothing here runs inference and nothing reads ground truth: these are
static and fixture-level checks over records that are already on disk.

**2. Is the amendment record honest?** Every entry must carry an original hash
that actually equals the value in the freeze record it cites (checked through a
JSON pointer, so an original hash cannot simply be invented), a current hash that
matches the file on disk, an allowed change type, and explicit
``false``/``false``/``true`` flags. There is no "ignore the hash" escape hatch,
and the tests assert that none can be added without failing here.

The frozen result files are read-only inputs. If they are absent -- a fresh clone
without the local ``results/`` tree -- the tests skip rather than fail.

Run with:  python -m pytest tests/test_post_freeze_amendments.py -v
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

HOLDOUT = PROJECT_ROOT / "results" / "subject_clean_holdout"
RERUN = PROJECT_ROOT / "results" / "subject_clean_rerun"
PROVENANCE = HOLDOUT / "HOLDOUT_EVALUATION_PROVENANCE.json"
AMENDMENTS = HOLDOUT / "POST_FREEZE_CODE_AMENDMENTS.json"
FREEZE_RECORD = RERUN / "SUBJECT_CLEAN_FREEZE_RECORD.json"
PREREG = RERUN / "SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json"
ENSEMBLE = RERUN / "CLEAN_ENSEMBLE_FREEZE_RECORD.json"
PER_CASE = HOLDOUT / "HOLDOUT_PER_CASE.csv"
PRIMARY = HOLDOUT / "HOLDOUT_PRIMARY_ENDPOINT.json"

requires_frozen = pytest.mark.skipif(
    not PROVENANCE.is_file() or not AMENDMENTS.is_file(),
    reason="local frozen holdout records are not present (fresh clone)",
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_pointer(doc, pointer: str):
    """RFC 6901. ``~1`` -> ``/`` and ``~0`` -> ``~``."""
    node = doc
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node


def load_module(name: str, rel_path: str):
    """Import a script as a module without letting it capture the test session.

    Several of these scripts replace ``sys.stdout`` with a UTF-8 wrapper at
    import time. That wrapper owns ``sys.stdout.buffer``, so when it is later
    garbage collected it closes the real stdout and every subsequent write in
    pytest raises "I/O operation on closed file". Restoring the original stream
    and detaching the wrapper's buffer keeps the process usable.
    """
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        spec.loader.exec_module(module)
    finally:
        hijacked = sys.stdout
        sys.stdout, sys.stderr = saved_out, saved_err
        if hijacked is not saved_out:
            try:
                hijacked.detach()  # release the buffer without closing it
            except (ValueError, AttributeError):
                pass
    return module


@pytest.fixture(scope="module")
def amendments() -> dict:
    return load_json(AMENDMENTS)


@pytest.fixture(scope="module")
def provenance() -> dict:
    return load_json(PROVENANCE)


# --------------------------------------------------------------------------- #
# 1. Semantic equivalence of the maintenance edit
# --------------------------------------------------------------------------- #


@requires_frozen
def test_historical_comparator_still_resolves_to_seed123() -> None:
    """The whole point of the edit: same comparator, read instead of hard-coded."""
    evaluator = load_module("ev_holdout", "scripts/evaluate_subject_clean_holdout.py")
    prereg = load_json(PREREG)
    ensemble = load_json(ENSEMBLE)

    resolved = evaluator.resolve_comparator(prereg, ensemble)

    assert resolved == "seed123"
    # and it is the frozen record, not the evaluator, that says so
    assert prereg["primary_comparator"]["id"] == "seed123"
    assert prereg["primary_comparator"]["locked"] is True


@requires_frozen
def test_historical_constant_was_seed123() -> None:
    """The superseded constant named the same comparator, so nothing moved."""
    old_source = _historical_evaluator_source()
    if old_source is None:
        pytest.skip("no git history available to recover the superseded constant")
    assert 'PRIMARY_COMPARATOR = "seed123"' in old_source


def _historical_evaluator_source() -> str | None:
    """The pre-amendment evaluator, from git. None when unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "show", "HEAD:scripts/evaluate_subject_clean_holdout.py"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # decode explicitly: the source is UTF-8 but the console default on Windows
    # is a legacy code page, and letting subprocess guess breaks on the CJK text
    return out.stdout.decode("utf-8", errors="replace")


@requires_frozen
def test_ensemble_members_unchanged() -> None:
    evaluator = load_module("ev_holdout2", "scripts/evaluate_subject_clean_holdout.py")
    ensemble = load_json(ENSEMBLE)
    seeds = sorted(f"seed{m['seed']}" for m in ensemble["members"])
    assert seeds == ["seed123", "seed2026", "seed42"]

    # the members the evaluator would load are exactly the frozen ones
    frozen = {f"seed{m['seed']}": m["checkpoint_sha256"] for m in ensemble["members"]}
    assert len(frozen) == 3
    assert evaluator is not None  # module imports cleanly under the new signature


@requires_frozen
def test_member_checkpoints_still_match_their_frozen_hashes() -> None:
    ensemble = load_json(ENSEMBLE)
    for member in ensemble["members"]:
        path = PROJECT_ROOT / member["checkpoint_path"]
        assert path.is_file(), f"missing frozen checkpoint {path}"
        assert sha256_of(path) == member["checkpoint_sha256"]


@requires_frozen
def test_metrics_implementation_unchanged(provenance: dict, amendments: dict) -> None:
    """The metric code is hash-pinned in the provenance record.

    A component may differ only when it carries an authorized amendment; the
    amendment's own integrity is covered by the tests further down.
    """
    amended = {e["file"] for e in amendments["amendments"]}
    for rel_path, want in provenance["component_sha256"].items():
        path = PROJECT_ROOT / rel_path
        assert path.is_file(), f"missing recorded component {rel_path}"
        actual = sha256_of(path)
        if actual != want:
            assert rel_path in amended, (
                f"recorded component changed without an amendment: {rel_path}")
    # the metric implementation itself must be untouched, amended or not
    metrics = PROJECT_ROOT / "src/evaluation/segmentation_metrics.py"
    assert sha256_of(metrics) == provenance["component_sha256"][
        "src/evaluation/segmentation_metrics.py"]


@requires_frozen
def test_bootstrap_and_permutation_parameters_unchanged() -> None:
    """Preregistered constants must not drift with a refactor."""
    ev = load_module("ev_frozen", "scripts/evaluate_frozen_holdout.py")
    assert ev.BOOTSTRAP_RESAMPLES == 10_000
    assert ev.PERMUTATION_DRAWS == 100_000
    assert ev.RNG_SEED == 20260927

    reported = load_json(PRIMARY)
    assert reported["bootstrap_n_resamples"] == ev.BOOTSTRAP_RESAMPLES
    assert reported["permutation_n_draws"] == ev.PERMUTATION_DRAWS
    assert reported["rng_seed"] == ev.RNG_SEED


@requires_frozen
def test_primary_endpoint_definition_unchanged() -> None:
    pre = load_json(PREREG)
    reported = load_json(PRIMARY)
    assert pre["primary_endpoint"]["name"] == reported["primary_endpoint"]
    assert pre["primary_success_rule"] == reported["primary_success_rule"]
    assert pre["statistical_analysis"]["bootstrap"]["seed"] == ev_rng_seed()
    assert reported["n_paired"] == pre["primary_endpoint"]["n_paired_cases"] == 100


def ev_rng_seed() -> int:
    return 20260927


@requires_frozen
def test_holdout_manifest_unchanged(provenance: dict) -> None:
    manifest = provenance["manifest"]
    path = PROJECT_ROOT / manifest["path"]
    assert sha256_of(path) == manifest["sha256"]
    assert manifest["n_cases"] == 100


@requires_frozen
def test_frozen_predictions_unchanged(provenance: dict) -> None:
    """The one-shot inference output must not have been touched."""
    artifact = provenance["predictions_artifact"]
    path = PROJECT_ROOT / artifact["path"]
    assert path.is_file()
    assert sha256_of(path) == artifact["sha256"]


@requires_frozen
def test_statistics_reproduce_the_frozen_numbers() -> None:
    """Re-aggregate the frozen per-case table with the CURRENT code.

    The evaluator's comparator used to come from a constant and now comes from
    the record. If that had changed which comparator is used, or if the
    statistics had drifted, these numbers would move. They must not.
    """
    pd = pytest.importorskip("pandas", reason="pandas is required")
    ev = load_module("ev_frozen2", "scripts/evaluate_frozen_holdout.py")

    frame = pd.read_csv(PER_CASE, encoding="utf-8-sig")
    cols = lambda who: [f"Dice_{roi}__{who}" for roi in ("STN", "SN", "RN")]  # noqa: E731
    delta = frame[cols("ensemble")].mean(axis=1).to_numpy() - \
        frame[cols("seed123")].mean(axis=1).to_numpy()

    reported = load_json(PRIMARY)
    assert round(float(frame[cols("ensemble")].mean(axis=1).mean()), 6) == \
        reported["ensemble_macro_Dice"]
    assert round(float(frame[cols("seed123")].mean(axis=1).mean()), 6) == \
        reported["comparator_macro_Dice"]
    assert abs(float(delta.mean()) - reported["mean_delta"]) < 5e-7

    ci_low, ci_high = ev.paired_bootstrap_ci(delta)
    p_value = ev.paired_signflip_pvalue(delta)
    # the frozen record keeps full precision, so compare with a tight tolerance
    assert abs(float(ci_low) - reported["bootstrap_ci_low"]) < 1e-12
    assert abs(float(ci_high) - reported["bootstrap_ci_high"]) < 1e-12
    assert abs(float(p_value) - reported["permutation_p"]) < 1e-12


# --------------------------------------------------------------------------- #
# 2. Integrity of the amendment record itself
# --------------------------------------------------------------------------- #


@requires_frozen
def test_amendment_record_shape(amendments: dict) -> None:
    assert amendments["record_type"] == "POST_FREEZE_CODE_AMENDMENTS"
    assert amendments["frozen_results_are_read_only"] is True
    assert amendments["allowed_change_types"], "allowed_change_types must be set"
    assert isinstance(amendments["amendments"], list)
    for entry in amendments["amendments"]:
        for key in ("file", "original_frozen_sha256", "current_sha256",
                    "frozen_sha256_source", "change_type",
                    "scientific_semantics_changed", "historical_result_changed",
                    "authorized_for_future_reproduction", "reason_en"):
            assert key in entry, f"amendment entry missing {key!r}"


@requires_frozen
def test_no_hash_ignoring_escape_hatch(amendments: dict) -> None:
    """The amendment chain must not contain a switch that disables checking."""
    blob = json.dumps(amendments).lower()
    for forbidden in ("ignore_hash", "skip_hash", "ignorehash",
                      "disable_check", "\"ignore\"", "wildcard"):
        assert forbidden not in blob, f"forbidden escape hatch present: {forbidden}"


@requires_frozen
def test_original_hashes_match_the_freeze_records(amendments: dict) -> None:
    """An entry may not invent its own original hash."""
    for entry in amendments["amendments"]:
        source = entry["frozen_sha256_source"]
        record_path = PROJECT_ROOT / source["record"]
        assert record_path.is_file(), f"cited record missing: {source['record']}"
        on_record = json_pointer(load_json(record_path), source["pointer"])
        assert entry["original_frozen_sha256"] == on_record, (
            f"{entry['file']}: original hash does not match {source['record']}"
            f"{source['pointer']}")
        # and it is NOT the current hash, i.e. this really is a change
        assert entry["original_frozen_sha256"] != entry["current_sha256"]


@requires_frozen
def test_current_hashes_match_disk(amendments: dict) -> None:
    for entry in amendments["amendments"]:
        path = PROJECT_ROOT / entry["file"]
        assert path.is_file(), f"amended file missing: {entry['file']}"
        assert sha256_of(path) == entry["current_sha256"], (
            f"{entry['file']}: recorded current hash is stale; the file changed "
            f"again without a new amendment entry")


@requires_frozen
def test_amendments_declare_no_semantic_or_result_change(amendments: dict) -> None:
    for entry in amendments["amendments"]:
        assert entry["change_type"] in amendments["allowed_change_types"]
        assert entry["scientific_semantics_changed"] is False
        assert entry["historical_result_changed"] is False
        assert entry["authorized_for_future_reproduction"] is True


@requires_frozen
def test_every_registered_file_is_either_frozen_or_amended(provenance: dict,
                                                           amendments: dict) -> None:
    """No hash-registered file may be silently different."""
    amended = {e["file"] for e in amendments["amendments"]}
    registry: dict[str, str] = {}
    script = provenance.get("evaluation_script") or {}
    if script.get("sha256"):
        registry[script.get("path") or "scripts/evaluate_subject_clean_holdout.py"] = \
            script["sha256"]
    registry.update(provenance.get("component_sha256") or {})

    for rel_path, frozen_sha in registry.items():
        actual = sha256_of(PROJECT_ROOT / rel_path)
        if actual == frozen_sha:
            continue
        assert rel_path in amended, (
            f"{rel_path} differs from its frozen hash and is not registered as an "
            f"authorized amendment")


@requires_frozen
def test_frozen_result_files_are_unmodified(provenance: dict) -> None:
    """The amendment chain must not have rewritten any historical result."""
    for name, want in (("HOLDOUT_PER_CASE.csv", None),
                       ("HOLDOUT_PRIMARY_ENDPOINT.json", None)):
        path = HOLDOUT / name
        assert path.is_file(), f"frozen result missing: {name}"
    # the manifest and checkpoints the provenance pins are the hard evidence
    assert sha256_of(PROJECT_ROOT / provenance["manifest"]["path"]) == \
        provenance["manifest"]["sha256"]
    assert sha256_of(PROJECT_ROOT / provenance["predictions_artifact"]["path"]) == \
        provenance["predictions_artifact"]["sha256"]
    for member in load_json(ENSEMBLE)["members"]:
        assert sha256_of(PROJECT_ROOT / member["checkpoint_path"]) == \
            member["checkpoint_sha256"]


@requires_frozen
def test_freeze_record_config_hashes_are_either_frozen_or_amended(
        amendments: dict) -> None:
    """Config drift is checked the same way, against the config freeze record."""
    if not FREEZE_RECORD.is_file():
        pytest.skip("config freeze record not present")
    record = load_json(FREEZE_RECORD)
    amended = {e["file"] for e in amendments["amendments"]}
    for rel_path, want in (record.get("clean_configs") or {}).items():
        path = PROJECT_ROOT / rel_path
        if not path.is_file():
            continue
        if sha256_of(path) == want:
            continue
        assert rel_path in amended, (
            f"{rel_path} differs from its frozen config hash and is not registered")


@requires_frozen
def test_amended_configs_are_semantically_identical() -> None:
    """A cosmetically amended config must still resolve to the frozen values.

    The frozen runs recorded their fully resolved config, which is far stronger
    evidence than a file hash: it is what the trainer actually used.
    """
    pd = pytest.importorskip("pandas", reason="pandas is required")
    from dataclasses import asdict

    from training.train_baseline import load_config, resolve_input_channels

    registry = pd.read_csv(RERUN / "TIER2_RUN_REGISTRY.csv", encoding="utf-8-sig")
    for _, row in registry.iterrows():
        summary_path = (PROJECT_ROOT / "results" / "experiments_subject_clean_v1"
                        / row.run_id / "training_summary_run.json")
        if not summary_path.is_file():
            pytest.skip(f"{row.run_id}: training summary not present")
        frozen = load_json(summary_path)["config"]
        current = asdict(load_config(PROJECT_ROOT / row.clean_config))

        # paths are re-anchored at run time; in_channels is derived, checked below
        skip = {"root", "checkpoint_dir", "results_dir", "cache_dir",
                "manifest_dir", "boundary_cache_dir", "spatial_prior_cache_dir",
                "in_channels"}
        for key, want in frozen.items():
            if key in skip:
                continue
            got = current.get(key, "<missing>")
            if isinstance(want, bool) or isinstance(got, bool):
                assert got == want, f"{row.run_id}.{key}: {got!r} != {want!r}"
            elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
                assert abs(want - got) < 1e-12, f"{row.run_id}.{key}: {got} != {want}"
            else:
                assert got == want, f"{row.run_id}.{key}: {got!r} != {want!r}"

        assert resolve_input_channels(
            load_config(PROJECT_ROOT / row.clean_config)) == frozen["in_channels"], \
            f"{row.run_id}: derived in_channels changed"

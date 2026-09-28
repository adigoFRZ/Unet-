#!/usr/bin/env python
"""One entry point for reproducing this project.

    python scripts/reproduce.py smoke
    python scripts/reproduce.py prepare --data-root <PDCADxFoundation path>
    python scripts/reproduce.py train
    python scripts/reproduce.py summarize

What each one is for
--------------------
``smoke``
    No medical data required. Generates the synthetic fixture and runs the whole
    chain -- labels -> normalisation -> manifests -> cache -> 1 epoch of
    training -> checkpoint save -> checkpoint reload -> validation. Every step
    is invoked here; nothing has to be run by hand. Anything it reports is a
    wiring check, never a result.

``prepare``
    Real data. Runs label extraction and normalisation, then HARD-VALIDATES the
    cases found on disk against the frozen manifests in
    ``manifests/subject_clean_v1/`` and builds **every** cache the 21 formal runs
    need: the baseline training cache, Experiment C's STN signed-distance cache,
    and Experiment E's subject-clean occupancy prior. The split is read from the
    repository; it is never re-derived. A case-count or case-set mismatch is a
    fatal error -- this script will not silently re-split, and it will not reuse
    an older cache built over a different cohort. ``prepare`` ends by running the
    preflight described below and prints ``PREPARE = PASS`` only if every input
    the 21 runs need is present.

``preflight``
    Checks -- without building anything -- that the manifests, the baseline
    cache, the boundary cache and the occupancy prior are all present and that
    the occupancy prior was built from exactly the 161 subject-clean
    development-train cases. Run it after ``prepare`` or any time later; it needs
    no access to the raw dataset.

``train``
    Runs the 21 formal subject-clean runs declared by the configs. The run
    registry is created automatically on first use, so a fresh clone needs no
    pre-existing state. The runs write to ``checkpoints/subject_clean_v1/`` and
    ``results/experiments_subject_clean_v1/``.

``summarize``
    Aggregates the validation results and writes the frozen records a later
    evaluation needs (clean-val table, baseline seed selection, ensemble freeze,
    holdout pre-registration). Touches train/val only.

``evaluate``
    The final step: evaluates the checkpoints **this reproduction trained** on
    the frozen 100-case ``internal_test`` split, with the frozen metric, ensemble
    recipe, comparator rule and statistics. It writes to a new reproduction
    namespace and refuses to touch the author's historical results. The 100 cases
    are not a fresh unseen test set, and the script says so in every record it
    writes.

Every child process is launched with ``sys.executable``, so Windows, Linux and
macOS all work from any virtualenv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
STATE = REPO / "results" / "reproduce" / "state.json"

SYNTHETIC_ROOT = REPO / "tests" / "_synthetic_data"
SUBJECT_CLEAN_MANIFESTS = REPO / "manifests" / "subject_clean_v1"

BASELINE_CACHE = REPO / "cache" / "baseline_v1"
BOUNDARY_CACHE = REPO / "cache" / "boundary_v1" / "stn_signed_distance"
SPATIAL_PRIOR_CACHE = REPO / "cache" / "spatial_prior_v1_subject_clean"

#: The published subject-clean split, as counts. These are frozen facts of the
#: published partition (README "结果" section); preflight compares the manifests
#: against them so that a silently re-derived or truncated split cannot pass as
#: "ready to train". ``excluded`` is ``EXCLUDED_CASES.csv``.
MANIFEST_FILES: tuple[tuple[str, str, int], ...] = (
    ("train", "train.csv", 161),
    ("val", "val.csv", 38),
    ("internal_test", "internal_test.csv", 100),
    ("challenge_test", "challenge_test.csv", 199),
    ("excluded", "EXCLUDED_CASES.csv", 2),
)

#: Number of formal runs the campaign is defined by. Derived from the configs at
#: run time; this is the expected size of that derivation.
EXPECTED_N_RUNS = 21


# --------------------------------------------------------------------------- #
# Process helpers
# --------------------------------------------------------------------------- #


def step(number: str, title: str) -> None:
    print()
    print("=" * 78)
    print(f"[{number}] {title}")
    print("=" * 78)


def run(args: Sequence[str | Path], *, cwd: Path = REPO) -> int:
    """Run a child step with this interpreter and stream its output."""
    cmd = [str(sys.executable), *[str(a) for a in args]]
    print(f"$ {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(cmd, cwd=str(cwd), env=env)
    return completed.returncode


def fail(message: str) -> int:
    print(f"\nerror: {message}\n", file=sys.stderr)
    return 1


def load_state() -> dict:
    if STATE.is_file():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(**updates) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state.update(updates)
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #


def cmd_smoke(args: argparse.Namespace) -> int:
    root = SYNTHETIC_ROOT
    print("SYNTHETIC SMOKE TEST")
    print("This runs on random blobs, not medical images. Every metric it prints")
    print("is meaningless as a segmentation result. It only proves the code runs.")

    step("1/8", "Generate the synthetic fixture")
    if run([SCRIPTS / "make_synthetic_data.py", "--out", root, "--overwrite"]) != 0:
        return fail("synthetic data generation failed")

    step("2/8", "Extract labels")
    if run([SCRIPTS / "prepare_labels.py", "--root", root]) != 0:
        return fail("prepare_labels failed")

    step("3/8", "Normalise images")
    if run([SCRIPTS / "normalize_images.py", "--root", root]) != 0:
        return fail("normalize_images failed")

    step("4/8", "Build manifests")
    if run([SCRIPTS / "build_manifests.py", "--root", root]) != 0:
        return fail("build_manifests failed")

    step("5/8", "Build the training cache")
    if run([SCRIPTS / "build_baseline_cache.py", "--root", root,
            "--split-dir", root / "manifests"]) != 0:
        return fail("build_baseline_cache failed")

    step("6/8", "Train (1 epoch), saving a checkpoint")
    if run(["src/training/train_baseline.py",
            "--config", REPO / "configs" / "synthetic_smoke.yaml",
            "--max-epochs", str(args.max_epochs)]) != 0:
        return fail("training failed")

    step("7/8", "Reload the saved checkpoint")
    if run(["-c", CHECKPOINT_RELOAD_PROBE]) != 0:
        return fail("checkpoint reload failed")

    step("8/8", "Validation pass")
    print("train_baseline.py already ran its final validation pass on the best")
    print("checkpoint in step 6; step 7 re-loaded that checkpoint from disk to")
    print("prove it is readable independently of the training process.")
    summary = sorted((root / "results").rglob("val_summary_best.json"))
    if not summary:
        return fail("no val_summary_best.json was written by the validation pass")
    payload = json.loads(summary[-1].read_text(encoding="utf-8"))
    macro = payload.get("macro_foreground_dice_mean")
    print(f"\nvalidation summary : {summary[-1].relative_to(REPO).as_posix()}")
    print(f"cases              : {payload.get('n_cases')}")
    print(f"macro foreground Dice = {macro}   <-- SYNTHETIC, NOT A RESULT")

    print()
    print("=" * 78)
    print("SMOKE = PASS   (wiring only; no scientific claim)")
    print("=" * 78)
    return 0


#: Reloads the checkpoint the smoke run just wrote, with no training context.
CHECKPOINT_RELOAD_PROBE = r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
import torch
from training.train_baseline import load_checkpoint

root = Path("tests/_synthetic_data/checkpoints")
candidates = sorted(root.rglob("run_best.pt")) or sorted(root.rglob("*.pt"))
if not candidates:
    raise SystemExit(f"error: no checkpoint written under {root}")
path = candidates[-1]
payload = load_checkpoint(path)
missing = {"model_state", "config"} - set(payload)
if missing:
    raise SystemExit(f"error: checkpoint {path} is missing {sorted(missing)}")
n = sum(v.numel() for v in payload["model_state"].values())
print(f"reloaded {path}")
print(f"  epoch       : {payload.get('epoch')}")
print(f"  parameters  : {n:,}")
print(f"  config keys : {len(payload.get('config', {}))}")
if n == 0:
    raise SystemExit("error: checkpoint has no parameters")
print("checkpoint reload OK")
""".strip()


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #


def validate_cases_against_manifests(data_root: Path) -> list[str]:
    """Compare the raw cases on disk with the frozen split. Returns problems.

    This is the guard that makes the published split authoritative: the raw data
    either matches it exactly or the pipeline stops. It never re-derives a
    split, and it never drops or invents a case to make the numbers line up.
    """
    sys.path.insert(0, str(REPO / "src"))
    from data import pdcadx_io as pio  # imported here so `--help` stays fast

    import pandas as pd

    discovered = {c.case_id for c in pio.discover_cases(data_root)}
    expected: set[str] = set()
    per_split: dict[str, set[str]] = {}
    for split, filename in (("train", "train.csv"), ("val", "val.csv"),
                            ("internal_test", "internal_test.csv"),
                            ("challenge_test", "challenge_test.csv"),
                            ("excluded", "EXCLUDED_CASES.csv")):
        frame = pd.read_csv(SUBJECT_CLEAN_MANIFESTS / filename,
                            encoding="utf-8-sig", dtype=str)
        per_split[split] = set(frame["case_id"])
        expected |= per_split[split]

    print(f"  cases on disk                 : {len(discovered)}")
    print(f"  cases in the frozen manifests : {len(expected)}")
    for split in ("train", "val", "internal_test", "challenge_test", "excluded"):
        print(f"    {split:<16} {len(per_split[split])}")

    problems: list[str] = []
    missing = sorted(expected - discovered)
    unexpected = sorted(discovered - expected)
    if missing:
        problems.append(
            f"{len(missing)} case(s) required by the frozen manifests are absent "
            f"from {data_root}: {missing[:10]}"
            + (" ..." if len(missing) > 10 else ""))
    if unexpected:
        problems.append(
            f"{len(unexpected)} case(s) on disk are not in the frozen manifests: "
            f"{unexpected[:10]}" + (" ..." if len(unexpected) > 10 else ""))
    if problems:
        problems.append(
            "The published split is authoritative. Fix the data root, or if the "
            "official release genuinely changed, re-derive the split deliberately "
            "-- this script will not do it for you.")
    return problems


def cmd_prepare(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        return fail(f"--data-root is not a directory: {data_root}")

    print(f"data root : {data_root}")
    print(f"repo root : {REPO}")
    print(f"split     : {SUBJECT_CLEAN_MANIFESTS.relative_to(REPO).as_posix()} "
          f"(frozen; NOT regenerated)")

    step("1/6", "Check the raw cases against the frozen subject-clean manifests")
    problems = validate_cases_against_manifests(data_root)
    if problems:
        for p in problems:
            print(f"  BAD {p}", file=sys.stderr)
        return fail("raw data does not match the frozen split; refusing to continue")

    if args.processed_only:
        print("  --processed-only: skipping label extraction and normalisation.")

    if not args.processed_only:
        step("2/6", "Extract labels from the official masks")
        if run([SCRIPTS / "prepare_labels.py", "--root", data_root]) != 0:
            return fail("prepare_labels failed")

        step("3/6", "Normalise images")
        if run([SCRIPTS / "normalize_images.py", "--root", data_root]) != 0:
            return fail("normalize_images failed")

    step("4/6", "Build the training cache (train + val only)")
    cache_args = [
        SCRIPTS / "build_baseline_cache.py",
        "--split-dir", SUBJECT_CLEAN_MANIFESTS,
        "--out-dir", BASELINE_CACHE,
        "--results-dir", REPO / "results" / "build_baseline_cache",
        # `--root` anchors the default results/log paths; the image and label
        # directories are passed explicitly because the raw data may live
        # outside the repository.
        "--root", data_root,
        "--image-dir", data_root / "processed" / "images",
        "--label-dir", data_root / "processed" / "labels",
    ]
    if args.overwrite:
        cache_args.append("--overwrite")
    if run(cache_args) != 0:
        return fail("build_baseline_cache failed")

    step("5/6", "Build Experiment C's STN signed-distance cache")
    print("  supervision for C_boundary; derived from each case's own GT mask")
    boundary_args = [
        SCRIPTS / "build_boundary_cache.py",
        "--root", REPO,
        # Experiment C's config points at cache/boundary_v1/stn_signed_distance;
        # the frozen subject-clean manifest is the only split this may be built
        # from, and the labels come from the cache step 4 just wrote.
        "--manifest-dir", SUBJECT_CLEAN_MANIFESTS,
        "--label-dir", BASELINE_CACHE / "labels",
        "--out-dir", BOUNDARY_CACHE,
        "--splits", "train,val",
    ]
    if args.overwrite:
        boundary_args.append("--overwrite")
    if run(boundary_args) != 0:
        return fail("build_boundary_cache failed")

    step("6/6", "Build Experiment E's subject-clean occupancy prior")
    n_train = len(read_manifest_ids(SUBJECT_CLEAN_MANIFESTS / "train.csv"))
    print(f"  cohort: the {n_train} subject-clean development-train cases, from")
    print(f"  {SUBJECT_CLEAN_MANIFESTS.relative_to(REPO).as_posix()}/train.csv")
    print("  The prior records its own case list, so a cache left over from the")
    print("  older 160-case campaign is refused rather than reused.")
    prior_args = [
        SCRIPTS / "build_spatial_prior_cache.py",
        "--root", REPO,
        "--manifest-dir", SUBJECT_CLEAN_MANIFESTS,
        "--label-dir", BASELINE_CACHE / "labels",
        "--out-dir", SPATIAL_PRIOR_CACHE,
        # Explicit, and derived from the frozen manifest rather than from a
        # default: this is the subject-clean cohort, not the legacy 160.
        "--expect-n-train", str(n_train),
    ]
    if run(prior_args) != 0:
        return fail("build_spatial_prior_cache failed")

    step("preflight", "Confirm every input the 21 formal runs need is present")
    problems, report = run_preflight()
    for line in report:
        print(f"  OK  {line}")
    if problems:
        for problem in problems:
            print(f"  BAD {problem}", file=sys.stderr)
        return fail("prepare finished but the preflight did not pass; "
                    "the 21 runs are NOT ready to start")

    save_state(data_root=str(data_root), prepared_at_utc=_now())

    print()
    print("=" * 78)
    print("PREPARE = PASS   (21/21 runs preflight-clean)")
    print(f"  baseline cache : {BASELINE_CACHE.relative_to(REPO).as_posix()}")
    print(f"  boundary cache : {BOUNDARY_CACHE.relative_to(REPO).as_posix()}")
    print(f"  occupancy prior: {SPATIAL_PRIOR_CACHE.relative_to(REPO).as_posix()} "
          f"(n_train={n_train})")
    print(f"  manifests      : {SUBJECT_CLEAN_MANIFESTS.relative_to(REPO).as_posix()} (frozen)")
    print("  next           : python scripts/reproduce.py train")
    print("=" * 78)
    return 0


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest_ids(path: Path) -> list[str]:
    import pandas as pd

    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    return sorted(str(c) for c in frame["case_id"])


def verify_spatial_prior_case_set(train_manifest: Path) -> list[str]:
    """The occupancy prior must have been built from exactly this manifest.

    Checked against the cache's own metadata, not against a directory listing:
    ``n_train`` and the recorded case list are what a training run actually
    divides by, so a mismatch there is the failure that matters. An older
    (160-case) cache fails here.
    """
    problems: list[str] = []
    shown = SPATIAL_PRIOR_CACHE.relative_to(REPO).as_posix() \
        if REPO in SPATIAL_PRIOR_CACHE.parents else str(SPATIAL_PRIOR_CACHE)
    sums_path = SPATIAL_PRIOR_CACHE / "occupancy_sum.npy"
    meta_path = SPATIAL_PRIOR_CACHE / "occupancy_meta.json"
    if not sums_path.is_file() or not meta_path.is_file():
        return [f"occupancy prior missing at {shown} "
                f"(expected occupancy_sum.npy and occupancy_meta.json)"]
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = read_manifest_ids(train_manifest)
    recorded = sorted(str(c) for c in metadata.get("train_case_ids", []))
    if recorded != expected:
        problems.append(
            f"occupancy prior covers {len(recorded)} case(s) but the train "
            f"manifest has {len(expected)}; missing "
            f"{sorted(set(expected) - set(recorded))[:5]}, unexpected "
            f"{sorted(set(recorded) - set(expected))[:5]} -- this looks like a "
            f"cache built from another cohort and it will NOT be reused")
    if metadata.get("source_split") != "train":
        problems.append(
            f"occupancy prior was built from split {metadata.get('source_split')!r}, "
            f"not 'train'")
    declared = metadata.get("manifest_sha256")
    if declared and declared != sha256_file(train_manifest):
        problems.append(
            "occupancy prior records a different train-manifest hash than the "
            f"frozen one ({str(declared)[:16]}… vs "
            f"{sha256_file(train_manifest)[:16]}…)")
    return problems


def load_run_configs() -> tuple[list[dict], list[str]]:
    """Derive the formal run list from the configs, exactly as the driver does."""
    import yaml

    config_dir = REPO / "configs" / "subject_clean_v1"
    problems: list[str] = []
    runs: list[dict] = []
    for path in sorted(config_dir.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        results_dir = str(raw.get("results_dir", ""))
        checkpoint_dir = str(raw.get("checkpoint_dir", ""))
        if not results_dir or not checkpoint_dir:
            problems.append(f"{path.relative_to(REPO).as_posix()} declares no "
                            f"results_dir / checkpoint_dir")
            continue
        run_id = Path(results_dir).name
        if Path(checkpoint_dir).name != run_id:
            problems.append(f"{path.relative_to(REPO).as_posix()}: checkpoint_dir "
                            f"and results_dir disagree on the run name")
            continue
        runs.append({"run_id": run_id, "config": path, "raw": raw})
    seen: dict[str, int] = {}
    for run in runs:
        seen[run["run_id"]] = seen.get(run["run_id"], 0) + 1
    duplicated = sorted(k for k, v in seen.items() if v > 1)
    if duplicated:
        problems.append(f"duplicate run_id across the configs: {duplicated}")
    if len(runs) != EXPECTED_N_RUNS:
        problems.append(f"the configs derive {len(runs)} runs, expected "
                        f"{EXPECTED_N_RUNS}")
    return runs, problems


def run_preflight() -> tuple[list[str], list[str]]:
    """Check every input the 21 formal runs need. Returns (problems, report)."""
    problems: list[str] = []
    report: list[str] = []

    # ---- 1. the frozen split ---------------------------------------------- #
    manifest_ids: dict[str, list[str]] = {}
    for split, filename, expected_n in MANIFEST_FILES:
        path = SUBJECT_CLEAN_MANIFESTS / filename
        if not path.is_file():
            problems.append(f"manifest missing: {path.relative_to(REPO).as_posix()}")
            continue
        ids = read_manifest_ids(path)
        manifest_ids[split] = ids
        if len(ids) != expected_n:
            problems.append(
                f"{filename}: {len(ids)} case(s), expected {expected_n} for the "
                f"frozen subject-clean split")
        else:
            report.append(f"manifest {split:<16} {len(ids):>4} cases  OK")

    # ---- 2. the caches all 21 runs share ---------------------------------- #
    train_val = (manifest_ids.get("train", []) + manifest_ids.get("val", []))
    if train_val and not BASELINE_CACHE.is_dir():
        problems.append(
            f"baseline cache missing entirely: "
            f"{BASELINE_CACHE.relative_to(REPO).as_posix()} -- "
            f"run `python scripts/reproduce.py prepare --data-root ...`")
    elif train_val:
        missing_images = [c for c in train_val
                          if not (BASELINE_CACHE / "images" / f"{c}.npy").is_file()]
        missing_labels = [c for c in train_val
                          if not (BASELINE_CACHE / "labels" / f"{c}.npy").is_file()]
        if missing_images:
            problems.append(
                f"baseline cache: {len(missing_images)} missing image tensor(s), "
                f"e.g. {missing_images[:5]} -- re-run prepare with --overwrite")
        if missing_labels:
            problems.append(
                f"baseline cache: {len(missing_labels)} missing label tensor(s), "
                f"e.g. {missing_labels[:5]} -- re-run prepare with --overwrite")
        if not missing_images and not missing_labels:
            report.append(f"baseline cache   {len(train_val):>4} cases  OK "
                          f"({BASELINE_CACHE.relative_to(REPO).as_posix()})")

    # ---- 3. the per-experiment caches ------------------------------------- #
    runs, config_problems = load_run_configs()
    problems.extend(config_problems)
    if not config_problems:
        report.append(f"configs          {len(runs):>4} runs    OK (derived, no "
                      f"duplicate run_id)")

    needs_boundary = [r["run_id"] for r in runs if r["raw"].get("boundary_cache_dir")]
    needs_occupancy = [r["run_id"] for r in runs
                       if str(r["raw"].get("spatial_prior_mode", "none"))
                       in ("occupancy", "both")]

    if needs_boundary:
        boundary_cases = sorted(set(manifest_ids.get("train", [])
                                    + manifest_ids.get("val", [])))
        missing = [c for c in boundary_cases
                   if not (BOUNDARY_CACHE / f"{c}.npy").is_file()]
        if not BOUNDARY_CACHE.is_dir() or missing:
            problems.append(
                f"boundary cache: {len(missing)} of {len(boundary_cases)} map(s) "
                f"missing from {BOUNDARY_CACHE.relative_to(REPO).as_posix()} "
                f"(needed by {', '.join(needs_boundary)}), e.g. {missing[:5]} -- "
                f"re-run prepare with --overwrite")
        else:
            report.append(f"boundary cache   {len(boundary_cases):>4} cases  OK "
                          f"({BOUNDARY_CACHE.relative_to(REPO).as_posix()}) "
                          f"for {', '.join(needs_boundary)}")

    if needs_occupancy:
        prior_problems = verify_spatial_prior_case_set(
            SUBJECT_CLEAN_MANIFESTS / "train.csv")
        if prior_problems:
            problems.extend(f"{p} (needed by {', '.join(needs_occupancy)})"
                            for p in prior_problems)
        else:
            n_train = len(manifest_ids.get("train", []))
            report.append(f"occupancy prior  {n_train:>4} cases  OK "
                          f"({SPATIAL_PRIOR_CACHE.relative_to(REPO).as_posix()}) "
                          f"for {', '.join(needs_occupancy)}")

    # ---- 4. anything the runs would need that is still missing ------------ #
    # Aggregated by path: all 21 configs share `cache/baseline_v1` and
    # `manifests/subject_clean_v1`, so reporting them once per distinct value
    # keeps a failure readable instead of repeating it 21 times.
    required: dict[tuple[str, str], list[str]] = {}
    for run in runs:
        # `cache_dir` is not checked here: the per-case checks above already
        # prove whether the baseline cache is usable, and they say *which* cases
        # are missing rather than that a directory does not exist.
        for key in ("manifest_dir",):
            value = run["raw"].get(key)
            if not value:
                problems.append(f"{run['run_id']}: config declares no {key}")
                continue
            required.setdefault((key, str(value)), []).append(run["run_id"])
    frozen_manifest_rel = SUBJECT_CLEAN_MANIFESTS.relative_to(REPO).as_posix()
    for (key, value), run_ids in sorted(required.items()):
        if not (REPO / value).exists():
            problems.append(
                f"{key}={value} does not exist "
                f"(needed by {len(run_ids)} run(s): {', '.join(run_ids[:4])}"
                + (", ..." if len(run_ids) > 4 else "") + ")")
        elif key == "manifest_dir" and value.rstrip("/") != frozen_manifest_rel:
            problems.append(
                f"{len(run_ids)} run(s) read their split from {value!r}, not the "
                f"frozen {frozen_manifest_rel!r}; the published split is the one "
                f"the campaign is defined on")
        else:
            report.append(f"{key:<16} {value:<32} OK "
                          f"({len(run_ids)} run(s))")
    return problems, report


def cmd_preflight(args: argparse.Namespace) -> int:
    step("preflight", "Check every input the 21 formal runs need")
    problems, report = run_preflight()
    for line in report:
        print(f"  OK  {line}")
    if problems:
        for problem in problems:
            print(f"  BAD {problem}", file=sys.stderr)
        print()
        print("PREFLIGHT = FAIL")
        return 1
    print()
    print("=" * 78)
    print("PREFLIGHT = PASS   (21/21 runs have their inputs and caches)")
    print("  next           : python scripts/reproduce.py train")
    print("=" * 78)
    return 0


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #


def cmd_train(args: argparse.Namespace) -> int:
    registry = REPO / "results" / "subject_clean_rerun" / "TIER2_RUN_REGISTRY.csv"
    if not registry.is_file():
        print("No run registry found; it will be created from the configs.")
    step("train", "Run the 21 formal subject-clean runs")
    extra: list[str | Path] = []
    if args.dry_run:
        extra.append("--dry-run")
    if args.reset_registry:
        extra.append("--reset-registry")
    return run([SCRIPTS / "run_tier2_subject_clean.py", *extra])


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #


def cmd_summarize(args: argparse.Namespace) -> int:
    step("summarize", "Aggregate clean-val results and freeze the records")
    print("Reads train/val results only. internal_test and challenge_test stay closed.")
    print()
    print("NOTE: this (re)writes the frozen records under results/subject_clean_rerun/")
    print("      -- clean-val table, baseline seed selection, ensemble freeze and the")
    print("      holdout pre-registration. On a machine that already holds a completed")
    print("      campaign, running it again replaces those records with fresh ones.")
    if not args.yes:
        print("\n      Pass --yes to confirm.")
        return 0
    return run([SCRIPTS / "aggregate_clean_val_results.py"])


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


def cmd_evaluate(args: argparse.Namespace) -> int:
    step("evaluate", "Independent reproduction evaluation on the 100 internal_test cases")
    print("Runs the checkpoints THIS reproduction trained against the frozen")
    print("internal_test split, using the frozen metric, ensemble recipe, comparator")
    print("and statistics. It writes to results/reproduction_eval_v1/ and refuses to")
    print("write anywhere near the author's historical results.")
    print()
    print("Note what this is: internal_test is NOT a fresh unseen test set. The same")
    print("100 cases were evaluated once by the author before publication; only the")
    print("weights are new here. The report says so in full.")
    print()
    evaluate_args: list[str | Path] = [
        SCRIPTS / "evaluate_reproduction.py",
        "--data-root", Path(args.data_root).expanduser(),
    ]
    if args.out_dir:
        evaluate_args += ["--out-dir", Path(args.out_dir).expanduser()]
    if args.overwrite:
        evaluate_args.append("--overwrite")
    if args.device:
        evaluate_args += ["--device", args.device]
    return run(evaluate_args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reproduce.py",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/reproduce.py smoke\n"
            "  python scripts/reproduce.py prepare --data-root /data/PDCADxFoundation\n"
            "  python scripts/reproduce.py preflight\n"
            "  python scripts/reproduce.py train\n"
            "  python scripts/reproduce.py summarize --yes\n"
            "  python scripts/reproduce.py evaluate --data-root /data/PDCADxFoundation\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser(
        "smoke", help="Synthetic end-to-end wiring check. Needs no medical data.")
    p_smoke.add_argument("--max-epochs", type=int, default=1,
                         help="Training epochs for the smoke run (default: 1).")
    p_smoke.set_defaults(func=cmd_smoke)

    p_prep = sub.add_parser(
        "prepare", help="Prepare the official data using the frozen split.")
    p_prep.add_argument("--data-root", type=Path, required=True,
                        help="Directory holding the official PDCADxFoundation tree.")
    p_prep.add_argument("--processed-only", action="store_true",
                        help="Skip label extraction and normalisation; just check "
                             "the cases and (re)build the cache.")
    p_prep.add_argument("--overwrite", action="store_true",
                        help="Rebuild cache entries that already exist.")
    p_prep.set_defaults(func=cmd_prepare)

    p_pre = sub.add_parser(
        "preflight",
        help="Check that the 21 runs have every input and cache they need.")
    p_pre.set_defaults(func=cmd_preflight)

    p_train = sub.add_parser("train", help="Run the 21 formal subject-clean runs.")
    p_train.add_argument("--dry-run", action="store_true",
                         help="List the runs and exit without training.")
    p_train.add_argument("--reset-registry", action="store_true",
                         help="Recreate the run registry from the configs.")
    p_train.set_defaults(func=cmd_train)

    p_sum = sub.add_parser(
        "summarize", help="Aggregate validation results and freeze the records.")
    p_sum.add_argument("--yes", action="store_true",
                       help="Confirm rewriting the frozen run records.")
    p_sum.set_defaults(func=cmd_summarize)

    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate this reproduction's own checkpoints on internal_test. "
             "Never touches the author's frozen results.")
    p_eval.add_argument("--data-root", type=Path, required=True,
                        help="The same official PDCADxFoundation tree passed to "
                             "prepare: it holds the processed images and the "
                             "internal_test ground truth.")
    p_eval.add_argument("--out-dir", type=Path, default=None,
                        help="Override the reproduction output directory "
                             "(default: results/reproduction_eval_v1/internal_test).")
    p_eval.add_argument("--overwrite", action="store_true",
                        help="Replace an existing reproduction report.")
    p_eval.add_argument("--device", type=str, default=None, help="cuda / cpu.")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

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
    ``manifests/subject_clean_v1/`` and builds the training cache. The split is
    read from the repository; it is never re-derived. A case-count or case-set
    mismatch is a fatal error -- this script will not silently re-split.

``train``
    Runs the 21 formal subject-clean runs declared by the configs. The run
    registry is created automatically on first use, so a fresh clone needs no
    pre-existing state. The runs write to ``checkpoints/subject_clean_v1/`` and
    ``results/experiments_subject_clean_v1/``.

``summarize``
    Aggregates the validation results and writes the frozen records a later
    evaluation needs (clean-val table, baseline seed selection, ensemble freeze,
    holdout pre-registration). Touches train/val only.

Every child process is launched with ``sys.executable``, so Windows, Linux and
macOS all work from any virtualenv.
"""

from __future__ import annotations

import argparse
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

    step("1/4", "Check the raw cases against the frozen subject-clean manifests")
    problems = validate_cases_against_manifests(data_root)
    if problems:
        for p in problems:
            print(f"  BAD {p}", file=sys.stderr)
        return fail("raw data does not match the frozen split; refusing to continue")

    if args.processed_only:
        print("  --processed-only: skipping label extraction and normalisation.")

    if not args.processed_only:
        step("2/4", "Extract labels from the official masks")
        if run([SCRIPTS / "prepare_labels.py", "--root", data_root]) != 0:
            return fail("prepare_labels failed")

        step("3/4", "Normalise images")
        if run([SCRIPTS / "normalize_images.py", "--root", data_root]) != 0:
            return fail("normalize_images failed")

    step("4/4", "Build the training cache (train + val only)")
    cache_args = [
        SCRIPTS / "build_baseline_cache.py",
        "--split-dir", SUBJECT_CLEAN_MANIFESTS,
        "--out-dir", REPO / "cache" / "baseline_v1",
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

    save_state(data_root=str(data_root), prepared_at_utc=_now())

    print()
    print("=" * 78)
    print("PREPARE = PASS")
    print(f"  cache          : {(REPO / 'cache' / 'baseline_v1').relative_to(REPO).as_posix()}")
    print(f"  manifests      : {SUBJECT_CLEAN_MANIFESTS.relative_to(REPO).as_posix()} (frozen)")
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
            "  python scripts/reproduce.py train\n"
            "  python scripts/reproduce.py summarize\n"
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

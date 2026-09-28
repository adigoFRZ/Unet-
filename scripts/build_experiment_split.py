#!/usr/bin/env python
"""Build the formal experiment split under ``manifests/experiment/``.

The official manifests are never modified. This script derives a separate,
frozen split for the experiment:

    official train (200, with GT)
        -> stratified by the `group` column into
           development train (160)  : the ONLY set allowed to design the crop
           development val   (40)   : evaluation of candidate crops only
    official val   (100, with GT) -> internal_test  : final blind-style check
    official test  (200, no GT)   -> challenge_test : untouched

Stratification uses the `group` column purely as an opaque category. Its medical
meaning is unknown and is deliberately not interpreted here.

Once written this split must not be re-randomised based on later model results.

Usage
-----
    python scripts/build_experiment_split.py --root .
    python scripts/build_experiment_split.py --root . --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import pdcadx_io as pio  # noqa: E402

LOGGER = logging.getLogger("build_experiment_split")

FIELDS: tuple[str, ...] = (
    "case_id", "split", "group", "T1_path", "QSM_path", "NM_path", "label_path",
)

#: fraction of the official train split reserved for development validation
DEV_VAL_FRACTION = 0.20


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the frozen experiment split from the official split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: <root>/manifests/experiment).")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Report dir (default: <root>/results/spatial_roi_design).")
    parser.add_argument("--seed", type=int, default=42, help="Stratification seed.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    LOGGER.addHandler(console)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    LOGGER.addHandler(handler)


def stratified_dev_split(
    groups: dict[str, str], seed: int, val_fraction: float
) -> tuple[list[str], list[str]]:
    """Split case ids into (develop, val) keeping the group ratio.

    Each group's id list is sorted before shuffling so the result depends only on
    the seed and the id set, never on filesystem enumeration order. Group sizes
    are handled exactly rather than approximately: the per-group validation count
    is rounded, and any shortfall/surplus is reconciled across groups so the total
    matches the requested fraction.
    """
    by_group: dict[str, list[str]] = {}
    for case_id, group in groups.items():
        by_group.setdefault(group, []).append(case_id)
    for ids in by_group.values():
        ids.sort()

    total = sum(len(v) for v in by_group.values())
    target_val = int(round(total * val_fraction))

    # Round each group independently, then correct the total on the largest group.
    val_counts = {g: int(round(len(ids) * val_fraction)) for g, ids in by_group.items()}
    diff = target_val - sum(val_counts.values())
    if diff != 0:
        biggest = max(by_group, key=lambda g: len(by_group[g]))
        val_counts[biggest] += diff

    rng = random.Random(seed)
    develop: list[str] = []
    validate: list[str] = []
    for group in sorted(by_group):
        ids = list(by_group[group])
        rng.shuffle(ids)
        n_val = max(0, min(val_counts[group], len(ids)))
        validate.extend(ids[:n_val])
        develop.extend(ids[n_val:])

    develop.sort()
    validate.sort()
    return develop, validate


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    out_dir = (args.out_dir or (root / "manifests" / "experiment")).resolve()
    results_dir = (args.results_dir or (root / "results" / "spatial_roi_design")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir / "build_experiment_split.log", args.verbose)

    # ---- official sources ------------------------------------------------- #
    official_train = pio.load_manifest_ids(root / "train_cases.csv")
    official_val = pio.load_manifest_ids(root / "val_cases.csv")
    if not official_train or not official_val:
        LOGGER.error("Could not read train_cases.csv / val_cases.csv")
        return 1

    cases = {c.case_id: c for c in pio.discover_cases(root)}
    official_test = {cid: "" for cid, c in cases.items() if c.split == "test"}

    LOGGER.info("Official: train=%d val=%d test=%d",
                len(official_train), len(official_val), len(official_test))

    # ---- derive the experiment split -------------------------------------- #
    develop_ids, devval_ids = stratified_dev_split(
        official_train, args.seed, DEV_VAL_FRACTION
    )
    LOGGER.info("Development: train=%d val=%d (seed=%d)",
                len(develop_ids), len(devval_ids), args.seed)

    split_of: dict[str, str] = {}
    for ids, name in ((develop_ids, "train"), (devval_ids, "val"),
                      (list(official_val), "internal_test"),
                      (list(official_test), "challenge_test")):
        for case_id in ids:
            split_of[case_id] = name

    group_of: dict[str, str] = {}
    group_of.update(official_train)
    group_of.update(official_val)

    # ---- leak check: no case may appear in two splits ---------------------- #
    # Checked against the raw id LISTS, not the dict: the dict cannot express a
    # collision, so only the lists can actually prove the splits are disjoint.
    all_ids = develop_ids + devval_ids + list(official_val) + list(official_test)
    counter = Counter(all_ids)
    collisions = sorted(cid for cid, n in counter.items() if n > 1)

    missing_dirs = sorted(set(split_of) - set(cases))
    if missing_dirs:
        LOGGER.warning("%d case ids have no directory on disk: %s",
                       len(missing_dirs), missing_dirs[:10])
    if collisions:
        LOGGER.error("Cases appearing in more than one split: %s", collisions[:10])
        return 1

    # ---- write the four manifests ----------------------------------------- #
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for case_id, split in tqdm(split_of.items(), desc="Writing manifests", unit="case"):
        record = cases.get(case_id)
        row: dict[str, Any] = {
            "case_id": case_id,
            "split": split,
            "group": group_of.get(case_id, ""),
        }
        for modality in pio.IMAGE_MODALITIES:
            processed = root / "processed" / "images" / f"{case_id}_{modality}.nii.gz"
            row[f"{modality}_path"] = processed.relative_to(root).as_posix()
        label = root / "processed" / "labels" / f"{case_id}_label.nii.gz"
        row["label_path"] = (
            label.relative_to(root).as_posix()
            if record is not None and "QSM_mask" in record.files else ""
        )
        rows_by_split.setdefault(split, []).append(row)

    for split in ("train", "val", "internal_test", "challenge_test"):
        subset = pd.DataFrame(rows_by_split.get(split, []), columns=list(FIELDS))
        subset = subset.sort_values("case_id").reset_index(drop=True)
        path = out_dir / f"{split}.csv"
        subset.to_csv(path, index=False, encoding="utf-8-sig")
        LOGGER.info("Wrote %s (%d cases)", path.name, len(subset))

    # ---- summary ----------------------------------------------------------- #
    # The stratification check compares the develop/val group ratio against the
    # official train pool it was drawn from.
    def group_counts(ids: Sequence[str]) -> dict[str, int]:
        return {str(k): int(v) for k, v in sorted(Counter(group_of.get(i, "") for i in ids).items())}

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "dev_val_fraction": DEV_VAL_FRACTION,
        "strategy": (
            "Official train stratified by the `group` column into development "
            "train/val. The medical meaning of `group` is unknown and is used "
            "only as an opaque stratification category."
        ),
        "provenance": {
            "train": "official train (train_cases.csv)",
            "val": "official train (train_cases.csv)",
            "internal_test": "official val (val_cases.csv)",
            "challenge_test": "official test (directory tree, no GT)",
        },
        "splits": {
            "train": {"n": len(develop_ids), "group_counts": group_counts(develop_ids),
                      "has_labels": True, "role": "crop design + training"},
            "val": {"n": len(devval_ids), "group_counts": group_counts(devval_ids),
                    "has_labels": True, "role": "candidate crop evaluation only"},
            "internal_test": {"n": len(official_val),
                              "group_counts": group_counts(list(official_val)),
                              "has_labels": True,
                              "role": "final blind-style coverage check; must not "
                                      "influence crop design"},
            "challenge_test": {"n": len(official_test), "group_counts": {},
                               "has_labels": False, "role": "held out; untouched"},
        },
        "official_counts": {"train": len(official_train), "val": len(official_val),
                            "test": len(official_test)},
        "integrity": {
            "total_cases": len(split_of),
            "n_duplicate_case_ids": len(collisions),
            "duplicate_case_ids": collisions[:50],
            "n_case_ids_without_directory": len(missing_dirs),
            "case_ids_without_directory": missing_dirs[:50],
            "sum_of_splits": len(develop_ids) + len(devval_ids)
                             + len(official_val) + len(official_test),
        },
        "leakage_policy": [
            "Fixed crop parameters are derived from development train ONLY.",
            "development val and internal_test may evaluate a candidate crop but "
            "must never influence its coordinates.",
            "Per-case GT must never determine that case's own crop.",
            "This split is frozen; it must not be re-randomised based on later "
            "model results.",
        ],
    }
    with (results_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    line = "=" * 66
    print()
    print(line)
    print("Experiment Split Summary")
    print(line)
    print(f"seed={args.seed}  dev_val_fraction={DEV_VAL_FRACTION}")
    print(f"{'split':<16}{'n':>5}  {'labels':>7}  group counts")
    for split, info in summary["splits"].items():
        print(f"{split:<16}{info['n']:>5}  {str(info['has_labels']):>7}  {info['group_counts']}")
    print(f"\ntotal distinct cases: {len(split_of)}  duplicates: {len(collisions)}")
    print(f"written to {out_dir}")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

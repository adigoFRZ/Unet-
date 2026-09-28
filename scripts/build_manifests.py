#!/usr/bin/env python
"""Build the train / val / test manifests for the processed dataset.

Writes ``manifests/train.csv``, ``manifests/val.csv`` and ``manifests/test.csv``
with the columns:

    case_id, split, T1_path, QSM_path, NM_path, label_path

Paths point at the **processed** files (``processed/images``, ``processed/labels``)
and are relative to the project root with forward slashes, so a manifest stays
valid if the project is moved or opened on another OS.

``label_path`` is empty for test cases, which have no public masks. Each row also
carries ``*_present`` flags recording whether the referenced file actually exists
at build time -- a manifest that silently points at missing files is worse than
one that says so.

Usage
-----
    python scripts/build_manifests.py --root .
    python scripts/build_manifests.py --root . --split-by csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import pdcadx_io as pio  # noqa: E402

LOGGER = logging.getLogger("build_manifests")

FIELDS: tuple[str, ...] = (
    "case_id", "split", "T1_path", "QSM_path", "NM_path", "label_path",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test manifests for the processed dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Processed image directory (default: <root>/processed/images).")
    parser.add_argument("--label-dir", type=Path, default=None,
                        help="Processed label directory (default: <root>/processed/labels).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Manifest directory (default: <root>/manifests).")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Run report directory (default: <root>/results/preprocessing).")
    parser.add_argument("--split-by", choices=("path", "csv"), default="path",
                        help="How the split is determined: from the raw directory "
                             "path, or from train_cases.csv / val_cases.csv.")
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


def relative(path: Path, root: Path) -> str:
    """Project-root-relative POSIX path, or the absolute path if not beneath root."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_split(
    record: pio.CaseRecord, split_by: str, csv_splits: dict[str, str]
) -> str:
    if split_by == "csv":
        # Test cases have no manifest entry, so fall back to the path.
        return csv_splits.get(record.case_id, record.split)
    return record.split


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: --root {root} is not a directory", file=sys.stderr)
        return 2

    image_dir = (args.image_dir or (root / "processed" / "images")).resolve()
    label_dir = (args.label_dir or (root / "processed" / "labels")).resolve()
    out_dir = (args.out_dir or (root / "manifests")).resolve()
    results_dir = (args.results_dir or (root / "results" / "preprocessing")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir / "build_manifests.log", args.verbose)

    csv_splits: dict[str, str] = {}
    for split, path in (("train", root / "train_cases.csv"), ("val", root / "val_cases.csv")):
        for case_id in pio.load_manifest_ids(path):
            csv_splits.setdefault(case_id, split)
    if csv_splits:
        LOGGER.info("Loaded %d case ids from the raw manifests", len(csv_splits))

    cases = pio.discover_cases(root)
    LOGGER.info("Discovered %d cases", len(cases))
    if not cases:
        LOGGER.error("No cases found under %s", root)
        return 1

    rows: list[dict[str, Any]] = []
    for record in tqdm(cases, desc="Building manifests", unit="case"):
        split = resolve_split(record, args.split_by, csv_splits)
        row: dict[str, Any] = {"case_id": record.case_id, "split": split}

        for modality in pio.IMAGE_MODALITIES:
            processed = image_dir / f"{record.case_id}_{modality}.nii.gz"
            row[f"{modality}_path"] = relative(processed, root)
            row[f"{modality}_present"] = processed.is_file()

        # Test cases have no public masks, so no label path is emitted for them.
        label_path = label_dir / f"{record.case_id}_label.nii.gz"
        has_source_mask = "QSM_mask" in record.files
        if has_source_mask:
            row["label_path"] = relative(label_path, root)
            row["label_present"] = label_path.is_file()
        else:
            row["label_path"] = ""
            row["label_present"] = False

        rows.append(row)

    frame = pd.DataFrame(rows)
    columns = list(FIELDS) + [c for c in frame.columns if c not in FIELDS]
    frame = frame[columns].sort_values(["split", "case_id"]).reset_index(drop=True)

    written: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        subset = frame[frame["split"] == split]
        # Only the six contract fields are written; the *_present flags are kept
        # in the summary rather than polluting the manifest schema.
        out_path = out_dir / f"{split}.csv"
        subset[list(FIELDS)].to_csv(out_path, index=False, encoding="utf-8-sig")
        written[split] = {
            "path": str(out_path),
            "n_cases": int(len(subset)),
            "n_with_labels": int(subset["label_present"].sum()) if len(subset) else 0,
            "n_images_missing": {
                m: int((~subset[f"{m}_present"]).sum()) for m in pio.IMAGE_MODALITIES
            } if len(subset) else {},
        }
        LOGGER.info("Wrote %s (%d cases)", out_path, len(subset))

    unknown = frame[~frame["split"].isin(["train", "val", "test"])]
    if not unknown.empty:
        LOGGER.warning("%d cases have an unrecognised split: %s",
                       len(unknown), sorted(unknown["case_id"])[:10])

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "split_by": args.split_by,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "total_cases": int(len(frame)),
        "split_counts": {str(k): int(v) for k, v in sorted(Counter(frame["split"]).items())},
        "splits": written,
        "n_cases_without_processed_images": int(
            (~frame[[f"{m}_present" for m in pio.IMAGE_MODALITIES]].any(axis=1)).sum()
        ),
    }
    with (results_dir / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    line = "=" * 62
    print()
    print(line)
    print("Manifest Summary")
    print(line)
    print(f"Split source : {args.split_by}")
    print(f"Total cases  : {summary['total_cases']}")
    for split, info in written.items():
        print(f"  {split:<6} n={info['n_cases']:<4} with_labels={info['n_with_labels']:<4} "
              f"missing_images={info['n_images_missing']}")
    if summary["n_cases_without_processed_images"]:
        print(f"\nNOTE: {summary['n_cases_without_processed_images']} cases have no "
              f"processed images yet -- run normalize_images.py to create them.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

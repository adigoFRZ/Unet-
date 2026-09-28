#!/usr/bin/env python
"""Generate project label maps from the raw QSM_mask volumes.

Mapping applied (all other raw labels become background):

    raw 0                  -> 0  background
    raw 9, 10              -> 1  STN
    raw 11, 12             -> 2  SN
    raw 13, 14             -> 3  RN
    everything else        -> 0

Raw labels 1-8 (Caudate / Putamen / Globus Pallidus / Thalamus) and 15-16
(Dentate) are deliberately discarded: they are outside the scope of this project.
They are recorded in the raw-label histogram so nothing is lost silently.

The raw dataset is READ-ONLY here. Output goes to ``processed/labels/`` with the
same shape, spacing, affine and orientation as the source mask.

Every written file is re-read and validated (unique values, per-class voxel
counts) before its row is accepted into the statistics table.

Usage
-----
    python scripts/prepare_labels.py --root .
    python scripts/prepare_labels.py --root . --overwrite
    python scripts/prepare_labels.py --root . --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import pdcadx_io as pio  # noqa: E402

LOGGER = logging.getLogger("prepare_labels")

VALID_CLASSES = {pio.BACKGROUND_CLASS, *pio.FOREGROUND_CLASSES}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 0/1/2/3 project label maps from raw QSM_mask volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Project root containing the raw data directories.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Label output directory (default: <root>/processed/labels).")
    parser.add_argument("--meta-dir", type=Path, default=None,
                        help="Metadata output directory (default: <root>/processed/metadata).")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Run report directory (default: <root>/results/preprocessing).")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma separated case ids to restrict processing to.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N cases (smoke testing).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute and overwrite existing label files.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the post-write re-read validation (not recommended).")
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


def verify_written_label(path: Path, expected_counts: dict[int, int]) -> dict[str, Any]:
    """Re-read a written label file and confirm its content and geometry.

    This is the check that catches a whole class of silent pipeline bugs -- a
    stale cache, a wrong dtype, a lost affine -- at the moment they are written
    rather than months later during training.
    """
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)

    unique = sorted(int(v) for v in np.unique(data))
    unexpected = [v for v in unique if v not in VALID_CLASSES]

    counts = {c: int(np.count_nonzero(data == c)) for c in pio.FOREGROUND_CLASSES}
    mismatch = {c: (counts[c], expected_counts.get(c, 0))
                for c in pio.FOREGROUND_CLASSES if counts[c] != expected_counts.get(c, 0)}

    return {
        "unique_values": unique,
        "unexpected_values": unexpected,
        "counts": counts,
        "count_mismatch": mismatch,
        "shape": tuple(int(s) for s in image.shape),
        "dtype": str(image.get_data_dtype()),
        "spacing": tuple(float(z) for z in image.header.get_zooms()[:3]),
        "axcodes": tuple(nib.aff2axcodes(image.affine)),
        "affine": np.asarray(image.affine, dtype=np.float64),
    }


def process_case(
    record: pio.CaseRecord,
    out_dir: Path,
    overwrite: bool,
    skip_verify: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Remap and write one case's labels. Returns (stats_row, warnings)."""
    warnings: list[dict[str, str]] = []
    case_id = record.case_id
    out_path = out_dir / f"{case_id}_label.nii.gz"

    source = record.files["QSM_mask"]
    raw, source_image = pio.load_array(source)

    label_map, raw_histogram = pio.remap_labels(raw)
    del raw

    counts = pio.class_counts(label_map)

    # ---- warnings on implausible / missing classes ------------------------ #
    for class_id in pio.FOREGROUND_CLASSES:
        name = pio.CLASS_NAMES[class_id]
        n = counts[class_id]
        if n == 0:
            warnings.append({
                "case_id": case_id, "split": record.split, "warning_type": "empty_class",
                "description": f"{name} (class {class_id}) has 0 voxels after remapping",
            })
        else:
            low, high = pio.CLASS_VOXEL_SANITY[class_id]
            if not (low <= n <= high):
                warnings.append({
                    "case_id": case_id, "split": record.split,
                    "warning_type": "implausible_class_size",
                    "description": f"{name} (class {class_id}) has {n} voxels, "
                                   f"outside the expected band [{low}, {high}]",
                })

    total_foreground = int(sum(counts.values()))
    if total_foreground == 0:
        warnings.append({
            "case_id": case_id, "split": record.split, "warning_type": "empty_label_map",
            "description": "no STN/SN/RN voxels remain after remapping",
        })

    # ---- write ------------------------------------------------------------ #
    if out_path.exists() and not overwrite:
        LOGGER.debug("%s exists; reusing (use --overwrite to recompute)", out_path.name)
    else:
        pio.save_like(
            source_image, label_map, out_path,
            dtype=np.uint8, description="PDCADx project labels 1=STN 2=SN 3=RN",
        )

    # ---- verify ----------------------------------------------------------- #
    verification: dict[str, Any] = {}
    if not skip_verify:
        verification = verify_written_label(out_path, counts)
        if verification["unexpected_values"]:
            raise ValueError(
                f"written label map contains unexpected values "
                f"{verification['unexpected_values']}"
            )
        if verification["count_mismatch"]:
            raise ValueError(
                f"written label counts differ from computed counts: "
                f"{verification['count_mismatch']}"
            )

        # Spatial metadata must survive the round trip unchanged.
        source_sig = pio.spatial_signature(source_image)
        if verification["shape"] != source_sig["shape"]:
            raise ValueError(
                f"shape changed: {verification['shape']} != {source_sig['shape']}"
            )
        if verification["axcodes"] != source_sig["axcodes"]:
            raise ValueError(
                f"orientation changed: {verification['axcodes']} != {source_sig['axcodes']}"
            )
        if not np.allclose(verification["affine"], source_sig["affine"], atol=0.0):
            raise ValueError("affine changed during label writing")
        if not np.allclose(verification["spacing"], source_sig["spacing"], atol=1e-6):
            raise ValueError(
                f"spacing changed: {verification['spacing']} != {source_sig['spacing']}"
            )

    voxel_volume_mm3 = float(np.prod(pio.spatial_signature(source_image)["spacing"]))
    row: dict[str, Any] = {
        "case_id": case_id,
        "split": record.split,
        "stn_voxels": counts[1],
        "sn_voxels": counts[2],
        "rn_voxels": counts[3],
        "total_foreground_voxels": total_foreground,
        "stn_volume_mm3": counts[1] * voxel_volume_mm3,
        "sn_volume_mm3": counts[2] * voxel_volume_mm3,
        "rn_volume_mm3": counts[3] * voxel_volume_mm3,
        "total_foreground_volume_mm3": total_foreground * voxel_volume_mm3,
        "foreground_fraction": total_foreground / float(label_map.size),
        "raw_labels_seen": "|".join(str(v) for v in sorted(raw_histogram)),
        "discarded_voxels": int(sum(
            c for v, c in raw_histogram.items() if v in pio.DISCARDED_QSM_LABELS
        )),
        "label_path": str(out_path),
    }
    if verification:
        row["verified_unique_values"] = "|".join(str(v) for v in verification["unique_values"])
        row["verified_dtype"] = verification["dtype"]

    return row, warnings


def build_summary(
    frame: pd.DataFrame, warnings: pd.DataFrame, args: argparse.Namespace
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(args.root.resolve()),
        "mapping": {
            "raw_to_class": {str(k): v for k, v in sorted(pio.QSM_LABEL_TO_CLASS.items())},
            "class_names": {str(k): v for k, v in sorted(pio.CLASS_NAMES.items())},
            "discarded_raw_labels": list(pio.DISCARDED_QSM_LABELS),
        },
        "n_cases": int(len(frame)),
        "split_counts": {str(k): int(v) for k, v in sorted(Counter(frame["split"]).items())},
        "n_warning_cases": int(warnings["case_id"].nunique()) if not warnings.empty else 0,
    }

    per_class: dict[str, Any] = {}
    for class_id in pio.FOREGROUND_CLASSES:
        name = pio.CLASS_NAMES[class_id]
        column = f"{name.lower()}_voxels"
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        per_class[name] = {
            "class_id": class_id,
            "n_cases": int(values.size),
            "n_cases_empty": int((values == 0).sum()),
            "min": int(values.min()),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "max": int(values.max()),
            "std": float(values.std()),
            "total_voxels": int(values.sum()),
        }
    summary["per_class"] = per_class

    total = pd.to_numeric(frame["total_foreground_voxels"], errors="coerce").dropna()
    if not total.empty:
        summary["total_foreground"] = {
            "min": int(total.min()), "median": float(total.median()),
            "mean": float(total.mean()), "max": int(total.max()),
        }

    summary["warnings_by_type"] = (
        {str(k): int(v) for k, v in sorted(Counter(warnings["warning_type"]).items())}
        if not warnings.empty else {}
    )
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    line = "=" * 62
    print()
    print(line)
    print("Label Generation Summary")
    print(line)
    print(f"Cases processed : {summary['n_cases']}")
    print(f"Splits          : {summary['split_counts']}")
    print(f"Cases with warnings: {summary['n_warning_cases']}")
    print()
    print(f"{'class':<6}{'n_cases':>9}{'empty':>7}{'min':>8}{'median':>10}{'mean':>10}{'max':>8}")
    for name, stats in summary["per_class"].items():
        print(f"{name:<6}{stats['n_cases']:>9}{stats['n_cases_empty']:>7}"
              f"{stats['min']:>8}{stats['median']:>10.0f}{stats['mean']:>10.0f}{stats['max']:>8}")
    if "total_foreground" in summary:
        t = summary["total_foreground"]
        print(f"\nTotal foreground voxels: min={t['min']} median={t['median']:.0f} "
              f"mean={t['mean']:.0f} max={t['max']}")
    if summary["warnings_by_type"]:
        print(f"\nWarnings: {summary['warnings_by_type']}")
    else:
        print("\nWarnings: none")
    print(line)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: --root {root} is not a directory", file=sys.stderr)
        return 2

    out_dir = (args.out_dir or (root / "processed" / "labels")).resolve()
    meta_dir = (args.meta_dir or (root / "processed" / "metadata")).resolve()
    results_dir = (args.results_dir or (root / "results" / "preprocessing")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir / "prepare_labels.log", args.verbose)

    cases = pio.discover_cases(root)
    with_mask = [c for c in cases if "QSM_mask" in c.files]
    LOGGER.info("Discovered %d cases, %d with QSM_mask", len(cases), len(with_mask))

    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        with_mask = [c for c in with_mask if c.case_id in wanted]
    if args.limit is not None:
        with_mask = with_mask[: args.limit]
    if not with_mask:
        LOGGER.error("No cases with QSM_mask to process.")
        return 1

    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []

    for record in tqdm(with_mask, desc="Preparing labels", unit="case"):
        try:
            row, case_warnings = process_case(
                record, out_dir, args.overwrite, args.skip_verify
            )
            rows.append(row)
            warnings.extend(case_warnings)
            for warning in case_warnings:
                LOGGER.warning("%s: %s", record.label, warning["description"])
        except Exception as exc:  # noqa: BLE001 - one bad case must not abort the run
            LOGGER.error("Case %s failed: %s", record.label, exc)
            LOGGER.debug("%s", traceback.format_exc())
            warnings.append({
                "case_id": record.case_id, "split": record.split,
                "warning_type": "processing_failed",
                "description": f"{type(exc).__name__}: {exc}",
            })

    if not rows:
        LOGGER.error("No cases were processed successfully.")
        return 1

    frame = pd.DataFrame(rows).sort_values(["split", "case_id"]).reset_index(drop=True)
    warning_frame = pd.DataFrame(
        warnings,
        columns=["case_id", "split", "warning_type", "description"],
    )
    if not warning_frame.empty:
        warning_frame = warning_frame.sort_values(
            ["warning_type", "split", "case_id"]
        ).reset_index(drop=True)

    stats_path = meta_dir / "label_statistics.csv"
    frame.to_csv(stats_path, index=False, encoding="utf-8-sig")
    warning_frame.to_csv(meta_dir / "label_warnings.csv", index=False, encoding="utf-8-sig")

    summary = build_summary(frame, warning_frame, args)
    with (results_dir / "label_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    LOGGER.info("Labels written to %s", out_dir)
    LOGGER.info("Statistics written to %s", stats_path)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

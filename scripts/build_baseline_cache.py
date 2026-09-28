#!/usr/bin/env python
"""Build the Baseline v1 training cache from the canonical processed dataset.

Why a cache exists: each training sample needs three gzip-compressed NIfTI
volumes decompressed and cropped. Doing that every epoch is pure repeated work.
The cache stores the already-cropped, already-transposed training tensors as
``.npy`` so the DataLoader only has to read contiguous arrays.

``processed/`` remains the canonical dataset. ``cache/`` is a derived artefact and
can be deleted and rebuilt at any time.

What is written, per case (development train + development val ONLY):

    cache/baseline_v1/images/<case_id>.npy   float32  (3, 32, 96, 96)  C,D,H,W
    cache/baseline_v1/labels/<case_id>.npy   uint8    (32, 96, 96)      D,H,W

``internal_test`` and ``challenge_test`` are deliberately NOT cached: they must
not be touched at this stage.

Every written file is re-read and validated before it is accepted.

Usage
-----
    python scripts/build_baseline_cache.py --root .
    python scripts/build_baseline_cache.py --root . --overwrite
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

from data import crop_spec as cs  # noqa: E402
from data import pdcadx_io as pio  # noqa: E402

LOGGER = logging.getLogger("build_baseline_cache")

#: Only these experiment splits may ever be cached at this stage.
CACHEABLE_SPLITS: tuple[str, ...] = ("train", "val")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen-crop Baseline v1 training cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Cache dir (default: <root>/cache/baseline_v1).")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Processed images (default: <root>/processed/images).")
    parser.add_argument("--label-dir", type=Path, default=None,
                        help="Processed labels (default: <root>/processed/labels).")
    parser.add_argument("--split-dir", type=Path, default=None,
                        help="Experiment split dir (default: <root>/manifests/experiment).")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Report dir (default: <root>/results/baseline_v1).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild files that already exist.")
    parser.add_argument("--limit", type=int, default=None, help="Smoke testing only.")
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


def build_case(
    case_id: str,
    image_paths: dict[str, Path],
    label_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop and transpose one case into tensor layout."""
    channels = []
    for modality in cs.CHANNEL_ORDER:
        path = image_paths.get(modality)
        if path is None:
            raise FileNotFoundError(f"{case_id}: missing {modality} image")
        volume = np.asanyarray(nib.load(str(path)).dataobj)
        if volume.shape[:3] != (300, 300, 70):
            raise ValueError(f"{case_id}/{modality}: unexpected shape {volume.shape}")
        # Crop + (X,Y,Z)->(Z,Y,X). Verified against the label's transform below.
        channels.append(cs.to_tensor_layout(volume).astype(np.float32))
        del volume

    image = np.stack(channels, axis=0)  # (C, D, H, W)
    if image.shape != (cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW):
        raise ValueError(f"{case_id}: image tensor shape {image.shape} is wrong")

    label_xyz = np.asanyarray(nib.load(str(label_path)).dataobj)
    if label_xyz.shape[:3] != (300, 300, 70):
        raise ValueError(f"{case_id}/label: unexpected shape {label_xyz.shape}")
    if np.issubdtype(label_xyz.dtype, np.floating):
        label_xyz = np.rint(label_xyz).astype(np.int64)
    label = cs.to_tensor_layout(label_xyz).astype(np.uint8)
    if label.shape != cs.CROP_SHAPE_DHW:
        raise ValueError(f"{case_id}: label tensor shape {label.shape} is wrong")

    return image, label


def verify_written(
    image_path: Path, label_path: Path
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Re-read the cache files and check shape, dtype, finiteness and labels."""
    image = np.load(image_path, mmap_mode=None)
    label = np.load(label_path, mmap_mode=None)
    problems: list[str] = []

    if image.shape != (cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW):
        problems.append(f"image shape {image.shape}")
    if label.shape != cs.CROP_SHAPE_DHW:
        problems.append(f"label shape {label.shape}")
    if image.dtype != np.float32:
        problems.append(f"image dtype {image.dtype}")
    if not np.issubdtype(label.dtype, np.integer):
        problems.append(f"label dtype {label.dtype}")
    if not np.isfinite(image).all():
        problems.append("image contains NaN/Inf")

    unique = sorted(int(v) for v in np.unique(label))
    if not set(unique).issubset({0, 1, 2, 3}):
        problems.append(f"label values {unique}")

    counts = {int(c): int(np.count_nonzero(label == c)) for c in cs.FOREGROUND_CLASSES}
    empty = [cs.CLASS_NAMES[c] for c, n in counts.items() if n == 0]
    if empty:
        problems.append(f"empty classes in crop: {empty}")

    info = {
        "image_shape": list(image.shape),
        "label_shape": list(label.shape),
        "image_dtype": str(image.dtype),
        "label_dtype": str(label.dtype),
        "label_unique": unique,
        "class_voxels": counts,
        "image_finite": bool(np.isfinite(image).all()),
        "problems": problems,
    }
    return info, image, label


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    out_dir = (args.out_dir or (root / "cache" / "baseline_v1")).resolve()
    image_dir = (args.image_dir or (root / "processed" / "images")).resolve()
    label_dir = (args.label_dir or (root / "processed" / "labels")).resolve()
    split_dir = (args.split_dir or (root / "manifests" / "experiment")).resolve()
    results_dir = (args.results_dir or (root / "results" / "baseline_v1")).resolve()
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir / "build_baseline_cache.log", args.verbose)

    LOGGER.info("Frozen crop: X%s Y%s Z%s -> XYZ %s -> DHW %s",
                cs.CROP_X, cs.CROP_Y, cs.CROP_Z,
                cs.CROP_SHAPE_XYZ, cs.CROP_SHAPE_DHW)

    # ---- which cases ------------------------------------------------------- #
    records: list[tuple[str, str]] = []
    for split in CACHEABLE_SPLITS:
        path = split_dir / f"{split}.csv"
        if not path.is_file():
            LOGGER.error("Missing %s; run build_experiment_split.py first.", path)
            return 2
        for case_id in pd.read_csv(path)["case_id"]:
            records.append((str(case_id), split))
    if args.limit is not None:
        records = records[: args.limit]
    LOGGER.info("Cases to cache: %d (%s)", len(records),
                {s: sum(1 for _, x in records if x == s) for s in CACHEABLE_SPLITS})

    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []

    for case_id, split in tqdm(records, desc="Building cache", unit="case"):
        image_out = out_dir / "images" / f"{case_id}.npy"
        label_out = out_dir / "labels" / f"{case_id}.npy"
        try:
            if not image_out.is_file() or not label_out.is_file() or args.overwrite:
                image_paths = {
                    m: image_dir / f"{case_id}_{m}.nii.gz" for m in cs.CHANNEL_ORDER
                }
                missing = [m for m, p in image_paths.items() if not p.is_file()]
                if missing:
                    raise FileNotFoundError(f"missing processed images: {missing}")
                label_path = label_dir / f"{case_id}_label.nii.gz"
                if not label_path.is_file():
                    raise FileNotFoundError(f"missing label {label_path.name}")

                image, label = build_case(case_id, image_paths, label_path)
                np.save(image_out, image)
                np.save(label_out, label)
                del image, label

            info, image, label = verify_written(image_out, label_out)
            if info["problems"]:
                raise ValueError(f"cache validation failed: {info['problems']}")

            row = {
                "case_id": case_id,
                "experiment_split": split,
                "image_path": str(image_out),
                "label_path": str(label_out),
                "image_shape": "x".join(str(v) for v in info["image_shape"]),
                "label_shape": "x".join(str(v) for v in info["label_shape"]),
                "image_dtype": info["image_dtype"],
                "label_dtype": info["label_dtype"],
                "label_unique": "|".join(str(v) for v in info["label_unique"]),
                **{f"{cs.CLASS_NAMES[c].lower()}_voxels": info["class_voxels"][c]
                   for c in cs.FOREGROUND_CLASSES},
                "foreground_voxels": int(sum(info["class_voxels"].values())),
                "crop_voxels": int(np.prod(cs.CROP_SHAPE_DHW)),
                "image_min": float(image.min()),
                "image_max": float(image.max()),
                "image_mean": float(image.mean()),
            }
            row["foreground_fraction"] = row["foreground_voxels"] / row["crop_voxels"]
            rows.append(row)
            del image, label
        except Exception as exc:  # noqa: BLE001 - one bad case must not abort
            LOGGER.error("Case %s failed: %s", case_id, exc)
            LOGGER.debug("%s", traceback.format_exc())
            warnings.append({"case_id": case_id, "split": split,
                             "warning_type": "cache_failed",
                             "description": f"{type(exc).__name__}: {exc}"})

    if not rows:
        LOGGER.error("No cases were cached successfully.")
        return 1

    frame = pd.DataFrame(rows).sort_values(["experiment_split", "case_id"])
    frame.to_csv(results_dir / "cache_index.csv", index=False, encoding="utf-8-sig")
    if warnings:
        pd.DataFrame(warnings, columns=["case_id", "split", "warning_type", "description"]) \
            .to_csv(results_dir / "cache_warnings.csv", index=False, encoding="utf-8-sig")

    # ---- cross-check against the split manifests --------------------------- #
    expected = {cid: s for cid, s in records}
    produced = set(frame["case_id"])
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crop": {
            "nifti_slices": {"X": list(cs.CROP_X), "Y": list(cs.CROP_Y),
                             "Z": list(cs.CROP_Z)},
            "bbox_inclusive_xyz": [list(cs.CROP_BBOX_INCLUSIVE_XYZ[0]),
                                   list(cs.CROP_BBOX_INCLUSIVE_XYZ[1])],
            "shape_xyz": list(cs.CROP_SHAPE_XYZ),
            "shape_dhw": list(cs.CROP_SHAPE_DHW),
            "channel_order": list(cs.CHANNEL_ORDER),
            "spacing_dhw_mm": list(cs.SPACING_DHW_MM),
        },
        "cache_dir": str(out_dir),
        "n_cases": int(len(frame)),
        "split_counts": {str(k): int(v) for k, v in
                         sorted(Counter(frame["experiment_split"]).items())},
        "n_expected": len(expected),
        "n_missing": len(set(expected) - produced),
        "missing_case_ids": sorted(set(expected) - produced)[:50],
        "n_unexpected": len(produced - set(expected)),
        "n_warning_cases": len({w["case_id"] for w in warnings}),
        "all_shapes_correct": bool(
            (frame["image_shape"] == "x".join(str(v) for v in
                                              (cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW))).all()
            and (frame["label_shape"] == "x".join(str(v) for v in cs.CROP_SHAPE_DHW)).all()
        ),
        "all_labels_in_range": bool(
            frame["label_unique"].apply(
                lambda s: set(int(v) for v in s.split("|")).issubset({0, 1, 2, 3})
            ).all()
        ),
        "all_classes_present_in_every_case": bool(
            (frame[[f"{cs.CLASS_NAMES[c].lower()}_voxels"
                    for c in cs.FOREGROUND_CLASSES]] > 0).all().all()
        ),
        "forbidden_splits_cached": sorted(
            set(w["split"] for w in warnings) - set(CACHEABLE_SPLITS)
        ),
    }
    for name in ("stn", "sn", "rn"):
        col = f"{name}_voxels"
        summary[f"{name}_in_crop"] = {
            "min": int(frame[col].min()), "median": float(frame[col].median()),
            "mean": float(frame[col].mean()), "max": int(frame[col].max()),
        }
    summary["foreground_fraction"] = {
        "min": float(frame["foreground_fraction"].min()),
        "median": float(frame["foreground_fraction"].median()),
        "max": float(frame["foreground_fraction"].max()),
    }

    with (results_dir / "cache_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    line = "=" * 68
    print()
    print(line)
    print("Baseline v1 cache summary")
    print(line)
    print(f"Crop (NIfTI XYZ) : {cs.CROP_SHAPE_XYZ}   slices X{cs.CROP_X} Y{cs.CROP_Y} Z{cs.CROP_Z}")
    print(f"Tensor (D,H,W)   : {cs.CROP_SHAPE_DHW}   image (C,D,H,W) = "
          f"({cs.IN_CHANNELS}, {', '.join(str(v) for v in cs.CROP_SHAPE_DHW)})")
    print(f"Channel order    : {cs.CHANNEL_ORDER}")
    print(f"Spacing (D,H,W)  : {cs.SPACING_DHW_MM} mm")
    print()
    print(f"Cases cached     : {summary['n_cases']}  {summary['split_counts']}")
    print(f"Missing vs split : {summary['n_missing']}")
    print(f"Shapes correct   : {summary['all_shapes_correct']}")
    print(f"Labels in range  : {summary['all_labels_in_range']}")
    print(f"All classes > 0  : {summary['all_classes_present_in_every_case']}")
    for name in ("stn", "sn", "rn"):
        s = summary[f"{name}_in_crop"]
        print(f"  {name.upper():<4} in crop: min={s['min']} median={s['median']:.0f} "
              f"mean={s['mean']:.0f} max={s['max']}")
    print(f"Foreground frac  : {summary['foreground_fraction']}")
    print(f"Warnings         : {summary['n_warning_cases']}")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

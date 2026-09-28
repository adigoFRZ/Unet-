#!/usr/bin/env python
"""Build an image-only inference cache for the SECONDARY FROZEN HOLDOUT.

Why this exists: the frozen baseline cache (`cache/baseline_v1`) contains only
the development train/val tensors by design -- `build_baseline_cache.py` has
`CACHEABLE_SPLITS = ("train", "val")` and never touched the holdout. The frozen
checkpoints were trained on `(3, 32, 96, 96)` crop tensors, so evaluating the
holdout at all requires producing the same tensors for those 100 cases.

This script does that, and nothing else:

  * **images only.** `build_image_tensor()` takes `(case_id, image_paths)` -- there
    is no label parameter, so no code path in this file can open a GT file. The
    holdout's labels are never read here.
  * **the same pipeline as training.** The image half of `build_baseline_cache.py`
    `build_case()` is reproduced verbatim: load `processed/images/<case>_<mod>.nii.gz`,
    `crop_spec.to_tensor_layout` (fixed crop, then the `(X,Y,Z)->(Z,Y,X)`
    transpose), `.astype(np.float32)`, stacked in `CHANNEL_ORDER` = (T1, QSM, NM).
    No new normalisation, no resampling, no interpolation, no per-case scaling.
  * **equivalence is measured, not asserted.** Before writing anything, the same
    pipeline is re-run over the *entire* frozen train/val cache and compared
    bitwise with `np.array_equal`. If a single tensor differs, the script refuses
    to build the holdout cache. Reading the source and claiming equivalence would
    not catch a different crop origin, a flipped axis or a dtype change; a bitwise
    comparison of all 200 frozen tensors does.

The frozen `build_baseline_cache.py` is deliberately NOT modified: unfreezing the
training cache builder to add a split would blur exactly the boundary the freeze
record exists to draw.

Usage
-----
    # verification only -- no holdout cache written
    python scripts/build_holdout_image_cache.py --root . --verify-only

    # verify, then build the holdout image cache
    python scripts/build_holdout_image_cache.py --root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import crop_spec as cs  # noqa: E402
from utils.paths import resolve_path, resolve_project_root  # noqa: E402

LOGGER = logging.getLogger("build_holdout_image_cache")

#: the split this script may read. Anything else is refused outright.
HOLDOUT_SPLIT = "internal_test"
#: the frozen cache whose semantics must be reproduced exactly
FROZEN_CACHE_REL = "cache/baseline_v1"
#: splits present in the frozen cache, used for the equivalence check
FROZEN_SPLITS = ("train", "val")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an image-only holdout inference cache "
                    "(no ground truth is read).")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest-dir", type=Path,
                        default=Path("manifests/experiment"))
    parser.add_argument("--image-dir", type=Path, default=Path("processed/images"))
    parser.add_argument("--frozen-cache", type=Path, default=Path(FROZEN_CACHE_REL))
    parser.add_argument("--out-dir", type=Path, default=Path("cache/holdout_frozen_v1"))
    parser.add_argument("--verify-only", action="store_true",
                        help="Run the equivalence check and stop before building.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# the pipeline -- images only, by construction
# --------------------------------------------------------------------------- #

def build_image_tensor(case_id: str, image_paths: dict[str, Path]) -> np.ndarray:
    """Crop + transpose one case's modalities into ``(3, D, H, W)`` float32.

    Verbatim reproduction of the image half of ``build_baseline_cache.build_case``.
    It has **no label argument and reads no label file** -- the signature is the
    guarantee, not a comment.
    """
    channels = []
    for modality in cs.CHANNEL_ORDER:
        path = image_paths.get(modality)
        if path is None:
            raise FileNotFoundError(f"{case_id}: missing {modality} image")
        volume = np.asanyarray(nib.load(str(path)).dataobj)
        if volume.shape[:3] != (300, 300, 70):
            raise ValueError(f"{case_id}/{modality}: unexpected shape {volume.shape}")
        # Crop + (X,Y,Z)->(Z,Y,X), identical to the frozen builder.
        channels.append(cs.to_tensor_layout(volume).astype(np.float32))
        del volume

    image = np.stack(channels, axis=0)  # (C, D, H, W)
    if image.shape != (cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW):
        raise ValueError(f"{case_id}: image tensor shape {image.shape} is wrong")
    return image


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_image_paths(image_dir: Path, case_id: str) -> dict[str, Path]:
    return {m: image_dir / f"{case_id}_{m}.nii.gz" for m in cs.CHANNEL_ORDER}


# --------------------------------------------------------------------------- #
# equivalence verification on the frozen cache
# --------------------------------------------------------------------------- #

def verify_equivalence(frozen_cache: Path, image_dir: Path,
                       manifest_dir: Path) -> dict[str, Any]:
    """Re-run this pipeline over the whole frozen cache and compare bitwise.

    Compared with ``np.array_equal`` rather than a tolerance: a float tolerance
    would hide exactly the failure this check exists to catch (a different crop
    origin or a resampled volume differs by far more than rounding, but a
    *slightly* wrong interpolation kernel might not).
    """
    report: dict[str, Any] = {"splits": {}, "all_equal": True, "n_compared": 0,
                              "mismatches": []}
    for split in FROZEN_SPLITS:
        manifest = manifest_dir / f"{split}.csv"
        frame = pd.read_csv(manifest)
        case_ids = sorted(str(c) for c in frame["case_id"])
        mismatched: list[str] = []
        checked = 0
        for case_id in case_ids:
            cached_path = frozen_cache / "images" / f"{case_id}.npy"
            if not cached_path.is_file():
                report["all_equal"] = False
                mismatched.append(f"{case_id}: missing from frozen cache")
                continue
            cached = np.load(cached_path)
            rebuilt = build_image_tensor(case_id, case_image_paths(image_dir, case_id))
            checked += 1
            report["n_compared"] += 1
            if rebuilt.dtype != cached.dtype or rebuilt.shape != cached.shape:
                report["all_equal"] = False
                mismatched.append(
                    f"{case_id}: dtype/shape {rebuilt.dtype}/{rebuilt.shape} vs "
                    f"{cached.dtype}/{cached.shape}")
            elif not np.array_equal(rebuilt, cached):
                delta = np.abs(rebuilt.astype(np.float64) - cached.astype(np.float64))
                report["all_equal"] = False
                mismatched.append(
                    f"{case_id}: values differ, max|delta|={delta.max():.3e}")
        report["splits"][split] = {"n_cases": len(case_ids), "n_compared": checked,
                                   "n_mismatched": len(mismatched),
                                   "mismatches": mismatched[:10]}
        report["mismatches"].extend(mismatched[:10])
        LOGGER.info("equivalence %-6s %d cases compared, %d mismatched",
                    split, checked, len(mismatched))
    return report


# --------------------------------------------------------------------------- #
# holdout cache build
# --------------------------------------------------------------------------- #

def build_holdout(root: Path, manifest_dir: Path, image_dir: Path,
                  out_dir: Path, overwrite: bool) -> dict[str, Any]:
    manifest = manifest_dir / f"{HOLDOUT_SPLIT}.csv"
    frame = pd.read_csv(manifest)
    case_ids = sorted(str(c) for c in frame["case_id"])

    if len(case_ids) != 100:
        raise ValueError(f"{HOLDOUT_SPLIT} manifest has {len(case_ids)} cases, expected 100")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"{HOLDOUT_SPLIT} manifest has duplicate case ids")

    # The holdout must not overlap the frozen cache's own splits -- a stray train
    # case here would silently turn a held-out evaluation into a resubstitution.
    for other in ("train", "val", "challenge_test"):
        other_path = manifest_dir / f"{other}.csv"
        if other_path.is_file():
            overlap = set(case_ids) & set(
                str(c) for c in pd.read_csv(other_path)["case_id"])
            if overlap:
                raise ValueError(
                    f"{HOLDOUT_SPLIT} overlaps {other} on {len(overlap)} case(s), "
                    f"e.g. {sorted(overlap)[:5]}")

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        out_path = out_dir / "images" / f"{case_id}.npy"
        if out_path.is_file() and not overwrite:
            image = np.load(out_path)
        else:
            paths = case_image_paths(image_dir, case_id)
            missing = [m for m, p in paths.items() if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"{case_id}: missing processed images {missing}")
            image = build_image_tensor(case_id, paths)
            np.save(out_path, image)

        if image.shape != (cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW):
            raise ValueError(f"{case_id}: cached shape {image.shape} is wrong")
        if image.dtype != np.float32:
            raise ValueError(f"{case_id}: cached dtype {image.dtype} is not float32")
        if not np.isfinite(image).all():
            raise ValueError(f"{case_id}: cache contains NaN/Inf")

        rows.append({
            "case_id": case_id,
            "image_tensor_path": str(out_path),
            "source_images": "|".join(
                str(image_dir / f"{case_id}_{m}.nii.gz") for m in cs.CHANNEL_ORDER),
            "shape": "x".join(str(v) for v in image.shape),
            "dtype": str(image.dtype),
            "channel_order": "|".join(cs.CHANNEL_ORDER),
            "image_sha256": sha256_array(image),
        })
        del image

    frame_out = pd.DataFrame(rows)
    (out_dir / "cache_index.csv").parent.mkdir(parents=True, exist_ok=True)
    frame_out.to_csv(out_dir / "cache_index.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "image-only inference cache for the SECONDARY FROZEN HOLDOUT",
        "n_cases": len(rows),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "crop": {
            "shape_xyz": list(cs.CROP_SHAPE_XYZ),
            "shape_dhw": list(cs.CROP_SHAPE_DHW),
            "slices": {"x": list(cs.CROP_X), "y": list(cs.CROP_Y),
                       "z": list(cs.CROP_Z)},
            "half_open": True,
        },
        "axis_transform": "to_tensor_layout: crop XYZ then transpose (2,1,0) -> (Z,Y,X) = (D,H,W)",
        "modalities": list(cs.CHANNEL_ORDER),
        "tensor_shape": [cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW],
        "dtype": "float32",
        "normalization_provenance": (
            "Inputs are processed/images/*.nii.gz produced by scripts/normalize_images.py "
            "(per-modality percentile clip [0.5,99.5]; T1/NM z-score, QSM median-IQR, "
            "computed over and applied only inside the foreground, background forced "
            "to exactly 0). This script applies NO further intensity transform."),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_cache_equivalence_verified": True,
        "GT_pixel_read": False,
        "labels_stored": False,
        "gt_derived_metadata_present": False,
        "note": ("This cache holds images only. No label tensor and no GT-derived "
                 "statistic is stored, so it cannot support model selection."),
    }
    (out_dir / "cache_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    root = resolve_project_root(args.root)
    manifest_dir = resolve_path(args.manifest_dir, root)
    image_dir = resolve_path(args.image_dir, root)
    frozen_cache = resolve_path(args.frozen_cache, root)
    out_dir = resolve_path(args.out_dir, root)

    LOGGER.info("Project root : %s", root)
    LOGGER.info("Images       : %s", image_dir)
    LOGGER.info("Frozen cache : %s", frozen_cache)
    LOGGER.info("Holdout out  : %s", out_dir)

    # ---- 1. equivalence on the frozen cache, BEFORE writing anything ------- #
    LOGGER.info("Verifying pipeline equivalence against the frozen cache ...")
    equivalence = verify_equivalence(frozen_cache, image_dir, manifest_dir)
    if not equivalence["all_equal"]:
        LOGGER.error("EQUIVALENCE FAILED on %d case(s); refusing to build the "
                     "holdout cache. First mismatches: %s",
                     len(equivalence["mismatches"]), equivalence["mismatches"][:5])
        return 2
    LOGGER.info("Equivalence OK: %d frozen tensors reproduced bitwise",
                equivalence["n_compared"])

    if args.verify_only:
        print()
        print("=" * 78)
        print("Equivalence verified (--verify-only): no holdout cache written.")
        print(f"  {equivalence['n_compared']} frozen train/val tensors reproduced bitwise")
        print("=" * 78)
        return 0

    # ---- 2. holdout image cache ------------------------------------------- #
    summary = build_holdout(root, manifest_dir, image_dir, out_dir, args.overwrite)
    summary["equivalence"] = equivalence

    # Write the gate report alongside the experiment outputs, not into the cache.
    gate_dir = (root / "results/experiments/baseline_deep_ensemble" /
                "holdout_cache_gate").resolve()
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "cache_build_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (gate_dir / "cache_index.csv").write_text(
        (out_dir / "cache_index.csv").read_text(encoding="utf-8-sig"),
        encoding="utf-8-sig")

    print()
    print("=" * 78)
    print("Holdout image cache built")
    print("=" * 78)
    print(f"  cases            : {summary['n_cases']}")
    print(f"  tensor           : {summary['tensor_shape']} {summary['dtype']}")
    print(f"  channels         : {summary['modalities']}")
    print(f"  crop             : XYZ {cs.CROP_SHAPE_XYZ} -> DHW {cs.CROP_SHAPE_DHW}")
    print(f"  manifest sha256  : {summary['manifest_sha256'][:32]}")
    print(f"  builder sha256   : {summary['builder_script_sha256'][:32]}")
    print(f"  equivalence      : {equivalence['n_compared']} frozen tensors, "
          f"all bitwise equal")
    print(f"  GT pixel read    : {summary['GT_pixel_read']}")
    print(f"  labels stored    : {summary['labels_stored']}")
    print("=" * 78)
    print(f"Written: {out_dir} and {gate_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

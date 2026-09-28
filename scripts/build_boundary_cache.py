#!/usr/bin/env python
"""Build the STN signed-distance cache for Experiment C (boundary supervision).

Computes, for every development-train / development-val case, the signed
distance map of that case's own ground-truth STN mask, using the real
anisotropic voxel spacing. This is *training supervision*, derived only from the
GT the model is already trained against -- no test split is read.

Symbol convention (this is the part that silently breaks boundary losses)
------------------------------------------------------------------------
    inside the GT  -> NEGATIVE distance
    outside the GT -> POSITIVE distance
    on the boundary -> ~0

so that ``mean(p_STN * phi)`` is *lowered* by raising the STN probability where
STN actually is, and *raised* by putting STN probability where it is not.

Stabilisation
-------------
Raw distances grow without bound towards the edge of the crop, and this crop is
64 mm across while STN has an equivalent radius of ~3.4 mm -- so an unclipped map
would be dominated by voxels that are tens of millimetres away and carry no
useful boundary information. Distances are therefore clipped to
``[-10, +10] mm`` and divided by 10, giving ``phi in [-1, 1]``. The clip range is
fixed here and is never adapted from validation data.

Axis/spacing
------------
Tensors are ``(D, H, W) = (Z, Y, X)``, so spacing must be
``(2.0, 0.6666667, 0.6666667)`` mm -- taken from :mod:`data.crop_spec`, not
hard-coded, so it cannot drift from the crop.

Usage
-----
    python scripts/build_boundary_cache.py --root .
    python scripts/build_boundary_cache.py --root . --verify-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import crop_spec as cs  # noqa: E402
from utils.paths import resolve_path, resolve_project_root  # noqa: E402

LOGGER = logging.getLogger("build_boundary_cache")

#: Only the development splits may be read. internal_test and challenge_test are
#: reserved; the cache builder refuses them outright rather than trusting the
#: caller to pass the right --splits value.
ALLOWED_SPLITS: tuple[str, ...] = ("train", "val")

#: Distance clip in millimetres, and the divisor that maps it to [-1, 1].
CLIP_MM = 10.0
NORMALISER = 10.0

#: Class id for STN. ``data.crop_spec`` is frozen and carries no STN constant, so
#: it is derived from that module's own CLASS_NAMES rather than restated as a
#: bare literal -- there is still only one place the mapping is written down.
STN: Final[int] = next(c for c, name in cs.CLASS_NAMES.items() if name == "STN")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the STN signed-distance cache.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--label-dir", type=Path,
                        default=Path("cache/baseline_v1/labels"),
                        help="Directory of cropped GT label .npy files (read-only).")
    parser.add_argument("--manifest-dir", type=Path,
                        default=Path("manifests/experiment"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("cache/boundary_v1/stn_signed_distance"))
    parser.add_argument("--splits", type=str, default="train,val",
                        help="Comma separated; only train and val are allowed.")
    parser.add_argument("--class-id", type=int, default=STN,
                        help="Class whose boundary is supervised (default STN).")
    parser.add_argument("--verify-only", action="store_true",
                        help="Do not write; re-check existing cache files instead.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute files that already exist.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def signed_distance_stn(
    label: np.ndarray, class_id: int = STN, spacing: Sequence[float] = cs.SPACING_DHW_MM
) -> tuple[np.ndarray, dict[str, Any]]:
    """Signed distance map of one class, clipped and normalised to [-1, 1].

    Negative inside, positive outside -- see the module docstring. Distances are
    euclidean in millimetres (``distance_transform_edt`` with ``sampling``), so
    the 2.0 mm through-plane spacing is honoured rather than being treated as
    equal to the in-plane 0.667 mm.
    """
    from scipy import ndimage

    if label.shape != cs.CROP_SHAPE_DHW:
        raise ValueError(
            f"label shape {label.shape} != frozen crop {cs.CROP_SHAPE_DHW}"
        )

    mask = label == class_id
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        # No GT for this class in this case. "Distance to the nearest GT voxel"
        # is undefined, so every voxel is treated as far outside rather than
        # silently producing an all-zero (i.e. "on the boundary") map.
        phi = np.ones(cs.CROP_SHAPE_DHW, dtype=np.float32)
        return phi, {"n_voxels": 0, "empty_mask": True}

    # edt(mask) is the distance to the nearest background voxel -> 0 outside.
    # edt(~mask) is the distance to the nearest foreground voxel -> 0 inside.
    # outside - inside is therefore positive outside and negative inside.
    distance_outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    distance_inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    signed = distance_outside - distance_inside

    clipped = np.clip(signed, -CLIP_MM, CLIP_MM) / NORMALISER
    phi = np.ascontiguousarray(clipped, dtype=np.float32)
    return phi, {
        "n_voxels": n_voxels,
        "empty_mask": False,
        "signed_min_mm": float(signed.min()),
        "signed_max_mm": float(signed.max()),
    }


def verify_map(phi: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    """Structural checks on one distance map, returned as a record."""
    mask = label == STN
    inside = phi[mask] if mask.any() else np.array([], dtype=np.float32)
    outside = phi[~mask]
    return {
        "shape_ok": phi.shape == cs.CROP_SHAPE_DHW,
        "dtype": str(phi.dtype),
        "finite": bool(np.isfinite(phi).all()),
        "min": float(phi.min()),
        "max": float(phi.max()),
        "range_ok": bool(phi.min() >= -1.0 - 1e-6 and phi.max() <= 1.0 + 1e-6),
        "inside_negative_fraction": float((inside <= 0).mean()) if inside.size else None,
        "outside_positive_fraction": float((outside >= 0).mean()),
        "n_voxels": int(mask.sum()),
        "values_exactly_zero": int((phi == 0).sum()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    root = resolve_project_root(args.root)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    bad = [s for s in splits if s not in ALLOWED_SPLITS]
    if bad:
        LOGGER.error("split(s) %s are not permitted; allowed: %s", bad, ALLOWED_SPLITS)
        return 2

    label_dir = resolve_path(args.label_dir, root)
    manifest_dir = resolve_path(args.manifest_dir, root)
    out_dir = resolve_path(args.out_dir, root)
    LOGGER.info("Project root : %s", root)
    LOGGER.info("Labels       : %s", label_dir)
    LOGGER.info("Output       : %s", out_dir)
    LOGGER.info("Spacing (DHW): %s mm", cs.SPACING_DHW_MM)
    LOGGER.info("Clip/norm    : +/-%g mm / %g", CLIP_MM, NORMALISER)

    case_ids: list[tuple[str, str]] = []
    for split in splits:
        manifest = manifest_dir / f"{split}.csv"
        if not manifest.is_file():
            LOGGER.error("missing manifest %s", manifest)
            return 2
        frame = pd.read_csv(manifest)
        for case_id in sorted(str(c) for c in frame["case_id"]):
            case_ids.append((split, case_id))
    LOGGER.info("Cases to process: %d", len(case_ids))

    if not args.verify_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    written = skipped = failed = 0
    start = time.perf_counter()

    for split, case_id in case_ids:
        label_path = label_dir / f"{case_id}.npy"
        out_path = out_dir / f"{case_id}.npy"
        try:
            if not label_path.is_file():
                raise FileNotFoundError(f"missing GT label {label_path}")
            label = np.load(label_path)

            if args.verify_only:
                if not out_path.is_file():
                    raise FileNotFoundError(f"missing cache file {out_path}")
                phi = np.load(out_path)
            elif out_path.is_file() and not args.overwrite:
                phi = np.load(out_path)
                skipped += 1
            else:
                phi, _ = signed_distance_stn(label, args.class_id, cs.SPACING_DHW_MM)
                np.save(out_path, phi)
                written += 1

            record = verify_map(phi, label)
            record.update({"case_id": case_id, "split": split})
            records.append(record)
            if not (record["shape_ok"] and record["finite"] and record["range_ok"]):
                LOGGER.error("VERIFICATION FAILED for %s: %s", case_id, record)
                failed += 1
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            LOGGER.error("case %s failed: %s", case_id, exc)
            failed += 1

    frame = pd.DataFrame(records)
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_list": list(splits),
        "n_cases": len(case_ids),
        "written": written,
        "already_present": skipped,
        "failed": failed,
        "class_id": args.class_id,
        "class_name": cs.CLASS_NAMES.get(args.class_id, str(args.class_id)),
        "spacing_dhw_mm": list(cs.SPACING_DHW_MM),
        "crop_shape_dhw": list(cs.CROP_SHAPE_DHW),
        "clip_mm": CLIP_MM,
        "normaliser": NORMALISER,
        "sign_convention": "negative inside GT, positive outside GT",
        "note": "Derived from this case's own ground-truth mask only. "
                "internal_test / challenge_test are never read.",
    }
    if not frame.empty:
        summary["checks"] = {
            "all_shapes_ok": bool(frame["shape_ok"].all()),
            "all_finite": bool(frame["finite"].all()),
            "all_ranges_ok": bool(frame["range_ok"].all()),
            "min_value": float(frame["min"].min()),
            "max_value": float(frame["max"].max()),
            "min_inside_negative_fraction": float(
                frame["inside_negative_fraction"].min(skipna=True)),
            "min_outside_positive_fraction": float(
                frame["outside_positive_fraction"].min()),
            "cases_with_empty_STN": int((frame["n_voxels"] == 0).sum()),
        }

    if not args.verify_only:
        (out_dir.parent / "boundary_cache_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        frame.to_csv(out_dir.parent / "boundary_cache_index.csv",
                     index=False, encoding="utf-8-sig")

    LOGGER.info("Done in %.1fs: written=%d already_present=%d failed=%d",
                time.perf_counter() - start, written, skipped, failed)
    if not frame.empty:
        checks = summary["checks"]
        print()
        print("=" * 74)
        print("STN signed-distance cache")
        print("=" * 74)
        print(f"cases                     : {summary['n_cases']}  (failed {failed})")
        print(f"shapes ok / finite / range: {checks['all_shapes_ok']} / "
              f"{checks['all_finite']} / {checks['all_ranges_ok']}")
        print(f"value range               : [{checks['min_value']:+.6f}, "
              f"{checks['max_value']:+.6f}]")
        print(f"min inside-negative frac  : {checks['min_inside_negative_fraction']:.6f}"
              f"   (want 1.0)")
        print(f"min outside-positive frac : {checks['min_outside_positive_fraction']:.6f}"
              f"   (want 1.0)")
        print(f"cases with empty STN      : {checks['cases_with_empty_STN']}")
        print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

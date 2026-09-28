#!/usr/bin/env python
"""Intensity normalisation for T1 / QSM / NM.

Per modality, computed over the foreground and applied only inside it:

    T1, NM : percentile clip [0.5, 99.5] -> z-score  (x - mean) / std
    QSM    : percentile clip [0.5, 99.5] -> median/IQR  (x - median) / IQR

Design decisions that matter and are easy to get wrong:

1. **Background is preserved as exactly zero.** Normalisation is applied only
   inside the foreground mask; every voxel outside it is set to 0.0. Without
   this, a z-score would turn QSM's exact-zero background into a constant
   non-zero value and silently destroy the background/foreground distinction
   that every downstream crop, patch and loss depends on.

2. **Statistics are computed on the clipped data**, so the reported mean/std (or
   median/IQR) are exactly the parameters that were applied. Storing parameters
   that do not reproduce the transform is worse than storing none.

3. **Percentile clipping comes before the scale estimate**, which makes both the
   clip and the scale robust to the bright outliers these modalities contain
   (T1 max reaches ~212000 while the non-zero median is ~4500).

Foreground definition is selectable:

    --foreground nonzero   (default) every voxel != 0
    --foreground head      Otsu-based head/tissue mask

The default matches the project specification. It is worth knowing that for T1
and NM "non-zero" is NOT "tissue": the audit measured T1 at ~81% non-zero, so most
of that set is noise floor. `--foreground head` restricts the statistics to actual
tissue and therefore yields a different (usually tighter) scale. Both are provided
so the choice can be made on evidence rather than assumption.

The raw dataset is READ-ONLY. Output goes to ``processed/images/``.

Usage
-----
    python scripts/normalize_images.py --root .
    python scripts/normalize_images.py --root . --foreground head
    python scripts/normalize_images.py --root . --modalities T1,QSM --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

LOGGER = logging.getLogger("normalize_images")

#: z-score modalities
ZSCORE_MODALITIES: tuple[str, ...] = ("T1", "NM")
#: robust median/IQR modality
ROBUST_MODALITIES: tuple[str, ...] = ("QSM",)

CLIP_LOW_PERCENTILE = 0.5
CLIP_HIGH_PERCENTILE = 99.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intensity-normalise T1 / QSM / NM, preserving spatial metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Project root containing the raw data directories.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Image output directory (default: <root>/processed/images).")
    parser.add_argument("--meta-dir", type=Path, default=None,
                        help="Metadata directory (default: <root>/processed/metadata).")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Run report directory (default: <root>/results/preprocessing).")
    parser.add_argument("--modalities", type=str, default="T1,QSM,NM",
                        help="Comma separated modalities to normalise.")
    parser.add_argument("--foreground", choices=("nonzero", "head"), default="nonzero",
                        help="How the foreground mask is defined.")
    parser.add_argument("--splits", type=str, default=None,
                        help="Comma separated splits to process, e.g. 'train,val'. "
                             "Default: every discovered case.")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma separated case ids to restrict processing to.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N cases (smoke testing).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute and overwrite existing outputs.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the post-write re-read validation.")
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


def foreground_mask(volume: np.ndarray, mode: str) -> np.ndarray:
    if mode == "head":
        return pio.estimate_head_mask(volume)
    return volume != 0


def normalize_volume(
    volume: np.ndarray, modality: str, fg_mode: str
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Normalise one volume. Returns (normalized, params, warnings)."""
    warnings: list[str] = []
    mask = foreground_mask(volume, fg_mode)
    n_mask = int(mask.sum())

    if n_mask < 100:
        raise ValueError(f"foreground mask has only {n_mask} voxels")

    values = volume[mask].astype(np.float64)
    p_low, p_high = np.percentile(values, [CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE])
    if not (math.isfinite(p_low) and math.isfinite(p_high)) or p_high <= p_low:
        warnings.append(
            f"degenerate clip range [{p_low}, {p_high}]; clipping disabled"
        )
        p_low, p_high = -math.inf, math.inf

    clipped = np.clip(volume.astype(np.float32), p_low, p_high)
    clipped_values = clipped[mask].astype(np.float64)

    params: dict[str, Any] = {
        "foreground_mode": fg_mode,
        "foreground_voxels": n_mask,
        "foreground_fraction": n_mask / float(volume.size),
        "clip_low": float(p_low),
        "clip_high": float(p_high),
        "clip_low_percentile": CLIP_LOW_PERCENTILE,
        "clip_high_percentile": CLIP_HIGH_PERCENTILE,
    }

    if modality in ZSCORE_MODALITIES:
        mean = float(clipped_values.mean())
        std = float(clipped_values.std())
        if std <= 0 or not math.isfinite(std):
            warnings.append(f"std is {std}; falling back to mean=0 std=1")
            mean, std = 0.0, 1.0
        normalized = (clipped - mean) / std
        params.update({"method": "zscore", "mean": mean, "std": std,
                       "median": float(np.median(clipped_values)),
                       "iqr": float(np.percentile(clipped_values, 75)
                                    - np.percentile(clipped_values, 25))})
    else:
        median = float(np.median(clipped_values))
        q25, q75 = np.percentile(clipped_values, [25, 75])
        iqr = float(q75 - q25)
        if iqr <= 0 or not math.isfinite(iqr):
            warnings.append(f"IQR is {iqr}; falling back to median=0 IQR=1")
            median, iqr = 0.0, 1.0
        normalized = (clipped - median) / iqr
        params.update({"method": "median_iqr", "median": median, "iqr": iqr,
                       "mean": float(clipped_values.mean()),
                       "std": float(clipped_values.std())})

    # Keep the background at exactly zero -- see the module docstring.
    normalized = normalized.astype(np.float32)
    normalized[~mask] = 0.0
    params["background_value"] = 0.0
    params["background_voxels"] = int(volume.size - n_mask)
    # Voxels that are exactly 0 *after* normalisation. This is deliberately not
    # the same as the background count: any foreground voxel whose value equals
    # the offset (the mean for z-score, the median for median/IQR) also lands on
    # exactly 0. For QSM the median sits inside the brain, so ~0.3% of brain
    # voxels collapse to 0 and `image != 0` stops being a valid background test.
    # Downstream code must take the background from the raw data or from an
    # explicit mask, never by thresholding the normalised image.
    params["zero_voxels_after"] = int(np.count_nonzero(normalized == 0))
    return normalized, params, warnings


def verify_written(
    path: Path,
    modality: str,
    reference: nib.Nifti1Image,
    params: dict[str, Any],
    foreground: np.ndarray,
) -> dict[str, Any]:
    """Re-read the written file and confirm both its geometry and its statistics.

    ``foreground`` is the mask derived from the RAW volume. The verification must
    use it rather than ``data != 0``: normalised foreground voxels can legitimately
    be exactly 0 (see ``zero_voxels_after``), and excluding them would bias the
    recomputed median/IQR away from the values that were actually applied.
    """
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    ref_sig = pio.spatial_signature(reference)

    result: dict[str, Any] = {
        "shape": tuple(int(s) for s in image.shape),
        "spacing": tuple(float(z) for z in image.header.get_zooms()[:3]),
        "axcodes": tuple(nib.aff2axcodes(image.affine)),
        "affine": np.asarray(image.affine, dtype=np.float64),
        "dtype": str(image.get_data_dtype()),
    }

    if result["shape"] != ref_sig["shape"]:
        raise ValueError(f"shape changed: {result['shape']} != {ref_sig['shape']}")
    if result["axcodes"] != ref_sig["axcodes"]:
        raise ValueError(f"orientation changed: {result['axcodes']} != {ref_sig['axcodes']}")
    if not np.allclose(result["affine"], ref_sig["affine"], atol=0.0):
        raise ValueError("affine changed during normalisation")
    if not np.allclose(result["spacing"], ref_sig["spacing"], atol=1e-6):
        raise ValueError(f"spacing changed: {result['spacing']} != {ref_sig['spacing']}")
    if not np.isfinite(data).all():
        raise ValueError("normalised output contains NaN or Inf")
    if foreground.shape != data.shape:
        raise ValueError("foreground mask shape does not match the written volume")

    # Background voxels must have stayed at exactly zero.
    if np.any(data[~foreground] != 0.0):
        n_bad = int(np.count_nonzero(data[~foreground] != 0.0))
        raise ValueError(f"{n_bad} background voxels are non-zero after normalisation")

    fg = data[foreground].astype(np.float64)
    result["n_foreground"] = int(fg.size)
    if fg.size > 100:
        result["fg_mean"] = float(fg.mean())
        result["fg_std"] = float(fg.std())
        result["fg_median"] = float(np.median(fg))
        result["fg_iqr"] = float(np.percentile(fg, 75) - np.percentile(fg, 25))

        # The transform must actually have taken effect, checked with the
        # statistic that this modality's method actually controls.
        if modality in ZSCORE_MODALITIES:
            if abs(result["fg_mean"]) > 0.05 or abs(result["fg_std"] - 1.0) > 0.05:
                raise ValueError(
                    f"z-score did not take effect: foreground mean={result['fg_mean']:.4f} "
                    f"std={result['fg_std']:.4f}"
                )
        else:
            if abs(result["fg_median"]) > 1e-3 or abs(result["fg_iqr"] - 1.0) > 1e-3:
                raise ValueError(
                    f"median/IQR did not take effect: foreground median="
                    f"{result['fg_median']:.6f} IQR={result['fg_iqr']:.6f}"
                )
    return result


def process_case(
    record: pio.CaseRecord,
    out_dir: Path,
    modalities: Sequence[str],
    fg_mode: str,
    overwrite: bool,
    skip_verify: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []

    for modality in modalities:
        source = record.files.get(modality)
        if source is None:
            warnings.append({
                "case_id": record.case_id, "split": record.split,
                "warning_type": "missing_modality",
                "description": f"{modality} is missing",
            })
            continue

        try:
            volume, image = pio.load_array(source)
            raw_foreground = foreground_mask(volume, fg_mode)
            normalized, params, vol_warnings = normalize_volume(volume, modality, fg_mode)
            del volume

            out_path = out_dir / f"{record.case_id}_{modality}.nii.gz"
            if not out_path.exists() or overwrite:
                pio.save_like(
                    image, normalized, out_path, dtype=np.float32,
                    description=f"{modality} normalized ({params['method']}, fg={fg_mode})",
                )
            del normalized

            verification: dict[str, Any] = {}
            if not skip_verify:
                verification = verify_written(
                    out_path, modality, image, params, raw_foreground
                )
            del raw_foreground

            row: dict[str, Any] = {
                "case_id": record.case_id,
                "split": record.split,
                "modality": modality,
                "method": params["method"],
                "foreground_mode": params["foreground_mode"],
                "foreground_voxels": params["foreground_voxels"],
                "foreground_fraction": params["foreground_fraction"],
                "background_voxels": params["background_voxels"],
                "zero_voxels_after": params["zero_voxels_after"],
                "clip_low": params["clip_low"],
                "clip_high": params["clip_high"],
                "mean": params["mean"],
                "std": params["std"],
                "median": params["median"],
                "iqr": params["iqr"],
                "image_path": str(out_path),
            }
            if verification:
                row["verified_fg_mean"] = verification.get("fg_mean", math.nan)
                row["verified_fg_std"] = verification.get("fg_std", math.nan)
                row["verified_fg_median"] = verification.get("fg_median", math.nan)
                row["verified_fg_iqr"] = verification.get("fg_iqr", math.nan)
                row["verified_dtype"] = verification["dtype"]
            rows.append(row)

            for message in vol_warnings:
                warnings.append({
                    "case_id": record.case_id, "split": record.split,
                    "warning_type": "normalization_fallback",
                    "description": f"{modality}: {message}",
                })
        except Exception as exc:  # noqa: BLE001 - one bad modality must not abort
            LOGGER.error("Case %s / %s failed: %s", record.label, modality, exc)
            LOGGER.debug("%s", traceback.format_exc())
            warnings.append({
                "case_id": record.case_id, "split": record.split,
                "warning_type": "processing_failed",
                "description": f"{modality}: {type(exc).__name__}: {exc}",
            })

    return rows, warnings


def build_summary(frame: pd.DataFrame, warnings: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(args.root.resolve()),
        "foreground_mode": args.foreground,
        "clip_percentiles": [CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE],
        "n_cases": int(frame["case_id"].nunique()) if not frame.empty else 0,
        "n_images": int(len(frame)),
        "splits": ({str(k): int(v) for k, v in sorted(Counter(frame["split"]).items())}
                   if not frame.empty else {}),
        "n_warning_cases": int(warnings["case_id"].nunique()) if not warnings.empty else 0,
        "warnings_by_type": ({str(k): int(v) for k, v in sorted(Counter(warnings["warning_type"]).items())}
                             if not warnings.empty else {}),
    }

    per_modality: dict[str, Any] = {}
    for modality in sorted(frame["modality"].unique()) if not frame.empty else []:
        sub = frame[frame["modality"] == modality]
        entry: dict[str, Any] = {"n_cases": int(len(sub)),
                                 "method": str(sub["method"].iloc[0])}
        for column in ("clip_low", "clip_high", "mean", "std", "median", "iqr",
                       "foreground_fraction", "verified_fg_mean", "verified_fg_std",
                       "verified_fg_median", "verified_fg_iqr"):
            if column not in sub:
                continue
            values = pd.to_numeric(sub[column], errors="coerce").dropna()
            if values.empty:
                continue
            entry[column] = {
                "min": float(values.min()), "median": float(values.median()),
                "mean": float(values.mean()), "max": float(values.max()),
                "std": float(values.std()),
            }
        per_modality[modality] = entry
    summary["per_modality"] = per_modality
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    line = "=" * 68
    print()
    print(line)
    print("Image Normalization Summary")
    print(line)
    print(f"Foreground definition : {summary['foreground_mode']}")
    print(f"Clip percentiles      : {summary['clip_percentiles']}")
    print(f"Cases / images        : {summary['n_cases']} / {summary['n_images']}")
    print(f"Splits                : {summary['splits']}")
    print(f"Cases with warnings   : {summary['n_warning_cases']}")
    print()
    for modality, entry in summary["per_modality"].items():
        print(f"--- {modality}  ({entry['method']}, n={entry['n_cases']}) ---")
        for key in ("clip_low", "clip_high", "mean", "std", "median", "iqr",
                    "foreground_fraction", "verified_fg_mean", "verified_fg_std"):
            if key not in entry:
                continue
            stats = entry[key]
            print(f"    {key:<20} min={stats['min']:>12.4f}  median={stats['median']:>10.4f}"
                  f"  max={stats['max']:>12.4f}")
        print()
    if summary["warnings_by_type"]:
        print(f"Warnings: {summary['warnings_by_type']}")
    else:
        print("Warnings: none")
    print(line)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: --root {root} is not a directory", file=sys.stderr)
        return 2

    out_dir = (args.out_dir or (root / "processed" / "images")).resolve()
    meta_dir = (args.meta_dir or (root / "processed" / "metadata")).resolve()
    results_dir = (args.results_dir or (root / "results" / "preprocessing")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    tag = "" if args.foreground == "nonzero" else f"_{args.foreground}"
    setup_logging(results_dir / f"normalize_images{tag}.log", args.verbose)

    modalities = tuple(m.strip().upper() for m in args.modalities.split(",") if m.strip())
    unknown = [m for m in modalities if m not in pio.IMAGE_MODALITIES]
    if unknown:
        LOGGER.error("Unknown modalities: %s", unknown)
        return 2

    cases = pio.discover_cases(root)
    LOGGER.info("Discovered %d cases; normalising %s", len(cases), modalities)

    if args.splits:
        wanted_splits = {s.strip().lower() for s in args.splits.split(",") if s.strip()}
        unknown_splits = wanted_splits - {"train", "val", "test"}
        if unknown_splits:
            LOGGER.error("Unknown splits: %s", sorted(unknown_splits))
            return 2
        cases = [c for c in cases if c.split in wanted_splits]
        LOGGER.info("Restricted to splits %s -> %d cases",
                    sorted(wanted_splits), len(cases))
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.case_id in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    cases = [c for c in cases if any(m in c.files for m in modalities)]
    if not cases:
        LOGGER.error("No cases to process.")
        return 1

    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []

    for record in tqdm(cases, desc="Normalizing", unit="case"):
        case_rows, case_warnings = process_case(
            record, out_dir, modalities, args.foreground, args.overwrite, args.skip_verify
        )
        rows.extend(case_rows)
        warnings.extend(case_warnings)

    if not rows:
        LOGGER.error("No images were normalised successfully.")
        return 1

    frame = pd.DataFrame(rows).sort_values(["split", "case_id", "modality"]).reset_index(drop=True)
    warning_frame = pd.DataFrame(
        warnings, columns=["case_id", "split", "warning_type", "description"]
    )

    params_path = meta_dir / f"normalization_params{tag}.csv"
    frame.to_csv(params_path, index=False, encoding="utf-8-sig")
    warning_frame.to_csv(
        meta_dir / f"normalization_warnings{tag}.csv", index=False, encoding="utf-8-sig"
    )

    summary = build_summary(frame, warning_frame, args)
    with (results_dir / f"normalization_summary{tag}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    LOGGER.info("Normalised images written to %s", out_dir)
    LOGGER.info("Parameters written to %s", params_path)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

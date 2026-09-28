#!/usr/bin/env python
"""Experiment G0 — STN failure-mechanism audit (development data only).

Reads only development-val artifacts. The frozen holdout is NOT touched: no
internal_test ground truth, no re-inference on it, no per-case holdout numbers.
The only holdout input is the already-published aggregate finding that STN is
badly over-segmented there, which motivates the mechanism question -- it does not
inform any parameter here.

Everything below is **exploratory**. n = 40, one development split, many
correlations. So each relationship is reported with its effect size and its raw
p, plus a Benjamini-Hochberg FDR across the family of tests, and the report leads
with patterns rather than with "significant findings". Nothing here changes the
preregistered Experiment F statistics.

Direction of the audit: it does not assume the answer. A volume/calibration
story, a boundary story, a spatial story and a class-confusion story each make
different, checkable predictions, and the report is explicit when the data
supports none of them.

Usage
-----
    # per-case and spatial analyses from existing prediction maps
    python scripts/analyze_stn_failure_mechanism.py --root .

    # additionally run one development-val pass to obtain p_STN for G0-E
    python scripts/analyze_stn_failure_mechanism.py --root . --with-probabilities
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import crop_spec as cs  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scipy import stats  # noqa: E402

from utils.paths import resolve_project_root  # noqa: E402

LOGGER = logging.getLogger("stn_failure_audit")

STN: int = next(c for c, name in cs.CLASS_NAMES.items() if name == "STN")

REHEARSAL = "results/experiments/baseline_deep_ensemble/holdout_evaluator_rehearsal"
OUT_DIR = "results/experiments/stn_failure_audit"
PROB_CACHE = "results/experiments/stn_failure_audit/stn_probability_cache"

#: member order in the saved prediction array
MEMBERS: tuple[str, ...] = ("ensemble", "seed42", "seed123", "seed2026")

#: number of boostrap draws for correlation CIs (exploratory)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260927


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STN failure-mechanism audit.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path(OUT_DIR))
    parser.add_argument("--with-probabilities", action="store_true",
                        help="Run one development-val pass to obtain p_STN (G0-E).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# statistics helpers (exploratory; effect size first)
# --------------------------------------------------------------------------- #

def spearman(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": int(x.size), "rho": float("nan"), "p": float("nan")}
    rho, p = stats.spearmanr(x, y)
    return {"n": int(x.size), "rho": float(rho), "p": float(p)}


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    """BH-FDR across a family of exploratory tests."""
    items = [(k, v) for k, v in p_values.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 1.0
    for rank, (key, value) in enumerate(reversed(items), start=1):
        index = m - rank + 1
        running = min(running, value * m / index)
        out[key] = running
    return out


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def centroid_of(mask: np.ndarray) -> tuple[float, float, float] | None:
    if not mask.any():
        return None
    return tuple(float(v) for v in np.argwhere(mask).mean(axis=0))


def bbox_of(mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    if not mask.any():
        return None
    idx = np.argwhere(mask)
    return tuple(int(v) for v in idx.min(axis=0)), tuple(int(v) for v in idx.max(axis=0))


def crop_margins(mask: np.ndarray) -> dict[str, float] | None:
    """Distance from the STN bounding box to each crop face, in voxels.

    The minimum over the six faces is the quantity that matters for a
    crop-boundary hypothesis: it is how close the structure comes to being cut.
    """
    box = bbox_of(mask)
    if box is None:
        return None
    (d0, h0, w0), (d1, h1, w1) = box
    depth, height, width = cs.CROP_SHAPE_DHW
    margins = {"d_lo": float(d0), "d_hi": float(depth - 1 - d1),
               "h_lo": float(h0), "h_hi": float(height - 1 - h1),
               "w_lo": float(w0), "w_hi": float(width - 1 - w1)}
    margins["min_margin"] = min(margins.values())
    return margins


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (mm) from every voxel to the nearest ``mask`` voxel."""
    from scipy import ndimage

    return ndimage.distance_transform_edt(~mask, sampling=cs.SPACING_DHW_MM)


# --------------------------------------------------------------------------- #
# optional probability pass (development val only)
# --------------------------------------------------------------------------- #

def build_probability_cache(root: Path, out_dir: Path, device: torch.device) -> Path:
    """One development-val forward pass, storing only the STN probability channel.

    Phase-A style: reads val images and the frozen checkpoints, writes p_STN. No
    ground truth is read here, and nothing about the holdout is involved.
    """
    from evaluate_frozen_holdout import (FROZEN_SEEDS, FrozenImageCache,
                                         load_frozen_models)

    case_ids = sorted(str(c) for c in pd.read_csv(
        root / "manifests/experiment/val.csv")["case_id"])
    cache = FrozenImageCache(root / "cache/baseline_v1/images", case_ids)
    models = load_frozen_models(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "p_stn.npy"
    stack = np.zeros((len(case_ids), len(MEMBERS), *cs.CROP_SHAPE_DHW), dtype=np.float32)
    with torch.no_grad():
        for index, case_id in enumerate(case_ids):
            image = torch.from_numpy(cache.get(case_id)[None]).to(device)
            with torch.inference_mode():
                probabilities = {s: F.softmax(models[s](image), dim=1)
                                 for s in FROZEN_SEEDS}
                ensemble = torch.stack(
                    [probabilities[s] for s in FROZEN_SEEDS]).mean(dim=0)
            stack[index, 0] = ensemble[0, STN].cpu().numpy()
            for offset, seed in enumerate(FROZEN_SEEDS, start=1):
                stack[index, offset] = probabilities[seed][0, STN].cpu().numpy()
    np.save(out_path, stack)
    (out_dir / "p_stn_meta.json").write_text(json.dumps({
        "source_split": "development val", "n_cases": len(case_ids),
        "members": list(MEMBERS), "channel": "p_STN",
        "ground_truth_read": False,
        "purpose": "G0-E probability diagnostics only; not a prediction artifact",
    }, indent=2), encoding="utf-8")
    LOGGER.info("probability cache written: %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_volume_bias(frame: pd.DataFrame, out: Path, dpi: int) -> dict[str, Any]:
    plt = _plt()
    figure, panels = plt.subplots(1, 3, figsize=(16, 4.6))
    colours = {"ensemble": "#ff3b30", "seed42": "#0a84ff", "seed123": "#34c759",
               "seed2026": "#ff9500"}
    for member, colour in colours.items():
        ratio = frame[f"ratio__{member}"]
        panels[0].hist(ratio, bins=16, alpha=0.5, label=member, color=colour)
        panels[1].scatter(frame[f"gtvox__{member}"], frame[f"predvox__{member}"],
                          s=22, alpha=0.75, label=member, color=colour)
    panels[0].axvline(1.0, color="black", lw=1.2)
    panels[0].axvline(1.2, color="grey", ls="--", lw=1.0)
    panels[0].set_xlabel("Predicted / GT STN volume")
    panels[0].set_ylabel("cases")
    panels[0].set_title("STN volume ratio (development val, n=40)")
    panels[0].legend(fontsize=8)
    panels[0].grid(alpha=0.25)

    limits = [0, float(frame[[c for c in frame if c.startswith('gtvox__')]].to_numpy().max()) * 1.1]
    panels[1].plot(limits, limits, color="black", lw=1.0, ls=":")
    panels[1].set_xlabel("GT STN volume (voxels)")
    panels[1].set_ylabel("predicted STN volume (voxels)")
    panels[1].set_title("Predicted vs GT volume")
    panels[1].legend(fontsize=8)
    panels[1].grid(alpha=0.25)

    for member, colour in colours.items():
        panels[2].scatter(frame[f"ratio__{member}"], frame[f"dice__{member}"],
                          s=22, alpha=0.75, label=member, color=colour)
    panels[2].axvline(1.0, color="black", lw=1.0, ls=":")
    panels[2].set_xlabel("Predicted / GT STN volume")
    panels[2].set_ylabel("STN Dice")
    panels[2].set_title("Volume ratio vs Dice")
    panels[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return {}


def plot_scatter(frame: pd.DataFrame, x_col: str, y_col: str, title: str,
                 xlabel: str, ylabel: str, out: Path, dpi: int,
                 vline: float | None = None) -> None:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(6.6, 5.4))
    colours = {"ensemble": "#ff3b30", "seed42": "#0a84ff", "seed123": "#34c759",
               "seed2026": "#ff9500"}
    for member, colour in colours.items():
        axis.scatter(frame[f"{x_col}__{member}"], frame[f"{y_col}__{member}"],
                     s=24, alpha=0.75, label=member, color=colour)
    if vline is not None:
        axis.axvline(vline, color="black", lw=1.0, ls=":")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_frequency_maps(maps: dict[str, np.ndarray], out: Path, dpi: int,
                        title: str) -> None:
    """Axial / coronal / sagittal maximum-intensity projections of a frequency map."""
    plt = _plt()
    keys = [k for k in ("GT STN occupancy", "FP frequency", "FN frequency")
            if k in maps]
    figure, panels = plt.subplots(len(keys), 3, figsize=(12, 3.4 * len(keys)),
                                  squeeze=False)
    for row, key in enumerate(keys):
        volume = maps[key]
        for column, (label, projection) in enumerate((
                ("axial", volume.max(axis=0)),
                ("coronal", volume.max(axis=1)),
                ("sagittal", volume.max(axis=2)))):
            axis = panels[row][column]
            image = axis.imshow(projection, cmap="magma", origin="lower",
                                aspect="auto", vmin=0, vmax=1)
            axis.set_title(f"{key} — {label}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            plt.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_probability_diagnostics(diagnostics: dict[str, Any], out: Path,
                                 dpi: int) -> None:
    plt = _plt()
    figure, panels = plt.subplots(1, 3, figsize=(16, 4.6))
    histograms = diagnostics.get("histograms", {})
    for key, colour in (("gt_stn", "#34c759"), ("fp", "#ff3b30"),
                        ("fn", "#0a84ff"), ("background", "#8e8e93")):
        if key in histograms:
            panels[0].plot(histograms[key]["centers"], histograms[key]["density"],
                           marker="o", ms=3, label=key, color=colour)
    panels[0].set_xlabel("p_STN")
    panels[0].set_ylabel("density")
    panels[0].set_title("p_STN by voxel category (ensemble)")
    panels[0].legend(fontsize=8)
    panels[0].grid(alpha=0.25)

    binned = diagnostics.get("fp_vs_distance", {})
    if binned:
        panels[1].plot(binned["bin_centers_mm"], binned["fp_rate"],
                       marker="o", color="#ff3b30")
        panels[1].set_xlabel("distance to GT STN boundary (mm)")
        panels[1].set_ylabel("fraction of shell voxels that are FP")
        panels[1].set_title("FP rate vs distance from the true boundary")
        panels[1].grid(alpha=0.25)

    summary = diagnostics.get("fp_confidence", {})
    if summary:
        labels = list(summary.keys())
        values = [summary[k] for k in labels]
        panels[2].bar(labels, values, color="#ff9500", edgecolor="black", lw=0.6)
        panels[2].set_title("FP voxels by p_STN band")
        panels[2].set_ylabel("fraction of all FP")
        panels[2].tick_params(axis="x", rotation=20)
        panels[2].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_crop_margin(frame: pd.DataFrame, out: Path, dpi: int) -> None:
    plt = _plt()
    figure, panels = plt.subplots(1, 3, figsize=(16, 4.6))
    for column, (y_key, label) in enumerate((("dice", "STN Dice"),
                                             ("hd95", "STN HD95 (mm)"),
                                             ("abs_ratio_err", "|Pred/GT - 1|"))):
        for member, colour in (("ensemble", "#ff3b30"), ("seed42", "#0a84ff")):
            # the margin is a property of the GT mask, identical for every member
            panels[column].scatter(frame["margin_min_margin"],
                                   frame[f"{y_key}__{member}"], s=22, alpha=0.75,
                                   label=member, color=colour)
        panels[column].set_xlabel("nearest STN-to-crop margin (voxels)")
        panels[column].set_ylabel(label)
        panels[column].set_title(f"{label} vs crop margin")
        panels[column].legend(fontsize=8)
        panels[column].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    root = resolve_project_root(args.root)
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rehearsal = (root / REHEARSAL).resolve()
    predictions_path = rehearsal / "predictions.npy"
    if not predictions_path.is_file():
        LOGGER.error("missing development-val prediction maps: %s", predictions_path)
        return 2
    predictions = np.load(predictions_path)
    per_case = pd.read_csv(rehearsal / "holdout_case_metrics.csv",
                           encoding="utf-8-sig")
    case_ids = list(per_case["case_id"])
    if predictions.shape[0] != len(case_ids):
        LOGGER.error("prediction/case count mismatch")
        return 2
    LOGGER.info("development val: %d cases, members %s", len(case_ids), MEMBERS)

    label_dir = root / "cache/baseline_v1/labels"
    gts = [np.load(label_dir / f"{c}.npy") for c in case_ids]

    # ---- per-case table ---------------------------------------------------- #
    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        row: dict[str, Any] = {"case_id": case_id}
        gt = gts[index]
        gt_mask = gt == STN
        row["gt_voxels_stn"] = int(gt_mask.sum())
        gt_centroid = centroid_of(gt_mask)
        gt_box = bbox_of(gt_mask)
        margins = crop_margins(gt_mask)
        depth, height, width = cs.CROP_SHAPE_DHW
        row["gt_centroid_d"] = gt_centroid[0] if gt_centroid else np.nan
        row["gt_centroid_h"] = gt_centroid[1] if gt_centroid else np.nan
        row["gt_centroid_w"] = gt_centroid[2] if gt_centroid else np.nan
        row["dist_to_crop_center_mm"] = float(np.linalg.norm(
            (np.asarray(gt_centroid) - np.asarray([depth / 2, height / 2, width / 2]))
            * np.asarray(cs.SPACING_DHW_MM))) if gt_centroid else np.nan
        if margins:
            row.update({f"margin_{k}": v for k, v in margins.items()})
        if gt_box:
            row["gt_bbox_size_d"] = gt_box[1][0] - gt_box[0][0] + 1
            row["gt_bbox_size_h"] = gt_box[1][1] - gt_box[0][1] + 1
            row["gt_bbox_size_w"] = gt_box[1][2] - gt_box[0][2] + 1

        for offset, member in enumerate(MEMBERS):
            prediction = predictions[index, offset]
            pred_mask = prediction == STN
            tp = int((pred_mask & gt_mask).sum())
            fp = int((pred_mask & ~gt_mask).sum())
            fn = int((~pred_mask & gt_mask).sum())
            row[f"predvox__{member}"] = int(pred_mask.sum())
            row[f"gtvox__{member}"] = int(gt_mask.sum())
            row[f"tp__{member}"] = tp
            row[f"fp__{member}"] = fp
            row[f"fn__{member}"] = fn
            row[f"ratio__{member}"] = (pred_mask.sum() / gt_mask.sum()
                                       if gt_mask.sum() else np.nan)
            row[f"abs_ratio_err__{member}"] = abs(row[f"ratio__{member}"] - 1.0)
            row[f"dice__{member}"] = float(per_case.loc[index, f"Dice_STN__{member}"])
            row[f"hd95__{member}"] = float(per_case.loc[index, f"HD95_STN_mm__{member}"])
            row[f"precision__{member}"] = float(per_case.loc[index, f"Precision_STN__{member}"])
            row[f"recall__{member}"] = float(per_case.loc[index, f"Recall_STN__{member}"])
            row[f"centroid_dist__{member}"] = float(
                per_case.loc[index, f"centroid_distance_mm_STN__{member}"])
            pred_centroid = centroid_of(pred_mask)
            row[f"centroid_shift__{member}"] = (
                float(np.linalg.norm((np.asarray(pred_centroid)
                                      - np.asarray(gt_centroid))
                                     * np.asarray(cs.SPACING_DHW_MM)))
                if pred_centroid and gt_centroid else np.nan)
            # class confusion: what the FPs actually are in the GT
            row[f"fp_from_background__{member}"] = int((pred_mask & ~gt_mask
                                                        & (gt == 0)).sum())
            row[f"fp_from_sn__{member}"] = int((pred_mask & (gt == 2)).sum())
            row[f"fp_from_rn__{member}"] = int((pred_mask & (gt == 3)).sum())
            # what the FNs were predicted as
            row[f"fn_as_background__{member}"] = int(((prediction == 0) & gt_mask).sum())
            row[f"fn_as_sn__{member}"] = int(((prediction == 2) & gt_mask).sum())
            row[f"fn_as_rn__{member}"] = int(((prediction == 3) & gt_mask).sum())
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "stn_failure_per_case.csv", index=False,
                 encoding="utf-8-sig")

    # ---- G0-A/B/F/G correlations ------------------------------------------- #
    summary: dict[str, Any] = {
        "experiment": "G0 — STN failure-mechanism audit",
        "data_used": "development val only (n=40)",
        "holdout_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "members": list(MEMBERS), "n_cases": len(case_ids),
        "marking": "EXPLORATORY — effect sizes first, FDR across the test family",
    }
    volume_bias: dict[str, Any] = {}
    for member in MEMBERS:
        ratio = frame[f"ratio__{member}"].to_numpy(float)
        volume_bias[member] = {
            "mean": float(np.nanmean(ratio)), "median": float(np.nanmedian(ratio)),
            "std": float(np.nanstd(ratio)),
            "iqr": [float(np.nanpercentile(ratio, 25)),
                    float(np.nanpercentile(ratio, 75))],
            "min": float(np.nanmin(ratio)), "max": float(np.nanmax(ratio)),
            "n_ratio_gt_1.2": int((ratio > 1.2).sum()),
            "n_ratio_gt_1.5": int((ratio > 1.5).sum()),
            "n_ratio_lt_0.8": int((ratio < 0.8).sum()),
        }
    summary["G0_A_volume_bias"] = volume_bias

    correlations: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    pairs = (("ratio", "dice"), ("ratio", "hd95"), ("ratio", "centroid_dist"),
             ("dice", "hd95"), ("dice", "centroid_dist"),
             ("hd95", "centroid_dist"), ("abs_ratio_err", "dice"),
             ("abs_ratio_err", "hd95"), ("min_margin", "dice"),
             ("min_margin", "hd95"), ("min_margin", "abs_ratio_err"),
             ("gtvox", "dice"))
    def column(key: str, member: str) -> str:
        """Member metrics are suffixed; GT-derived geometry is per-case, not
        per-member, and the margin columns carry a ``margin_`` prefix."""
        for candidate in (f"{key}__{member}", key, f"margin_{key}"):
            if candidate in frame.columns:
                return candidate
        raise KeyError(key)

    for member in ("ensemble", "seed42"):
        for left, right in pairs:
            key = f"{left}_vs_{right}__{member}"
            result = spearman(frame[column(left, member)], frame[column(right, member)])
            correlations[key] = result
            if np.isfinite(result["p"]):
                p_values[key] = result["p"]
    adjusted = benjamini_hochberg(p_values)
    for key, value in adjusted.items():
        correlations[key]["bh_fdr"] = value
    summary["G0_B_correlations"] = correlations

    # ---- G0-G size tertiles ------------------------------------------------- #
    sizes = frame["gtvox__ensemble"].to_numpy(float)
    cut_small, cut_large = np.percentile(sizes, [33.333, 66.667])
    groups = np.where(sizes <= cut_small, "small",
                      np.where(sizes <= cut_large, "medium", "large"))
    frame["size_group"] = groups
    size_summary: dict[str, Any] = {"cutpoints_voxels": [float(cut_small), float(cut_large)]}
    for member in MEMBERS:
        entry: dict[str, Any] = {}
        arrays = {}
        for group in ("small", "medium", "large"):
            mask = groups == group
            arrays[group] = frame.loc[mask, f"dice__{member}"].to_numpy(float)
            entry[group] = {
                "n": int(mask.sum()),
                "dice_mean": float(np.nanmean(arrays[group])),
                "hd95_mean": float(frame.loc[mask, f"hd95__{member}"].mean()),
                "precision_mean": float(frame.loc[mask, f"precision__{member}"].mean()),
                "recall_mean": float(frame.loc[mask, f"recall__{member}"].mean()),
                "ratio_mean": float(frame.loc[mask, f"ratio__{member}"].mean()),
            }
        try:
            entry["kruskal_dice"] = {
                "statistic": float(stats.kruskal(*arrays.values()).statistic),
                "p": float(stats.kruskal(*arrays.values()).pvalue)}
        except ValueError as exc:
            entry["kruskal_dice"] = {"error": str(exc)}
        size_summary[member] = entry
    summary["G0_G_size_dependence"] = size_summary

    # ---- G0-C/D spatial maps and class confusion ---------------------------- #
    occ = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.float32)
    fp_map = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.float32)
    fn_map = np.zeros(cs.CROP_SHAPE_DHW, dtype=np.float32)
    confusion = {member: {"fp_from_background": 0, "fp_from_sn": 0, "fp_from_rn": 0,
                          "fn_as_background": 0, "fn_as_sn": 0, "fn_as_rn": 0}
                 for member in MEMBERS}
    for index, case_id in enumerate(case_ids):
        gt = gts[index]
        gt_mask = gt == STN
        occ += gt_mask.astype(np.float32)
        prediction = predictions[index, MEMBERS.index("ensemble")]
        pred_mask = prediction == STN
        fp_map += (pred_mask & ~gt_mask).astype(np.float32)
        fn_map += (~pred_mask & gt_mask).astype(np.float32)
        for member in MEMBERS:
            pm = predictions[index, MEMBERS.index(member)] == STN
            confusion[member]["fp_from_background"] += int((pm & ~gt_mask & (gt == 0)).sum())
            confusion[member]["fp_from_sn"] += int((pm & (gt == 2)).sum())
            confusion[member]["fp_from_rn"] += int((pm & (gt == 3)).sum())
            confusion[member]["fn_as_background"] += int(((predictions[index, MEMBERS.index(member)] == 0) & gt_mask).sum())
            confusion[member]["fn_as_sn"] += int(((predictions[index, MEMBERS.index(member)] == 2) & gt_mask).sum())
            confusion[member]["fn_as_rn"] += int(((predictions[index, MEMBERS.index(member)] == 3) & gt_mask).sum())
    total = float(len(case_ids))
    maps = {"GT STN occupancy": occ / total, "FP frequency": fp_map / total,
            "FN frequency": fn_map / total}
    plot_frequency_maps(maps, out_dir / "stn_fp_frequency_map.png", args.dpi,
                        "Experiment G0 — STN FP/FN frequency (development val, "
                        "ensemble, n=40)")
    plot_frequency_maps({"FN frequency": maps["FN frequency"]},
                        out_dir / "stn_fn_frequency_map.png", args.dpi,
                        "Experiment G0 — STN FN frequency (development val, ensemble)")

    confusion_rows = []
    for member, counts in confusion.items():
        fp_total = sum(counts[k] for k in counts if k.startswith("fp_"))
        fn_total = sum(counts[k] for k in counts if k.startswith("fn_"))
        row = {"member": member, "fp_total": fp_total, "fn_total": fn_total}
        for key, value in counts.items():
            row[key] = value
            row[f"{key}_frac"] = (value / fp_total if key.startswith("fp_") and fp_total
                                  else value / fn_total if fn_total else np.nan)
        confusion_rows.append(row)
    pd.DataFrame(confusion_rows).to_csv(out_dir / "stn_confusion_summary.csv",
                                        index=False, encoding="utf-8-sig")
    summary["G0_D_class_confusion"] = confusion_rows

    # ---- G0-E probability diagnostics --------------------------------------- #
    probability_diagnostics: dict[str, Any] = {"available": False}
    prob_path = (root / PROB_CACHE / "p_stn.npy")
    if args.with_probabilities or prob_path.is_file():
        if args.with_probabilities and not prob_path.is_file():
            build_probability_cache(root, root / PROB_CACHE, device)
        probabilities = np.load(prob_path)
        entry = MEMBERS.index("ensemble")
        gt_stn_values, fp_values, fn_values, bg_values = [], [], [], []
        fp_confidence = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                         "0.6-0.8": 0, "0.8-1.0": 0}
        shell_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
        shell_hits = np.zeros(shell_edges.size - 1)
        shell_totals = np.zeros(shell_edges.size - 1)
        for index in range(len(case_ids)):
            gt_mask = gts[index] == STN
            probability = probabilities[index, entry]
            prediction = predictions[index, entry] == STN
            fp = prediction & ~gt_mask
            fn = ~prediction & gt_mask
            gt_stn_values.append(probability[gt_mask])
            fp_values.append(probability[fp])
            fn_values.append(probability[fn])
            bg_values.append(probability[~gt_mask][::37])   # subsample the background
            for band, (low, high) in zip(fp_confidence, ((0, .2), (.2, .4), (.4, .6),
                                                         (.6, .8), (.8, 1.01))):
                fp_confidence[band] += int(((probability >= low) & (probability < high)
                                            & fp).sum())
            distance = distance_to_mask(gt_mask)
            for shell in range(shell_edges.size - 1):
                shell_mask = (distance >= shell_edges[shell]) & (distance < shell_edges[shell + 1])
                shell_totals[shell] += int((shell_mask & ~gt_mask).sum())
                shell_hits[shell] += int((shell_mask & fp).sum())
        fp_total = sum(fp_confidence.values())
        probability_diagnostics = {
            "available": True,
            "source": "development val, one forward pass; no ground truth read in phase A",
            "p_stn_summary": {},
            "fp_confidence": {k: (v / fp_total if fp_total else 0.0)
                              for k, v in fp_confidence.items()},
            "histograms": {},
            "fp_vs_distance": {
                "bin_centers_mm": [float((shell_edges[i] + shell_edges[i + 1]) / 2)
                                   for i in range(shell_edges.size - 1)],
                "fp_rate": [float(shell_hits[i] / shell_totals[i])
                            if shell_totals[i] else 0.0
                            for i in range(shell_edges.size - 1)],
                "shell_voxel_totals": [int(v) for v in shell_totals],
            },
        }
        for name, values in (("gt_stn", gt_stn_values), ("fp", fp_values),
                             ("fn", fn_values), ("background", bg_values)):
            concatenated = np.concatenate(values) if values else np.array([])
            if concatenated.size:
                counts, edges = np.histogram(concatenated, bins=20, range=(0, 1), density=True)
                probability_diagnostics["histograms"][name] = {
                    "centers": [float((edges[i] + edges[i + 1]) / 2) for i in range(20)],
                    "density": [float(v) for v in counts]}
                probability_diagnostics["p_stn_summary"][name] = {
                    "n_voxels": int(concatenated.size),
                    "mean": float(concatenated.mean()),
                    "median": float(np.median(concatenated)),
                    "p90": float(np.percentile(concatenated, 90)),
                    "frac_above_0.5": float((concatenated > 0.5).mean())}
        plot_probability_diagnostics(probability_diagnostics,
                                     out_dir / "stn_probability_diagnostics.png",
                                     args.dpi)
    summary["G0_E_probability_diagnostics"] = probability_diagnostics

    # ---- G0-H directional shell analysis ------------------------------------ #
    # Decides whether the boundary error is at the sampling limit: the grid is
    # 2.0 mm through-plane (D) but 0.667 mm in-plane (H, W), so a one-voxel
    # displacement costs three times as much in D. If the FP shell is
    # predominantly one voxel in D, the error is a partial-volume / resolution
    # artefact rather than a correctable in-plane boundary offset.
    direction_counts = {f"{axis}{sign}": 0 for axis in "dhw" for sign in ("_plus", "_minus")}
    direction_totals = dict(direction_counts)
    for index in range(len(case_ids)):
        gt_mask = gts[index] == STN
        prediction = predictions[index, MEMBERS.index("ensemble")] == STN
        for axis_position, axis in enumerate("dhw"):
            for sign, shift in (("_plus", 1), ("_minus", -1)):
                moved = np.zeros_like(gt_mask)
                source = [slice(None)] * 3
                target = [slice(None)] * 3
                if shift == 1:
                    source[axis_position] = slice(0, -1)
                    target[axis_position] = slice(1, None)
                else:
                    source[axis_position] = slice(1, None)
                    target[axis_position] = slice(0, -1)
                moved[tuple(target)] = gt_mask[tuple(source)]
                moved &= ~gt_mask                 # exclude voxels that are still GT
                key = f"{axis}{sign}"
                direction_totals[key] += int(moved.sum())
                direction_counts[key] += int((moved & prediction).sum())
    summary["G0_H_directional_shell"] = {
        "note": ("FP rate on the shell one voxel outside the GT STN mask, per "
                 "direction. D is 2.0 mm, H and W are 0.667 mm."),
        "voxel_spacing_mm": {"d": cs.SPACING_DHW_MM[0], "h": cs.SPACING_DHW_MM[1],
                             "w": cs.SPACING_DHW_MM[2]},
        "per_direction": {
            key: {"shell_voxels": int(direction_totals[key]),
                  "fp_voxels": int(direction_counts[key]),
                  "fp_rate": (float(direction_counts[key] / direction_totals[key])
                              if direction_totals[key] else 0.0)}
            for key in direction_counts},
    }

    # ---- G0-F crop margin ---------------------------------------------------- #
    crop_corr = {}
    for member in ("ensemble", "seed42"):
        for left, right in (("min_margin", "dice"), ("min_margin", "hd95"),
                            ("min_margin", "abs_ratio_err")):
            crop_corr[f"{left}_vs_{right}__{member}"] = spearman(
                frame[column(left, member)], frame[column(right, member)])
    summary["G0_F_crop_position"] = {
        "correlations": crop_corr,
        "gt_centroid_distance_to_crop_center_mm": {
            "mean": float(frame["dist_to_crop_center_mm"].mean()),
            "median": float(frame["dist_to_crop_center_mm"].median()),
            "max": float(frame["dist_to_crop_center_mm"].max())},
        "min_margin_voxels": {"min": float(frame["margin_min_margin"].min()),
                              "median": float(frame["margin_min_margin"].median())},
    }
    plot_crop_margin(frame, out_dir / "stn_crop_margin_analysis.png", args.dpi)

    # ---- section 6: ensemble vs baseline ------------------------------------ #
    delta_rows = []
    for index, case_id in enumerate(case_ids):
        row = {"case_id": case_id,
               "baseline_dice": frame.loc[index, "dice__seed42"],
               "ensemble_dice": frame.loc[index, "dice__ensemble"],
               "delta_dice": frame.loc[index, "dice__ensemble"] - frame.loc[index, "dice__seed42"],
               "delta_hd95": frame.loc[index, "hd95__ensemble"] - frame.loc[index, "hd95__seed42"],
               "delta_precision": frame.loc[index, "precision__ensemble"] - frame.loc[index, "precision__seed42"],
               "delta_recall": frame.loc[index, "recall__ensemble"] - frame.loc[index, "recall__seed42"],
               "delta_ratio_error": (frame.loc[index, "abs_ratio_err__ensemble"]
                                     - frame.loc[index, "abs_ratio_err__seed42"]),
               "gt_volume": frame.loc[index, "gtvox__seed42"],
               "baseline_ratio": frame.loc[index, "ratio__seed42"],
               "baseline_hd95": frame.loc[index, "hd95__seed42"],
               "baseline_centroid_dist": frame.loc[index, "centroid_dist__seed42"],
               "min_crop_margin": frame.loc[index, "margin_min_margin"]}
        delta_rows.append(row)
    delta_frame = pd.DataFrame(delta_rows)
    delta_frame.to_csv(out_dir / "ensemble_delta_analysis.csv", index=False,
                       encoding="utf-8-sig")
    delta_corr = {}
    for column in ("baseline_dice", "gt_volume", "baseline_ratio", "baseline_hd95",
                   "baseline_centroid_dist", "min_crop_margin"):
        delta_corr[f"delta_dice_vs_{column}"] = spearman(delta_frame[column],
                                                         delta_frame["delta_dice"])
    summary["section6_ensemble_benefit"] = {
        "mean_delta_dice": float(delta_frame["delta_dice"].mean()),
        "median_delta_dice": float(delta_frame["delta_dice"].median()),
        "n_cases_improved": int((delta_frame["delta_dice"] > 0).sum()),
        "n_cases_worsened": int((delta_frame["delta_dice"] < 0).sum()),
        "delta_dice_by_baseline_quartile": {
            f"Q{q+1}": float(delta_frame.loc[
                pd.qcut(delta_frame["baseline_dice"], 4, labels=False) == q,
                "delta_dice"].mean()) for q in range(4)},
        "correlations": delta_corr,
    }

    plot_volume_bias(frame, out_dir / "stn_volume_bias.png", args.dpi)
    plot_scatter(frame, "ratio", "dice", "STN Dice vs volume ratio (development val)",
                 "Predicted / GT STN volume", "STN Dice",
                 out_dir / "stn_ratio_vs_dice.png", args.dpi, vline=1.0)
    plot_scatter(frame, "ratio", "hd95", "STN HD95 vs volume ratio (development val)",
                 "Predicted / GT STN volume", "STN HD95 (mm)",
                 out_dir / "stn_ratio_vs_hd95.png", args.dpi, vline=1.0)
    plot_scatter(frame, "centroid_dist", "dice",
                 "STN Dice vs centroid distance (development val)",
                 "centroid distance (mm)", "STN Dice",
                 out_dir / "stn_dice_vs_centroid.png", args.dpi)

    (out_dir / "stn_failure_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    # ---- console ----------------------------------------------------------- #
    line = "=" * 96
    print()
    print(line)
    print("Experiment G0 — STN failure-mechanism audit (development val, n=40)")
    print(line)
    for member in ("ensemble", "seed42"):
        v = volume_bias[member]
        print(f"  {member:<9} ratio mean {v['mean']:.3f} median {v['median']:.3f} "
              f"| >1.2 {v['n_ratio_gt_1.2']}/40 | >1.5 {v['n_ratio_gt_1.5']}/40 "
              f"| <0.8 {v['n_ratio_lt_0.8']}/40")
    print(line)
    print("  Spearman (ensemble):")
    for left, right in pairs:
        c = correlations[f"{left}_vs_{right}__ensemble"]
        if not np.isfinite(c["rho"]):
            continue
        fdr = c.get("bh_fdr", float("nan"))
        print(f"    {left:<14} vs {right:<14} rho {c['rho']:+.3f}  p {c['p']:.4g}"
              f"  BH-FDR {fdr:.3g}")
    print(line)
    print("  class confusion (ensemble FP source / FN destination):")
    e = [r for r in confusion_rows if r["member"] == "ensemble"][0]
    print(f"    FP total {e['fp_total']}: background {e['fp_from_background_frac']:.3f} "
          f"SN {e['fp_from_sn_frac']:.3f} RN {e['fp_from_rn_frac']:.3f}")
    print(f"    FN total {e['fn_total']}: background {e['fn_as_background_frac']:.3f} "
          f"SN {e['fn_as_sn_frac']:.3f} RN {e['fn_as_rn_frac']:.3f}")
    print(line)
    if probability_diagnostics.get("available"):
        print("  FP by p_STN band (ensemble):")
        for band, value in probability_diagnostics["fp_confidence"].items():
            print(f"    p_STN {band}: {value:.3f}")
    else:
        print("  probability diagnostics: NOT RUN (pass --with-probabilities)")
    print(line)
    print(f"Written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Baseline v1 error analysis on the frozen development-validation split.

READ-ONLY with respect to the model: the best checkpoint is loaded and used for
inference only. Nothing is retrained, no loss or hyper-parameter is changed, and
no prediction is post-processed. This stage exists purely to explain where the
frozen baseline fails, and in particular why STN is weaker than SN and RN.

Scope: development val (40 cases) only. internal_test and challenge_test are
never touched.

What it produces, under ``results/baseline_v1/error_analysis/``:

    per_case_metrics.csv        per (case, class): volumes, ratio, TP/FP/FN,
                                Dice/Precision/Recall, HD95, centroids, distance
    left_right_stn.csv          per-side STN metrics from raw QSM_mask labels 9/10
    worst_STN_cases.csv         worst 5 by STN Dice
    best_STN_cases.csv          best 5 by STN Dice
    worst_macro_cases.csv       worst 5 by macro foreground Dice
    stn_correlations.json       Pearson + Spearman of STN Dice vs 4 covariates
    error_analysis_summary.json aggregate statistics and failure counts
    *.png                       scatter plots and overlays

Usage
-----
    python scripts/analyze_baseline_errors.py --root .
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import crop_spec as cs  # noqa: E402
from data import pdcadx_io as pio  # noqa: E402
from data.segmentation_dataset import SegmentationDataset  # noqa: E402
from evaluation import segmentation_metrics as sm  # noqa: E402
from models.anisotropic_unet3d import AnisotropicUNet3D, UNet3DConfig  # noqa: E402
from training.train_baseline import (  # noqa: E402
    load_checkpoint,
    load_config,
    resolve_config_paths,
)

LOGGER = logging.getLogger("analyze_baseline_errors")

#: raw QSM_mask label -> anatomical side.
#: Verified empirically: axis 0 axes toward L (left), so a LOWER x index is the
#: patient's RIGHT. Label 9 sits at centroid_x ~= 135, label 10 at ~= 165.
RAW_LABEL_SIDE: dict[int, str] = {9: "right", 10: "left"}

OVER_SEGMENTATION_RATIO = 1.2
UNDER_SEGMENTATION_RATIO = 0.8
STN_LOW_DICE = 0.65
STN_HIGH_DICE = 0.80


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Error analysis of the frozen Baseline v1 on development val.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--config", type=Path, default=Path("configs/subject_clean_v1/baseline_v1.yaml"))
    parser.add_argument("--run-id", type=str, default="formal_seed42")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Default: <checkpoint_dir>/<run-id>/run_best.pt")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <results_dir>/<run-id>/error_analysis")
    parser.add_argument("--split", type=str, default="val",
                        help="Only val is permitted at this stage.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--verbose", "-v", action="store_true")
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


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: SegmentationDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Predictions for every case, keyed by case id. No post-processing."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        argmax = logits.argmax(dim=1).cpu().numpy()
        for index, case_id in enumerate(batch["case_id"]):
            predictions[case_id] = argmax[index].astype(np.uint8)
    return predictions


def raw_side_masks(case_id: str, raw_mask_path: Path) -> dict[str, np.ndarray]:
    """Crop the raw QSM_mask to the frozen crop and split STN into left/right.

    Used for error analysis only. The training labels are untouched: this is a
    separate view of the raw annotation, never fed back into the model.
    """
    raw = np.asanyarray(nib.load(str(raw_mask_path)).dataobj)
    if np.issubdtype(raw.dtype, np.floating):
        raw = np.rint(raw).astype(np.int32)
    cropped = cs.to_tensor_layout(raw.astype(np.int32))
    return {side: (cropped == label) for label, side in RAW_LABEL_SIDE.items()}


def split_prediction_by_side(
    prediction: np.ndarray, side_masks: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float | None]:
    """Split the STN prediction into left/right using a mid-sagittal plane.

    Comparing the WHOLE STN prediction against one side's GT would charge every
    opposite-side voxel as a false positive and report a Dice of ~0.47 for a model
    that is actually doing much better -- the two sides would be indistinguishable
    and the number would be meaningless. So the prediction is first divided at the
    midline between the two GT side centroids, and each half is scored against its
    own side's GT.

    The plane is derived from the raw annotation. The left/right axis is W (the
    third axis of the (D, H, W) = (Z, Y, X) layout).
    """
    pred_stn = prediction == 1
    present = {s: m for s, m in side_masks.items() if m.any()}
    if "left" not in present or "right" not in present:
        return {"left": np.zeros_like(pred_stn), "right": np.zeros_like(pred_stn)}, None

    left_w = pio.voxel_centroid(present["left"])[2]
    right_w = pio.voxel_centroid(present["right"])[2]
    midline = (left_w + right_w) / 2.0

    w_grid = np.arange(pred_stn.shape[2])[None, None, :]
    return {
        "left": pred_stn & (w_grid >= midline),
        "right": pred_stn & (w_grid < midline),
    }, float(midline)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def case_class_metrics(
    prediction: np.ndarray, truth: np.ndarray, class_id: int
) -> dict[str, Any]:
    """Volumes, overlap counts, centroid and centroid distance for one class."""
    pred_mask = prediction == class_id
    true_mask = truth == class_id

    tp = int(np.count_nonzero(pred_mask & true_mask))
    fp = int(np.count_nonzero(pred_mask & ~true_mask))
    fn = int(np.count_nonzero(~pred_mask & true_mask))
    gt_voxels = tp + fn
    pred_voxels = tp + fp

    result: dict[str, Any] = {
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        "volume_ratio": (pred_voxels / gt_voxels) if gt_voxels > 0 else math.nan,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": sm.dice_from_counts(tp, fp, fn),
        "precision": sm.precision_from_counts(tp, fp, fn),
        "recall": sm.recall_from_counts(tp, fp, fn),
        "hd95_mm": sm.hausdorff_95_mm(pred_mask, true_mask, cs.SPACING_DHW_MM),
        "gt_voxels_mm3": gt_voxels * float(np.prod(cs.SPACING_DHW_MM)),
        "pred_voxels_mm3": pred_voxels * float(np.prod(cs.SPACING_DHW_MM)),
    }

    if true_mask.any():
        gt_centroid = pio.voxel_centroid(true_mask)
        result["gt_centroid_dhw"] = "x".join(f"{v:.2f}" for v in gt_centroid)
    else:
        gt_centroid = None
        result["gt_centroid_dhw"] = ""

    if pred_mask.any():
        pred_centroid = pio.voxel_centroid(pred_mask)
        result["pred_centroid_dhw"] = "x".join(f"{v:.2f}" for v in pred_centroid)
    else:
        pred_centroid = None
        result["pred_centroid_dhw"] = ""

    if gt_centroid is not None and pred_centroid is not None:
        delta = np.asarray(pred_centroid) - np.asarray(gt_centroid)
        # Distance in millimetres: D is 2.0 mm, H/W are 0.667 mm. A voxel-space
        # distance would understate through-plane displacement threefold.
        result["centroid_distance_mm"] = float(np.sqrt(np.sum(
            (delta * np.asarray(cs.SPACING_DHW_MM)) ** 2)))
        result["centroid_delta_d"] = float(delta[0])
        result["centroid_delta_h"] = float(delta[1])
        result["centroid_delta_w"] = float(delta[2])
    else:
        result["centroid_distance_mm"] = math.nan
        result["centroid_delta_d"] = math.nan
        result["centroid_delta_h"] = math.nan
        result["centroid_delta_w"] = math.nan

    return result


def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Pearson and Spearman, each with its p-value and sample size."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out: dict[str, Any] = {"n": int(x.size)}
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        out.update({"pearson_r": math.nan, "pearson_p": math.nan,
                    "spearman_rho": math.nan, "spearman_p": math.nan})
        return out
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    out.update({
        "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue),
    })
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def scatter_with_fit(
    x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str,
    out_path: Path, dpi: int, annotate_labels: Sequence[str] | None = None,
) -> None:
    figure, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(x, y, s=42, color="#0a84ff", edgecolor="white", linewidth=0.7, zorder=3)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3 and np.ptp(x[mask]) > 0:
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 50)
        ax.plot(xs, slope * xs + intercept, color="#ff3b30", lw=1.6,
                linestyle="--", zorder=2)
        r = stats.pearsonr(x[mask], y[mask])
        rho = stats.spearmanr(x[mask], y[mask])
        ax.set_title(f"{title}\nPearson r = {r.statistic:.3f} (p={r.pvalue:.3g})   "
                     f"Spearman rho = {rho.statistic:.3f} (p={rho.pvalue:.3g})")
    else:
        ax.set_title(title)

    if annotate_labels is not None:
        for xi, yi, label in zip(x, y, annotate_labels):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(label, (xi, yi), fontsize=6.5, alpha=0.75,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_distribution(frame: pd.DataFrame, out_path: Path, dpi: int) -> None:
    figure, panels = plt.subplots(1, 2, figsize=(13, 5))
    colours = {"STN": "#ff3b30", "SN": "#34c759", "RN": "#0a84ff"}

    dice_data = [frame[f"dice_{n}"].dropna().values for n in cs.CLASS_NAMES.values()]
    box = panels[0].boxplot(dice_data, tick_labels=list(cs.CLASS_NAMES.values()),
                            patch_artist=True, widths=0.55, showmeans=True)
    for patch, name in zip(box["boxes"], cs.CLASS_NAMES.values()):
        patch.set_facecolor(colours[name])
        patch.set_alpha(0.55)
    panels[0].axhline(1.0, color="grey", linestyle=":", lw=1)
    panels[0].set_ylabel("Dice")
    panels[0].set_title("Per-class Dice across the 40 validation cases")
    panels[0].set_ylim(0, 1.05)
    panels[0].grid(alpha=0.25, axis="y")

    ratio_data = [frame[f"volume_ratio_{n}"].dropna().values for n in cs.CLASS_NAMES.values()]
    box2 = panels[1].boxplot(ratio_data, tick_labels=list(cs.CLASS_NAMES.values()),
                             patch_artist=True, widths=0.55, showmeans=True)
    for patch, name in zip(box2["boxes"], cs.CLASS_NAMES.values()):
        patch.set_facecolor(colours[name])
        patch.set_alpha(0.55)
    panels[1].axhline(1.0, color="grey", linestyle=":", lw=1)
    panels[1].axhspan(UNDER_SEGMENTATION_RATIO, OVER_SEGMENTATION_RATIO,
                      color="green", alpha=0.08)
    panels[1].set_ylabel("predicted volume / GT volume")
    panels[1].set_title("Per-class volume ratio (shaded band = 0.8-1.2)")
    panels[1].grid(alpha=0.25, axis="y")

    figure.suptitle("Baseline v1 error analysis — development val (n=40)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_case_overlay(
    case_id: str,
    image: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    metrics_row: pd.Series,
    out_path: Path,
    dpi: int,
) -> None:
    """Axial / coronal / sagittal views of GT vs prediction for one case.

    The tensors are (D, H, W) = (Z, Y, X). Each view is produced by slicing the
    axis being collapsed and taking the centroid of the GT as the slice index,
    so the structures of interest are actually visible.
    """
    colours = {"STN": "#ff3b30", "SN": "#34c759", "RN": "#0a84ff"}

    # Choose, per axis, the slice containing the MOST STN — not the union
    # centroid and not the centroid of STN itself.
    #
    # STN is a small structure (~5 voxels tall in D) that sits a couple of voxels
    # above the union centroid of STN/SN/RN, so a slice chosen from the union
    # centroid can miss it entirely and the red contour silently disappears from
    # an "STN error" figure. A centroid can also fall between slices. Taking the
    # arg-max of the STN mask guarantees the structure is actually on screen.
    stn_truth = truth == 1
    index: list[int] = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        if stn_truth.any():
            profile = stn_truth.sum(axis=other)
            index.append(int(np.argmax(profile)))
        else:
            index.append(truth.shape[axis] // 2)
    if not stn_truth.any():
        LOGGER.warning("%s: no STN in GT; using mid-volume slices", case_id)

    views = (
        ("axial", 0, lambda a: a[index[0]]),
        ("coronal", 1, lambda a: a[:, index[1], :]),
        ("sagittal", 2, lambda a: a[:, :, index[2]]),
    )

    figure, panels = plt.subplots(2, 3, figsize=(13.5, 9.0))
    for column, (name, _axis, take) in enumerate(views):
        background = take(image)
        gt_slice = take(truth)
        pred_slice = take(prediction)

        for row, (label, mask, style) in enumerate(
            (("ground truth", gt_slice, "solid"),
             ("prediction", pred_slice, "dashed"))
        ):
            ax = panels[row][column]
            ax.imshow(background, cmap="gray", origin="lower", aspect="equal")
            for class_id, class_name in cs.CLASS_NAMES.items():
                layer = mask == class_id
                if layer.any():
                    ax.contour(layer.astype(float), levels=[0.5],
                               colors=[colours[class_name]], linewidths=1.3,
                               linestyles=style, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{name} — {label}", fontsize=9)

    dice = metrics_row.get("dice_STN", math.nan)
    precision = metrics_row.get("precision_STN", math.nan)
    recall = metrics_row.get("recall_STN", math.nan)
    # Column keys come from case_class_metrics(); HD95 is "hd95_mm_STN", not
    # "hd95_STN_mm" -- reading the wrong key silently rendered "nan".
    hd95 = metrics_row.get("hd95_mm_STN", math.nan)
    gt_vol = metrics_row.get("gt_voxels_STN", math.nan)
    pred_vol = metrics_row.get("pred_voxels_STN", math.nan)
    ratio = metrics_row.get("volume_ratio_STN", math.nan)

    figure.suptitle(
        f"{case_id} — STN Dice {dice:.3f} | Precision {precision:.3f} | "
        f"Recall {recall:.3f} | HD95 {hd95:.2f} mm\n"
        f"GT volume {gt_vol:.0f} vox | predicted {pred_vol:.0f} vox | "
        f"ratio {ratio:.2f}",
        fontsize=12,
    )
    handles = [Line2D([0], [0], color=colours[n], lw=2, label=n)
               for n in cs.CLASS_NAMES.values()]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.02, 1, 0.93))
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()

    if args.split != "val":
        print(f"error: only --split val is permitted at this stage "
              f"(got {args.split!r}); internal_test and challenge_test are frozen.",
              file=sys.stderr)
        return 2

    # Same rule as the trainer: a missing config is an error, never a silent
    # fall back to the dataclass defaults.
    if not args.config.is_file():
        print(f"\nerror: config file not found: {args.config}\n"
              f"       Refusing to fall back to built-in defaults.\n",
              file=sys.stderr)
        return 4
    config = load_config(args.config)
    resolve_config_paths(config, root)

    results_dir = Path(config.results_dir) / args.run_id
    checkpoint = args.checkpoint or (Path(config.checkpoint_dir) / args.run_id / "run_best.pt")
    out_dir = args.out_dir or (results_dir / "error_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "analyze_baseline_errors.log", args.verbose)

    if not checkpoint.is_file():
        LOGGER.error("Checkpoint not found: %s", checkpoint)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)
    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Split: %s (development validation only)", args.split)

    # ---- model + data (inference only) ------------------------------------ #
    payload = load_checkpoint(checkpoint)
    stored = payload.get("config", {}) or {}
    model = AnisotropicUNet3D(UNet3DConfig(
        in_channels=int(stored.get("in_channels", config.in_channels)),
        num_classes=int(stored.get("num_classes", config.num_classes)),
        base_channels=int(stored.get("base_channels", config.base_channels)),
    )).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    LOGGER.info("Loaded best checkpoint from epoch %s (best metric %.4f)",
                payload.get("epoch"), payload.get("best_metric", float("nan")))

    dataset = SegmentationDataset(root, args.split, cache_dir=config.cache_dir,
                                  manifest_dir=config.manifest_dir)
    predictions = run_inference(model, dataset, config.batch_size, device)
    LOGGER.info("Predicted %d cases", len(predictions))

    cases_by_id = {c.case_id: c for c in pio.discover_cases(root)}

    # ---- per-case per-class metrics --------------------------------------- #
    rows: list[dict[str, Any]] = []
    left_right_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []

    for case_id in dataset.case_ids:
        try:
            index = dataset.case_ids.index(case_id)
            truth = dataset[index]["label"].numpy()
            prediction = predictions[case_id]

            row: dict[str, Any] = {"case_id": case_id}
            for class_id, name in cs.CLASS_NAMES.items():
                metrics = case_class_metrics(prediction, truth, class_id)
                for key, value in metrics.items():
                    if key in ("gt_centroid_dhw", "pred_centroid_dhw"):
                        row[f"{key.replace('_dhw','')}_{name}"] = value
                    else:
                        row[f"{key}_{name}"] = value

            union_pred = np.isin(prediction, list(cs.CLASS_NAMES))
            union_true = np.isin(truth, list(cs.CLASS_NAMES))
            row["macro_foreground_dice"] = float(np.nanmean(
                [row[f"dice_{n}"] for n in cs.CLASS_NAMES.values()]))

            # Union centroid (for slice selection in the overlays).
            if union_true.any():
                row["union_gt_centroid_dhw"] = "x".join(
                    f"{v:.2f}" for v in pio.voxel_centroid(union_true))
            rows.append(row)

            # ---- left / right STN from the RAW annotation ------------------ #
            record = cases_by_id.get(case_id)
            if record is not None and "QSM_mask" in record.files:
                sides = raw_side_masks(case_id, record.files["QSM_mask"])
                pred_sides, midline = split_prediction_by_side(prediction, sides)
                for side, gt_side in sorted(sides.items()):
                    pred_side = pred_sides[side]
                    tp = int(np.count_nonzero(pred_side & gt_side))
                    fp = int(np.count_nonzero(pred_side & ~gt_side))
                    fn = int(np.count_nonzero(~pred_side & gt_side))
                    # Count of this side's GT that the model placed on the OTHER
                    # side (a side-assignment error rather than a miss).
                    crossed = int(np.count_nonzero(
                        pred_sides["right" if side == "left" else "left"] & gt_side))
                    left_right_rows.append({
                        "case_id": case_id,
                        "side": side,
                        "raw_label": next(k for k, v in RAW_LABEL_SIDE.items() if v == side),
                        "midline_w": midline,
                        "gt_voxels": tp + fn,
                        "pred_voxels": tp + fp,
                        "tp": tp, "fp": fp, "fn": fn,
                        "gt_on_opposite_side": crossed,
                        "dice": sm.dice_from_counts(tp, fp, fn),
                        "precision": sm.precision_from_counts(tp, fp, fn),
                        "recall": sm.recall_from_counts(tp, fp, fn),
                        "hd95_mm": sm.hausdorff_95_mm(pred_side, gt_side, cs.SPACING_DHW_MM),
                    })
        except Exception as exc:  # noqa: BLE001 - one bad case must not abort
            LOGGER.error("Case %s failed: %s", case_id, exc)
            LOGGER.debug("%s", traceback.format_exc())
            warnings.append({"case_id": case_id, "warning_type": "analysis_failed",
                             "description": f"{type(exc).__name__}: {exc}"})

    if not rows:
        LOGGER.error("No cases analysed.")
        return 1

    frame = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    frame.to_csv(out_dir / "per_case_metrics.csv", index=False, encoding="utf-8-sig")

    left_right = pd.DataFrame(left_right_rows)
    if not left_right.empty:
        left_right.to_csv(out_dir / "left_right_stn.csv", index=False, encoding="utf-8-sig")

    # ---- rankings ---------------------------------------------------------- #
    frame.sort_values("dice_STN").head(5).to_csv(
        out_dir / "worst_STN_cases.csv", index=False, encoding="utf-8-sig")
    frame.sort_values("dice_STN", ascending=False).head(5).to_csv(
        out_dir / "best_STN_cases.csv", index=False, encoding="utf-8-sig")
    frame.sort_values("macro_foreground_dice").head(5).to_csv(
        out_dir / "worst_macro_cases.csv", index=False, encoding="utf-8-sig")

    # ---- STN correlations -------------------------------------------------- #
    label_list = frame["case_id"].tolist()
    stn_correlations = {
        "stn_dice_vs_gt_volume": correlations(
            frame["gt_voxels_STN"].to_numpy(float), frame["dice_STN"].to_numpy(float)),
        "stn_dice_vs_volume_ratio": correlations(
            frame["volume_ratio_STN"].to_numpy(float), frame["dice_STN"].to_numpy(float)),
        "stn_dice_vs_centroid_distance": correlations(
            frame["centroid_distance_mm_STN"].to_numpy(float), frame["dice_STN"].to_numpy(float)),
        "stn_dice_vs_hd95": correlations(
            frame["hd95_mm_STN"].to_numpy(float), frame["dice_STN"].to_numpy(float)),
        "stn_precision_vs_recall": correlations(
            frame["recall_STN"].to_numpy(float), frame["precision_STN"].to_numpy(float)),
    }
    with (out_dir / "stn_correlations.json").open("w", encoding="utf-8") as handle:
        json.dump(stn_correlations, handle, indent=2, ensure_ascii=False, default=str)

    # ---- summary ----------------------------------------------------------- #
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "split": args.split,
        "n_cases": int(len(frame)),
        "note": "Analysis only. No retraining, no loss change, no post-processing.",
        "per_class": {},
        "stn_focus": {},
        "left_right_stn": {},
        "correlations": stn_correlations,
        "worst_5_stn": frame.sort_values("dice_STN").head(5)["case_id"].tolist(),
        "best_5_stn": frame.sort_values("dice_STN", ascending=False).head(5)["case_id"].tolist(),
        "worst_5_macro": frame.sort_values("macro_foreground_dice").head(5)["case_id"].tolist(),
    }

    for name in cs.CLASS_NAMES.values():
        dice = frame[f"dice_{name}"]
        ratio = frame[f"volume_ratio_{name}"]
        summary["per_class"][name] = {
            "dice": {"mean": float(dice.mean()), "std": float(dice.std()),
                     "median": float(dice.median()), "min": float(dice.min()),
                     "max": float(dice.max())},
            "precision_mean": float(frame[f"precision_{name}"].mean()),
            "recall_mean": float(frame[f"recall_{name}"].mean()),
            "hd95_mm_mean": float(frame[f"hd95_mm_{name}"].mean()),
            "gt_volume": {"mean": float(frame[f"gt_voxels_{name}"].mean()),
                          "std": float(frame[f"gt_voxels_{name}"].std()),
                          "median": float(frame[f"gt_voxels_{name}"].median())},
            "pred_volume": {"mean": float(frame[f"pred_voxels_{name}"].mean()),
                            "std": float(frame[f"pred_voxels_{name}"].std()),
                            "median": float(frame[f"pred_voxels_{name}"].median())},
            "volume_ratio": {"mean": float(ratio.mean()), "std": float(ratio.std()),
                             "median": float(ratio.median())},
            "fp_total": int(frame[f"fp_{name}"].sum()),
            "fn_total": int(frame[f"fn_{name}"].sum()),
            "centroid_distance_mm": {
                "mean": float(frame[f"centroid_distance_mm_{name}"].mean()),
                "std": float(frame[f"centroid_distance_mm_{name}"].std()),
                "median": float(frame[f"centroid_distance_mm_{name}"].median()),
            },
        }

    ratio = frame["volume_ratio_STN"]
    summary["stn_focus"] = {
        "over_segmentation_cases": int((ratio > OVER_SEGMENTATION_RATIO).sum()),
        "under_segmentation_cases": int((ratio < UNDER_SEGMENTATION_RATIO).sum()),
        "within_0.8_1.2_cases": int(((ratio >= UNDER_SEGMENTATION_RATIO)
                                     & (ratio <= OVER_SEGMENTATION_RATIO)).sum()),
        "ratio_thresholds": [UNDER_SEGMENTATION_RATIO, OVER_SEGMENTATION_RATIO],
        "dice_below_0.65_cases": int((frame["dice_STN"] < STN_LOW_DICE).sum()),
        "dice_above_0.80_cases": int((frame["dice_STN"] > STN_HIGH_DICE).sum()),
        "precision_lt_recall_cases": int(
            (frame["precision_STN"] < frame["recall_STN"]).sum()),
        "fp_total": int(frame["fp_STN"].sum()),
        "fn_total": int(frame["fn_STN"].sum()),
    }

    if not left_right.empty:
        per_side = {}
        for side, group in left_right.groupby("side"):
            per_side[side] = {
                "raw_label": int(group["raw_label"].iloc[0]),
                "n_cases": int(len(group)),
                "dice_mean": float(group["dice"].mean()),
                "dice_std": float(group["dice"].std()),
                "precision_mean": float(group["precision"].mean()),
                "recall_mean": float(group["recall"].mean()),
                "gt_volume_mean": float(group["gt_voxels"].mean()),
                "pred_volume_mean": float(group["pred_voxels"].mean()),
                "gt_on_opposite_side_total": int(group["gt_on_opposite_side"].sum()),
            }
        summary["left_right_stn"] = per_side
        pivot = left_right.pivot(index="case_id", columns="side", values="dice")
        if {"left", "right"}.issubset(pivot.columns):
            difference = (pivot["left"] - pivot["right"]).dropna()
            summary["left_right_stn_diff"] = {
                "mean_left_minus_right": float(difference.mean()),
                "std": float(difference.std()),
                "n_left_better": int((difference > 0).sum()),
                "n_right_better": int((difference < 0).sum()),
                # Paired test: the two sides come from the same subject, so a
                # paired test is the right comparison, not an unpaired one.
                "paired_ttest_p": float(stats.ttest_rel(
                    pivot["left"].dropna(), pivot["right"].dropna()).pvalue),
            }

    with (out_dir / "error_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    if warnings:
        pd.DataFrame(warnings).to_csv(out_dir / "analysis_warnings.csv",
                                      index=False, encoding="utf-8-sig")

    # ---- STN focus table --------------------------------------------------- #
    stn_columns = ["case_id", "dice_STN", "precision_STN", "recall_STN",
                   "volume_ratio_STN", "gt_voxels_STN", "pred_voxels_STN",
                   "fp_STN", "fn_STN", "hd95_mm_STN", "centroid_distance_mm_STN"]
    frame[stn_columns].sort_values("dice_STN").to_csv(
        out_dir / "stn_case_table.csv", index=False, encoding="utf-8-sig")

    # ---- figures ----------------------------------------------------------- #
    try:
        scatter_with_fit(
            frame["gt_voxels_STN"].to_numpy(float), frame["dice_STN"].to_numpy(float),
            "STN GT volume (voxels)", "STN Dice",
            "STN Dice vs GT volume", out_dir / "STN_dice_vs_gt_volume.png",
            args.dpi, label_list)
        scatter_with_fit(
            frame["volume_ratio_STN"].to_numpy(float), frame["dice_STN"].to_numpy(float),
            "predicted / GT volume", "STN Dice",
            "STN Dice vs volume ratio", out_dir / "STN_dice_vs_volume_ratio.png",
            args.dpi, label_list)
        scatter_with_fit(
            frame["recall_STN"].to_numpy(float), frame["precision_STN"].to_numpy(float),
            "STN Recall", "STN Precision",
            "STN Precision vs Recall", out_dir / "STN_precision_vs_recall.png",
            args.dpi, label_list)
        scatter_with_fit(
            frame["hd95_mm_STN"].to_numpy(float), frame["dice_STN"].to_numpy(float),
            "STN HD95 (mm)", "STN Dice",
            "STN Dice vs HD95", out_dir / "STN_dice_vs_hd95.png",
            args.dpi, label_list)
        plot_per_class_distribution(
            frame, out_dir / "per_class_dice_distribution.png", args.dpi)
        LOGGER.info("Wrote 5 analysis figures")
    except Exception as exc:  # noqa: BLE001 - figures must not fail the analysis
        LOGGER.warning("Figure rendering failed: %s", exc)
        LOGGER.debug("%s", traceback.format_exc())

    # ---- per-case overlays ------------------------------------------------- #
    overlay_dir = out_dir / "case_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    try:
        worst = frame.sort_values("dice_STN").head(5)
        median_row = frame.iloc[(frame["dice_STN"] - frame["dice_STN"].median())
                                .abs().argsort()[:3]]
        chosen = [(cid, "worst") for cid in worst["case_id"]] + \
                 [(cid, "typical") for cid in median_row["case_id"]]
        for case_id, tag in chosen:
            index = dataset.case_ids.index(case_id)
            sample = dataset[index]
            image = sample["image"][0].numpy()     # T1 channel, (D, H, W)
            truth = sample["label"].numpy()
            metrics_row = frame[frame["case_id"] == case_id].iloc[0]
            plot_case_overlay(case_id, image, truth, predictions[case_id], metrics_row,
                              overlay_dir / f"{tag}_{case_id}_STN.png", args.dpi)
        LOGGER.info("Wrote %d case overlay figures to %s", len(chosen), overlay_dir)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Overlay rendering failed: %s", exc)
        LOGGER.debug("%s", traceback.format_exc())

    # ---- console summary --------------------------------------------------- #
    line = "=" * 74
    print()
    print(line)
    print("Baseline v1 Error Analysis — development val (n=%d)" % len(frame))
    print(line)
    header = f"{'class':<6}{'Dice':>18}{'Prec':>9}{'Rec':>9}{'VolRatio':>11}{'FP':>8}{'FN':>8}{'CDist mm':>10}"
    print(header)
    for name in cs.CLASS_NAMES.values():
        entry = summary["per_class"][name]
        print(f"{name:<6}{entry['dice']['mean']:>10.4f} ± {entry['dice']['std']:<6.4f}"
              f"{entry['precision_mean']:>9.3f}{entry['recall_mean']:>9.3f}"
              f"{entry['volume_ratio']['mean']:>11.3f}"
              f"{entry['fp_total']:>8}{entry['fn_total']:>8}"
              f"{entry['centroid_distance_mm']['mean']:>10.2f}")
    stn = summary["stn_focus"]
    print()
    print(f"STN over-segmentation (ratio > {OVER_SEGMENTATION_RATIO}): {stn['over_segmentation_cases']}")
    print(f"STN under-segmentation (ratio < {UNDER_SEGMENTATION_RATIO}): {stn['under_segmentation_cases']}")
    print(f"STN within 0.8-1.2            : {stn['within_0.8_1.2_cases']}")
    print(f"STN Dice < 0.65               : {stn['dice_below_0.65_cases']}")
    print(f"STN Dice > 0.80               : {stn['dice_above_0.80_cases']}")
    print(f"STN precision < recall cases  : {stn['precision_lt_recall_cases']}")
    if "left_right_stn" in summary and summary["left_right_stn"]:
        print()
        for side, entry in sorted(summary["left_right_stn"].items()):
            print(f"  {side:<6} STN (raw label {entry['raw_label']}): "
                  f"Dice {entry['dice_mean']:.4f} ± {entry['dice_std']:.4f}  "
                  f"GT vol {entry['gt_volume_mean']:.0f}  pred vol {entry['pred_volume_mean']:.0f}")
        diff = summary.get("left_right_stn_diff")
        if diff:
            print(f"  left - right Dice: {diff['mean_left_minus_right']:+.4f} "
                  f"(paired t-test p={diff['paired_ttest_p']:.3g}; "
                  f"left better in {diff['n_left_better']}, right better in {diff['n_right_better']})")
    print()
    print(f"worst 5 STN : {summary['worst_5_stn']}")
    print(f"best 5 STN  : {summary['best_5_stn']}")
    print(f"worst 5 macro: {summary['worst_5_macro']}")
    print(line)
    print(f"Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

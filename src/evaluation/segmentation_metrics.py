"""Segmentation metrics for Baseline v1.

Two rules shape this module:

1. **Everything is computed per case, then averaged over cases.**
   Flattening a batch and computing one pooled Dice gives a different (and wrong)
   number: a case with a large structure would dominate a case with a small one.
   ``evaluate_case`` handles one case; ``aggregate_cases`` reports mean and std
   across cases. The batch is never pooled.

2. **Background is never part of the macro Dice.**
   Background is >99% of the crop, so including it produces a number that is
   always ~0.99 and tells you nothing. ``macro_foreground_dice`` averages only
   STN, SN and RN.

HD95, when requested, is measured in **millimetres** using the true anisotropic
spacing ``(2.0, 0.667, 0.667)`` for ``(D, H, W)``. Voxel-unit distances would
understate through-plane error by roughly 3x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy import ndimage

from data import crop_spec as cs


# --------------------------------------------------------------------------- #
# Per-case metrics
# --------------------------------------------------------------------------- #


@dataclass
class CaseMetrics:
    """Metrics for a single case."""

    case_id: str
    dice: dict[int, float]
    precision: dict[int, float]
    recall: dict[int, float]
    tp: dict[int, int]
    fp: dict[int, int]
    fn: dict[int, int]
    macro_foreground_dice: float
    hd95_mm: dict[int, float] | None = None
    #: classes where the prediction was empty (TP + FP == 0)
    empty_prediction: dict[int, bool] = field(default_factory=dict)
    #: classes genuinely absent from the GT
    empty_gt: dict[int, bool] = field(default_factory=dict)

    def to_per_case_csv_row(self) -> dict[str, Any]:
        """Row for the reported per-case evaluation CSV.

        Uses the capitalised column names required by the evaluation spec
        (``Dice_STN``, ``HD95_STN_mm``, ...). :meth:`to_row` keeps the lowercase
        keys used internally by aggregation and the training history.
        """
        row: dict[str, Any] = {"case_id": self.case_id}
        for class_id, name in cs.CLASS_NAMES.items():
            row[f"Dice_{name}"] = self.dice.get(class_id, math.nan)
            row[f"Precision_{name}"] = self.precision.get(class_id, math.nan)
            row[f"Recall_{name}"] = self.recall.get(class_id, math.nan)
            row[f"HD95_{name}_mm"] = (self.hd95_mm or {}).get(class_id, math.nan)
            # Diagnostics kept alongside so a low score can be explained without
            # re-running inference.
            row[f"TP_{name}"] = self.tp.get(class_id, 0)
            row[f"FP_{name}"] = self.fp.get(class_id, 0)
            row[f"FN_{name}"] = self.fn.get(class_id, 0)
            # Explicit voxel counts: a Dice of 0 is ambiguous between "predicted
            # nothing" and "predicted the wrong place", and those need very
            # different responses.
            row[f"PredVoxels_{name}"] = int(self.tp.get(class_id, 0) + self.fp.get(class_id, 0))
            row[f"GTVoxels_{name}"] = int(self.tp.get(class_id, 0) + self.fn.get(class_id, 0))
            row[f"empty_prediction_{name}"] = bool(
                self.empty_prediction.get(class_id, False)
            )
            row[f"empty_gt_{name}"] = bool(self.empty_gt.get(class_id, False))
        row["macro_foreground_dice"] = self.macro_foreground_dice
        return row

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"case_id": self.case_id}
        for class_id, name in cs.CLASS_NAMES.items():
            row[f"dice_{name}"] = self.dice.get(class_id, math.nan)
            row[f"precision_{name}"] = self.precision.get(class_id, math.nan)
            row[f"recall_{name}"] = self.recall.get(class_id, math.nan)
            row[f"tp_{name}"] = self.tp.get(class_id, 0)
            row[f"fp_{name}"] = self.fp.get(class_id, 0)
            row[f"fn_{name}"] = self.fn.get(class_id, 0)
            row[f"empty_prediction_{name}"] = bool(self.empty_prediction.get(class_id, False))
            row[f"empty_gt_{name}"] = bool(self.empty_gt.get(class_id, False))
            if self.hd95_mm:
                row[f"hd95_{name}_mm"] = self.hd95_mm.get(class_id, math.nan)
        row["macro_foreground_dice"] = self.macro_foreground_dice
        return row


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    """Dice = 2TP / (2TP + FP + FN).

    An empty prediction against a non-empty GT scores **0.0**, not NaN. This
    matters: NaN would be dropped by the aggregator, so a model that predicted
    nothing at all would have its failures silently removed from the mean and
    look *better* than one that tried and partly succeeded.

    NaN is reserved for a class genuinely absent from both prediction and GT,
    where the metric is undefined rather than zero.
    """
    denom = 2 * tp + fp + fn
    if denom == 0:
        return math.nan
    return float(2 * tp) / float(denom)


def precision_from_counts(tp: int, fp: int, fn: int) -> float:
    """Precision = TP / (TP + FP).

    With no positive predictions the ratio is 0/0. That is **0.0** when the GT
    actually contains the class (a total miss must not score well), and NaN only
    when the class is absent from the GT as well, where precision is undefined.
    """
    if tp + fp == 0:
        return 0.0 if (tp + fn) > 0 else math.nan
    return float(tp) / float(tp + fp)


def recall_from_counts(tp: int, fp: int, fn: int) -> float:
    """Recall = TP / (TP + FN). NaN only when the GT contains no such class."""
    if tp + fn == 0:
        return math.nan
    return float(tp) / float(tp + fn)


def evaluate_case(
    logits: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    case_id: str = "",
    compute_hd95: bool = False,
) -> CaseMetrics:
    """Metrics for ONE case.

    ``logits``: (C, D, H, W) raw scores, or (N, C, D, H, W) with N == 1.
    ``target``: (D, H, W) integer labels, or (N, D, H, W) with N == 1.
    """
    logits_np = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    target_np = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)

    if logits_np.ndim == 5:
        if logits_np.shape[0] != 1:
            raise ValueError(
                f"evaluate_case expects a single case, got batch of {logits_np.shape[0]}. "
                f"Metrics must be computed per case, never pooled over a batch."
            )
        logits_np = logits_np[0]
    if target_np.ndim == 4:
        if target_np.shape[0] != 1:
            raise ValueError(
                f"evaluate_case expects a single case, got batch of {target_np.shape[0]}."
            )
        target_np = target_np[0]

    prediction = logits_np.argmax(axis=0)

    dice: dict[int, float] = {}
    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    tp: dict[int, int] = {}
    fp: dict[int, int] = {}
    fn: dict[int, int] = {}
    empty_prediction: dict[int, bool] = {}
    empty_gt: dict[int, bool] = {}

    for class_id in cs.FOREGROUND_CLASSES:
        pred_c = prediction == class_id
        true_c = target_np == class_id
        tp_c = int(np.count_nonzero(pred_c & true_c))
        fp_c = int(np.count_nonzero(pred_c & ~true_c))
        fn_c = int(np.count_nonzero(~pred_c & true_c))
        tp[class_id] = tp_c
        fp[class_id] = fp_c
        fn[class_id] = fn_c
        empty_prediction[class_id] = (tp_c + fp_c) == 0
        empty_gt[class_id] = (tp_c + fn_c) == 0
        dice[class_id] = dice_from_counts(tp_c, fp_c, fn_c)
        precision[class_id] = precision_from_counts(tp_c, fp_c, fn_c)
        recall[class_id] = recall_from_counts(tp_c, fp_c, fn_c)

    # Macro over foreground ONLY. Background is deliberately not a member.
    valid = [dice[c] for c in cs.FOREGROUND_CLASSES if not math.isnan(dice[c])]
    macro = float(np.mean(valid)) if valid else math.nan

    hd95: dict[int, float] | None = None
    if compute_hd95:
        hd95 = {}
        for class_id in cs.FOREGROUND_CLASSES:
            hd95[class_id] = hausdorff_95_mm(
                prediction == class_id, target_np == class_id, cs.SPACING_DHW_MM
            )

    return CaseMetrics(case_id=case_id, dice=dice, precision=precision, recall=recall,
                       tp=tp, fp=fp, fn=fn, macro_foreground_dice=macro, hd95_mm=hd95,
                       empty_prediction=empty_prediction, empty_gt=empty_gt)


# --------------------------------------------------------------------------- #
# HD95 in millimetres
# --------------------------------------------------------------------------- #


def hausdorff_95_mm(
    pred_mask: np.ndarray, true_mask: np.ndarray, spacing: Sequence[float]
) -> float:
    """Symmetric 95th-percentile surface distance, in millimetres.

    ``spacing`` must be the physical voxel size in the same axis order as the
    masks. Passing ``sampling=spacing`` makes ``distance_transform_edt`` return
    distances in mm rather than voxels, which matters a lot here: a 1-voxel error
    along D is 2.0 mm but only 0.667 mm along H/W.
    """
    pred_mask = np.asarray(pred_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)
    if not pred_mask.any() or not true_mask.any():
        return math.nan

    structure = ndimage.generate_binary_structure(pred_mask.ndim, 1)
    pred_surface = pred_mask & ~ndimage.binary_erosion(pred_mask, structure=structure)
    true_surface = true_mask & ~ndimage.binary_erosion(true_mask, structure=structure)
    if not pred_surface.any() or not true_surface.any():
        return math.nan

    dt_to_true = ndimage.distance_transform_edt(~true_surface, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    d_pred_to_true = dt_to_true[pred_surface]
    d_true_to_pred = dt_to_pred[true_surface]
    if d_pred_to_true.size == 0 or d_true_to_pred.size == 0:
        return math.nan
    return float(max(np.percentile(d_pred_to_true, 95), np.percentile(d_true_to_pred, 95)))


# --------------------------------------------------------------------------- #
# Aggregation across cases
# --------------------------------------------------------------------------- #

AGGREGATE_KEYS: tuple[str, ...] = tuple(
    [f"dice_{n}" for n in cs.CLASS_NAMES.values()]
    + [f"precision_{n}" for n in cs.CLASS_NAMES.values()]
    + [f"recall_{n}" for n in cs.CLASS_NAMES.values()]
    + ["macro_foreground_dice"]
)

#: HD95 keys are aggregated separately: they are only computed on the final
#: best-checkpoint pass, so a run that never requests them simply has none.
HD95_KEYS: tuple[str, ...] = tuple(f"hd95_{n}_mm" for n in cs.CLASS_NAMES.values())


def _summarise(values: np.ndarray, n_cases: int) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        # n_cases is the full cohort; n_valid is how many contributed a finite
        # value. They differ only where a metric is undefined for some cases
        # (e.g. HD95 when nothing was predicted), and that gap is the point.
        "n_cases": int(n_cases),
        "n_valid": int(values.size),
    }


def aggregate_cases(
    case_metrics: Iterable[CaseMetrics], include_hd95: bool = True
) -> dict[str, dict[str, float]]:
    """Mean/std/median/min/max over cases, per metric.

    ``std`` is the population standard deviation across cases (ddof=0), i.e. the
    spread of per-case performance. It is NOT the std over voxels.

    HD95 is summarised over the cases where it is defined (``n_valid`` says how
    many). Empty predictions are **not** silently dropped: they are counted in
    ``empty_prediction_<class>`` so a model that predicts nothing cannot look
    good by having its failures excluded.
    """
    metrics_list = list(case_metrics)
    rows = [m.to_row() for m in metrics_list]
    if not rows:
        return {}

    result: dict[str, dict[str, float]] = {}
    for key in AGGREGATE_KEYS:
        values = np.array([r[key] for r in rows if key in r], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        result[key] = _summarise(values, len(metrics_list))

    if include_hd95:
        for key in HD95_KEYS:
            values = np.array([r[key] for r in rows if key in r], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            result[key] = _summarise(values, len(metrics_list))

    # Per-class failure counts, reported rather than dropped.
    for class_id, name in cs.CLASS_NAMES.items():
        n_empty_pred = sum(1 for m in metrics_list
                           if m.empty_prediction.get(class_id, False))
        n_empty_gt = sum(1 for m in metrics_list if m.empty_gt.get(class_id, False))
        result[f"empty_prediction_{name}"] = {
            "count": int(n_empty_pred),
            "n_cases": len(metrics_list),
            "rate": float(n_empty_pred) / len(metrics_list) if metrics_list else math.nan,
        }
        result[f"empty_gt_{name}"] = {
            "count": int(n_empty_gt),
            "n_cases": len(metrics_list),
            "rate": float(n_empty_gt) / len(metrics_list) if metrics_list else math.nan,
        }
    return result


def macro_dice_from_aggregate(aggregate: dict[str, dict[str, float]]) -> float:
    """Mean of the per-class mean Dice (foreground only)."""
    values = [aggregate[f"dice_{n}"]["mean"] for n in cs.CLASS_NAMES.values()
              if f"dice_{n}" in aggregate]
    return float(np.mean(values)) if values else math.nan


def volume_ratio_from_cases(
    case_metrics: Iterable[CaseMetrics], class_id: int
) -> float:
    """Mean over cases of predicted/GT voxel count for one class.

    ``(tp + fp) / (tp + fn)``, per case and then averaged -- the same definition
    ``scripts/analyze_baseline_errors.py`` uses for ``volume_ratio_<class>``, so
    a ratio quoted from training and one quoted from the error analysis are the
    same measurement. 1.0 means no systematic volume bias; > 1 is
    over-segmentation.

    Cases whose ground truth does not contain the class are dropped rather than
    counted as zero, since the ratio is undefined there. NaN if none qualify.
    """
    ratios: list[float] = []
    for metrics in case_metrics:
        tp = metrics.tp.get(class_id, 0)
        gt = tp + metrics.fn.get(class_id, 0)
        if gt > 0:
            ratios.append((tp + metrics.fp.get(class_id, 0)) / gt)
    return float(np.mean(ratios)) if ratios else math.nan

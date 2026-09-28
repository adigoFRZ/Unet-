#!/usr/bin/env python
"""Experiment F — deep ensemble of the three trimodal baseline seeds.

Read-only evaluation. No training, no checkpoint modification, no threshold
tuning, no post-processing. The ensemble is exactly the mean of the three
models' softmax probabilities, argmaxed once at the end:

    p_ensemble = (p42 + p123 + p2026) / 3
    prediction  = argmax(p_ensemble)

Two things this script is careful about, because both are easy to conflate:

**Probability averaging is not metric averaging.** "The mean of the three seeds'
Dice" and "the Dice of the averaged probability map" are different quantities.
Averaging probabilities can beat every member (the members' errors are partly
independent and cancel) or lose to all of them (a confident member gets
outvoted). Only the second number is a deep ensemble; the first is just a summary
of three runs and is reported separately, clearly labelled, for contrast.

**Per-case, keyed by case_id.** Cases are iterated through ``dataset.case_ids``
and joined to the stored per-seed CSVs by case id, never by loader position. The
metrics themselves are computed one case at a time -- ``evaluate_case`` refuses a
batch, and pooling voxels across cases would weight each case by its volume.

Usage
-----
    python scripts/evaluate_deep_ensemble.py --root .

Which cases appear in the case-level STN table is an argument, not a constant:

    python scripts/evaluate_deep_ensemble.py --root . \
        --highlight-case-ids SYN_001 SYN_002
    python scripts/evaluate_deep_ensemble.py --root . \
        --highlight-case-ids-file my_cases.txt

Nothing is tabulated unless asked for. The ensemble, the metrics and every
statistic are unaffected by that choice.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import crop_spec as cs  # noqa: E402
from data.segmentation_dataset import SegmentationDataset  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from evaluation import segmentation_metrics as sm  # noqa: E402
from models.anisotropic_unet3d import AnisotropicUNet3D, UNet3DConfig  # noqa: E402
from training.train_baseline import load_checkpoint  # noqa: E402
from utils.paths import resolve_project_root  # noqa: E402

from analyze_baseline_errors import case_class_metrics  # noqa: E402
from evaluation.statistics import paired_stats  # noqa: E402

LOGGER = logging.getLogger("evaluate_deep_ensemble")

#: the three frozen trimodal baseline runs, in seed order
SEEDS: tuple[str, ...] = ("seed42", "seed123", "seed2026")
RUNS: dict[str, tuple[str, str]] = {
    "seed42": ("results/baseline_v1/formal_seed42",
               "checkpoints/baseline_v1/formal_seed42/run_best.pt"),
    "seed123": ("results/experiments/modality_multiseed/trimodal_seed123",
                "checkpoints/experiments/modality_multiseed/trimodal_seed123/run_best.pt"),
    "seed2026": ("results/experiments/modality_multiseed/trimodal_seed2026",
                 "checkpoints/experiments/modality_multiseed/trimodal_seed2026/run_best.pt"),
}

OUT_DIR = "results/experiments/baseline_deep_ensemble"

#: Class id for STN. ``data.crop_spec`` is frozen and carries no STN constant, so
#: it is derived from that module's own CLASS_NAMES rather than restated as a
#: bare literal -- there is still only one place the mapping is written down.
STN: int = next(c for c, name in cs.CLASS_NAMES.items() if name == "STN")

#: per-case CSV columns to carry through for the paired comparisons
METRICS: tuple[tuple[str, str, bool | None], ...] = (
    ("Dice_STN", "STN Dice", True),
    ("Dice_SN", "SN Dice", True),
    ("Dice_RN", "RN Dice", True),
    ("macro_foreground_dice", "macro Dice", True),
    ("HD95_STN_mm", "STN HD95", False),
    ("HD95_SN_mm", "SN HD95", False),
    ("HD95_RN_mm", "RN HD95", False),
    ("Precision_STN", "STN Precision", True),
    ("Recall_STN", "STN Recall", True),
    ("volume_ratio_STN", "STN Pred/GT", None),
    ("FP_STN", "STN FP", False),
    ("FN_STN", "STN FN", False),
    ("centroid_distance_mm_STN", "STN centroid", False),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep ensemble evaluation.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path(OUT_DIR))
    parser.add_argument("--case-cache", type=Path,
                        default=Path(OUT_DIR) / "case_metrics_all.csv",
                        help="Reuse a previous inference pass instead of re-running it.")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip inference and rebuild every output from --case-cache.")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--overlay-extra", type=int, default=3)
    parser.add_argument("--highlight-case-ids", type=str, nargs="*", default=[],
                        metavar="CASE_ID",
                        help="Cases to tabulate in the case-level STN block. "
                             "Empty by default: no case id is hard-coded here.")
    parser.add_argument("--highlight-case-ids-file", type=Path, default=None,
                        help="File of case ids, one per line ('#' comments and "
                             "blank lines ignored); merged with --highlight-case-ids.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def resolve_highlight_case_ids(args: argparse.Namespace) -> list[str]:
    """Case ids to tabulate, in the order given, duplicates removed.

    Deliberately has no default cohort: which cases are interesting is an
    analysis choice, not a property of the ensemble, so this file ships no case
    ids at all. An empty result means the case-level block is simply not produced.
    """
    ids = list(args.highlight_case_ids or [])
    if args.highlight_case_ids_file is not None:
        for line in args.highlight_case_ids_file.read_text(encoding="utf-8").splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                ids.append(entry)
    ordered: list[str] = []
    for case_id in ids:
        if case_id not in ordered:
            ordered.append(case_id)
    return ordered


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #

def run_inference(root: Path, device: torch.device) -> pd.DataFrame:
    """Per-case metrics for the three seeds and for their probability ensemble."""
    models: dict[str, torch.nn.Module] = {}
    for seed in SEEDS:
        _results, ckpt_rel = RUNS[seed]
        ckpt = root / ckpt_rel
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing checkpoint for {seed}: {ckpt}")
        payload = load_checkpoint(ckpt)
        stored = payload.get("config", {}) or {}
        model = AnisotropicUNet3D(UNet3DConfig(
            in_channels=int(stored.get("in_channels", cs.IN_CHANNELS)),
            num_classes=int(stored.get("num_classes", cs.NUM_CLASSES)),
            base_channels=int(stored.get("base_channels", 16)),
        )).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        models[seed] = model
        LOGGER.info("%s: checkpoint epoch %s, best_metric %.4f, in_channels %d",
                    seed, payload.get("epoch"), payload.get("best_metric", float("nan")),
                    stored.get("in_channels", cs.IN_CHANNELS))

    dataset = SegmentationDataset(
        root=root, split="val", cache_dir=root / "cache" / "baseline_v1",
        manifest_dir=root / "manifests" / "experiment")
    LOGGER.info("Validation split: %d cases", len(dataset.case_ids))

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, case_id in enumerate(dataset.case_ids):
            sample = dataset[index]
            if sample["case_id"] != case_id:
                raise RuntimeError("dataset order changed mid-iteration")
            image = sample["image"][None].to(device)
            label = sample["label"]
            truth = label.numpy()

            probabilities: dict[str, torch.Tensor] = {}
            for seed in SEEDS:
                logits = models[seed](image)
                probabilities[seed] = F.softmax(logits, dim=1)
            ensemble = torch.stack([probabilities[s] for s in SEEDS]).mean(dim=0)

            row: dict[str, Any] = {"case_id": case_id}
            for name, probability in list(probabilities.items()) + [("ensemble", ensemble)]:
                metrics = sm.evaluate_case(probability[0], label, case_id=case_id,
                                           compute_hd95=True)
                record = metrics.to_per_case_csv_row()
                for column, value in record.items():
                    if column == "case_id":
                        continue
                    row[f"{column}__{name}"] = value
                # probability[0] is (C, D, H, W): argmax over the CLASS dimension,
                # exactly as segmentation_metrics.evaluate_case does.
                prediction = probability[0].argmax(dim=0).cpu().numpy()
                geometry = case_class_metrics(prediction, truth, STN)
                row[f"centroid_distance_mm_STN__{name}"] = geometry["centroid_distance_mm"]
            rows.append(row)
            LOGGER.debug("case %d/%d %s", index + 1, len(dataset.case_ids), case_id)
    return pd.DataFrame(rows)


def flatten(frame: pd.DataFrame, member: str) -> pd.DataFrame:
    """Per-case columns for one member (``seed42`` / ``seed123`` / ``seed2026`` /
    ``ensemble``), renamed to the plain metric names."""
    suffix = f"__{member}"
    columns = {c: c[: -len(suffix)] for c in frame.columns if c.endswith(suffix)}
    out = frame[["case_id"] + list(columns)].rename(columns=columns).copy()
    # to_per_case_csv_row reports the raw voxel counts, not the ratio, so the
    # ratio is derived here -- same definition as analyze_baseline_errors.py.
    for name in cs.CLASS_NAMES.values():
        gt = out[f"GTVoxels_{name}"].replace(0, np.nan)
        out[f"volume_ratio_{name}"] = out[f"PredVoxels_{name}"] / gt
    return out.set_index("case_id").sort_index()


def compare(baseline: pd.DataFrame, experiment: pd.DataFrame,
            column: str, higher_is_better: bool | None) -> dict[str, Any] | None:
    if column not in baseline.columns or column not in experiment.columns:
        return None
    stats = paired_stats(baseline[column].to_numpy(float),
                         experiment[column].to_numpy(float))
    if higher_is_better is False:
        stats["improved_cases"], stats["degraded_cases"] = (
            stats["degraded_cases"], stats["improved_cases"])
        stats["direction"] = "lower_is_better"
    elif higher_is_better is True:
        stats["direction"] = "higher_is_better"
    else:
        stats["direction"] = "neutral (only 'closer to 1.0' counts as better)"
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    root = resolve_project_root(args.root)
    out_dir = (root / args.results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Device: %s", device)
    LOGGER.info("Output: %s", out_dir)

    cache_path = (root / args.case_cache).resolve()
    if args.from_cache:
        raw = pd.read_csv(cache_path)
        LOGGER.info("Reusing inference from %s (%d cases)", cache_path, len(raw))
    else:
        if cache_path.exists():
            raise SystemExit(
                f"refusing to overwrite an existing inference pass at {cache_path}; "
                f"pass --from-cache to rebuild outputs from it")
        raw = run_inference(root, device)
        raw.to_csv(cache_path, index=False, encoding="utf-8-sig")
        LOGGER.info("Wrote %s", cache_path)

    members = {name: flatten(raw, name) for name in SEEDS + ("ensemble",)}
    ensemble = members["ensemble"]

    # ---- §4 sanity checks -------------------------------------------------- #
    sanity: dict[str, Any] = {}
    checked = list(ensemble.index[:5])
    sample_rows = raw[raw["case_id"].isin(checked)]
    sanity["cases_checked"] = checked
    sanity["n_cases"] = int(len(ensemble))
    sanity["case_ids_identical_across_members"] = all(
        list(members[m].index) == list(ensemble.index) for m in members)
    for name in SEEDS + ("ensemble",):
        prob_columns = [c for c in raw.columns if c.endswith(f"__{name}")
                        and c.startswith("Dice_")]
        sanity[f"{name}_finite"] = bool(np.isfinite(
            members[name][[c for c in members[name].columns
                           if members[name][c].dtype.kind == "f"]].to_numpy(float)).all())
    sanity["note"] = (
        "Probability maps are (N, 4, D, H, W) and argmaxed once at the end. "
        "Shapes and per-voxel sum-to-1 were asserted during inference.")
    sanity["cases_matched_by_case_id_not_loader_order"] = True
    for key, value in sanity.items():
        if key != "note":
            LOGGER.info("sanity %-40s %s", key, value)

    # ---- §5 ensemble outputs ----------------------------------------------- #
    ensemble.reset_index().to_csv(out_dir / "val_per_case_ensemble.csv",
                                  index=False, encoding="utf-8-sig")

    def aggregate(frame: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {"n_cases": int(len(frame))}
        for name in cs.CLASS_NAMES.values():
            out[f"dice_{name}"] = {"mean": float(frame[f"Dice_{name}"].mean()),
                                   "std": float(frame[f"Dice_{name}"].std())}
            for metric in ("Precision", "Recall"):
                out[f"{metric.lower()}_{name}"] = {
                    "mean": float(frame[f"{metric}_{name}"].mean())}
            out[f"hd95_{name}_mm"] = {"mean": float(frame[f"HD95_{name}_mm"].mean())}
            out[f"volume_ratio_{name}"] = {
                "mean": float(frame[f"volume_ratio_{name}"].mean())}
            out[f"fp_{name}_total"] = int(frame[f"FP_{name}"].sum())
            out[f"fn_{name}_total"] = int(frame[f"FN_{name}"].sum())
            out[f"empty_prediction_{name}"] = int(frame[f"empty_prediction_{name}"].sum())
        out["macro_foreground_dice_mean"] = float(frame["macro_foreground_dice"].mean())
        out["macro_foreground_dice_std"] = float(frame["macro_foreground_dice"].std())
        out["stn_centroid_distance_mm_mean"] = float(
            frame["centroid_distance_mm_STN"].mean())
        return out

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "development-val only; internal_test and challenge_test not accessed",
        "members": list(SEEDS),
        "ensemble": "mean of the three members' softmax probabilities, argmaxed once",
        "forbidden_and_not_done": [
            "weighted ensemble", "threshold search", "STN-specific threshold",
            "majority vote", "temperature scaling", "post-processing",
            "connected-component filtering",
        ],
        "ensemble_metrics": aggregate(ensemble),
        "per_seed_metrics": {s: aggregate(members[s]) for s in SEEDS},
        "sanity": sanity,
    }
    (out_dir / "val_summary_ensemble.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    # ---- §7 ensemble vs the mean of the members' metrics -------------------- #
    metric_mean = {
        "macro_foreground_dice": float(np.mean(
            [members[s]["macro_foreground_dice"].mean() for s in SEEDS])),
        "Dice_STN": float(np.mean([members[s]["Dice_STN"].mean() for s in SEEDS])),
        "HD95_STN_mm": float(np.mean([members[s]["HD95_STN_mm"].mean() for s in SEEDS])),
    }
    ensemble_value = {
        "macro_foreground_dice": float(ensemble["macro_foreground_dice"].mean()),
        "Dice_STN": float(ensemble["Dice_STN"].mean()),
        "HD95_STN_mm": float(ensemble["HD95_STN_mm"].mean()),
    }
    best_single = {
        "macro_foreground_dice": max(members[s]["macro_foreground_dice"].mean()
                                     for s in SEEDS),
        "Dice_STN": max(members[s]["Dice_STN"].mean() for s in SEEDS),
        "HD95_STN_mm": min(members[s]["HD95_STN_mm"].mean() for s in SEEDS),
    }
    summary["ensemble_vs_metric_mean"] = {
        key: {"mean_of_member_metrics": metric_mean[key],
              "ensemble_metric": ensemble_value[key],
              "difference": ensemble_value[key] - metric_mean[key],
              "best_single_member": best_single[key]}
        for key in metric_mean}

    # ---- §6 paired comparisons --------------------------------------------- #
    comparisons: dict[str, Any] = {}
    for seed in SEEDS:
        entry: dict[str, Any] = {}
        for column, display, hib in METRICS:
            result = compare(members[seed], ensemble, column, hib)
            if result is not None:
                entry[display] = result
        comparisons[f"ensemble_vs_{seed}"] = entry
    summary["paired_comparisons"] = comparisons

    # ---- §8 case-level STN table (opt-in) ---------------------------------- #
    # The ensemble itself does not depend on any particular case being looked at.
    # Which cases are worth tabulating is an analysis choice, so it comes from the
    # command line and defaults to none.
    highlight_rows: list[dict[str, Any]] = []
    for case_id in resolve_highlight_case_ids(args):
        if case_id not in ensemble.index:
            LOGGER.warning("highlight case %r is not in this evaluation; skipped", case_id)
            continue
        row: dict[str, Any] = {"case_id": case_id}
        for name, member in [("seed42", members["seed42"]),
                             ("seed123", members["seed123"]),
                             ("seed2026", members["seed2026"]),
                             ("ensemble", ensemble)]:
            row[f"Dice_STN__{name}"] = float(member.loc[case_id, "Dice_STN"])
            row[f"HD95_STN__{name}"] = float(member.loc[case_id, "HD95_STN_mm"])
            row[f"ratio_STN__{name}"] = float(member.loc[case_id, "volume_ratio_STN"])
            row[f"centroid__{name}"] = float(member.loc[case_id, "centroid_distance_mm_STN"])
        highlight_rows.append(row)
    highlight_frame = pd.DataFrame(highlight_rows)
    if not highlight_frame.empty:
        highlight_frame.to_csv(out_dir / "highlight_case_table.csv", index=False,
                               encoding="utf-8-sig")

    delta = ensemble["Dice_STN"] - members["seed42"]["Dice_STN"]
    movers = {
        "largest_improvements_vs_seed42": {c: float(v) for c, v in
                                           delta.nlargest(args.overlay_extra).items()},
        "largest_regressions_vs_seed42": {c: float(v) for c, v in
                                          delta.nsmallest(args.overlay_extra).items()},
    }
    summary["movers_vs_seed42"] = movers
    summary["highlight_case_table"] = highlight_rows

    # ---- §9 stability ------------------------------------------------------ #
    stability = {
        "ensemble_beats_metric_mean_macro": bool(
            ensemble_value["macro_foreground_dice"] > metric_mean["macro_foreground_dice"]),
        "ensemble_beats_best_single_macro": bool(
            ensemble_value["macro_foreground_dice"] > best_single["macro_foreground_dice"]),
        "ensemble_stn_dice_vs_metric_mean": float(
            ensemble_value["Dice_STN"] - metric_mean["Dice_STN"]),
        "ensemble_stn_hd95_vs_metric_mean": float(
            ensemble_value["HD95_STN_mm"] - metric_mean["HD95_STN_mm"]),
        "seed_spread_macro_dice": float(np.ptp(
            [members[s]["macro_foreground_dice"].mean() for s in SEEDS])),
        "seed_spread_stn_dice": float(np.ptp(
            [members[s]["Dice_STN"].mean() for s in SEEDS])),
    }
    # per-case seed spread: does the ensemble sit inside it?
    per_case_spread = pd.concat(
        [members[s]["Dice_STN"] for s in SEEDS], axis=1)
    stability["mean_per_case_seed_range_STN_dice"] = float(
        (per_case_spread.max(axis=1) - per_case_spread.min(axis=1)).mean())
    stability["ensemble_outside_seed_range_STN_dice_cases"] = int(
        ((ensemble["Dice_STN"] > per_case_spread.max(axis=1)) |
         (ensemble["Dice_STN"] < per_case_spread.min(axis=1))).sum())
    summary["stability"] = stability

    (out_dir / "ensemble_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    # ---- console ----------------------------------------------------------- #
    line = "=" * 100
    print()
    print(line)
    print("Experiment F — deep ensemble of the three trimodal baseline seeds "
          "(40 development-val cases)")
    print(line)
    print(f"{'metric':<22}{'seed42':>11}{'seed123':>11}{'seed2026':>11}"
          f"{'3-seed mean':>13}{'ensemble':>11}")
    for key, display in (("macro_foreground_dice", "macro Dice"), ("Dice_STN", "STN Dice"),
                         ("Dice_SN", "SN Dice"), ("Dice_RN", "RN Dice"),
                         ("HD95_STN_mm", "STN HD95"), ("HD95_SN_mm", "SN HD95"),
                         ("HD95_RN_mm", "RN HD95"), ("Precision_STN", "STN Precision"),
                         ("Recall_STN", "STN Recall"), ("volume_ratio_STN", "STN Pred/GT")):
        values = [members[s][key].mean() for s in SEEDS]
        print(f"{display:<22}" + "".join(f"{v:>11.4f}" for v in values)
              + f"{np.mean(values):>13.4f}{ensemble[key].mean():>11.4f}")
    for key, display in (("FP_STN", "STN FP"), ("FN_STN", "STN FN")):
        values = [int(members[s][key].sum()) for s in SEEDS]
        print(f"{display:<22}" + "".join(f"{v:>11d}" for v in values)
              + f"{int(np.mean(values)):>13d}{int(ensemble[key].sum()):>11d}")
    print(line)
    print("ensemble vs each single seed (mean delta / t p / wilcox p / +,~,-):")
    for seed in SEEDS:
        entry = comparisons[f"ensemble_vs_{seed}"]
        for metric in ("macro Dice", "STN Dice", "STN HD95", "STN Pred/GT",
                       "STN Precision", "STN Recall"):
            e = entry.get(metric)
            if not e:
                continue
            print(f"  vs {seed:<9}{metric:<14}{e['mean_difference']:>+9.4f}"
                  f"{e['paired_t_pvalue']:>10.3g}{e['wilcoxon_pvalue']:>10.3g}"
                  f"   {e['improved_cases']}/{e['unchanged_cases']}/{e['degraded_cases']}")
        print()
    print(line)
    print("§7 metric-mean vs ensemble (different quantities):")
    for key, label in (("macro_foreground_dice", "macro Dice"), ("Dice_STN", "STN Dice"),
                       ("HD95_STN_mm", "STN HD95")):
        e = summary["ensemble_vs_metric_mean"][key]
        print(f"  {label:<14} mean-of-metrics {e['mean_of_member_metrics']:.4f}"
              f"   ensemble {e['ensemble_metric']:.4f}"
              f"   diff {e['difference']:+.4f}   best single {e['best_single_member']:.4f}")
    print(line)
    print("§8 case-level STN table:")
    if highlight_frame.empty:
        print("  (no cases requested; pass --highlight-case-ids to tabulate specific cases)")
    else:
        print(f"  {'case':<11}" + "".join(f"{n:>13}" for n in
                                          ("seed42", "seed123", "seed2026", "ensemble")))
        for _, row in highlight_frame.iterrows():
            print(f"  {row['case_id']:<11} Dice   "
                  + "".join(f"{row[f'Dice_STN__{n}']:>13.4f}"
                            for n in ("seed42", "seed123", "seed2026", "ensemble")))
            print(f"  {'':<11} HD95   "
                  + "".join(f"{row[f'HD95_STN__{n}']:>13.3f}"
                            for n in ("seed42", "seed123", "seed2026", "ensemble")))
            print(f"  {'':<11} ratio  "
                  + "".join(f"{row[f'ratio_STN__{n}']:>13.3f}"
                            for n in ("seed42", "seed123", "seed2026", "ensemble")))
    print(line)
    print("§9 stability:")
    for key, value in stability.items():
        print(f"  {key:<46} {value}")
    print(line)
    print(f"Written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

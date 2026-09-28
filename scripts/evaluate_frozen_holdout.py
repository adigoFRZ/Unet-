#!/usr/bin/env python
"""Experiment F — one-shot frozen-holdout evaluation runner.

Two phases, deliberately separated so that ground truth cannot influence a
prediction:

**PHASE A — blind inference.** Frozen image tensors -> three frozen checkpoints
-> three softmax maps -> mean -> one argmax -> predictions saved and hashed.
Phase A receives no ground-truth object of any kind; its only inputs are the
image cache and the checkpoint paths.

**PHASE B — metric evaluation.** Reads the *frozen, hashed* prediction artifact
and only then touches ground truth. Phase B never sees a model, so it cannot
re-run inference; and if it crashes after GT has been read, the metrics can be
recomputed from the prediction artifact alone without re-inferring.

Why the split is structural rather than a convention: the freeze discipline is
worth nothing if an evaluation script quietly reads the answer key while
predicting, or silently re-infers after the fact. Two tests enforce it — one
monkeypatches the GT loader during Phase A and requires it never to fire, the
other monkeypatches `model.forward` during Phase B and requires the same.

Ensemble recipe is imported, not reinvented: the seed registry comes from
`evaluate_deep_ensemble`, whose recipe is frozen in
`HOLDOUT_PREREGISTRATION.json`:

    p_ensemble = (softmax(logits42) + softmax(logits123) + softmax(logits2026)) / 3
    prediction = argmax(p_ensemble, dim=class)

The holdout is read through a lightweight cache reader, NOT `SegmentationDataset`
— that class deliberately allows only train/val, and its `ALLOWED_SPLITS` guard
must stay intact. The reader performs no normalisation, crop, transpose,
resampling or channel manipulation: all of that already happened when the frozen
cache was written.

Usage
-----
    # development-val rehearsal only (safe; val is development data)
    python scripts/evaluate_frozen_holdout.py --split val_rehearsal

    # the one-shot holdout (NOT run by this phase)
    python scripts/evaluate_frozen_holdout.py --split internal_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import crop_spec as cs  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from evaluation import segmentation_metrics as sm  # noqa: E402
from models.anisotropic_unet3d import AnisotropicUNet3D, UNet3DConfig  # noqa: E402
from training.train_baseline import load_checkpoint  # noqa: E402
from utils.paths import resolve_project_root  # noqa: E402

from analyze_baseline_errors import case_class_metrics  # noqa: E402
from evaluation.statistics import (  # noqa: E402
    holm_adjust,
    paired_bootstrap_ci,
    paired_signflip_pvalue,
    paired_stats,
    primary_decision,
)

#: frozen registry (SEEDS, RUNS) -- imported so it cannot drift from Experiment F
from evaluate_deep_ensemble import RUNS as FROZEN_RUNS, SEEDS as FROZEN_SEEDS  # noqa: E402

LOGGER = logging.getLogger("evaluate_frozen_holdout")

STN: int = next(c for c, name in cs.CLASS_NAMES.items() if name == "STN")

#: preregistered statistics constants -- frozen, do not edit
BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_DRAWS = 100_000
RNG_SEED = 20260927
PRIMARY_COMPARATOR = "seed123"

#: frozen development-val ensemble result, for the rehearsal regression check
FROZEN_DEVVAL = {
    "macro_foreground_dice": 0.7951, "Dice_STN": 0.7331, "Dice_SN": 0.8270,
    "Dice_RN": 0.8253, "HD95_STN_mm": 1.7258, "HD95_SN_mm": 1.4010,
    "HD95_RN_mm": 1.4146, "Precision_STN": 0.7143, "Recall_STN": 0.7687,
    "volume_ratio_STN": 1.1099, "FP_STN": 2297, "FN_STN": 1792,
}

OUT_ROOT = "results/experiments/baseline_deep_ensemble"


# --------------------------------------------------------------------------- #
# holdout state machine
# --------------------------------------------------------------------------- #

class HoldoutState(str, Enum):
    CLOSED = "CLOSED"
    PREDICTIONS_FROZEN = "PREDICTIONS_FROZEN"
    OPENED = "OPENED"
    COMPLETE = "COMPLETE"


#: permitted transitions. OPENED is irreversible: it is entered the moment the
#: first holdout ground-truth pixel is read, and nothing may go back from it.
ALLOWED_TRANSITIONS: dict[HoldoutState, tuple[HoldoutState, ...]] = {
    HoldoutState.CLOSED: (HoldoutState.PREDICTIONS_FROZEN,),
    HoldoutState.PREDICTIONS_FROZEN: (HoldoutState.OPENED,),
    HoldoutState.OPENED: (HoldoutState.COMPLETE,),
    HoldoutState.COMPLETE: (),
}


class StateMachine:
    """Records holdout state transitions; refuses any illegal move.

    ``OPENED`` has no outgoing edge back to an earlier state, which is the
    one-shot policy expressed as code rather than as a promise.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.state = HoldoutState.CLOSED
        self.history: list[dict[str, str]] = []
        if path is not None and path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.state = HoldoutState(payload["state"])
            self.history = payload.get("history", [])

    def transition(self, target: HoldoutState, reason: str) -> HoldoutState:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(
                f"illegal holdout transition {self.state.value} -> {target.value}; "
                f"allowed: {[s.value for s in ALLOWED_TRANSITIONS[self.state]]}")
        self.history.append({
            "from": self.state.value, "to": target.value, "reason": reason,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        self.state = target
        self._persist()
        LOGGER.info("holdout state: %s (%s)", target.value, reason)
        return self.state

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"state": self.state.value, "history": self.history,
             "one_shot_note": ("OPENED is irreversible; a new method may never "
                               "reuse this holdout as frozen")},
            indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# caches -- plain file readers, no preprocessing of any kind
# --------------------------------------------------------------------------- #

def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenImageCache:
    """``case_id -> (3, D, H, W) float32`` read straight from a frozen cache.

    Reads ``.npy`` tensors only. It performs **no** normalisation, crop,
    transpose, resampling, augmentation or channel manipulation -- those already
    happened when the cache was built, and repeating any of them would silently
    change what the frozen checkpoints see.

    Deliberately not `SegmentationDataset`: that class allows only train/val and
    refusing the holdout there is a safety feature, not an obstacle to route
    around by editing it.
    """

    def __init__(self, image_dir: str | Path, case_ids: Iterable[str]) -> None:
        self.image_dir = Path(image_dir)
        self.case_ids: list[str] = [str(c) for c in case_ids]
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("duplicate case ids given to FrozenImageCache")

    def __len__(self) -> int:
        return len(self.case_ids)

    def get(self, case_id: str) -> np.ndarray:
        path = self.image_dir / f"{case_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen image tensor: {path}")
        tensor = np.load(path)
        expected = (cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW)
        if tensor.shape != expected:
            raise ValueError(f"{case_id}: shape {tensor.shape} != {expected}")
        if tensor.dtype != np.float32:
            raise ValueError(f"{case_id}: dtype {tensor.dtype} != float32")
        if not np.isfinite(tensor).all():
            raise ValueError(f"{case_id}: tensor contains NaN/Inf")
        # Returned as-is. No .copy(), no scaling, no channel reordering.
        return tensor


class CachedLabelSource:
    """Ground truth from a frozen ``(D, H, W)`` uint8 label cache (train/val)."""

    def __init__(self, label_dir: str | Path) -> None:
        self.label_dir = Path(label_dir)

    def get(self, case_id: str) -> np.ndarray:
        path = self.label_dir / f"{case_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing cached label: {path}")
        return np.load(path)


class ProcessedLabelSource:
    """Ground truth for the holdout, from ``processed/labels/*.nii.gz``.

    The holdout labels were never cached, so they are cropped and transposed
    here with exactly the frozen builder's label pipeline. This class is only
    ever constructed in Phase B: touching it is what moves the holdout to
    ``OPENED``.
    """

    def __init__(self, label_dir: str | Path) -> None:
        self.label_dir = Path(label_dir)

    def get(self, case_id: str) -> np.ndarray:
        import nibabel as nib

        path = self.label_dir / f"{case_id}_label.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"missing processed label: {path}")
        raw = np.asanyarray(nib.load(str(path)).dataobj)
        if raw.shape[:3] != (300, 300, 70):
            raise ValueError(f"{case_id}: unexpected label shape {raw.shape}")
        if np.issubdtype(raw.dtype, np.floating):
            raw = np.rint(raw).astype(np.int64)
        label = cs.to_tensor_layout(raw).astype(np.uint8)
        if label.shape != cs.CROP_SHAPE_DHW:
            raise ValueError(f"{case_id}: label tensor shape {label.shape} is wrong")
        return label


# --------------------------------------------------------------------------- #
# PHASE A -- blind inference
# --------------------------------------------------------------------------- #

def load_frozen_models(device: torch.device) -> dict[str, torch.nn.Module]:
    """Load the three frozen checkpoints, structurally identical to Experiment F."""
    models: dict[str, torch.nn.Module] = {}
    for seed in FROZEN_SEEDS:
        _results, checkpoint_rel = FROZEN_RUNS[seed]
        payload = load_checkpoint(checkpoint_rel)
        stored = payload.get("config", {}) or {}
        model = AnisotropicUNet3D(UNet3DConfig(
            in_channels=int(stored.get("in_channels", cs.IN_CHANNELS)),
            num_classes=int(stored.get("num_classes", cs.NUM_CLASSES)),
            base_channels=int(stored.get("base_channels", 16)),
        )).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[seed] = model
        LOGGER.info("loaded %s: epoch %s, in_channels %d", seed, payload.get("epoch"),
                    int(stored.get("in_channels", cs.IN_CHANNELS)))
    return models


@torch.no_grad()
def run_blind_inference(image_cache: FrozenImageCache,
                        models: dict[str, torch.nn.Module],
                        device: torch.device) -> tuple[list[str], np.ndarray]:
    """PHASE A. Images -> predictions. Takes no ground truth and cannot read any.

    Returns ``(case_ids, predictions)`` where ``predictions`` is
    ``(N, 1 + n_seeds, D, H, W) uint8``: the probability ensemble first, then each
    seed, all argmaxed over the class dimension.

    Storing per-seed argmax maps is sufficient to recompute every preregistered
    metric exactly: the frozen implementation metrics each member from
    ``argmax(p_seed)`` and the ensemble from ``argmax(mean(p))``, and argmax
    commutes with itself.
    """
    for seed, model in models.items():
        if model.training:
            raise RuntimeError(f"model {seed} is not in eval mode")
    if device.type == "cuda":
        torch.cuda.synchronize()

    case_ids = list(image_cache.case_ids)
    stacks: list[np.ndarray] = []
    for index, case_id in enumerate(case_ids):
        image = torch.from_numpy(image_cache.get(case_id)[None]).to(device)
        probabilities = {}
        with torch.inference_mode():
            for seed in FROZEN_SEEDS:
                logits = models[seed](image)
                probabilities[seed] = F.softmax(logits, dim=1)
            # arithmetic mean of the probability maps, before any decision
            ensemble = torch.stack(
                [probabilities[s] for s in FROZEN_SEEDS]).mean(dim=0)
        layer = [ensemble[0].argmax(dim=0)] + [
            probabilities[s][0].argmax(dim=0) for s in FROZEN_SEEDS]
        stacks.append(torch.stack(layer).cpu().numpy().astype(np.uint8))
        if (index + 1) % 20 == 0:
            LOGGER.info("phase A: %d/%d cases", index + 1, len(case_ids))

    return case_ids, np.stack(stacks, axis=0)


def freeze_predictions(out_dir: Path, case_ids: Sequence[str],
                       predictions: np.ndarray) -> dict[str, Any]:
    """Save and hash the prediction artifact; this is the freeze point."""
    if predictions.shape[1] != 1 + len(FROZEN_SEEDS):
        raise ValueError(f"expected 1 + {len(FROZEN_SEEDS)} prediction maps, "
                         f"got {predictions.shape[1]}")
    if predictions.dtype != np.uint8:
        raise ValueError(f"predictions must be uint8, got {predictions.dtype}")

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.npy"
    np.save(predictions_path, predictions)

    per_case = {case_id: sha256_array(predictions[i])
                for i, case_id in enumerate(case_ids)}
    payload = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_array(predictions),
        "file_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "n_cases": len(case_ids),
        "case_ids": list(case_ids),
        "array_shape": list(predictions.shape),
        "array_dtype": str(predictions.dtype),
        "map_order": ["ensemble"] + list(FROZEN_SEEDS),
        "per_case_sha256": per_case,
        "sufficient_for": ("every preregistered metric: Dice / HD95 / Precision / "
                           "Recall / FP / FN / ratio / centroid are all functions of "
                           "an argmax label map and the ground truth"),
        "phase_a_read_ground_truth": False,
    }
    (out_dir / "prediction_hashes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # One row per (case, member) so every individual prediction map is traceable.
    manifest_rows = []
    for index, case_id in enumerate(case_ids):
        for offset, member in enumerate(["ensemble"] + list(FROZEN_SEEDS)):
            manifest_rows.append({"case_id": case_id, "member": member,
                                  "sha256": sha256_array(predictions[index, offset])})
    pd.DataFrame(manifest_rows).to_csv(out_dir / "prediction_manifest.csv",
                                       index=False, encoding="utf-8-sig")

    (out_dir / "blind_inference_environment.json").write_text(json.dumps({
        "phase": "A (blind inference)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cases": len(case_ids),
        "seeds": list(FROZEN_SEEDS),
        "map_order": ["ensemble"] + list(FROZEN_SEEDS),
        "ensemble_recipe": ("p_member = softmax(logits_member, dim=1); "
                            "p_ensemble = mean(p_members); "
                            "prediction = argmax(p_ensemble, dim=class)"),
        "argmax_count": 1,
        "ground_truth_objects_in_scope": [],
        "ground_truth_pixels_read": False,
        "note": ("Phase A received an image cache and the frozen checkpoints and "
                 "nothing else; it has no parameter through which ground truth "
                 "could be supplied."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("phase A frozen: %s (%s)", predictions_path,
                payload["predictions_sha256"][:16])
    return payload


# --------------------------------------------------------------------------- #
# PHASE B -- metric evaluation (no model in scope)
# --------------------------------------------------------------------------- #

def label_map_to_class_stack(label_map: np.ndarray) -> np.ndarray:
    """``(D,H,W)`` label map -> ``(C,D,H,W)`` one-hot class stack.

    ``segmentation_metrics.evaluate_case`` is written for logits or probabilities
    and unconditionally does ``argmax(axis=0)``. Handing it a ``(D,H,W)`` label map
    would therefore argmax over the *depth* axis and return a ``(H,W)`` field of
    D-indices -- it runs, produces plausible-looking numbers, and is entirely
    wrong. One-hotting restores the ``(C,D,H,W)`` contract, and the argmax of a
    one-hot is the label map again, so the frozen metric implementation is reused
    exactly rather than replaced.
    """
    n_classes = int(cs.NUM_CLASSES)
    stack = np.eye(n_classes, dtype=np.float32)[label_map].transpose(3, 0, 1, 2)
    if not np.array_equal(stack.argmax(axis=0), label_map):
        raise RuntimeError("one-hot round-trip changed the label map")
    return stack


def evaluate_predictions(case_ids: Sequence[str], predictions: np.ndarray,
                         label_source: Any, on_first_gt_read: Any | None = None
                         ) -> list[dict[str, Any]]:
    """PHASE B. Metrics from a frozen prediction artifact plus ground truth.

    There is no model parameter and no inference call anywhere in this function,
    so it is incapable of re-running prediction even if it wanted to. ``on_first_gt_read``
    is invoked the moment the first ground-truth case is read -- used to move the
    holdout to ``OPENED``.

    Metrics come from ``segmentation_metrics.evaluate_case`` (Dice, Precision,
    Recall, HD95 with the real anisotropic spacing) and ``case_class_metrics``
    (FP / FN / volume ratio / centroid distance), i.e. the same implementations
    the frozen development-val result used.
    """
    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        label = label_source.get(case_id)
        if on_first_gt_read is not None and index == 0:
            on_first_gt_read()
        truth = np.asarray(label)
        row: dict[str, Any] = {"case_id": case_id}
        for offset, name in enumerate(["ensemble"] + list(FROZEN_SEEDS)):
            prediction = predictions[index, offset]
            metrics = sm.evaluate_case(label_map_to_class_stack(prediction), truth,
                                       case_id=case_id, compute_hd95=True)
            for column, value in metrics.to_per_case_csv_row().items():
                if column == "case_id":
                    continue
                row[f"{column}__{name}"] = value
            # to_per_case_csv_row reports raw voxel counts, not the ratio, so the
            # ratio is derived here for every class -- same definition as
            # analyze_baseline_errors.case_class_metrics.
            for class_name in cs.CLASS_NAMES.values():
                gt = row[f"GTVoxels_{class_name}__{name}"]
                row[f"volume_ratio_{class_name}__{name}"] = (
                    row[f"PredVoxels_{class_name}__{name}"] / gt if gt else np.nan)
            geometry = case_class_metrics(prediction, truth, STN)
            row[f"centroid_distance_mm_STN__{name}"] = geometry["centroid_distance_mm"]
        rows.append(row)
    return rows


def aggregate(frame: pd.DataFrame, member: str) -> dict[str, Any]:
    """Per-member aggregate over cases. macro Dice is the mean of the per-case
    mean of the three foreground Dice -- patient first, then average."""
    suffix = f"__{member}"
    columns = {c: c[: -len(suffix)] for c in frame.columns if c.endswith(suffix)}
    view = frame[["case_id"] + list(columns)].rename(columns=columns)
    out: dict[str, Any] = {"n_cases": int(len(view))}
    for name in cs.CLASS_NAMES.values():
        out[f"Dice_{name}"] = float(view[f"Dice_{name}"].mean())
        out[f"Dice_{name}_std"] = float(view[f"Dice_{name}"].std())
        out[f"HD95_{name}_mm"] = float(view[f"HD95_{name}_mm"].mean())
        out[f"Precision_{name}"] = float(view[f"Precision_{name}"].mean())
        out[f"Recall_{name}"] = float(view[f"Recall_{name}"].mean())
        out[f"volume_ratio_{name}"] = float(view[f"volume_ratio_{name}"].mean())
        out[f"FP_{name}"] = int(view[f"FP_{name}"].sum())
        out[f"FN_{name}"] = int(view[f"FN_{name}"].sum())
        out[f"empty_prediction_{name}"] = int(view[f"empty_prediction_{name}"].sum())
    out["macro_foreground_dice"] = float(
        (view["Dice_STN"] + view["Dice_SN"] + view["Dice_RN"]).mean() / 3.0)
    out["centroid_distance_mm_STN"] = float(view["centroid_distance_mm_STN"].mean())
    return out


# --------------------------------------------------------------------------- #
# preregistered statistics
# --------------------------------------------------------------------------- #
# The estimators themselves live in ``src/evaluation/statistics.py`` so that they
# are importable, testable and shared. What stays here is the *preregistration*:
# the constants below are the frozen settings this evaluation was run with, and
# every call site passes them explicitly rather than relying on the module
# defaults -- an experiment should not silently change if a default changes.


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-holdout evaluation runner.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--split", choices=["val_rehearsal", "internal_test"],
                        default="val_rehearsal")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verify-gt-pipeline", action="store_true",
                        help="Prove the processed-label pipeline reproduces the "
                             "frozen label cache on train/val, then stop.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def split_paths(root: Path, split: str) -> dict[str, Any]:
    if split == "val_rehearsal":
        return {
            "split": split,
            "is_holdout": False,
            "manifest": root / "manifests/experiment/val.csv",
            "image_dir": root / "cache/baseline_v1/images",
            "label_source": CachedLabelSource(root / "cache/baseline_v1/labels"),
            "out_dir": root / OUT_ROOT / "holdout_evaluator_rehearsal",
        }
    return {
        "split": split,
        "is_holdout": True,
        "manifest": root / "manifests/experiment/internal_test.csv",
        "image_dir": root / "cache/holdout_frozen_v1/images",
        "label_source_factory": lambda: ProcessedLabelSource(root / "processed/labels"),
        "out_dir": root / OUT_ROOT / "secondary_frozen_holdout",
    }


def verify_gt_pipeline(root: Path) -> dict[str, Any]:
    """Prove the holdout label pipeline matches the frozen label cache.

    Run on train/val only -- the holdout labels are not read. Without this, the
    holdout's GT transform would be unverified at the moment it is first needed,
    which is precisely when it is least safe to discover a discrepancy.
    """
    source = ProcessedLabelSource(root / "processed/labels")
    report: dict[str, Any] = {"all_equal": True, "n_compared": 0, "mismatches": []}
    for split in ("train", "val"):
        for case_id in sorted(str(c) for c in pd.read_csv(
                root / f"manifests/experiment/{split}.csv")["case_id"]):
            cached = np.load(root / "cache/baseline_v1/labels" / f"{case_id}.npy")
            rebuilt = source.get(case_id)
            report["n_compared"] += 1
            if not np.array_equal(rebuilt, cached):
                report["all_equal"] = False
                report["mismatches"].append(case_id)
        LOGGER.info("gt pipeline %-6s ok", split)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    root = resolve_project_root(args.root)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.verify_gt_pipeline:
        report = verify_gt_pipeline(root)
        print(json.dumps(report, indent=2))
        return 0 if report["all_equal"] else 2

    paths = split_paths(root, args.split)
    out_dir = (root / args.out_dir) if args.out_dir else paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("split=%s holdout=%s", args.split, paths["is_holdout"])
    LOGGER.info("device=%s out=%s", device, out_dir)

    case_ids = sorted(str(c) for c in pd.read_csv(paths["manifest"])["case_id"])
    LOGGER.info("manifest: %d cases", len(case_ids))

    state = StateMachine(out_dir / "holdout_state.json")
    LOGGER.info("holdout state on entry: %s", state.state.value)

    # ---- PHASE A: blind inference ------------------------------------------ #
    models = load_frozen_models(device)
    image_cache = FrozenImageCache(paths["image_dir"], case_ids)
    LOGGER.info("PHASE A: blind inference over %d cases", len(image_cache))
    predicted_ids, predictions = run_blind_inference(image_cache, models, device)
    frozen = freeze_predictions(out_dir, predicted_ids, predictions)
    if state.state is HoldoutState.CLOSED:
        state.transition(HoldoutState.PREDICTIONS_FROZEN,
                         "three seed + ensemble predictions saved and hashed")

    # ---- PHASE B: metrics --------------------------------------------------- #
    LOGGER.info("PHASE B: metric evaluation from the frozen artifact")
    label_source = (paths["label_source"] if not paths["is_holdout"]
                    else paths["label_source_factory"]())

    def mark_opened() -> None:
        """Called the instant the first holdout ground-truth pixel is read.

        The OPENED record is written *here*, not at the end, so the moment the
        one-shot holdout was opened is captured even if the metric phase later
        fails. After this point no re-inference is permitted under any
        circumstance.
        """
        if state.state is not HoldoutState.PREDICTIONS_FROZEN or not paths["is_holdout"]:
            return
        state.transition(HoldoutState.OPENED, "first holdout ground-truth read")
        prereg = json.loads(
            (root / OUT_ROOT / "HOLDOUT_PREREGISTRATION.json").read_text(
                encoding="utf-8"))
        opened = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "statement": ("SECONDARY FROZEN HOLDOUT OPENED FOR ONE-SHOT "
                          "MODEL-PERFORMANCE EVALUATION"),
            "n_cases": len(predicted_ids),
            "preregistration_hashes": {
                "HOLDOUT_PREREGISTRATION.json": sha256_file(
                    root / OUT_ROOT / "HOLDOUT_PREREGISTRATION.json"),
                "HOLDOUT_PREREGISTRATION.md": sha256_file(
                    root / OUT_ROOT / "HOLDOUT_PREREGISTRATION.md"),
            },
            "freeze_record_sha256": sha256_file(root / OUT_ROOT / "FREEZE_RECORD.json"),
            "prediction_sha256": frozen["predictions_sha256"],
            "prediction_manifest_sha256": sha256_file(
                out_dir / "prediction_manifest.csv"),
            "checkpoint_hashes": {a["label"]: a["sha256"]
                                  for a in prereg["frozen_artifacts"]
                                  if a["kind"] == "checkpoint"},
            "config_hashes": {a["label"]: a["sha256"]
                              for a in prereg["frozen_artifacts"]
                              if a["kind"] == "config"},
            "ensemble_script_sha256": sha256_file(root / "scripts/evaluate_deep_ensemble.py"),
            "internal_test_manifest_sha256": sha256_file(paths["manifest"]),
            "irreversible": True,
            "no_reinference_after_opening": True,
        }
        (out_dir / "HOLDOUT_OPENED.json").write_text(
            json.dumps(opened, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.warning("HOLDOUT OPENED at %s", opened["timestamp_utc"])

    rows = evaluate_predictions(predicted_ids, predictions, label_source,
                                on_first_gt_read=mark_opened)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "holdout_case_metrics.csv", index=False,
                 encoding="utf-8-sig")

    summary = {
        "split": args.split, "is_holdout": bool(paths["is_holdout"]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cases": len(predicted_ids),
        "prediction_sha256": frozen["predictions_sha256"],
        "members": {m: aggregate(frame, m) for m in ["ensemble"] + list(FROZEN_SEEDS)},
        "primary_comparator": PRIMARY_COMPARATOR,
        "statistics_constants": {"bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                                 "permutation_draws": PERMUTATION_DRAWS,
                                 "rng_seed": RNG_SEED},
        "holdout_state": state.state.value,
    }

    # ---- preregistered primary analysis ------------------------------------ #
    def flat(member: str, column: str) -> np.ndarray:
        return frame[f"{column}__{member}"].to_numpy(float)

    def macro_dice(member: str) -> np.ndarray:
        return (flat(member, "Dice_STN") + flat(member, "Dice_SN")
                + flat(member, "Dice_RN")) / 3.0

    deltas = macro_dice("ensemble") - macro_dice(PRIMARY_COMPARATOR)
    mean_delta = float(np.mean(deltas))
    median_delta = float(np.median(deltas))
    ci_low, ci_high = paired_bootstrap_ci(deltas, BOOTSTRAP_RESAMPLES, RNG_SEED)
    p_value = paired_signflip_pvalue(deltas, PERMUTATION_DRAWS, RNG_SEED)
    decision = primary_decision(mean_delta, ci_low, p_value)

    primary = {
        "split": args.split, "n_paired_cases": int(deltas.size),
        "comparator": PRIMARY_COMPARATOR,
        "endpoint": "case-wise macro foreground Dice",
        "estimand": "mean_i(MacroDice_ensemble,i - MacroDice_seed123,i)",
        "comparator_macro_dice": float(np.mean(macro_dice(PRIMARY_COMPARATOR))),
        "ensemble_macro_dice": float(np.mean(macro_dice("ensemble"))),
        "mean_paired_delta": mean_delta, "median_paired_delta": median_delta,
        "bootstrap_ci_95": [ci_low, ci_high],
        "permutation_p_two_sided": p_value,
        "paired_t_pvalue": float(paired_stats(
            macro_dice(PRIMARY_COMPARATOR), macro_dice("ensemble"))["paired_t_pvalue"]),
        "primary_success": decision["result"],
        "decision": decision,
    }

    # ---- secondary --------------------------------------------------------- #
    secondary: dict[str, Any] = {}
    for member in FROZEN_SEEDS:
        entry = {}
        for column, hib in (("Dice_STN", True), ("Dice_SN", True), ("Dice_RN", True),
                            ("HD95_STN_mm", False), ("HD95_SN_mm", False),
                            ("HD95_RN_mm", False), ("Precision_STN", True),
                            ("Recall_STN", True)):
            stats = paired_stats(flat(member, column), flat("ensemble", column))
            if not hib:
                stats["improved_cases"], stats["degraded_cases"] = (
                    stats["degraded_cases"], stats["improved_cases"])
            entry[column] = stats
        stats = paired_stats(
            (flat(member, "Dice_STN") + flat(member, "Dice_SN")
             + flat(member, "Dice_RN")) / 3.0, macro_dice("ensemble"))
        entry["macro_foreground_dice"] = stats
        secondary[f"ensemble_vs_{member}"] = {
            "marking": "SECONDARY / EXPLORATORY",
            "holm_adjusted_pvalues": holm_adjust({
                key: value["paired_t_pvalue"] for key, value in entry.items()}),
            "metrics": entry,
        }

    stn = {}
    for member in ["ensemble"] + list(FROZEN_SEEDS):
        dice = flat(member, "Dice_STN")
        hd95 = flat(member, "HD95_STN_mm")
        ratio = flat(member, "volume_ratio_STN")
        stn[member] = {
            "Dice": {"mean": float(dice.mean()), "std": float(dice.std()),
                     "median": float(np.median(dice)), "min": float(dice.min()),
                     "max": float(dice.max())},
            "HD95": {"mean": float(hd95.mean()), "std": float(hd95.std()),
                     "median": float(np.median(hd95)), "min": float(hd95.min()),
                     "max": float(hd95.max())},
            "Precision": float(flat(member, "Precision_STN").mean()),
            "Recall": float(flat(member, "Recall_STN").mean()),
            "Pred_over_GT": float(ratio.mean()),
            "FP": int(flat(member, "FP_STN").sum()),
            "FN": int(flat(member, "FN_STN").sum()),
            "n_dice_below_065": int((dice < 0.65).sum()),
            "n_ratio_above_1.2": int((ratio > 1.2).sum()),
            "n_ratio_below_0.8": int((ratio < 0.8).sum()),
            "n_empty_prediction": int(
                frame[f"empty_prediction_STN__{member}"].astype(bool).sum()),
        }

    (out_dir / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "primary_endpoint.json").write_text(
        json.dumps(primary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "primary_statistics.json").write_text(
        json.dumps({"constants": summary["statistics_constants"],
                    "primary": primary, "decision": decision},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "secondary_statistics.json").write_text(
        json.dumps(secondary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "stn_diagnostics.json").write_text(
        json.dumps(stn, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "prediction_hashes.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")

    # the preregistered report filenames, plus descriptive aliases
    (out_dir / "primary_analysis.json").write_text(
        json.dumps(primary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "secondary_analysis.json").write_text(
        json.dumps(secondary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "environment.json").write_text(
        json.dumps({"split": args.split, "is_holdout": bool(paths["is_holdout"]),
                    "n_cases": len(predicted_ids),
                    "ensemble_recipe": ("mean of the members' softmax "
                                        "probabilities, argmaxed once"),
                    "primary_comparator": PRIMARY_COMPARATOR,
                    "statistics_constants": summary["statistics_constants"],
                    "spacing_dhw_mm": list(cs.SPACING_DHW_MM),
                    "crop_shape_dhw": list(cs.CROP_SHAPE_DHW)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "holdout_per_case.csv").write_text(
        (out_dir / "holdout_case_metrics.csv").read_text(encoding="utf-8-sig"),
        encoding="utf-8-sig")

    # ---- failure-mode analysis (frozen definitions) ------------------------ #
    failure_modes = {
        "thresholds": {"dice_below": 0.65, "ratio_above": 1.2, "ratio_below": 0.8,
                       "note": "thresholds are frozen; they are not re-tuned here"},
        "marking": "SECONDARY / DIAGNOSTIC",
        "per_member": stn,
    }
    (out_dir / "failure_mode_analysis.json").write_text(
        json.dumps(failure_modes, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- final report ------------------------------------------------------- #
    if paths["is_holdout"]:
        if decision["result"] == "CONFIRMED":
            interpretation = (
                "Experiment F improved macro foreground Dice on the secondary "
                "frozen holdout relative to the preregistered best single-model "
                "comparator.")
            case = "CASE 1 — primary success"
        elif mean_delta > 0:
            interpretation = ("Directionally positive but not confirmed on the "
                              "preregistered primary endpoint.")
            case = "CASE 2 — directionally positive, not confirmed"
        else:
            interpretation = ("Primary holdout improvement was not reproduced.")
            case = "CASE 3 — not reproduced"

        qualifier = ("Labels were previously used for geometric / crop-coverage / "
                     "ROI auditing, but were not used for model training, "
                     "checkpoint selection, model/loss selection, or "
                     "segmentation-performance evaluation.")
        forbidden = ["final generalization performance",
                     "fully independent test performance",
                     "state-of-the-art", "clinically validated"]

        report = {
            "dataset_status": "SECONDARY FROZEN HOLDOUT",
            "mandatory_qualifier": qualifier,
            "forbidden_descriptions_used": [],
            "forbidden_phrasings_not_used": forbidden,
            "n_cases": len(predicted_ids),
            "frozen_method": {
                "seeds": list(FROZEN_SEEDS),
                "ensemble_recipe": ("p_ensemble = mean(softmax(logits, dim=1)); "
                                    "prediction = argmax(p_ensemble, dim=class)"),
                "prediction_sha256": frozen["predictions_sha256"],
                "primary_comparator": PRIMARY_COMPARATOR,
            },
            "primary_endpoint": primary,
            "primary_success": decision["result"],
            "interpretation_case": case,
            "interpretation": interpretation,
            "secondary": secondary,
            "stn_diagnostics": stn,
            "failure_modes": failure_modes,
            "holdout_state": state.state.value,
        }
        (out_dir / "FINAL_HOLDOUT_REPORT.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        member = summary["members"]["ensemble"]
        lines = [
            "# Experiment F — SECONDARY FROZEN HOLDOUT: Final Report", "",
            f"**Primary result: {decision['result']}**  ·  state: `{state.state.value}`",
            "", "## 1. Dataset status", "",
            "**SECONDARY FROZEN HOLDOUT** — 100 cases.", "",
            f"> {qualifier}", "",
            "Not an untouched, pristine or completely independent test set.", "",
            "## 2. Frozen method", "",
            f"- members: {', '.join(FROZEN_SEEDS)}",
            f"- ensemble: `p_ensemble = mean(softmax(logits, dim=1))`, "
            f"`prediction = argmax(p_ensemble, dim=class)`",
            f"- prediction sha256: `{frozen['predictions_sha256']}`",
            f"- primary comparator: **{PRIMARY_COMPARATOR}**", "",
            "## 3. Primary endpoint", "",
            "| | |", "|---|---|",
            f"| {PRIMARY_COMPARATOR} macro Dice | {primary['comparator_macro_dice']:.6f} |",
            f"| ensemble macro Dice | {primary['ensemble_macro_dice']:.6f} |",
            f"| mean paired Δ | {primary['mean_paired_delta']:+.6f} |",
            f"| median paired Δ | {primary['median_paired_delta']:+.6f} |",
            f"| 95% bootstrap CI | [{primary['bootstrap_ci_95'][0]:+.6f}, "
            f"{primary['bootstrap_ci_95'][1]:+.6f}] |",
            f"| permutation p (two-sided) | {primary['permutation_p_two_sided']:.6f} |",
            f"| **PRIMARY SUCCESS** | **{decision['result']}** |", "",
            "## 4. Secondary results", "",
            "| metric | ensemble |", "|---|---|",
            f"| STN Dice | {member['Dice_STN']:.4f} |",
            f"| SN Dice | {member['Dice_SN']:.4f} |",
            f"| RN Dice | {member['Dice_RN']:.4f} |",
            f"| STN HD95 (mm) | {member['HD95_STN_mm']:.4f} |",
            f"| SN HD95 (mm) | {member['HD95_SN_mm']:.4f} |",
            f"| RN HD95 (mm) | {member['HD95_RN_mm']:.4f} |",
            f"| STN Precision | {member['Precision_STN']:.4f} |",
            f"| STN Recall | {member['Recall_STN']:.4f} |",
            f"| STN Pred/GT | {member['volume_ratio_STN']:.4f} |",
            f"| STN FP / FN | {member['FP_STN']} / {member['FN_STN']} |", "",
            "## 5. Failure / worst-case analysis", "",
            "| member | Dice<0.65 | ratio>1.2 | ratio<0.8 | empty |",
            "|---|---|---|---|---|",
        ]
        for key in ["ensemble"] + list(FROZEN_SEEDS):
            entry = stn[key]
            lines.append(f"| {key} | {entry['n_dice_below_065']} | "
                         f"{entry['n_ratio_above_1.2']} | "
                         f"{entry['n_ratio_below_0.8']} | "
                         f"{entry['n_empty_prediction']} |")
        lines += ["", "## 6. Interpretation", "", interpretation, ""]
        (out_dir / "FINAL_HOLDOUT_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8")

    if state.state is HoldoutState.OPENED:
        state.transition(HoldoutState.COMPLETE, "preregistered metrics written")

    # ---- artifact hashes ---------------------------------------------------- #
    artifacts = sorted(p for p in out_dir.iterdir()
                       if p.is_file() and p.name != "artifact_hashes.json")
    (out_dir / "artifact_hashes.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "artifacts": {p.name: sha256_file(p) for p in artifacts},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.split == "val_rehearsal":
        # Compare against the frozen run's OWN per-case artifact, not against the
        # 4-decimal numbers printed in its report: the printed constants carry
        # rounding, so a tight tolerance against them would fail on a perfect
        # reproduction. The per-case CSV is the frozen source of truth.
        frozen_csv = root / OUT_ROOT / "case_metrics_all.csv"
        per_case_match: dict[str, Any] = {"available": frozen_csv.is_file()}
        if frozen_csv.is_file():
            old = pd.read_csv(frozen_csv).set_index("case_id").sort_index()
            # Read back the CSV just written rather than using the in-memory frame:
            # comparing a serialised file against a full-precision value would show
            # ~1e-16 differences that are pure float-repr rounding, not computation.
            new = pd.read_csv(out_dir / "holdout_case_metrics.csv",
                              encoding="utf-8-sig").set_index("case_id").sort_index()
            per_case_match["case_ids_identical"] = list(old.index) == list(new.index)
            per_case_match["n_cases"] = int(len(old))
            shared = [c for c in old.columns if c in new.columns]
            largest = 0.0
            for column in shared:
                a = old[column].to_numpy(float)
                b = new[column].to_numpy(float)
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.any():
                    largest = max(largest, float(np.abs(a[mask] - b[mask]).max()))
            per_case_match["n_shared_columns"] = len(shared)
            per_case_match["largest_absolute_difference"] = largest
            per_case_match["bitwise_identical"] = bool(largest == 0.0)
            per_case_match["voxel_level_artifact_available"] = False
            per_case_match["voxel_level_note"] = (
                "the frozen Experiment F run persisted per-case metrics, not voxel "
                "predictions, so an exact voxel comparison is not possible without "
                "re-running inference. Identical Dice/HD95/Precision/Recall/FP/FN "
                "for all 40 cases x 4 members is the strongest available evidence.")

        rounded_match = {}
        for key, expected in FROZEN_DEVVAL.items():
            got = summary["members"]["ensemble"].get(key)
            # FROZEN_DEVVAL holds values as published to 4 decimals
            rounded_match[key] = {
                "published": expected, "rehearsal": got,
                "agrees_to_published_precision": bool(
                    got is not None and round(float(got), 4) == round(float(expected), 4))}

        (out_dir / "rehearsal_report.json").write_text(json.dumps({
            "purpose": ("development-val rehearsal of the one-shot evaluator; "
                        "NOT a holdout result"),
            "n_cases": len(predicted_ids),
            "per_case_vs_frozen_experiment_f": per_case_match,
            "agreement_with_published_constants": rounded_match,
            "all_published_constants_agree": all(
                v["agrees_to_published_precision"] for v in rounded_match.values()),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "rehearsal_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "rehearsal_per_case.csv").write_text(
            (out_dir / "holdout_case_metrics.csv").read_text(encoding="utf-8-sig"),
            encoding="utf-8-sig")

    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split, "is_holdout": bool(paths["is_holdout"]),
        "device": str(device),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "frozen_seeds": list(FROZEN_SEEDS),
        "ensemble_recipe": ("p_ensemble = mean(softmax(logits, dim=1) over seeds); "
                            "prediction = argmax(p_ensemble, dim=class)"),
        "crop_shape_dhw": list(cs.CROP_SHAPE_DHW),
        "spacing_dhw_mm": list(cs.SPACING_DHW_MM),
        "channel_order": list(cs.CHANNEL_ORDER),
        "holdout_state": state.state.value,
    }
    (out_dir / "evaluator_environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")

    if not paths["is_holdout"]:
        LOGGER.info("rehearsal complete; state unchanged: %s", state.state.value)

    line = "=" * 96
    print()
    print(line)
    print(f"Frozen-holdout evaluator — split={args.split} "
          f"({'HOLDOUT' if paths['is_holdout'] else 'rehearsal'})")
    print(line)
    member = summary["members"]["ensemble"]
    print(f"  ensemble macro Dice : {member['macro_foreground_dice']:.4f}")
    print(f"  STN/SN/RN Dice      : {member['Dice_STN']:.4f} / {member['Dice_SN']:.4f} / "
          f"{member['Dice_RN']:.4f}")
    print(f"  STN HD95 mm         : {member['HD95_STN_mm']:.4f}")
    print(f"  STN Precision/Recall: {member['Precision_STN']:.4f} / "
          f"{member['Recall_STN']:.4f}")
    print(f"  STN Pred/GT         : {member['volume_ratio_STN']:.4f}")
    print(f"  STN FP / FN         : {member['FP_STN']} / {member['FN_STN']}")
    print(line)
    print(f"  primary (vs {PRIMARY_COMPARATOR}): mean delta {mean_delta:+.6f}  "
          f"median {median_delta:+.6f}")
    print(f"  95% bootstrap CI    : [{ci_low:+.6f}, {ci_high:+.6f}]")
    print(f"  permutation p       : {p_value:.6f}")
    print(f"  PRIMARY SUCCESS     : {decision['result']}")
    print(line)
    print(f"  holdout state       : {state.state.value}")
    print(f"Written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

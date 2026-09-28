#!/usr/bin/env python
"""INDEPENDENT REPRODUCTION EVALUATION on the fixed 100-case ``internal_test``.

What this produces, in one sentence
-----------------------------------
Metrics for the models **this reproduction trained itself**, on the frozen
``internal_test`` split published with the repository, using the project's
already-frozen metric, ensemble recipe, comparator rule and statistics.

What it is NOT -- read this before quoting any number it prints
---------------------------------------------------------------
* **Not the author's historical evaluation.** Those results are the frozen,
  one-shot evaluation under ``results/subject_clean_holdout/`` (and the original
  Experiment F under ``results/experiments/baseline_deep_ensemble/``). This script
  refuses to write anywhere inside either namespace, so it cannot overwrite them.
* **Not a fresh, unseen test set.** The same 100 subject-clean cases were already
  evaluated once by the author before this repository was published. What a
  reproduction can claim is *weight-level* independence: the checkpoints loaded
  here were trained from scratch and never took part in any model selection on
  these cases. Novelty of the *data* does not hold, and nothing this script emits
  may be described as "a fresh holdout", "never used before" or "fully
  independent new test set".
* **Not a new experiment.** Nothing scientific is chosen here: no threshold, no
  metric definition, no ensemble member set, no comparator.

Protections (none are optional, none can be switched off)
---------------------------------------------------------
* The comparator is read from the **local** frozen pre-registration and must be
  ``locked`` and be one of the frozen ensemble members. It is never hard-coded and
  never chosen here.
* The ensemble recipe is the frozen one -- ``softmax(logits, dim=1)`` per member,
  arithmetic mean of the probability maps, a single ``argmax`` at the end --
  executed by :func:`evaluate_frozen_holdout.run_blind_inference`.
* The metric and the paired statistics are the frozen implementations
  (``src/evaluation/segmentation_metrics.py``,
  ``src/evaluation/statistics.py``) with the **preregistered constants**; the
  script refuses to run if the pre-registration asks for different constants.
* The ``internal_test`` image tensors are built with the identical pipeline used
  for the training cache and verified **bitwise** against that cache before a
  single forward pass happens.
* The ``internal_test`` case set must not overlap ``train`` / ``val`` /
  ``challenge_test``.
* Ground truth is read from ``--data-root`` (the official processed tree). No
  repository-internal ``processed/labels`` path is assumed.

Usage
-----
    python scripts/evaluate_reproduction.py --data-root <PDCADxFoundation path>
    python scripts/reproduce.py evaluate --data-root <PDCADxFoundation path>
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import build_holdout_image_cache as BHIC  # noqa: E402
import evaluate_frozen_holdout as EFH  # noqa: E402
from data import crop_spec as cs  # noqa: E402
from evaluation.statistics import (  # noqa: E402
    paired_bootstrap_ci,
    paired_signflip_pvalue,
    primary_decision,
)
from models.anisotropic_unet3d import AnisotropicUNet3D, UNet3DConfig  # noqa: E402
from training.train_baseline import load_checkpoint  # noqa: E402

LOGGER = logging.getLogger("evaluate_reproduction")

#: Namespaces that hold the author's frozen / historical artifacts. Writing into
#: any of them is refused outright -- the reproduction evaluation is a *separate*
#: record, and merging the two is exactly the error this guard exists to prevent.
FROZEN_NAMESPACES: tuple[str, ...] = (
    "results/subject_clean_holdout",              # author's one-shot corrected re-evaluation
    "results/experiments/baseline_deep_ensemble",  # author's original Experiment F holdout
    "results/experiments_subject_clean_v1",        # author's 21 formal runs
    "results/subject_clean_rerun",                 # author's frozen records
)

#: Record kind written by this script. Distinct from every author-side record.
RECORD_KIND = "INDEPENDENT_REPRODUCTION_EVALUATION"

#: The novelty statement is a constant string so that it cannot be quietly
#: reworded per run. It is copied verbatim into the report and the markdown.
NOVELTY_STATEMENT = (
    "The 100 internal_test cases are the SAME cases the author evaluated once "
    "before this repository was published. This is NOT a fresh, never-used or "
    "unseen test set, and it is NOT a new independent holdout. Only weight-level "
    "independence holds: the checkpoints evaluated here were trained from scratch "
    "and took no part in any model selection on these cases."
)

DEFAULT_MANIFEST = "manifests/subject_clean_v1/internal_test.csv"
DEFAULT_IMAGE_CACHE = "cache/reproduction_eval_v1/images"
DEFAULT_REFERENCE_CACHE = "cache/baseline_v1"
DEFAULT_OUT_DIR = "results/reproduction_eval_v1/internal_test"
DEFAULT_FREEZE_RECORD = "results/subject_clean_rerun/CLEAN_ENSEMBLE_FREEZE_RECORD.json"
DEFAULT_PREREGISTRATION = "results/subject_clean_rerun/SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_case_ids(manifest: Path) -> list[str]:
    frame = pd.read_csv(manifest, encoding="utf-8-sig", dtype=str)
    if "case_id" not in frame.columns:
        raise SystemExit(f"error: {manifest} has no case_id column")
    case_ids = [str(c) for c in frame["case_id"]]
    if len(set(case_ids)) != len(case_ids):
        raise SystemExit(f"error: {manifest} contains duplicate case ids")
    return case_ids


def assert_outside_frozen_namespaces(out_dir: Path, root: Path) -> None:
    """Refuse to write into any namespace that holds frozen author results."""
    resolved = out_dir.resolve()
    for relative in FROZEN_NAMESPACES:
        frozen = (root / relative).resolve()
        if resolved == frozen or frozen in resolved.parents:
            raise SystemExit(
                f"error: output directory {resolved} is inside the frozen namespace "
                f"{relative}.\n"
                "       That directory holds the author's existing results; a "
                "reproduction evaluation must never write there.\n"
                "       Use the default --out-dir, or another directory outside "
                f"{relative}."
            )


def resolve_manifest_paths(manifest: Path, data_root: Path) -> dict[str, Path]:
    """Per-case GT label path, read from the manifest and anchored at --data-root.

    The repository-internal ``processed/labels`` path is deliberately *not*
    assumed: the manifest carries the path, ``--data-root`` locates it. A manifest
    without usable label paths is an error, not a cue to guess a default.
    """
    frame = pd.read_csv(manifest, encoding="utf-8-sig", dtype=str)
    if "label_path" not in frame.columns:
        raise SystemExit(
            f"error: {manifest} has no label_path column, so the ground truth for "
            "this split cannot be located from --data-root."
        )
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for case_id, raw in zip(frame["case_id"], frame["label_path"]):
        relative = str(raw).strip()
        if not relative or relative.lower() == "nan":
            missing.append(str(case_id))
            continue
        paths[str(case_id)] = (data_root / relative).resolve()
    if missing:
        raise SystemExit(
            f"error: {len(missing)} case(s) in {manifest.name} have no label_path, "
            f"e.g. {missing[:5]}.\n"
            "       The published internal_test labels come from the official "
            "PDCADxFoundation release; make sure --data-root points at it."
        )
    return paths


def image_paths_from_manifest(manifest: Path, data_root: Path,
                              case_ids: Sequence[str]) -> dict[str, dict[str, Path]]:
    """Per-case ``{modality: path}`` from the manifest, anchored at --data-root."""
    frame = pd.read_csv(manifest, encoding="utf-8-sig", dtype=str)
    frame = frame.set_index("case_id")
    out: dict[str, dict[str, Path]] = {}
    problems: list[str] = []
    for case_id in case_ids:
        if case_id not in frame.index:
            problems.append(f"{case_id}: absent from the manifest")
            continue
        row = frame.loc[case_id]
        entry: dict[str, Path] = {}
        for modality in cs.CHANNEL_ORDER:
            column = f"{modality}_path"
            raw = str(row.get(column, "")).strip()
            if not raw or raw.lower() == "nan":
                problems.append(f"{case_id}: no {column}")
                continue
            entry[modality] = (data_root / raw).resolve()
        out[case_id] = entry
    if problems:
        raise SystemExit(
            "error: image paths cannot be resolved from the manifest:\n  "
            + "\n  ".join(problems[:10])
        )
    return out


# --------------------------------------------------------------------------- #
# image cache
# --------------------------------------------------------------------------- #

def build_or_verify_image_cache(manifest: Path, manifest_dir: Path, data_root: Path,
                                image_cache_dir: Path, reference_cache: Path,
                                case_ids: Sequence[str], *,
                                overwrite: bool) -> dict[str, Any]:
    """Build the internal_test image tensors and prove the pipeline is unchanged.

    Two independent things happen here, and both are required:

    1. **equivalence.** The training-cache pipeline is re-run over the whole
       reference cache (train + val) and compared *bitwise*. This is what makes
       "the evaluation tensors are the same tensors the trainer would have built"
       a measured fact rather than a comment.
    2. **build.** The internal_test tensors are written with the identical
       function. ``build_image_tensor`` has no label argument, so no code path in
       this step can open a ground-truth file.
    """
    manifest_dir = manifest_dir.resolve()
    reference_cache = reference_cache.resolve()
    if not reference_cache.is_dir():
        raise SystemExit(
            f"error: the reference cache {reference_cache} is missing, so the "
            "image pipeline cannot be verified.\n"
            "       Run `python scripts/reproduce.py prepare --data-root ...` first; "
            "the reference cache is what the training run uses."
        )

    equivalence = BHIC.verify_equivalence(reference_cache, data_root / "processed" / "images",
                                          manifest_dir)
    if not equivalence["all_equal"]:
        raise SystemExit(
            "error: the internal_test image pipeline does NOT reproduce the "
            "training cache bitwise:\n  "
            + "\n  ".join(equivalence["mismatches"][:10])
            + "\n       Refusing to evaluate on tensors the trainer would not have built."
        )

    image_dir = data_root / "processed" / "images"
    image_cache_dir = image_cache_dir.resolve()
    (image_cache_dir / "images").mkdir(parents=True, exist_ok=True)

    image_paths = image_paths_from_manifest(manifest, data_root, case_ids)
    built = reused = 0
    for case_id in case_ids:
        out_path = image_cache_dir / "images" / f"{case_id}.npy"
        if out_path.is_file() and not overwrite:
            tensor = np.load(out_path)
            reused += 1
        else:
            paths = image_paths[case_id]
            absent = [m for m, p in paths.items() if not p.is_file()]
            if absent:
                raise SystemExit(
                    f"error: {case_id}: missing processed image(s) {absent} under "
                    f"{data_root}. Run `prepare` against this --data-root first."
                )
            tensor = BHIC.build_image_tensor(case_id, paths)
            np.save(out_path, tensor)
            built += 1
        expected = (cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW)
        if tensor.shape != expected:
            raise SystemExit(f"error: {case_id}: cached shape {tensor.shape} != {expected}")
        if tensor.dtype != np.float32:
            raise SystemExit(f"error: {case_id}: cached dtype {tensor.dtype} is not float32")
        if not np.isfinite(tensor).all():
            raise SystemExit(f"error: {case_id}: image tensor contains NaN/Inf")

    LOGGER.info("image cache %s: %d built, %d reused", image_cache_dir, built, reused)
    return {
        "path": str(image_cache_dir.relative_to(REPO)) if REPO in image_cache_dir.parents
                else str(image_cache_dir),
        "n_cases": len(case_ids),
        "built": built,
        "reused": reused,
        "pipeline_equivalence_with_training_cache": {
            "reference_cache": str(reference_cache),
            "all_equal": equivalence["all_equal"],
            "n_compared": equivalence["n_compared"],
            "per_split": equivalence["splits"],
        },
        "reads_ground_truth": False,
    }


# --------------------------------------------------------------------------- #
# frozen records
# --------------------------------------------------------------------------- #

def load_frozen_records(freeze_path: Path, prereg_path: Path) -> tuple[dict, dict]:
    for path, name in ((freeze_path, "ensemble freeze record"),
                       (prereg_path, "pre-registration")):
        if not path.is_file():
            raise SystemExit(
                f"error: {name} not found: {path}\n"
                "       These records are written by "
                "`python scripts/reproduce.py summarize --yes`, which selects the "
                "comparator on the development validation split.\n"
                "       The comparator must never be chosen here, so this script "
                "will not fall back to a default."
            )
    return (json.loads(freeze_path.read_text(encoding="utf-8")),
            json.loads(prereg_path.read_text(encoding="utf-8")))


def resolve_comparator(prereg: dict, freeze: dict) -> str:
    """The comparator, read from the frozen pre-registration.

    Identical logic to the author-side evaluator: the block must exist, be
    ``locked``, name a frozen ensemble member and carry that member's checkpoint
    path. A comparator that is hard-coded here, or chosen after the fact, is
    refused.
    """
    block = prereg.get("primary_comparator")
    if not isinstance(block, dict) or not block.get("id"):
        raise SystemExit(
            "error: the pre-registration has no primary_comparator.id; the "
            "comparator is chosen on the validation split and frozen before any "
            "test evaluation. Refusing to guess one now."
        )
    if not block.get("locked", False):
        raise SystemExit("error: primary_comparator is not marked locked.")
    comparator = str(block["id"])
    frozen = {f"seed{m['seed']}": m for m in freeze["members"]}
    if comparator not in frozen:
        raise SystemExit(
            f"error: the pre-registration names comparator {comparator!r}, which is "
            f"not a frozen ensemble member (members: {sorted(frozen)})."
        )
    member = frozen[comparator]
    if block.get("checkpoint") and block["checkpoint"] != member["checkpoint_path"]:
        raise SystemExit(
            f"error: comparator checkpoint {block['checkpoint']} does not match the "
            f"frozen member {member['checkpoint_path']}."
        )
    return comparator


def check_statistics_match_preregistration(prereg: dict) -> dict[str, Any]:
    """The frozen constants must equal what the pre-registration asked for."""
    frozen_stats = {
        "bootstrap_n_resamples": EFH.BOOTSTRAP_RESAMPLES,
        "permutation_n_draws": EFH.PERMUTATION_DRAWS,
        "rng_seed": EFH.RNG_SEED,
    }
    declared = (prereg.get("statistical_analysis") or {})
    wanted = {
        "bootstrap_n_resamples": (declared.get("bootstrap") or {}).get("n_resamples"),
        "permutation_n_draws": (declared.get("permutation") or {}).get("n_draws"),
        "rng_seed": (declared.get("bootstrap") or {}).get("seed"),
    }
    mismatch = {k: (wanted[k], frozen_stats[k]) for k in frozen_stats
                if wanted[k] is not None and int(wanted[k]) != int(frozen_stats[k])}
    if mismatch:
        raise SystemExit(
            "error: the frozen statistics constants differ from the "
            f"pre-registration (preregistered value, frozen value): {mismatch}.\n"
            "       The statistical calculation may not be changed here."
        )
    return {**frozen_stats, "matches_preregistration": True}


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

def load_members(freeze: dict, root: Path, device: Any) -> tuple[dict[str, Any], list[dict]]:
    """Load this reproduction's ensemble members from the local freeze record."""
    records: list[dict] = []
    models: dict[str, Any] = {}
    for member in freeze["members"]:
        path = (root / str(member["checkpoint_path"])).resolve()
        if not path.is_file():
            raise SystemExit(
                f"error: ensemble member checkpoint missing: {path}\n"
                "       Train the runs first (`python scripts/reproduce.py train`) "
                "and re-run summarize."
            )
        digest = sha256_file(path)
        if member.get("checkpoint_sha256") and digest != member["checkpoint_sha256"]:
            raise SystemExit(
                f"error: {path} does not match the hash in the freeze record "
                f"({digest[:16]}… != {member['checkpoint_sha256'][:16]}…)."
            )
        payload = load_checkpoint(path)
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
        name = f"seed{member['seed']}"
        models[name] = model
        records.append({
            "member": name,
            "checkpoint_path": str(member["checkpoint_path"]),
            "checkpoint_sha256": digest,
            "best_epoch": member.get("best_epoch"),
            "trained_by_this_reproduction": True,
        })
        LOGGER.info("loaded %s: %s (epoch %s, sha %s…)", name, path,
                    payload.get("epoch"), digest[:16])

    if set(models) != set(EFH.FROZEN_SEEDS):
        raise SystemExit(
            f"error: this script evaluates the frozen three-member ensemble "
            f"{sorted(EFH.FROZEN_SEEDS)}, but the freeze record lists "
            f"{sorted(models)}.\n"
            "       The member set is part of the frozen ensemble recipe, so it is "
            "not re-chosen here."
        )
    return models, records


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent reproduction evaluation on internal_test "
                    "(the author's frozen results are never touched).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None,
                        help="Project root every relative path is anchored to "
                             "(default: the repository this script lives in).")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory holding the official PDCADxFoundation tree "
                             "(the same one passed to `prepare`).")
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST),
                        help=f"internal_test manifest (default: {DEFAULT_MANIFEST}).")
    parser.add_argument("--image-cache-dir", type=Path, default=Path(DEFAULT_IMAGE_CACHE),
                        help=f"Where the internal_test image tensors live "
                             f"(default: {DEFAULT_IMAGE_CACHE}).")
    parser.add_argument("--reference-cache", type=Path, default=Path(DEFAULT_REFERENCE_CACHE),
                        help="Training cache the image pipeline is verified against "
                             f"(default: {DEFAULT_REFERENCE_CACHE}).")
    parser.add_argument("--reference-manifest-dir", type=Path, default=None,
                        help="Directory holding the reference cache's train/val "
                             "manifests (default: the directory of --manifest).")
    parser.add_argument("--freeze-record", type=Path, default=Path(DEFAULT_FREEZE_RECORD),
                        help=f"Local ensemble freeze record (default: {DEFAULT_FREEZE_RECORD}).")
    parser.add_argument("--preregistration", type=Path, default=Path(DEFAULT_PREREGISTRATION),
                        help=f"Local pre-registration (default: {DEFAULT_PREREGISTRATION}).")
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR),
                        help=f"Where this reproduction's results are written "
                             f"(default: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing reproduction report in --out-dir. "
                             "This never touches the author's namespaces.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    from utils.paths import resolve_path, resolve_project_root

    root = resolve_project_root(args.root)
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"error: --data-root is not a directory: {data_root}")

    manifest = resolve_path(args.manifest, root)
    image_cache_dir = resolve_path(args.image_cache_dir, root)
    reference_cache = resolve_path(args.reference_cache, root)
    out_dir = resolve_path(args.out_dir, root)
    assert_outside_frozen_namespaces(out_dir, root)

    report_path = out_dir / "REPRODUCTION_EVALUATION_REPORT.json"
    if report_path.is_file() and not args.overwrite:
        raise SystemExit(
            f"error: {report_path} already exists.\n"
            "       Pass --overwrite to replace this reproduction's own report "
            "(the author's frozen results are never touched either way)."
        )

    print("=" * 88)
    print("INDEPENDENT REPRODUCTION EVALUATION -- internal_test")
    print("=" * 88)
    print("This evaluates the checkpoints THIS reproduction trained, on the frozen")
    print("internal_test split published with the repository. It is not the author's")
    print("historical evaluation, and internal_test is not a fresh unseen test set:")
    print(f"  {NOVELTY_STATEMENT}")
    print("=" * 88)

    if not manifest.is_file():
        raise SystemExit(f"error: internal_test manifest not found: {manifest}")
    case_ids = read_case_ids(manifest)

    # ---- frozen records, comparator, statistics --------------------------- #
    freeze, prereg = load_frozen_records(resolve_path(args.freeze_record, root),
                                         resolve_path(args.preregistration, root))
    comparator = resolve_comparator(prereg, freeze)
    stats_constants = check_statistics_match_preregistration(prereg)
    member_names = [f"seed{m['seed']}" for m in freeze["members"]]
    print(f"members    : {member_names}")
    print(f"comparator : {comparator}  (read from the frozen pre-registration)")

    if not prereg.get("written_before_any_internal_test_access", False):
        raise SystemExit(
            "error: the pre-registration does not declare "
            "written_before_any_internal_test_access=True. A comparator chosen "
            "after the test has been looked at is not a pre-registration."
        )

    # ---- case-set integrity ------------------------------------------------ #
    expected_n = (prereg.get("dataset_identity") or {}).get("n_cases")
    if expected_n is not None and int(expected_n) != len(case_ids):
        raise SystemExit(
            f"error: the pre-registration covers {expected_n} cases but {manifest} "
            f"has {len(case_ids)}. The evaluation set may not be changed here."
        )
    declared_sha = (prereg.get("dataset_identity") or {}).get("manifest_sha256")
    if declared_sha and sha256_file(manifest) != declared_sha:
        raise SystemExit(
            f"error: {manifest} does not match the manifest hash recorded in the "
            "pre-registration. The evaluation set may not be changed here."
        )
    for other in ("train", "val", "challenge_test"):
        other_path = manifest.parent / f"{other}.csv"
        if other_path.is_file():
            overlap = set(case_ids) & set(read_case_ids(other_path))
            if overlap:
                raise SystemExit(
                    f"error: internal_test overlaps {other} on {len(overlap)} case(s), "
                    f"e.g. {sorted(overlap)[:5]}."
                )
    print(f"cases      : {len(case_ids)} from {manifest}")
    print(f"data root  : {data_root}")
    print(f"output     : {out_dir}")

    # ---- image cache -------------------------------------------------------- #
    reference_manifest_dir = (resolve_path(args.reference_manifest_dir, root)
                              if args.reference_manifest_dir else manifest.parent)
    cache_record = build_or_verify_image_cache(
        manifest, reference_manifest_dir, data_root, image_cache_dir,
        reference_cache, case_ids, overwrite=args.overwrite)
    print(f"image cache: {image_cache_dir}  "
          f"(pipeline verified bitwise on "
          f"{cache_record['pipeline_equivalence_with_training_cache']['n_compared']} "
          f"training tensors)")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device     : {device}")

    # ---- PHASE A: inference on this reproduction's own weights ------------- #
    models, member_records = load_members(freeze, root, device)
    image_cache = EFH.FrozenImageCache(image_cache_dir / "images", case_ids)
    infer_start = datetime.now(timezone.utc)
    ids, predictions = EFH.run_blind_inference(image_cache, models, device)
    infer_end = datetime.now(timezone.utc)
    print(f"PHASE A    : predictions {predictions.shape} "
          f"{(infer_end - infer_start).total_seconds():.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "REPRODUCTION_PREDICTIONS.npy", predictions)

    # ---- PHASE B: ground truth, from --data-root only ---------------------- #
    label_paths = resolve_manifest_paths(manifest, data_root)
    label_dirs = {p.parent for p in label_paths.values()}
    if len(label_dirs) != 1:
        raise SystemExit(
            "error: the manifest's label paths do not share one directory: "
            f"{sorted(str(d) for d in label_dirs)[:3]}"
        )
    label_dir = label_dirs.pop()
    missing = [c for c in ids if not label_paths[c].is_file()]
    if missing:
        raise SystemExit(
            f"error: {len(missing)} ground-truth label(s) missing under {label_dir}, "
            f"e.g. {missing[:5]}.\n"
            "       internal_test labels come from the official release; prepare "
            "must have been run against this --data-root."
        )
    LOGGER.info("PHASE B: ground truth from %s (labels are never cached in the repo)",
                label_dir)

    rows = EFH.evaluate_predictions(ids, predictions, EFH.ProcessedLabelSource(label_dir))
    per_case = pd.DataFrame(rows)
    per_case.to_csv(out_dir / "REPRODUCTION_PER_CASE.csv", index=False,
                    encoding="utf-8-sig")

    # ---- frozen statistics -------------------------------------------------- #
    seeds = [f"seed{m['seed']}" for m in freeze["members"]]
    agg = {name: EFH.aggregate(per_case, name) for name in ["ensemble"] + seeds}
    macro = lambda member: float(agg[member]["macro_foreground_dice"])  # noqa: E731

    classes = tuple(n for n in cs.CLASS_NAMES.values())
    d_ens = per_case[[f"Dice_{c}__ensemble" for c in classes]].mean(axis=1).to_numpy()
    d_cmp = per_case[[f"Dice_{c}__{comparator}" for c in classes]].mean(axis=1).to_numpy()
    deltas = d_ens - d_cmp
    ci_low, ci_high = paired_bootstrap_ci(deltas, EFH.BOOTSTRAP_RESAMPLES, EFH.RNG_SEED)
    p_value = paired_signflip_pvalue(deltas, EFH.PERMUTATION_DRAWS, EFH.RNG_SEED)
    decision = primary_decision(float(deltas.mean()), ci_low, p_value)

    primary = {
        "primary_endpoint": "case-wise paired macro foreground Dice difference",
        "comparison": f"ensemble vs {comparator}",
        "n_paired": int(len(deltas)),
        "ensemble_macro_Dice": round(macro("ensemble"), 6),
        "comparator_macro_Dice": round(macro(comparator), 6),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "permutation_p": p_value,
        "primary_success_rule": prereg.get("primary_success_rule"),
        "decision": decision,
    }
    (out_dir / "REPRODUCTION_PRIMARY_ENDPOINT.json").write_text(
        json.dumps(primary, indent=2, ensure_ascii=False), encoding="utf-8")

    secondary = pd.DataFrame([
        {"member": member, "n_cases": agg[member]["n_cases"],
         **{k: v for k, v in agg[member].items() if k != "n_cases"}}
        for member in ["ensemble", comparator] + [s for s in seeds if s != comparator]
    ])
    secondary.to_csv(out_dir / "REPRODUCTION_SECONDARY_ENDPOINTS.csv", index=False,
                     encoding="utf-8-sig")

    report = {
        "record_type": "REPRODUCTION_EVALUATION",
        "evaluation_kind": RECORD_KIND,
        "evaluation_kind_description": (
            "Models trained from scratch by this reproduction, evaluated on the "
            "frozen subject-clean internal_test split."
        ),
        "started_utc": infer_start.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "distinct_from": {
            "author_historical_corrected_fixed_re_evaluation": {
                "kind": "AUTHOR_HISTORICAL_FIXED_RE_EVALUATION",
                "location": "results/subject_clean_holdout/ (not written here)",
                "same_100_cases": True,
                "touched_by_this_run": False,
            },
            "statement": (
                "Both evaluations use the same 100 cases. The difference is the "
                "weights: the author's record used the frozen historical "
                "checkpoints; this record uses checkpoints trained by this "
                "reproduction. The two must not be merged or presented as one."
            ),
        },
        "test_set": {
            "manifest": str(manifest.relative_to(root)) if root in manifest.parents
                        else str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "n_cases": len(case_ids),
            "is_a_fresh_unseen_test_set": False,
            "is_a_never_used_holdout": False,
            "novelty_statement": NOVELTY_STATEMENT,
            "claimable": "weight-level independence only",
        },
        "data_root": str(data_root),
        "label_source": str(label_dir),
        "image_cache": cache_record,
        "members": member_records,
        "comparator": comparator,
        "comparator_source": str(resolve_path(args.preregistration, root)),
        "ensemble_recipe": freeze.get("ensemble_recipe"),
        "statistics": stats_constants,
        "prediction_artifact": "REPRODUCTION_PREDICTIONS.npy",
        "primary": primary,
        "aggregates": agg,
        "not_done_here": [
            "no comparator / threshold / metric / ensemble-recipe choice",
            "no scientific hyper-parameter change",
            "no write into any author namespace",
            "no challenge_test access",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    (out_dir / "REPRODUCTION_EVALUATION_REPORT.md").write_text(
        _markdown(report, per_case, seeds, comparator, agg), encoding="utf-8")

    print()
    print("=" * 88)
    print(f"INDEPENDENT REPRODUCTION EVALUATION (n={len(ids)})")
    print("=" * 88)
    print(f"  ensemble macro Dice   : {macro('ensemble'):.4f}")
    print(f"  comparator macro Dice : {macro(comparator):.4f}  ({comparator})")
    print(f"  mean delta            : {float(deltas.mean()):+.6f}")
    print(f"  95% bootstrap CI      : [{ci_low:+.6f}, {ci_high:+.6f}]")
    print(f"  permutation p         : {p_value:.6f}")
    print(f"  PRIMARY SUCCESS       : {decision['result']}")
    print("-" * 88)
    print("  These numbers are a REPRODUCTION result, not the author's historical")
    print("  result, and internal_test is not a fresh unseen test set:")
    print(f"    {NOVELTY_STATEMENT}")
    print(f"  written to: {out_dir}")
    print("=" * 88)
    return 0


def _markdown(report: dict, per_case: pd.DataFrame, seeds: Sequence[str],
              comparator: str, agg: dict) -> str:
    classes = tuple(cs.CLASS_NAMES.values())
    lines = [
        "# Independent reproduction evaluation — internal_test",
        "",
        f"- record kind: `{report['evaluation_kind']}`",
        f"- cases: **{report['test_set']['n_cases']}** "
        f"(`{report['test_set']['manifest']}`, sha256 `{report['test_set']['manifest_sha256'][:16]}…`)",
        f"- members: {', '.join(seeds)} (trained by this reproduction)",
        f"- comparator: `{comparator}` (read from the frozen pre-registration)",
        f"- generated: {report['finished_utc']}",
        "",
        "## This is not the author's result, and not a fresh test set",
        "",
        report["test_set"]["novelty_statement"],
        "",
        report["distinct_from"]["statement"],
        "",
        "## Primary endpoint",
        "",
        f"- ensemble macro Dice: **{report['primary']['ensemble_macro_Dice']:.6f}**",
        f"- comparator macro Dice ({comparator}): "
        f"**{report['primary']['comparator_macro_Dice']:.6f}**",
        f"- mean paired Δ: {report['primary']['mean_delta']:+.6f}",
        f"- 95% bootstrap CI: [{report['primary']['bootstrap_ci_low']:+.6f}, "
        f"{report['primary']['bootstrap_ci_high']:+.6f}]",
        f"- permutation p: {report['primary']['permutation_p']:.6f}",
        f"- decision: **{report['primary']['decision']['result']}**",
        "",
        "## Per-member aggregates",
        "",
        "| member | macro Dice | " + " | ".join(f"Dice_{c}" for c in classes) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in classes) + " |",
    ]
    for member in ["ensemble", comparator] + [s for s in seeds if s != comparator]:
        a = agg[member]
        lines.append(
            f"| {member} | {a['macro_foreground_dice']:.6f} | "
            + " | ".join(f"{a[f'Dice_{c}']:.6f}" for c in classes) + " |")
    lines += [
        "",
        "## Not done here",
        "",
        *[f"- {item}" for item in report["not_done_here"]],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

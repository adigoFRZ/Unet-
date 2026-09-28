#!/usr/bin/env python
"""Build the Experiment E occupancy-prior cache.

Computes, for each foreground class, the per-voxel count of development-train
cases whose ground truth marks that voxel:

    S_k(v) = sum over the N development-train cases of 1[Y_i(v) = k]

and stores it as one ``(3, D, H, W)`` integer volume plus a metadata sidecar.
The full N maps are *not* stored -- a training case's leave-one-out prior is
obtained by subtracting its own mask at load time (see ``data.spatial_prior``).

N is not a constant of this script: it is the number of cases in the train
manifest it is given, and the published subject-clean manifest
(``manifests/subject_clean_v1/train.csv``) holds **161**. The old 160-case
campaign is a different cohort, and a cache built from it must never be used --
which is why the cache records its own case list and is refused whenever that
list does not match the manifest exactly.

Naming: this is a **training-set occupancy map** / **empirical spatial prior**.
It is NOT an anatomical atlas and NOT a registered probability atlas -- no
cross-subject registration exists in this project (see the Experiment E preflight
audit). The cases share a voxel grid, not a common anatomical space.

Data-source rules, enforced here rather than left to the caller:
  * only development-train labels are read;
  * validation / internal_test / challenge_test labels are never opened;
  * the train manifest must contain exactly ``--expect-n-train`` cases (by
    default, however many the manifest holds) and every label must exist;
  * after building, the cache's own recorded case list is compared against the
    manifest case set -- a mismatch is a hard failure, not a warning.

Usage
-----
    python scripts/build_spatial_prior_cache.py --root \\
        --manifest-dir manifests/subject_clean_v1 \\
        --out-dir cache/spatial_prior_v1_subject_clean --expect-n-train 161
    python scripts/build_spatial_prior_cache.py --root . --verify-only
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import crop_spec as cs  # noqa: E402
from data.spatial_prior import (  # noqa: E402
    OCCUPANCY_CHANNEL_NAMES,
    OCCUPANCY_CLASSES,
    OccupancyPrior,
)
from utils.paths import resolve_path, resolve_project_root  # noqa: E402

LOGGER = logging.getLogger("build_spatial_prior_cache")

#: only this split may feed the prior
SOURCE_SPLIT = "train"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the occupancy prior cache.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--label-dir", type=Path,
                        default=Path("cache/baseline_v1/labels"))
    parser.add_argument("--manifest-dir", type=Path,
                        default=Path("manifests/experiment"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("cache/spatial_prior_v1"))
    parser.add_argument("--expect-n-train", type=int, default=None,
                        help="Refuse to build unless the train split has this many "
                             "cases. Default: however many the manifest holds "
                             "(no hard-coded cohort size).")
    parser.add_argument("--verify-only", action="store_true",
                        help="Re-check an existing cache without rewriting it.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def sha256_array(array: np.ndarray) -> str:
    """Content hash of an array, including dtype and shape."""
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


def leave_one_out_dice(prior: OccupancyPrior, label_dir: Path) -> dict[str, Any]:
    """Per-class LOO Dice between each training case's GT and the population map.

    Answers "how much does the average mask actually tell you about a *held-out*
    case?" -- the quantity that determines whether the prior can carry
    information. Descriptive only; this is not a model result.
    """
    out: dict[str, Any] = {}
    majority = {k: (prior.fixed()[i] > 0.5)
                for i, k in enumerate(OCCUPANCY_CLASSES)}

    for column, class_id in enumerate(OCCUPANCY_CLASSES):
        dice: list[float] = []
        for case_id in prior.case_ids:
            label = np.load(label_dir / f"{case_id}.npy")
            mask = label == class_id
            total = int(mask.sum()) + int(majority[class_id].sum())
            if total == 0:
                continue
            dice.append(2.0 * float((mask & majority[class_id]).sum()) / total)
        values = np.asarray(dice, dtype=float)
        out[cs.CLASS_NAMES[class_id]] = {
            "n_cases": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "n_below_0.3": int((values < 0.3).sum()),
        }
    return out


def occupancy_sanity(prior: OccupancyPrior, label_dir: Path) -> dict[str, Any]:
    """Per-class summary of the prior: how concentrated is it, and how much of a
    held-out case does it explain?"""
    probability = prior.fixed()
    per_class: dict[str, Any] = {}
    mean_volumes: list[float] = []
    for column, class_id in enumerate(OCCUPANCY_CLASSES):
        name = cs.CLASS_NAMES[class_id]
        volumes = [
            float((np.load(label_dir / f"{c}.npy") == class_id).sum())
            for c in prior.case_ids
        ]
        mean_volume = float(np.mean(volumes))
        mean_volumes.append(mean_volume)
        per_class[name] = {
            "mean_gt_volume_voxels": mean_volume,
            "max_probability": float(probability[column].max()),
            "voxels_p_gt_0_5": int((probability[column] > 0.5).sum()),
            "voxels_p_gt_0_1": int((probability[column] > 0.1).sum()),
            "consensus_over_mean_volume": (
                float((probability[column] > 0.5).sum()) / mean_volume
                if mean_volume else None
            ),
        }
    per_class["_loo_dice_vs_majority_map"] = leave_one_out_dice(prior, label_dir)
    return per_class


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    root = resolve_project_root(args.root)
    label_dir = resolve_path(args.label_dir, root)
    manifest_dir = resolve_path(args.manifest_dir, root)
    out_dir = resolve_path(args.out_dir, root)

    LOGGER.info("Project root : %s", root)
    LOGGER.info("Labels       : %s", label_dir)
    LOGGER.info("Manifest     : %s", manifest_dir / f"{SOURCE_SPLIT}.csv")
    LOGGER.info("Output       : %s", out_dir)

    manifest_path = manifest_dir / f"{SOURCE_SPLIT}.csv"
    if not manifest_path.is_file():
        LOGGER.error("missing train manifest %s", manifest_path)
        return 2
    frame = pd.read_csv(manifest_path, encoding="utf-8-sig")
    case_ids = sorted(str(c) for c in frame["case_id"])

    # ---- hard preconditions ------------------------------------------------ #
    # The expected cohort size is an argument, or -- with no argument -- whatever
    # the given manifest holds. It is deliberately NOT a module constant: the
    # legacy campaign had 160 development-train cases and the published
    # subject-clean split has 161, so a hard-coded size would either refuse a
    # correct build or, far worse, accept a prior built over the wrong cohort.
    expect_n = args.expect_n_train if args.expect_n_train is not None else len(case_ids)
    LOGGER.info("Cohort size: manifest has %d case(s); expecting %d",
                len(case_ids), expect_n)
    if len(case_ids) != expect_n:
        LOGGER.error("train manifest has %d cases, expected %d; refusing to build "
                     "a prior over the wrong cohort", len(case_ids), expect_n)
        return 2
    if len(set(case_ids)) != len(case_ids):
        LOGGER.error("train manifest contains duplicate case ids")
        return 2
    missing = [c for c in case_ids if not (label_dir / f"{c}.npy").is_file()]
    if missing:
        LOGGER.error("%d label(s) missing, e.g. %s", len(missing), missing[:5])
        return 2

    sums_path = out_dir / "occupancy_sum.npy"
    meta_path = out_dir / "occupancy_meta.json"

    if args.verify_only:
        if not sums_path.is_file():
            LOGGER.error("no cache at %s", sums_path)
            return 2
        prior = OccupancyPrior.load(out_dir, expected_case_ids=case_ids)
        LOGGER.info("Cache verified: %d cases, shape %s, dtype %s",
                    prior.n_train, prior.sums.shape, prior.sums.dtype)
        sums = prior.sums
    else:
        LOGGER.info("Summing %d development-train label maps...", len(case_ids))
        prior = OccupancyPrior.build(label_dir, case_ids)
        sums = prior.sums
        if sums.dtype != np.uint16:
            LOGGER.error("unexpected dtype %s", sums.dtype)
            return 2
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(sums_path, sums)
        LOGGER.info("Wrote %s (%d bytes)", sums_path, sums_path.stat().st_size)

    # ---- explicit case-set verification (both modes) ----------------------- #
    recorded = sorted(str(c) for c in prior.case_ids)
    if recorded != case_ids:
        missing = sorted(set(case_ids) - set(recorded))
        extra = sorted(set(recorded) - set(case_ids))
        LOGGER.error("the cache's case set does not match the train manifest "
                     "(%d recorded vs %d expected; missing %s; unexpected %s); "
                     "refusing to report success",
                     len(recorded), len(case_ids), missing[:5], extra[:5])
        return 2
    if int(sums.max()) > prior.n_train or sums.shape[1:] != cs.CROP_SHAPE_DHW:
        LOGGER.error("cache contents inconsistent with its case list")
        return 2

    # ---- sanity (read-only, always run) ------------------------------------ #
    sanity = occupancy_sanity(prior, label_dir)

    metadata: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "training-set occupancy prior (empirical spatial prior)",
        "not_an_atlas": (
            "No cross-subject registration exists in this project. These counts "
            "are per-voxel frequencies on the shared acquisition grid, not "
            "probabilities in a standardised anatomical space."),
        "source_split": SOURCE_SPLIT,
        "train_case_ids": list(prior.case_ids),
        "n_train": prior.n_train,
        "shape": list(sums.shape),
        "dtype": str(sums.dtype),
        "class_order": [int(c) for c in OCCUPANCY_CLASSES],
        "class_names": [cs.CLASS_NAMES[c] for c in OCCUPANCY_CLASSES],
        "channel_names": list(OCCUPANCY_CHANNEL_NAMES),
        "crop_shape_dhw": list(cs.CROP_SHAPE_DHW),
        "raw_shape_xyz": list(cs.RAW_SHAPE_XYZ),
        "label_dir": str(label_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sums_sha256": sha256_array(sums),
        "sanity": sanity,
    }
    if not args.verify_only:
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        LOGGER.info("Wrote %s", meta_path)

        # Round trip: re-read the cache that was just written, with the manifest's
        # case list as the expectation. This is a check on the artifact, not on
        # the in-memory object -- what matters is what a training run will load,
        # and a cache left over from the 160-case campaign fails right here.
        reloaded = OccupancyPrior.load(out_dir, expected_case_ids=case_ids)
        if reloaded.n_train != len(case_ids):
            LOGGER.error("cache on disk records %d case(s), manifest has %d",
                         reloaded.n_train, len(case_ids))
            return 2
        LOGGER.info("Case-set check on the written cache: %d case(s) == manifest",
                    reloaded.n_train)

    # ---- console ----------------------------------------------------------- #
    print()
    print("=" * 82)
    print("Experiment E — occupancy prior (training-set occupancy map)")
    print("=" * 82)
    print(f"source            : {SOURCE_SPLIT} split, {prior.n_train} cases")
    print(f"array             : {sums.shape} {sums.dtype}  max count {int(sums.max())}")
    print(f"sums sha256       : {metadata['sums_sha256'][:32]}")
    print(f"manifest sha256   : {metadata['manifest_sha256'][:32]}")
    print()
    print(f"{'class':<6}{'mean vol':>10}{'max P':>9}{'P>0.5':>8}{'P>0.1':>8}"
          f"{'cons/vol':>10}{'LOO Dice':>10}{'<0.3':>7}")
    for class_id in OCCUPANCY_CLASSES:
        name = cs.CLASS_NAMES[class_id]
        row = sanity[name]
        loo = sanity["_loo_dice_vs_majority_map"][name]
        print(f"{name:<6}{row['mean_gt_volume_voxels']:>10.1f}"
              f"{row['max_probability']:>9.3f}{row['voxels_p_gt_0_5']:>8d}"
              f"{row['voxels_p_gt_0_1']:>8d}"
              f"{row['consensus_over_mean_volume']:>10.2f}"
              f"{loo['median']:>10.3f}{loo['n_below_0.3']:>7d}")
    print()
    print("LOO Dice = per-case Dice between that case's GT and the population")
    print("majority map (P>0.5) built WITHOUT that case. Descriptive prior audit,")
    print("not a model result.")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

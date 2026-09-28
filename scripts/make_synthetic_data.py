#!/usr/bin/env python
"""Generate a tiny SYNTHETIC dataset so the pipeline can be exercised end to end.

This is **not medical data and not a result**. It is random blobs on a grid that
matches the real dataset's geometry, written in the same directory layout, so
that `prepare_labels.py -> normalize_images.py -> build_manifests.py ->
build_baseline_cache.py -> train_baseline.py --smoke-overfit` can be run without
the real (licensed, access-controlled) PDCADxFoundation data.

Nothing produced here is evidence about segmentation quality. A model trained on
it learns to find three blobs; the Dice numbers it reports are meaningless. Use
it to check that the code runs and the wiring is intact, nothing more.

Output layout (mirrors the provider's, see src/data/pdcadx_io.py):

    <out>/
      train/<case_id>/{T1,QSM,NM}.nii.gz  {QSM_mask,NM_mask}.nii.gz
      val/<case_id>/{T1,QSM,NM}.nii.gz    {QSM_mask,NM_mask}.nii.gz
      train_cases.csv        case_id,group   (no header)
      val_cases.csv

Usage
-----
    python scripts/make_synthetic_data.py                     # default sizes
    python scripts/make_synthetic_data.py --n-train 2 --n-val 1
    python scripts/make_synthetic_data.py --out tests/_synthetic_data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import crop_spec as cs  # noqa: E402

LOGGER = logging.getLogger("make_synthetic_data")

#: The frozen raw grid. Synthetic volumes use it so every downstream constant
#: (crop indices, coordinate normalisation in the spatial prior) still applies.
SHAPE_XYZ = cs.RAW_SHAPE_XYZ
AFFINE = np.diag([*cs.SPACING_XYZ_MM, 1.0]).astype(np.float64)

#: Where the three structures sit, in NIfTI (X, Y, Z) voxel coordinates. All
#: three are inside the frozen crop (X/Y 103:199, Z 8:40) with margin.
STRUCTURE_CENTRES = {
    "STN": ((150, 150, 22), 4.0, (9, 10)),
    "SN": ((150, 140, 20), 5.0, (11, 12)),
    "RN": ((150, 130, 22), 5.0, (13, 14)),
}


def _blob(shape: tuple[int, int, int], centre: tuple[int, int, int],
          radius: float) -> np.ndarray:
    grid = np.ogrid[: shape[0], : shape[1], : shape[2]]
    squared = sum((axis - c) ** 2 for axis, c in zip(grid, centre))
    return squared <= radius ** 2


def _head(shape: tuple[int, int, int]) -> np.ndarray:
    """A smooth ellipsoid standing in for the head, so the foreground mask is
    not the whole volume and `normalize_images.py` has something to normalise."""
    grid = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    ellipsoid = (
        ((grid[0] - cx) / (shape[0] * 0.42)) ** 2
        + ((grid[1] - cy) / (shape[1] * 0.42)) ** 2
        + ((grid[2] - cz) / (shape[2] * 0.45)) ** 2
    )
    return ellipsoid <= 1.0


def build_case(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One synthetic case: three modalities plus the two mask volumes."""
    head = _head(SHAPE_XYZ)
    # Smooth white noise so the images have spatial structure; a box filter is
    # enough and runs in a fraction of a second on this grid.
    rng_field = rng.normal(0.0, 1.0, size=SHAPE_XYZ).astype(np.float32)
    smoothed = ndimage.uniform_filter(rng_field, size=5, mode="constant")

    qsm_mask = np.zeros(SHAPE_XYZ, dtype=np.uint8)
    for _name, (centre, radius, labels) in STRUCTURE_CENTRES.items():
        # Jitter the centre so cases differ, then split the blob on X around the
        # jittered centre to give the structure its left/right label pair.
        jitter = tuple(int(round(v)) for v in rng.normal(0, 2, size=3))
        moved = tuple(c + j for c, j in zip(centre, jitter))
        blob = _blob(SHAPE_XYZ, moved, radius)
        lower = blob.copy()
        lower[moved[0]:] = False
        upper = blob.copy()
        upper[: moved[0]] = False
        qsm_mask[lower] = labels[0]
        qsm_mask[upper] = labels[1]

    # NM_mask carries only the NM-visible SN, labels 1/2.
    nm_mask = np.zeros(SHAPE_XYZ, dtype=np.uint8)
    nm_mask[qsm_mask == 11] = 1
    nm_mask[qsm_mask == 12] = 2

    # Background stays exactly 0 outside the head, which is the convention every
    # downstream step assumes. QSM is allowed to go negative.
    t1 = np.where(head, 100.0 + 20.0 * smoothed, 0.0).astype(np.float32)
    qsm = np.where(head, smoothed * 0.3 - 0.05, 0.0).astype(np.float32)
    nm = np.where(head, 50.0 + 10.0 * np.abs(smoothed), 0.0).astype(np.float32)
    return {"T1": t1, "QSM": qsm, "NM": nm,
            "QSM_mask": qsm_mask, "NM_mask": nm_mask}


def write_case(directory: Path, arrays: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        image = nib.Nifti1Image(array, AFFINE)
        image.header.set_xyzt_units("mm")
        nib.save(image, str(directory / f"{name}.nii.gz"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a small synthetic dataset for pipeline smoke tests.")
    parser.add_argument("--out", type=Path, default=Path("tests/_synthetic_data"),
                        help="Directory to write into (default: %(default)s).")
    parser.add_argument("--n-train", type=int, default=3)
    parser.add_argument("--n-val", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")

    root = args.out.resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        LOGGER.error("%s already exists and is not empty; pass --overwrite to replace it.",
                     root)
        return 3

    total = args.n_train + args.n_val
    per_case_mb = 3 * int(np.prod(SHAPE_XYZ)) * 4 / 1e6
    LOGGER.warning("Writing SYNTHETIC data (random blobs, not medical images).")
    LOGGER.info("Grid %s at %s mm; about %.0f MB per case.",
                SHAPE_XYZ, cs.SPACING_XYZ_MM, per_case_mb)

    rng = np.random.default_rng(args.seed)
    rows: dict[str, list[tuple[str, int]]] = {"train": [], "val": []}
    for split, count, offset in (("train", args.n_train, 0), ("val", args.n_val, 500)):
        for index in range(count):
            case_id = f"SYN_{split.upper()}_{offset + index:03d}"
            write_case(root / split / case_id, build_case(rng))
            rows[split].append((case_id, index % 2))
            LOGGER.info("%s/%s written", split, case_id)

    # The provider ships these headerless, as case_id,group.
    for split, entries in rows.items():
        path = root / f"{split}_cases.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            for case_id, group in entries:
                handle.write(f"{case_id},{group}\n")
        LOGGER.info("wrote %s (%d case(s))", path, len(entries))

    LOGGER.warning("Done. This is synthetic data: any metric computed on it is "
                   "meaningless as a segmentation result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

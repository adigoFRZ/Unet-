"""Shared I/O and conventions for the PDCADxFoundation DBS segmentation project.

This module is the single source of truth for:

* how the raw dataset is laid out and discovered on disk,
* the official QSM_mask label -> project class mapping,
* reading and writing NIfTI volumes while preserving spatial metadata.

It is deliberately read-only with respect to the raw data directories: nothing
here writes anywhere except to a caller-supplied output path.

Label semantics (confirmed from the PDCADxFoundation documentation):

    QSM_mask  1/2   Caudate Nucleus
              3/4   Putamen
              5/6   Globus Pallidus
              7/8   Thalamus
              9/10  STN   <- project class 1
              11/12 SN    <- project class 2
              13/14 RN    <- project class 3
              15/16 Dentate Nucleus
    NM_mask   1/2   SN (NM-visible)
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------- #
# Naming conventions
# --------------------------------------------------------------------------- #

NIFTI_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii")

IMAGE_MODALITIES: tuple[str, ...] = ("T1", "QSM", "NM")
MASK_MODALITIES: tuple[str, ...] = ("QSM_mask", "NM_mask")
ALL_MODALITIES: tuple[str, ...] = IMAGE_MODALITIES + MASK_MODALITIES

IMAGE_ALIASES: dict[str, str] = {
    "t1": "T1",
    "t1w": "T1",
    "t1mprage": "T1",
    "mprage": "T1",
    "qsm": "QSM",
    "nm": "NM",
    "neuromelanin": "NM",
}
MASK_SUFFIXES: tuple[str, ...] = ("_mask", "-mask", "_seg", "-seg", "_label", "-label")

SPLIT_TOKENS: dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}

# Directories that are project scaffolding, never raw data.
SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git", ".venv", "venv", "env", "__pycache__", ".idea", ".vscode", ".claude",
        "src", "scripts", "configs", "manifests", "results", "checkpoints", "tests",
        "docs", "processed", "node_modules",
    }
)

# --------------------------------------------------------------------------- #
# Official label mapping
# --------------------------------------------------------------------------- #

#: raw QSM_mask label value -> project foreground class id
QSM_LABEL_TO_CLASS: dict[int, int] = {
    9: 1, 10: 1,   # STN
    11: 2, 12: 2,  # SN
    13: 3, 14: 3,  # RN
}

#: project class id -> structure name
CLASS_NAMES: dict[int, str] = {1: "STN", 2: "SN", 3: "RN"}

#: classes kept in the generated label maps; everything else becomes background
FOREGROUND_CLASSES: tuple[int, ...] = (1, 2, 3)

#: raw labels that are collapsed to background (documented for transparency)
DISCARDED_QSM_LABELS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 15, 16)

#: the NM SN label ids, kept here so downstream code does not re-hardcode them
NM_SN_LABELS: tuple[int, ...] = (1, 2)

BACKGROUND_CLASS = 0

#: plausible per-case voxel-count sanity band for each class. Values far outside
#: these are surfaced as warnings, never silently corrected. Bands are wide on
#: purpose: the audit measured STN 21-187, SN 260-873, RN 88-314 voxels.
CLASS_VOXEL_SANITY: dict[int, tuple[int, int]] = {
    1: (5, 2000),     # STN
    2: (50, 5000),    # SN
    3: (20, 3000),    # RN
}


# --------------------------------------------------------------------------- #
# Filename / path helpers
# --------------------------------------------------------------------------- #


def strip_nifti_suffix(filename: str) -> str | None:
    """Return ``filename`` without its NIfTI suffix, or ``None`` if not NIfTI."""
    lower = filename.lower()
    for suffix in NIFTI_SUFFIXES:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def is_nifti(path: Path) -> bool:
    return strip_nifti_suffix(path.name) is not None


def classify_filename(path: Path) -> tuple[str | None, bool]:
    """Map a NIfTI filename to ``(modality, is_mask)``; ``(None, False)`` if unknown."""
    stem = strip_nifti_suffix(path.name)
    if stem is None:
        return None, False

    lower = stem.lower()
    is_mask = False
    for suffix in MASK_SUFFIXES:
        if lower.endswith(suffix):
            is_mask = True
            lower = lower[: -len(suffix)]
            break

    normalized = re.sub(r"[^0-9a-zA-Z]+", "", lower)
    image_modality = IMAGE_ALIASES.get(normalized)
    if image_modality is None:
        return (f"{stem}_mask" if is_mask else stem), is_mask
    return (f"{image_modality}_mask" if is_mask else image_modality), is_mask


def infer_split(rel_dir: Path) -> str | None:
    """Infer the split name from a case directory's path via token matching."""
    for part in rel_dir.parts:
        for token in re.split(r"[^0-9a-zA-Z]+", part.lower()):
            if token in SPLIT_TOKENS:
                return SPLIT_TOKENS[token]
    return None


def _walk_pruned(root: Path) -> Iterator[tuple[str, list[str], list[str]]]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith(".")
        )
        yield dirpath, dirnames, filenames


@dataclass
class CaseRecord:
    """One case directory and the NIfTI files it contains."""

    case_id: str
    case_dir: Path
    split: str
    files: dict[str, Path] = field(default_factory=dict)

    def has(self, *modalities: str) -> bool:
        return all(m in self.files for m in modalities)

    @property
    def label(self) -> str:
        return f"{self.split}/{self.case_id}"


def discover_cases(root: Path) -> list[CaseRecord]:
    """Recursively discover every case directory containing recognised NIfTI files.

    Discovery is driven entirely by the files on disk -- no path is hardcoded,
    so the redundant nested directories (and the misspelled train wrapper) in
    this dataset are handled automatically.
    """
    cases: list[CaseRecord] = []
    for dirpath, _dirnames, filenames in _walk_pruned(root):
        nifti = [Path(dirpath) / f for f in sorted(filenames) if is_nifti(Path(f))]
        if not nifti:
            continue

        case_dir = Path(dirpath)
        record = CaseRecord(
            case_id=case_dir.name,
            case_dir=case_dir,
            split=infer_split(case_dir.relative_to(root)) or "unknown",
        )
        for path in nifti:
            modality, _ = classify_filename(path)
            if modality in ALL_MODALITIES and modality not in record.files:
                record.files[modality] = path
        if record.files:
            cases.append(record)

    cases.sort(key=lambda c: (c.split, c.case_id))
    return cases


def load_manifest_ids(path: Path) -> dict[str, str]:
    """Load a headerless ``case_id[,group]`` CSV into ``{case_id: group}``."""
    mapping: dict[str, str] = {}
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0]:
                mapping.setdefault(parts[0], parts[1] if len(parts) > 1 else "")
    return mapping


# --------------------------------------------------------------------------- #
# Volume I/O
# --------------------------------------------------------------------------- #


def load_image(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def load_array(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a volume as a native-dtype ndarray plus its NIfTI image.

    Uses ``np.asanyarray(dataobj)`` rather than ``get_fdata()`` so the data is not
    silently upcast to float64 -- that keeps peak memory at one copy of the volume.
    """
    image = load_image(path)
    return np.asanyarray(image.dataobj), image


def spatial_signature(image: nib.Nifti1Image) -> dict[str, object]:
    """The spatial metadata that every derived file must preserve exactly."""
    header = image.header
    return {
        "shape": tuple(int(s) for s in image.shape),
        "spacing": tuple(float(z) for z in header.get_zooms()[:3]),
        "axcodes": tuple(nib.aff2axcodes(image.affine)),
        "affine": np.asarray(image.affine, dtype=np.float64),
        "sform_code": int(header["sform_code"]),
        "qform_code": int(header["qform_code"]),
        "dtype": str(image.get_data_dtype()),
    }


def save_like(
    reference: nib.Nifti1Image,
    data: np.ndarray,
    out_path: Path,
    dtype: np.dtype | type = np.float32,
    description: str = "",
) -> None:
    """Write ``data`` to ``out_path`` reusing ``reference``'s spatial metadata.

    Only the data block and dtype change; affine, qform/sform codes, spacing and
    orientation all come from the reference image, so the output is on exactly the
    same voxel grid as the raw volume. No resampling or reorientation happens here.

    The reference image is never modified.
    """
    data = np.asarray(data, dtype=dtype)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    # Scaling is meaningless once we write explicit float data; clear it so no
    # reader accidentally applies a stale slope/intercept.
    header["scl_slope"] = 1.0
    header["scl_inter"] = 0.0
    if description:
        header["descrip"] = description[:79]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference.affine, header)
    image.set_qform(reference.get_qform(), code=int(header["qform_code"]))
    image.set_sform(reference.get_sform(), code=int(header["sform_code"]))
    nib.save(image, str(out_path))


def remap_labels(raw: np.ndarray) -> tuple[np.ndarray, Counter]:
    """Collapse a raw QSM_mask into project classes ``{0, 1, 2, 3}``.

    Returns the remapped uint8 label map and a Counter of the raw label values
    that were seen (so the mapping can be audited after the fact).

    Nothing is merged across structures: 9 and 10 both become STN because they are
    the left/right members of one bilateral structure, not because they are
    different tissues.
    """
    if np.issubdtype(raw.dtype, np.floating):
        rounded = np.rint(raw)
        if np.any(np.abs(raw - rounded) > 1e-4):
            raise ValueError("QSM_mask contains non-integer label values")
        raw = rounded

    raw_int = raw.astype(np.int32, copy=False)
    values, counts = np.unique(raw_int, return_counts=True)
    seen = Counter({int(v): int(c) for v, c in zip(values, counts)})

    out = np.zeros(raw_int.shape, dtype=np.uint8)
    for raw_label, class_id in QSM_LABEL_TO_CLASS.items():
        out[raw_int == raw_label] = class_id
    return out, seen


def class_counts(label_map: np.ndarray) -> dict[int, int]:
    """Voxel count for each foreground class present in ``label_map``."""
    counts = {}
    for class_id in FOREGROUND_CLASSES:
        counts[class_id] = int(np.count_nonzero(label_map == class_id))
    return counts


def format_bbox(box: tuple[Sequence[int], Sequence[int]] | None) -> str:
    return "" if box is None else "x".join(str(int(v)) for v in box[0]) + ".." + "x".join(
        str(int(v)) for v in box[1]
    )


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu's between-class-variance threshold for a 1D sample of intensities."""
    finite = values[np.isfinite(values)]
    if finite.size < 100:
        return float("nan")
    hist, edges = np.histogram(finite, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total <= 0:
        return float("nan")

    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_total = float(np.dot(hist, centers))
    cum_sum = np.cumsum(hist * centers)

    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return float("nan")

    mean_bg = np.divide(cum_sum, weight_bg, out=np.zeros_like(cum_sum), where=weight_bg > 0)
    mean_fg = np.divide(sum_total - cum_sum, weight_fg,
                        out=np.zeros_like(cum_sum), where=weight_fg > 0)
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    variance[~valid] = -1.0
    return float(centers[int(np.argmax(variance))])


#: Above this non-zero fraction a volume has a noise floor rather than a true
#: zero background, so tissue must be separated with a threshold.
ZERO_BACKGROUND_MAX_FRACTION = 0.50
MIN_HEAD_FRACTION = 0.02
MAX_HEAD_FRACTION = 0.70


def estimate_head_mask(volume: np.ndarray) -> np.ndarray:
    """Approximate head/tissue support for one volume.

    Needed because these volumes are not background-free in a consistent way:
    QSM is exactly zero outside the head, but T1 and NM carry a non-zero noise
    floor over most of the FOV (T1 is ~81% non-zero). Treating "non-zero" as
    "tissue" therefore means two different things depending on the modality.

    Two regimes:
      * exact-zero background (QSM): ``volume != 0`` already is the head. Running
        Otsu here would be wrong -- QSM values straddle zero, so an intensity
        split would keep only the positive lobe and halve the brain.
      * noise-floor modality (T1, NM): split tissue from noise with Otsu computed
        on the non-zero values only.

    The mask is a review/normalisation aid only; it is never written as a label.
    """
    nonzero = volume != 0
    nonzero_fraction = float(nonzero.mean())
    if nonzero_fraction <= 0.0:
        return nonzero

    if nonzero_fraction <= ZERO_BACKGROUND_MAX_FRACTION:
        body = nonzero
    else:
        sampled = volume[::4, ::4, ::4] if volume.size > 4_000_000 else volume
        threshold = otsu_threshold(sampled[sampled != 0])
        body = nonzero & (volume > threshold) if np.isfinite(threshold) else nonzero
        if not (MIN_HEAD_FRACTION <= float(body.mean()) <= MAX_HEAD_FRACTION):
            body = nonzero

    if not body.any():
        return nonzero

    labels, count = ndimage.label(body)
    if count > 1:
        sizes = np.bincount(labels.reshape(-1))
        sizes[0] = 0
        body = labels == int(np.argmax(sizes))
    body = ndimage.binary_fill_holes(body)

    if float(body.mean()) < MIN_HEAD_FRACTION:
        return nonzero
    return body


def voxel_centroid(mask: np.ndarray) -> tuple[float, ...]:
    """True voxel-weighted centroid, in voxel index space.

        centroid[axis] = sum(index * n_voxels_at_index) / n_foreground_voxels

    This is the mean coordinate over ALL foreground voxels. It is NOT the
    centroid of the occupied-slice projection: that variant weights every
    occupied slice equally regardless of how many voxels it holds, which pulls
    the result toward thin tails of a structure. The audit script's
    ``_centroid_voxel`` uses the projection form -- measured error on this
    dataset reaches 13.3 voxels (~9 mm) for elongated structures such as the
    caudate, so it must not be reused for spatial ROI design.

    Verified against ``np.argwhere(mask).mean(axis=0)``.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    total = int(mask.sum())
    if total == 0:
        return tuple(float("nan") for _ in range(mask.ndim))

    coords: list[float] = []
    for axis in range(mask.ndim):
        other_axes = tuple(a for a in range(mask.ndim) if a != axis)
        counts = mask.sum(axis=other_axes).astype(np.int64)
        indices = np.arange(counts.size, dtype=np.int64)
        coords.append(float((counts * indices).sum() / total))
    return tuple(coords)


def voxel_bbox(mask: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Axis-aligned bounding box of a boolean mask, as (min_idx, max_idx) per axis."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    mins: list[int] = []
    maxs: list[int] = []
    for axis in range(mask.ndim):
        other_axes = tuple(a for a in range(mask.ndim) if a != axis)
        idx = np.flatnonzero(np.any(mask, axis=other_axes))
        if idx.size == 0:
            return None
        mins.append(int(idx[0]))
        maxs.append(int(idx[-1]))
    return tuple(mins), tuple(maxs)

"""Experiment E -- explicit spatial priors as extra input channels.

Two priors are provided, and the naming matters as much as the maths:

**Coordinate channels** (``coords``). Three channels giving each voxel's
normalised position on the *raw* acquisition grid, computed once and then
cropped with the frozen crop:

    Cx(x) = 2x/(300-1) - 1      varies along W
    Cy(y) = 2y/(300-1) - 1      varies along H
    Cz(z) = 2z/(70-1)  - 1      varies along D

Normalisation happens on the raw grid and **must not be repeated after the
crop** -- the cropped ranges are deliberately asymmetric
(Cx/Cy in [-0.311, +0.324], Cz in [-0.768, +0.130]) because the frozen crop is
not centred on the raw volume. Re-normalising to [-1, 1] afterwards would
silently destroy the absolute position information the channel exists to carry.

**Training-set occupancy prior** (``occupancy``). Three channels giving, per
voxel and per class, the fraction of development-train cases whose ground truth
marks it:

    S_k(v)  = sum_i M_i,k(v)          over the 160 development-train cases
    P_k(v)  = S_k(v) / 160            for validation
    P_k^{-i}(v) = (S_k(v) - M_i,k(v)) / 159   for training case i

Naming, deliberately: this is an **empirical spatial prior** / **training-set
occupancy map**, NOT an anatomical atlas and NOT a registered probability atlas.
There is no cross-subject registration anywhere in this project -- the 160 cases
share a voxel grid (identical shape, spacing and zero-translation affine), not a
common anatomical space. "Atlas" would imply a standardised space that was never
established.

Why leave-one-out is mandatory, not an optimisation: feeding ``S_k/160`` to the
model while training on case i puts case i's *own* ground truth into its input.
That is a self-referential leak -- the model could reduce the loss by reading the
answer rather than by segmenting. Training therefore subtracts the current case's
own mask; validation uses the full 160-case map, which is leak-free because no
validation case contributed to it.

Boundary: every prior channel is appended AFTER the image channels, never
inserted before them, and is attached in this module rather than by
``SegmentationDataset``. Two consequences that matter:
  * ``sample["image"][0]`` stays T1, so the existing prediction overlays and any
    channel-0 assumption keep working unchanged;
  * the priors never pass through ``normalize_images.py``, so intensity
    normalisation (percentile clip + z-score, background forced to zero) can
    never rescale or zero out a coordinate or a prior value.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from data import crop_spec as cs
from utils.paths import resolve_path

try:  # torch is an optional heavy dependency for the pure-numpy tooling
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - only hit before torch is installed
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        """Placeholder so the module imports without torch installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required to use the Experiment E prior wrapper."
            )


LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #

MODE_NONE = "none"
MODE_COORDS = "coords"
MODE_OCCUPANCY = "occupancy"
MODE_BOTH = "both"

#: channel counts contributed by each mode
_PRIOR_CHANNELS: dict[str, int] = {
    MODE_NONE: 0,
    MODE_COORDS: 3,
    MODE_OCCUPANCY: 3,
    MODE_BOTH: 6,
}

#: labels for the prior channels, in the order they are appended
COORDINATE_CHANNEL_NAMES: tuple[str, ...] = ("Cx", "Cy", "Cz")
OCCUPANCY_CHANNEL_NAMES: tuple[str, ...] = ("P_STN", "P_SN", "P_RN")

#: class order of the occupancy cache's class axis
OCCUPANCY_CLASSES: tuple[int, ...] = cs.FOREGROUND_CLASSES

#: number of development-train cases the occupancy prior is defined over
N_TRAIN = 160

#: the split the occupancy prior may be built from, and the only one it uses
TRAIN_SPLIT = "train"


def resolve_spatial_prior_mode(value: str | None) -> str:
    """Validate a ``spatial_prior_mode`` value; ``None`` means ``"none"``."""
    mode = MODE_NONE if value is None else str(value)
    if mode not in _PRIOR_CHANNELS:
        raise ValueError(
            f"unknown spatial_prior_mode {mode!r}; expected one of "
            f"{sorted(_PRIOR_CHANNELS)} (or None for {MODE_NONE!r})"
        )
    return mode


def spatial_prior_channel_count(mode: str) -> int:
    """Number of extra input channels a prior mode contributes."""
    return _PRIOR_CHANNELS[resolve_spatial_prior_mode(mode)]


# --------------------------------------------------------------------------- #
# coordinate channels
# --------------------------------------------------------------------------- #

def _axis_coordinate(index_count: int) -> np.ndarray:
    """``2*i/(n-1) - 1`` over ``i = 0 .. n-1`` -- normalised on the RAW grid."""
    if index_count < 2:
        raise ValueError(f"axis must have at least 2 samples, got {index_count}")
    return 2.0 * np.arange(index_count, dtype=np.float64) / (index_count - 1) - 1.0


def build_coordinate_channels() -> np.ndarray:
    """The three coordinate channels, ``(3, D, H, W)`` float32.

    Built on the raw ``(X, Y, Z)`` grid and passed through the *same*
    ``crop_spec.to_tensor_layout`` the images use, so the two can never drift
    apart: a change to the crop moves both together.

    No re-normalisation after cropping -- see the module docstring.
    """
    nx, ny, nz = cs.RAW_SHAPE_XYZ
    x = _axis_coordinate(nx)
    y = _axis_coordinate(ny)
    z = _axis_coordinate(nz)

    cx = np.broadcast_to(x[:, None, None], (nx, ny, nz))
    cy = np.broadcast_to(y[None, :, None], (nx, ny, nz))
    cz = np.broadcast_to(z[None, None, :], (nx, ny, nz))

    stacked = np.stack([np.ascontiguousarray(cs.to_tensor_layout(plane))
                        for plane in (cx, cy, cz)], axis=0)
    if stacked.shape != (3, *cs.CROP_SHAPE_DHW):
        raise ValueError(
            f"coordinate channels have shape {stacked.shape}, expected "
            f"{(3, *cs.CROP_SHAPE_DHW)}"
        )
    return stacked.astype(np.float32)


_COORDINATE_CACHE: np.ndarray | None = None


def coordinate_channels() -> np.ndarray:
    """``build_coordinate_channels()``, computed once per process.

    The array is read-only by convention: callers must not modify it in place,
    since every sample of every batch shares it.
    """
    global _COORDINATE_CACHE
    if _COORDINATE_CACHE is None:
        _COORDINATE_CACHE = build_coordinate_channels()
    return _COORDINATE_CACHE


# --------------------------------------------------------------------------- #
# occupancy prior
# --------------------------------------------------------------------------- #

class OccupancyPrior:
    """Class-wise voxel occupancy counts over the development-train cases.

    Holds only the per-class *sum* ``S_k`` (3, D, H, W) plus the identity of the
    cases that produced it -- never 160 full maps. A training case's
    leave-one-out map is obtained by subtracting its own mask at load time, which
    costs one comparison and keeps the cache at ~176 KB.
    """

    def __init__(self, sums: np.ndarray, case_ids: Sequence[str],
                 metadata: dict[str, Any] | None = None) -> None:
        if sums.ndim != 4 or sums.shape[0] != len(OCCUPANCY_CLASSES):
            raise ValueError(
                f"occupancy sums must be ({len(OCCUPANCY_CLASSES)}, D, H, W); got "
                f"{tuple(sums.shape)}"
            )
        if sums.shape[1:] != cs.CROP_SHAPE_DHW:
            raise ValueError(
                f"occupancy sums must be on the frozen crop "
                f"{cs.CROP_SHAPE_DHW}; got {tuple(sums.shape[1:])}"
            )
        if not np.issubdtype(sums.dtype, np.integer):
            raise ValueError(f"occupancy sums must be integer, got {sums.dtype}")
        case_ids = tuple(str(c) for c in case_ids)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("duplicate case ids in the occupancy cache")
        if int(sums.max()) > len(case_ids):
            raise ValueError(
                f"occupancy count {int(sums.max())} exceeds the number of cases "
                f"({len(case_ids)}) -- the cache is inconsistent with its case list"
            )
        if (sums < 0).any():
            raise ValueError("occupancy sums contain negative counts")

        self.sums = sums
        self.case_ids = case_ids
        self.metadata = dict(metadata or {})
        self._index = {c: i for i, c in enumerate(case_ids)}

    # ---- construction ---------------------------------------------------- #

    @staticmethod
    def build(label_dir: str | Path, case_ids: Sequence[str]) -> "OccupancyPrior":
        """Sum the per-class masks of ``case_ids`` from the label cache.

        Only the ids given are read. The caller is responsible for passing the
        development-train manifest's case list -- this function has no way to
        tell a train id from a val id, so it must not be the only guard.
        """
        label_dir = Path(label_dir)
        case_ids = tuple(str(c) for c in case_ids)
        if not case_ids:
            raise ValueError("cannot build an occupancy prior from zero cases")

        sums = np.zeros((len(OCCUPANCY_CLASSES), *cs.CROP_SHAPE_DHW), dtype=np.uint16)
        for case_id in case_ids:
            path = label_dir / f"{case_id}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"missing label for {case_id}: {path}")
            label = np.load(path)
            if label.shape != cs.CROP_SHAPE_DHW:
                raise ValueError(
                    f"{case_id}: label shape {label.shape} != {cs.CROP_SHAPE_DHW}"
                )
            for column, class_id in enumerate(OCCUPANCY_CLASSES):
                sums[column] += (label == class_id).astype(np.uint16)

        return OccupancyPrior(sums, case_ids,
                              {"n_train": len(case_ids),
                               "label_source": str(label_dir)})

    @classmethod
    def load(cls, cache_dir: str | Path, expected_case_ids: Sequence[str] | None = None
             ) -> "OccupancyPrior":
        """Load a cache written by ``scripts/build_spatial_prior_cache.py``.

        ``expected_case_ids`` (the development-train manifest) is checked exactly
        against what the cache recorded, so a cache built from the wrong split --
        or from a different manifest revision -- is refused rather than silently
        used.
        """
        cache_dir = Path(cache_dir)
        sums_path = cache_dir / "occupancy_sum.npy"
        meta_path = cache_dir / "occupancy_meta.json"
        if not sums_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"occupancy cache incomplete in {cache_dir}; run "
                f"scripts/build_spatial_prior_cache.py first"
            )
        sums = np.load(sums_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        prior = cls(sums, metadata.get("train_case_ids", []), metadata)

        if expected_case_ids is not None:
            expected = sorted(str(c) for c in expected_case_ids)
            if sorted(prior.case_ids) != expected:
                raise ValueError(
                    "occupancy cache was built from a different case set than the "
                    f"manifest: cache n={len(prior.case_ids)}, manifest "
                    f"n={len(expected)}"
                )
        return prior

    # ---- queries --------------------------------------------------------- #

    @property
    def n_train(self) -> int:
        return len(self.case_ids)

    def fixed(self) -> np.ndarray:
        """``S_k / n_train`` -- the prior used for validation."""
        return (self.sums.astype(np.float32) / float(self.n_train))

    def leave_one_out(self, case_index: int, label: np.ndarray) -> np.ndarray:
        """``(S_k - M_i,k) / (n_train - 1)`` for training case ``case_index``.

        ``label`` is that case's own project label map, which the caller already
        has as part of the sample -- so the subtraction costs no extra I/O and
        cannot accidentally use a stale copy.
        """
        if self.n_train < 2:
            raise ValueError("leave-one-out needs at least 2 training cases")
        if label.shape != cs.CROP_SHAPE_DHW:
            raise ValueError(
                f"label shape {label.shape} != {cs.CROP_SHAPE_DHW}")

        own = np.stack([(label == class_id) for class_id in OCCUPANCY_CLASSES])
        counts = self.sums.astype(np.float32) - own.astype(np.float32)
        if counts.min() < 0:
            raise ValueError(
                "negative leave-one-out count -- the supplied label is not a "
                "member of this occupancy prior's case set"
            )
        return counts / float(self.n_train - 1)


# --------------------------------------------------------------------------- #
# dataset wrapper
# --------------------------------------------------------------------------- #

class SpatialPriorDataset(Dataset):
    """Appends Experiment E's prior channels to a base dataset's images.

    The base dataset is untouched: priors are attached here, after the image has
    been loaded and (already) intensity-normalised, so nothing in the
    preprocessing or normalisation path can see them. Channel order is always
    ``[image modalities..., priors...]`` so channel 0 stays T1.

    ``split`` selects the occupancy semantics and is deliberately explicit: the
    train split gets leave-one-out maps, everything else gets the fixed
    160-case map. A leakage-aware wrapper should not have to guess which it is.
    """

    def __init__(self, base: Dataset, mode: str, *, split: str,
                 occupancy: OccupancyPrior | None = None) -> None:
        self.base = base
        self.split = split
        self.mode = resolve_spatial_prior_mode(mode)
        self.channels = spatial_prior_channel_count(self.mode)

        wants_occupancy = self.mode in (MODE_OCCUPANCY, MODE_BOTH)
        if wants_occupancy and occupancy is None:
            raise ValueError(
                f"spatial_prior_mode={self.mode!r} needs an occupancy prior, but "
                f"none was supplied"
            )
        if not wants_occupancy and occupancy is not None:
            raise ValueError(
                f"spatial_prior_mode={self.mode!r} does not use an occupancy "
                f"prior; got one anyway"
            )
        if split == "train" and wants_occupancy and occupancy is not None:
            # A training sample must be a member of the prior's case set, or the
            # leave-one-out subtraction would remove the wrong (absent) mask.
            base_ids = set(str(c) for c in getattr(base, "case_ids", []))
            missing = base_ids - set(occupancy.case_ids)
            if missing:
                raise ValueError(
                    f"{len(missing)} training case(s) are absent from the occupancy "
                    f"prior, e.g. {sorted(missing)[:5]}; the prior must be built "
                    f"from exactly these cases"
                )
        self.occupancy = occupancy
        #: mirror the base's case list so subsetting and case listing keep working
        self.case_ids: list[str] = list(getattr(base, "case_ids", []))

        if self.mode == MODE_NONE:
            LOGGER.info("SpatialPriorDataset: mode=none (images are passed through)")
        else:
            LOGGER.info("SpatialPriorDataset: mode=%s split=%s -> +%d channel(s)",
                        self.mode, split, self.channels)

    # ---- transparent forwarding ------------------------------------------ #

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so this cannot shadow the
        # attributes defined above. Keeps `describe()`, `modalities`, ... working.
        return getattr(self.__dict__["base"], name)

    def __len__(self) -> int:
        return len(self.base)

    def prior_for(self, index: int, label: np.ndarray) -> np.ndarray | None:
        """The prior block for one sample, ``(channels, D, H, W)`` or ``None``.

        Train + occupancy -> leave-one-out; anything else -> the fixed map.
        """
        if self.mode == MODE_NONE:
            return None
        blocks: list[np.ndarray] = []
        if self.mode in (MODE_COORDS, MODE_BOTH):
            blocks.append(coordinate_channels())
        if self.mode in (MODE_OCCUPANCY, MODE_BOTH):
            if self.split == "train":
                blocks.append(self.occupancy.leave_one_out(index, label))
            else:
                blocks.append(self.occupancy.fixed())
        return np.concatenate(blocks, axis=0).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required for the prior wrapper")

        label = sample["label"].numpy() if torch.is_tensor(sample["label"]) \
            else np.asarray(sample["label"])
        prior = self.prior_for(index, label)
        if prior is None:
            return sample

        image = sample["image"]
        prior_tensor = torch.from_numpy(np.ascontiguousarray(prior)).float()
        if prior_tensor.shape[1:] != image.shape[1:]:
            raise ValueError(
                f"prior spatial shape {tuple(prior_tensor.shape[1:])} does not "
                f"match the image {tuple(image.shape[1:])}"
            )
        # Appended, never prepended: channel 0 must stay T1 for the overlays.
        sample["image"] = torch.cat([image, prior_tensor], dim=0)
        return sample


def build_spatial_prior(
    root: str | Path, config: Any, split: str,
) -> tuple[OccupancyPrior | None, str]:
    """Resolve the prior mode and (if needed) load the occupancy cache.

    Returns ``(occupancy_or_None, mode)``. The cache is always validated against
    the **development-train manifest**, whatever split is being built: the
    occupancy map is defined over the 160 training cases, so a validation loader
    must still be using a cache built from exactly those cases. Checking against
    the split actually requested would let a val-built (leaking) cache pass.
    """
    import pandas as pd

    mode = resolve_spatial_prior_mode(getattr(config, "spatial_prior_mode", None))
    if mode not in (MODE_OCCUPANCY, MODE_BOTH):
        return None, mode

    cache_dir = getattr(config, "spatial_prior_cache_dir", None)
    if not cache_dir:
        raise ValueError(
            f"spatial_prior_mode={mode!r} requires spatial_prior_cache_dir to be set"
        )

    manifest_dir = resolve_path(getattr(config, "manifest_dir"), root)
    manifest_path = manifest_dir / f"{TRAIN_SPLIT}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing train manifest {manifest_path}")
    train_ids = sorted(str(c) for c in pd.read_csv(manifest_path)["case_id"])

    resolved = resolve_path(cache_dir, root)
    occupancy = OccupancyPrior.load(resolved, expected_case_ids=train_ids)
    if occupancy.n_train != len(train_ids):
        raise ValueError(
            f"occupancy prior spans {occupancy.n_train} cases but the train "
            f"manifest has {len(train_ids)}"
        )
    LOGGER.info("Occupancy prior: %d train cases from %s (building split=%s)",
                occupancy.n_train, resolved, split)
    return occupancy, mode

"""Dataset over the frozen-crop Baseline v1 cache.

Returns tensors already in PyTorch ``(C, D, H, W)`` / ``(D, H, W)`` layout, so the
axis convention is decided once at cache-build time (see ``crop_spec``) and never
re-derived here. That is deliberate: a silent (X,Y,Z) vs (D,H,W) mix-up between
image and label would still run, and would train a network on misaligned data.

Baseline v1 constraints:
  * no random patch sampling -- the crop is fixed and identical for every case
  * no augmentation -- augmentation is a separate later experiment
  * no GT-guided re-centring -- the crop never depends on a case's own mask

Iteration order is deterministic: manifests are read and sorted by ``case_id``, so
a given seed always yields the same sequence regardless of filesystem ordering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from data import crop_spec as cs
from data.augmentation import SegmentationAugmentor
from utils.paths import resolve_path, resolve_project_root

try:  # torch is an optional heavy dependency for the pure-data tooling
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - only hit before torch is installed
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        """Placeholder so the module imports without torch installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required to use SegmentationDataset. "
                "Install it with: pip install torch --index-url "
                "https://download.pytorch.org/whl/cu126"
            )


LOGGER = logging.getLogger(__name__)

#: Splits this dataset is allowed to serve. Evaluation-only splits are excluded
#: on purpose: internal_test and challenge_test must not be touched at this stage.
ALLOWED_SPLITS: tuple[str, ...] = ("train", "val")


class SegmentationDataset(Dataset):
    """Fixed-crop multimodal MRI dataset for Baseline v1.

    Parameters
    ----------
    root:
        Project root.
    split:
        ``"train"`` or ``"val"`` -- the development splits only.
    cache_dir:
        Defaults to ``<root>/cache/baseline_v1``.
    manifest_dir:
        Defaults to ``<root>/manifests/experiment``.
    verify_files:
        Check that every referenced ``.npy`` exist while building the index.
    modalities:
        Which input modalities to return, a subset of ``crop_spec.CHANNEL_ORDER``
        in the order they should appear as channels. ``None`` (the default) means
        all of them, in the frozen order -- so the Baseline v1 output is
        unchanged. Used by the modality ablation, which must feed the model only
        the modalities under test rather than zeroing the others out: zeroing
        would leave the network's first layer sized for three inputs and would
        not be an ablation of the input at all.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        cache_dir: str | Path | None = None,
        manifest_dir: str | Path | None = None,
        verify_files: bool = True,
        augment: bool = False,
        augment_seed: int | None = None,
        modalities: Sequence[str] | None = None,
    ) -> None:
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"split {split!r} is not available at this stage; "
                f"allowed: {ALLOWED_SPLITS}. internal_test / challenge_test are "
                f"reserved and must not be read here."
            )

        # Anchor every path to the project root rather than the current working
        # directory, so launching from elsewhere still reads the right data.
        self.root = resolve_project_root(root)
        self.split = split
        self.cache_dir = (resolve_path(cache_dir, self.root) if cache_dir
                          else self.root / "cache" / "baseline_v1")
        self.manifest_dir = (resolve_path(manifest_dir, self.root) if manifest_dir
                             else self.root / "manifests" / "experiment")

        manifest_path = self.manifest_dir / f"{split}.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest {manifest_path}")

        frame = pd.read_csv(manifest_path)
        if "case_id" not in frame.columns:
            raise ValueError(f"{manifest_path} has no case_id column")

        # Sorted for determinism, independent of how the manifest was written.
        self.case_ids: list[str] = sorted(str(c) for c in frame["case_id"])
        self.image_dir = self.cache_dir / "images"
        self.label_dir = self.cache_dir / "labels"

        # Modality selection. Kept as *names* rather than raw indices so a config
        # cannot silently mean the wrong channel if the cache order ever moves,
        # and resolved to indices once, here.
        self.modalities: tuple[str, ...] = (
            tuple(cs.CHANNEL_ORDER) if modalities is None else tuple(modalities)
        )
        unknown = [m for m in self.modalities if m not in cs.CHANNEL_ORDER]
        if unknown:
            raise ValueError(
                f"unknown modality/modalities {unknown}; the cache provides "
                f"{tuple(cs.CHANNEL_ORDER)}"
            )
        if not self.modalities:
            raise ValueError("at least one modality must be selected")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError(f"duplicate modalities in {self.modalities}")
        self.channel_indices: tuple[int, ...] = tuple(
            cs.CHANNEL_ORDER.index(m) for m in self.modalities
        )
        #: True when every channel is selected in the frozen order, in which case
        #: indexing is skipped entirely so the default output is bit-identical.
        self._all_channels_in_order: bool = (
            self.channel_indices == tuple(range(len(cs.CHANNEL_ORDER)))
        )

        missing: list[str] = []
        if verify_files:
            for case_id in self.case_ids:
                if not (self.image_dir / f"{case_id}.npy").is_file():
                    missing.append(f"image:{case_id}")
                if not (self.label_dir / f"{case_id}.npy").is_file():
                    missing.append(f"label:{case_id}")
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cache file(s) missing, e.g. {missing[:5]}. "
                f"Run scripts/build_baseline_cache.py first."
            )

        # Augmentation is a training-only concept. Enabling it on an evaluation
        # split would make the reported metric depend on a random draw and
        # invalidate any comparison against the baseline, so it is refused here
        # rather than left to the caller to remember.
        if augment and split != "train":
            raise ValueError(
                f"augmentation requested for split {split!r}. Only the training "
                f"split may be augmented; validation must stay deterministic."
            )
        self.augment = bool(augment)
        self.augmentor: SegmentationAugmentor | None = (
            SegmentationAugmentor(
                channel_order=cs.CHANNEL_ORDER,
                seed=augment_seed if augment_seed is not None else 42,
            )
            if augment else None
        )

        LOGGER.info("SegmentationDataset split=%s: %d cases from %s (augment=%s, "
                    "modalities=%s -> %d channel(s))",
                    split, len(self.case_ids), self.cache_dir, self.augment,
                    "/".join(self.modalities), len(self.modalities))

    # ---- index ------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_id = self.case_ids[index]

        image = np.load(self.image_dir / f"{case_id}.npy")
        label = np.load(self.label_dir / f"{case_id}.npy")

        # The cache writes (C, D, H, W) and (D, H, W). Assert rather than assume:
        # a wrong shape here means the cache is stale or was built with a
        # different crop, and continuing would corrupt training silently.
        from data import crop_spec as cs

        if image.shape != (cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW):
            raise ValueError(
                f"{case_id}: image shape {image.shape} != "
                f"{(cs.IN_CHANNELS, *cs.CROP_SHAPE_DHW)}"
            )
        if label.shape != cs.CROP_SHAPE_DHW:
            raise ValueError(
                f"{case_id}: label shape {label.shape} != {cs.CROP_SHAPE_DHW}"
            )

        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required to build tensors")

        if self.augmentor is not None:
            image, label = self.augmentor(image, label)
            # A nearest-neighbour resample cannot invent a class, but assert it
            # anyway: a bad interpolation order upstream would silently corrupt
            # every augmented label, and that is not something to discover later.
            if not set(np.unique(label).tolist()).issubset({0, 1, 2, 3}):
                raise ValueError(
                    f"{case_id}: augmented label contains values outside 0-3: "
                    f"{sorted(np.unique(label).tolist())}"
                )

        # Select the requested modalities AFTER augmentation: the augmentor is
        # written against the full channel order and applies its per-modality
        # intensity transforms by name, so it must see all channels.
        if not self._all_channels_in_order:
            image = image[list(self.channel_indices)]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "case_id": case_id,
        }

    # ---- convenience ------------------------------------------------------ #

    def describe(self) -> dict[str, Any]:
        from data import crop_spec as cs

        return {
            "split": self.split,
            "n_cases": len(self.case_ids),
            "cache_dir": str(self.cache_dir),
            "manifest": str(self.manifest_dir / f"{self.split}.csv"),
            "image_shape": [len(self.modalities), *cs.CROP_SHAPE_DHW],
            "label_shape": list(cs.CROP_SHAPE_DHW),
            "channel_order": list(cs.CHANNEL_ORDER),
            "modalities": list(self.modalities),
            "channel_indices": list(self.channel_indices),
            "spacing_dhw_mm": list(cs.SPACING_DHW_MM),
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

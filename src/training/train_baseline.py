"""Baseline v1 training framework.

Includes the mandatory tiny-set overfit acceptance test as a mode
(``--smoke-overfit N``) rather than a separate script, so the overfit test
exercises exactly the same dataset, model, loss and optimiser code that real
training will use -- a separate implementation could pass while the real path is
broken.

Split discipline
----------------
Only ``train`` and ``val`` are ever constructed. ``internal_test`` and
``challenge_test`` are not reachable from this module: ``SegmentationDataset``
rejects those split names outright. Model selection uses validation macro
foreground Dice only.

Usage
-----
    # full training (NOT run automatically at this stage)
    python -m src.training.train_baseline --config configs/subject_clean_v1/baseline_v1.yaml

    # tiny-set overfit acceptance test
    python -m src.training.train_baseline --config configs/subject_clean_v1/baseline_v1.yaml \
        --smoke-overfit 2 --max-steps 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data import crop_spec as cs  # noqa: E402
from data.interface_soft_target import (  # noqa: E402
    BAND_MM as SOFT_BAND_MM,
    interface_soft_target,
)
from data.interface_soft_target_mass_conserving import (  # noqa: E402
    mass_conserving_soft_target,
)
from data.segmentation_dataset import SegmentationDataset  # noqa: E402
from data.spatial_prior import (  # noqa: E402
    MODE_NONE as SPATIAL_PRIOR_NONE,
    OccupancyPrior,
    SpatialPriorDataset,
    build_spatial_prior,
    resolve_spatial_prior_mode,
    spatial_prior_channel_count,
)
from evaluation import segmentation_metrics as sm  # noqa: E402
from losses.boundary_dice_ce import (  # noqa: E402
    BoundaryDiceCELoss,
    BoundaryDiceCELossConfig,
)
from losses.dice_ce import DiceCELoss, DiceCELossConfig  # noqa: E402
from losses.stn_interface_soft_ce import (  # noqa: E402
    InterfaceSoftDiceCELoss,
    InterfaceSoftDiceCELossConfig,
)
from losses.tversky_dice_ce import (  # noqa: E402
    STN,
    HybridOverlapCELoss,
    HybridOverlapCELossConfig,
)
from models.anisotropic_unet3d import (  # noqa: E402
    AnisotropicUNet3D,
    UNet3DConfig,
    count_parameters,
)
from utils.paths import resolve_path, resolve_project_root  # noqa: E402

LOGGER = logging.getLogger("train_baseline")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class BaselineConfig:
    """Everything needed to reproduce a run."""

    # data
    root: str = "."
    cache_dir: str = "cache/baseline_v1"
    manifest_dir: str = "manifests/experiment"

    # model
    base_channels: int = 16
    in_channels: int = cs.IN_CHANNELS
    num_classes: int = cs.NUM_CLASSES
    #: Which modalities to feed the model, a subset of ``crop_spec.CHANNEL_ORDER``.
    #: ``None`` (the default) means all three in the frozen order, so Baseline v1
    #: is unchanged. When set, ``in_channels`` is DERIVED from it rather than read
    #: from the config, so the two can never disagree -- see resolve_modalities.
    modalities: Sequence[str] | None = None
    #: Experiment E: which spatial priors to append as extra input channels.
    #: "none" (the default) adds nothing, so Baseline v1 and Experiments A-D are
    #: unaffected by this field's existence. "coords" adds Cx/Cy/Cz, "occupancy"
    #: adds the train-set occupancy maps P_STN/P_SN/P_RN, "both" adds all six.
    spatial_prior_mode: str = "none"
    #: Directory holding the occupancy cache (required for "occupancy"/"both").
    spatial_prior_cache_dir: str | None = None

    # optimisation
    seed: int = 42
    batch_size: int = 2
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    amp: bool = True
    max_epochs: int = 200
    early_stopping_patience: int = 40
    num_workers: int = 0
    #: Page-lock host staging buffers for faster host->device copies. Purely a
    #: throughput setting: it cannot change weights, losses or metrics. Disable it
    #: on a machine short of physical RAM, where the pinned allocator can fail and
    #: surface as a misleading "CUDA error: out of memory" even though the GPU has
    #: plenty of free VRAM.
    pin_memory: bool = True
    grad_accumulation_steps: int = 1
    grad_clip_norm: float | None = 1.0

    # loss
    ce_weight: float = 1.0
    #: For "dice_ce" this weights the foreground soft-Dice term. For
    #: "stn_tversky" it weights the WHOLE hybrid overlap term (STN + SN + RN),
    #: not the Dice part alone -- reusing the key keeps one knob per loss term
    #: instead of two names for the same thing.
    dice_weight: float = 1.0
    #: Which criterion to build. "dice_ce" is the frozen Baseline v1 loss and is
    #: the default, so every pre-existing config still takes the baseline path.
    #: "stn_tversky" is Experiment B.
    loss_type: str = "dice_ce"
    #: Experiment B only: class given the asymmetric Tversky index, and the
    #: coefficients on FALSE POSITIVES (alpha) and FALSE NEGATIVES (beta).
    #: alpha > beta means over-segmentation is punished harder than
    #: under-segmentation. Ignored unless loss_type == "stn_tversky".
    tversky_class: int = STN
    tversky_alpha: float = 0.6
    tversky_beta: float = 0.4
    #: Experiment C only: weight on the STN signed-distance boundary term, and
    #: the class it supervises. Ignored unless loss_type == "stn_boundary".
    boundary_weight: float = 0.10
    boundary_class: int = STN
    #: Directory of precomputed STN signed-distance maps. When set, the TRAIN
    #: split is wrapped so each sample also carries its map. Left None, no split
    #: is wrapped and nothing changes for any other experiment.
    boundary_cache_dir: str | None = None
    #: Experiment G1 only: replace the STN component of the TRAINING target with
    #: the interface-centred soft map. False (the default) leaves the baseline and
    #: every other experiment on the hard one-hot label, so their behaviour is
    #: unaffected by this field's existence.
    soft_boundary_enabled: bool = False
    #: Half-width of the soft ramp in mm. Pre-registered; never tuned.
    soft_boundary_band_mm: float = SOFT_BAND_MM
    #: Which soft target to build. "interface" is Experiment G1, whose outside
    #: shell carries a +4.104% net STN mass bias. "mass_conserving" is G1-MC,
    #: which rescales that shell per case so the case total is exactly the hard
    #: count. Only the target builder differs; the loss is shared.
    soft_boundary_mode: str = "interface"

    # augmentation (Experiment A). Off by default so the baseline is unchanged
    # when this flag is absent from a config.
    augmentation_enabled: bool = False
    augmentation_seed: int | None = None

    # bookkeeping
    checkpoint_dir: str = "checkpoints/baseline_v1"
    results_dir: str = "results/baseline_v1"


def load_config(path: str | Path | None) -> BaselineConfig:
    """Load a YAML config, falling back to the dataclass defaults."""
    config = BaselineConfig()
    if path is None:
        return config
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    known = {f for f in asdict(config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for key, value in raw.items():
        setattr(config, key, value)
    return config


def set_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed every RNG and configure cuDNN for reproducibility.

    Returns a record of what was configured, including an explicit note about
    whether bitwise determinism is actually guaranteed. Full bitwise determinism
    is NOT claimed: forcing ``torch.use_deterministic_algorithms(True)`` would
    reject some of the kernels this network relies on, and the instruction is not
    to break the implementation for determinism's sake.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    record = {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": torch.cuda.is_available(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "bitwise_deterministic": False,
        "note": (
            "cuDNN is in deterministic mode and autotuning is off. Bitwise "
            "reproducibility is still not guaranteed: some CUDA kernels used here "
            "are non-deterministic by default and enforcing "
            "torch.use_deterministic_algorithms(True) would reject them. Run-to-run "
            "variation from this source is expected to be far smaller than the "
            "seed-to-seed variation of training itself."
        ),
    }
    LOGGER.info("Seeding: %s", record["note"])
    return record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash(start_dir: Path) -> str | None:
    """Current git commit, or None when the project is not a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=start_dir,
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - absence of git must not break a run
        pass
    return None


def record_environment(
    run_results_dir: Path,
    config: BaselineConfig,
    config_path: Path | None,
    manifest_dir: Path,
    seeding: dict[str, Any],
) -> dict[str, Any]:
    """Write environment.json + run_config.yaml so a run can be identified later.

    Hashes the experiment manifests and the config file: a recorded accuracy
    number is only meaningful if you can prove which data split and which config
    produced it.
    """
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_gb": (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else None
        ),
        "gpu_capability": (
            f"{torch.cuda.get_device_properties(0).major}."
            f"{torch.cuda.get_device_properties(0).minor}"
            if torch.cuda.is_available() else None
        ),
        "git_commit": git_commit_hash(run_results_dir.parent),
        "seeding": seeding,
        "frozen_config": asdict(config),
        "crop": {
            "shape_xyz": list(cs.CROP_SHAPE_XYZ),
            "shape_dhw": list(cs.CROP_SHAPE_DHW),
            "slices": {"x": list(cs.CROP_X), "y": list(cs.CROP_Y), "z": list(cs.CROP_Z)},
            "channel_order": list(cs.CHANNEL_ORDER),
            "spacing_dhw_mm": list(cs.SPACING_DHW_MM),
        },
        "config_file": str(config_path) if config_path else None,
        "config_sha256": sha256_file(config_path) if config_path and config_path.is_file() else None,
        "manifest_hashes": {},
    }

    # Experiment E: record which prior the run used, and the cache's own
    # identity + content hash, so a reported number can be tied to a specific
    # occupancy map rather than to "whatever was in the directory at the time".
    prior_mode = resolve_spatial_prior_mode(config.spatial_prior_mode)
    env["spatial_prior"] = {
        "mode": prior_mode,
        "channels": spatial_prior_channel_count(prior_mode),
        "image_channels": len(resolve_modalities(config)),
        "in_channels": config.in_channels,
        "cache_dir": config.spatial_prior_cache_dir,
        "cache_metadata": None,
    }
    if config.spatial_prior_cache_dir:
        meta_path = resolve_path(config.spatial_prior_cache_dir,
                                 Path(config.root)) / "occupancy_meta.json"
        if meta_path.is_file():
            cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            env["spatial_prior"]["cache_metadata"] = {
                "path": str(meta_path),
                "n_train": cache_meta.get("n_train"),
                "sums_sha256": cache_meta.get("sums_sha256"),
                "manifest_sha256": cache_meta.get("manifest_sha256"),
                "class_names": cache_meta.get("class_names"),
                "kind": cache_meta.get("kind"),
                "not_an_atlas": cache_meta.get("not_an_atlas"),
            }

    for split in ("train", "val", "internal_test", "challenge_test"):
        path = manifest_dir / f"{split}.csv"
        if path.is_file():
            env["manifest_hashes"][split] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "n_lines": sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1,
            }

    run_results_dir.mkdir(parents=True, exist_ok=True)
    with (run_results_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=2, ensure_ascii=False, default=str)

    if config_path and config_path.is_file():
        shutil.copyfile(config_path, run_results_dir / "run_config.yaml")

    LOGGER.info("Environment recorded: torch %s / cuda %s / cudnn %s / gpu %s",
                env["torch_version"], env["torch_cuda_version"],
                env["cudnn_version"], env["gpu_name"])
    LOGGER.info("Config sha256    : %s", env["config_sha256"])
    for split, info in env["manifest_hashes"].items():
        LOGGER.info("Manifest %-15s n=%-4s sha256=%s",
                    split, info["n_lines"], info["sha256"][:16])
    return env


class TrainingHalt(RuntimeError):
    """Raised when a numerical or I/O safety check fails.

    Signals that training must stop immediately WITHOUT altering any
    hyper-parameter. The instruction is explicit: stop and report, never
    auto-adjust and continue.
    """


def check_finite(name: str, tensor: torch.Tensor, epoch: int, step: int) -> None:
    if not torch.isfinite(tensor).all():
        raise TrainingHalt(
            f"non-finite {name} at epoch {epoch} step {step}: "
            f"{tensor.detach().flatten()[:5].tolist()}"
        )


#: How many consecutive non-finite gradients (under AMP) before the run is
#: declared diverged rather than merely skipping an overflowing step.
MAX_CONSECUTIVE_NONFINITE_GRADIENTS = 8


def check_gradients_finite(model: nn.Module, epoch: int, step: int) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise TrainingHalt(
                f"non-finite gradient in {name} at epoch {epoch} step {step}"
            )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def resolve_config_paths(config: BaselineConfig, root_override: Path | None = None) -> Path:
    """Anchor every configured path to the project root, in place.

    Config values are typically relative (``cache/baseline_v1``). Without this
    they would be resolved against the shell's working directory, so running the
    trainer from another directory would read the wrong place. Returns the
    resolved project root.
    """
    project_root = resolve_project_root(
        root_override if root_override is not None else config.root
    )
    config.root = str(project_root)
    config.cache_dir = str(resolve_path(config.cache_dir, project_root))
    config.manifest_dir = str(resolve_path(config.manifest_dir, project_root))
    config.checkpoint_dir = str(resolve_path(config.checkpoint_dir, project_root))
    config.results_dir = str(resolve_path(config.results_dir, project_root))
    return project_root


#: Criterion component key -> training-history column name. A criterion may report
#: extra diagnostic components; anything not listed here becomes ``train_<key>``.
#:
#: This map exists because the alternative -- deriving the column from the key
#: alone -- would rename the Baseline's ``train_total_loss`` to
#: ``train_loss_loss`` and silently change the column set of every baseline run.
#: With the map, Baseline v1 keeps exactly the columns it has always had.
COMPONENT_COLUMNS: dict[str, str] = {
    "loss": "train_total_loss",
    "ce": "train_ce_loss",
    "dice": "train_dice_loss",
    "overlap": "train_overlap_loss",
    "boundary": "train_boundary_loss",
}


def build_criterion(config: BaselineConfig) -> nn.Module:
    """Build the criterion selected by ``config.loss_type``.

    ``"dice_ce"`` constructs exactly the object this trainer built before this
    factory existed, so Baseline v1 is unaffected by its presence.
    """
    if config.loss_type == "dice_ce":
        return DiceCELoss(DiceCELossConfig(
            ce_weight=config.ce_weight, dice_weight=config.dice_weight,
        ))
    if config.loss_type == "stn_tversky":
        return HybridOverlapCELoss(HybridOverlapCELossConfig(
            ce_weight=config.ce_weight,
            overlap_weight=config.dice_weight,
            tversky_class=config.tversky_class,
            tversky_alpha=config.tversky_alpha,
            tversky_beta=config.tversky_beta,
        ))
    if config.loss_type == "stn_boundary":
        return BoundaryDiceCELoss(BoundaryDiceCELossConfig(
            ce_weight=config.ce_weight,
            dice_weight=config.dice_weight,
            boundary_weight=config.boundary_weight,
            boundary_class=config.boundary_class,
        ))
    if config.loss_type == "stn_interface_soft":
        return InterfaceSoftDiceCELoss(InterfaceSoftDiceCELossConfig(
            ce_weight=config.ce_weight,
            dice_weight=config.dice_weight,
        ))
    raise ValueError(
        f"unknown loss_type {config.loss_type!r}; expected 'dice_ce', "
        f"'stn_tversky', 'stn_boundary' or 'stn_interface_soft'"
    )


def resolve_input_channels(config: BaselineConfig) -> int:
    """Total model input width = image modalities + spatial-prior channels.

    The single place ``in_channels`` is decided. It is never read from the config
    file, so a YAML value can never disagree with what the Dataset actually
    emits -- the failure mode that would otherwise show up as a confusing shape
    error deep inside the first convolution.
    """
    image_channels = len(resolve_modalities(config))
    prior_channels = spatial_prior_channel_count(config.spatial_prior_mode)
    total = image_channels + prior_channels
    if total <= 0:
        raise ValueError("input must have at least one channel")
    return total


def resolve_modalities(config: BaselineConfig) -> tuple[str, ...]:
    """Modalities to feed the model, validated against the frozen channel order.

    ``None`` resolves to every modality in ``crop_spec.CHANNEL_ORDER``, which is
    what Baseline v1 uses. The returned tuple is also what ``in_channels`` is
    derived from, so the model's input width and the data's channel count cannot
    drift apart.
    """
    modalities = (tuple(cs.CHANNEL_ORDER) if config.modalities is None
                  else tuple(config.modalities))
    if not modalities:
        raise ValueError("modalities must select at least one modality")
    unknown = [m for m in modalities if m not in cs.CHANNEL_ORDER]
    if unknown:
        raise ValueError(
            f"unknown modality/modalities {unknown}; available: "
            f"{tuple(cs.CHANNEL_ORDER)}"
        )
    if len(set(modalities)) != len(modalities):
        raise ValueError(f"duplicate modalities in {modalities}")
    return modalities


class BoundarySupervisionDataset(Dataset):
    """Wraps a :class:`SegmentationDataset`, adding the STN signed-distance map.

    A wrapper rather than an extra option on ``SegmentationDataset``: the
    baseline's dataset must keep returning exactly what it has always returned,
    and this extra supervision exists only for the one experiment that uses it.
    Only the training split is wrapped -- the boundary map is training
    supervision, and validation never builds a criterion.
    """

    def __init__(self, base: Dataset, boundary_dir: str | Path) -> None:
        self.base = base
        self.boundary_dir = Path(boundary_dir)
        # Mirror the base's public surface so callers (overfit subsetting, case
        # listing) keep working unchanged on the wrapped object.
        self.case_ids: list[str] = list(base.case_ids)  # type: ignore[attr-defined]
        missing = [c for c in self.case_ids
                   if not (self.boundary_dir / f"{c}.npy").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} boundary map(s) missing from {self.boundary_dir}, "
                f"e.g. {missing[:5]}. Run scripts/build_boundary_cache.py first."
            )
        LOGGER.info("BoundarySupervisionDataset: %d case(s) from %s",
                    len(self.case_ids), self.boundary_dir)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        case_id = sample["case_id"]
        phi = np.load(self.boundary_dir / f"{case_id}.npy")
        if phi.shape != cs.CROP_SHAPE_DHW:
            raise ValueError(
                f"{case_id}: boundary map shape {phi.shape} != frozen crop "
                f"{cs.CROP_SHAPE_DHW}"
            )
        if not np.isfinite(phi).all():
            raise ValueError(f"{case_id}: boundary map contains NaN/Inf")
        sample["boundary"] = torch.from_numpy(
            np.ascontiguousarray(phi)).float()
        return sample


#: Soft-target builders, by config name. Both take ``(label, band_mm)`` and
#: return ``(C, D, H, W)``. Only the target differs between them -- G1-MC exists
#: because the G1 interface target carries a +4.104% net STN mass bias that the
#: audits traced to an outside/inside interface-voxel count asymmetry, and it
#: corrects that by rescaling the outside shell per case. The loss, the geometry,
#: the band and the tie rule are shared, so this is the entire difference.
SOFT_BOUNDARY_MODES: dict[str, Any] = {
    "interface": interface_soft_target,
    "mass_conserving": mass_conserving_soft_target,
}


def resolve_soft_boundary_mode(mode: str) -> str:
    """Validate a ``soft_boundary_mode`` value and return it unchanged."""
    if mode not in SOFT_BOUNDARY_MODES:
        raise ValueError(
            f"unknown soft_boundary_mode {mode!r}; expected one of "
            f"{sorted(SOFT_BOUNDARY_MODES)}"
        )
    return mode


class InterfaceSoftTargetDataset(Dataset):
    """Wraps a :class:`SegmentationDataset`, adding the soft STN target.

    A wrapper rather than an option on ``SegmentationDataset``, for the same
    reason as :class:`BoundarySupervisionDataset`: the baseline's dataset must
    keep returning exactly what it always has. Only the training split is
    wrapped -- validation must keep the hard ground truth so every reported
    metric stays comparable, and the criterion is never built for validation.

    The target is derived from the case's own GT label, so no cache is needed and
    nothing can go stale. It costs ~6 ms/case (the geometry is computed on a
    window around the structure, not the whole crop).
    """

    def __init__(
        self,
        base: Dataset,
        band_mm: float = SOFT_BAND_MM,
        mode: str = "interface",
    ) -> None:
        self.base = base
        self.band_mm = float(band_mm)
        self.mode = resolve_soft_boundary_mode(mode)
        self.builder = SOFT_BOUNDARY_MODES[self.mode]
        self.case_ids: list[str] = list(base.case_ids)  # type: ignore[attr-defined]
        LOGGER.info("InterfaceSoftTargetDataset: %d case(s), band=%.7f mm, mode=%s",
                    len(self.case_ids), self.band_mm, self.mode)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        label = sample["label"].numpy()
        soft = self.builder(label, band_mm=self.band_mm)
        sample["soft_target"] = torch.from_numpy(soft).float()
        return sample


def build_dataloader(
    config: BaselineConfig, split: str, shuffle: bool, augment: bool = False
) -> tuple[Dataset, DataLoader]:
    """Build a loader. ``augment`` is honoured for the train split only.

    ``SegmentationDataset`` raises if augmentation is requested for any other
    split, so the evaluation path cannot accidentally become stochastic.
    """
    dataset: Dataset = SegmentationDataset(
        root=config.root, split=split,
        cache_dir=config.cache_dir, manifest_dir=config.manifest_dir,
        augment=augment and split == "train",
        augment_seed=(config.augmentation_seed
                      if config.augmentation_seed is not None else config.seed),
        modalities=config.modalities,
    )
    # Experiment C only: attach the STN signed-distance map to training samples.
    # Train split only -- validation never builds a criterion, so wrapping it
    # would just add I/O.
    if config.boundary_cache_dir and split == "train":
        dataset = BoundarySupervisionDataset(
            dataset, resolve_path(config.boundary_cache_dir, Path(config.root)))

    # Experiment G1 only: attach the interface-centred soft STN target to training
    # samples. Train split only -- validation keeps the hard GT, and no criterion
    # is built for it.
    if config.soft_boundary_enabled and split == "train":
        dataset = InterfaceSoftTargetDataset(
            dataset, band_mm=config.soft_boundary_band_mm,
            mode=config.soft_boundary_mode)

    # Experiment E: append spatial-prior channels, applied last (outermost) so it
    # sees the fully assembled sample. Only `image` is extended and the priors go
    # AFTER the modalities, so channel 0 stays T1 for the prediction overlays.
    prior_mode = resolve_spatial_prior_mode(config.spatial_prior_mode)
    if prior_mode != SPATIAL_PRIOR_NONE:
        if augment and split == "train":
            # Nothing in the augmentation pipeline transforms a prior, so an
            # augmented image would no longer line up with its coordinates or its
            # occupancy map. Refuse rather than train on silently misaligned
            # supervision.
            raise ValueError(
                "Experiment E v1 does not support augmentation: spatial priors "
                "would not receive the same spatial transform as the image. "
                "Set augmentation_enabled=false, or implement a synchronised "
                "spatial transform for priors first."
            )
        occupancy, prior_mode = build_spatial_prior(config.root, config, split)
        dataset = SpatialPriorDataset(dataset, prior_mode, split=split,
                                      occupancy=occupancy)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
    return dataset, loader


# --------------------------------------------------------------------------- #
# Train / eval steps
# --------------------------------------------------------------------------- #


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    config: BaselineConfig,
    max_steps: int | None = None,
    step_offset: int = 0,
    epoch: int = 0,
) -> tuple[dict[str, float], int]:
    """One pass over ``loader``. Returns (mean losses, number of optimizer steps).

    Raises :class:`TrainingHalt` on a non-finite loss or gradient. The run stops
    there rather than adjusting anything and continuing.

    The accumulated components are read off whatever the criterion returns rather
    than a hard-coded key set, so a criterion with extra diagnostic terms is
    logged (and finiteness-checked) too.
    """
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0
    steps = 0
    optimiser.zero_grad(set_to_none=True)
    step_index = step_offset
    nonfinite_gradient_count = 0
    consecutive_nonfinite = 0
    amp_skipped_steps = 0

    for batch_index, batch in enumerate(loader):
        if max_steps is not None and step_index >= max_steps:
            break

        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        # Present only for experiments that supply extra supervision (Experiment
        # C's boundary maps). Every other configuration has no such key, so the
        # criterion is called with the same two arguments it always was.
        boundary = batch.get("boundary")
        if boundary is not None:
            boundary = boundary.to(device, non_blocking=True)
        # Present only for Experiment G1, and only on the training split. When
        # absent the criterion is called with the hard label exactly as before.
        soft_target = batch.get("soft_target")
        if soft_target is not None:
            soft_target = soft_target.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
            logits = model(image)
            if soft_target is not None:
                loss, components = criterion(logits, soft_target)
            elif boundary is None:
                loss, components = criterion(logits, label)
            else:
                loss, components = criterion(logits, label, boundary)

        # Safety: a non-finite loss must stop the run, not be optimised through.
        # Every reported component is checked, not just the three the baseline
        # happens to report.
        for key, value in components.items():
            check_finite(key, value, epoch, step_index)

        if not totals:
            totals = {key: 0.0 for key in components}
        elif set(components) != set(totals):
            # A criterion whose key set changes between batches would otherwise
            # surface as a confusing KeyError, or be quietly half-ignored.
            raise TrainingHalt(
                f"criterion component keys changed mid-epoch at epoch {epoch} "
                f"step {step_index}: {sorted(totals)} -> {sorted(components)}"
            )

        loss = loss / config.grad_accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_boundary = (batch_index + 1) % config.grad_accumulation_steps == 0
        is_last = batch_index + 1 == len(loader)
        if is_boundary or is_last:
            if scaler is not None:
                # ---- AMP path ---------------------------------------------- #
                # A non-finite gradient here is NOT automatically an error.
                # With fp16 autocast, GradScaler deliberately scales the loss up
                # until the scaled gradients overflow, then detects that and
                # SKIPS the step (reducing the scale). That is the mechanism
                # working as intended, and it is transient.
                #
                # Halting on the first occurrence would abort a healthy run: a
                # real Experiment A run was lost this way at epoch 61 with a
                # perfectly finite, monotonically decreasing loss. So a single
                # overflow is absorbed; only a *sustained* run of them means the
                # training has actually diverged.
                scaler.unscale_(optimiser)
                gradients_finite = all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.grad is not None
                )
                if gradients_finite:
                    if config.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip_norm)
                else:
                    # Deliberately do NOT clip: clipping a non-finite gradient
                    # can zero it out, after which GradScaler would no longer
                    # recognise the overflow and would take a step on garbage.
                    nonfinite_gradient_count += 1
                    consecutive_nonfinite += 1
                    if consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE_GRADIENTS:
                        raise TrainingHalt(
                            f"{consecutive_nonfinite} consecutive non-finite "
                            f"gradients ending at epoch {epoch} step {step_index} "
                            f"-- training has diverged, not a transient AMP skip"
                        )

                scale_before = scaler.get_scale()
                scaler.step(optimiser)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    amp_skipped_steps += 1
                if gradients_finite:
                    consecutive_nonfinite = 0
            else:
                # ---- non-AMP path: a non-finite gradient is a hard error --- #
                check_gradients_finite(model, epoch, step_index)
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            steps += 1
            step_index += 1

        for key, value in components.items():
            totals[key] += float(value.item())
        n_batches += 1

    if not totals:
        # An empty loader would otherwise return an empty dict and push a
        # meaningless zero row into the history. Stop and say so instead.
        raise TrainingHalt(
            f"loader produced no batches at epoch {epoch}; refusing to report a "
            f"zero-filled loss row"
        )

    amp_stats = {
        "amp_skipped_steps": amp_skipped_steps,
        "nonfinite_gradient_steps": nonfinite_gradient_count,
    }
    return ({k: v / max(n_batches, 1) for k, v in totals.items()}, steps, amp_stats)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    compute_hd95: bool = False,
) -> tuple[list[sm.CaseMetrics], dict[str, dict[str, float]]]:
    """Per-case metrics over the validation set, then aggregated across cases."""
    model.eval()
    per_case: list[sm.CaseMetrics] = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(image)
        # One case at a time: metrics are never pooled across a batch.
        for index in range(image.shape[0]):
            per_case.append(sm.evaluate_case(
                logits[index], label[index],
                case_id=batch["case_id"][index],
                compute_hd95=compute_hd95,
            ))

    return per_case, sm.aggregate_cases(per_case)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    best_metric: float,
    config: BaselineConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": asdict(config),
            "seed": config.seed,
            "class_names": cs.CLASS_NAMES,
            "crop_shape_dhw": list(cs.CROP_SHAPE_DHW),
            "channel_order": list(cs.CHANNEL_ORDER),
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_training_curves(history: pd.DataFrame, out_path: Path, dpi: int = 120) -> None:
    """Training loss and validation Dice curves.

    Purely a rendering of the recorded history -- nothing here feeds back into
    training or changes any result.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, panels = plt.subplots(1, 2, figsize=(14, 5))

    # Draw whichever loss components this run actually recorded. A hard-coded
    # list would raise KeyError for a criterion with a different component set --
    # and since this whole function is wrapped in an `except Exception` by the
    # caller, that failure would be a single warning line and no figure at all.
    components = [c for c in history.columns
                  if c.startswith("train_") and c.endswith("_loss")]
    if "train_total_loss" in components:
        components.remove("train_total_loss")
        panels[0].plot(history["epoch"], history["train_total_loss"],
                       label="total", lw=1.8)
    for column in components:
        panels[0].plot(history["epoch"], history[column],
                       label=column[len("train_"):-len("_loss")], lw=1.2, alpha=0.85)
    panels[0].set_xlabel("epoch")
    panels[0].set_ylabel("loss")
    panels[0].set_title("Training loss")
    panels[0].grid(alpha=0.25)
    panels[0].legend()

    for name, colour in (("STN", "#ff3b30"), ("SN", "#34c759"), ("RN", "#0a84ff")):
        column = f"val_Dice_{name}"
        if column in history:
            panels[1].plot(history["epoch"], history[column], label=name,
                           color=colour, lw=1.6)
    if "val_macro_Dice" in history:
        panels[1].plot(history["epoch"], history["val_macro_Dice"],
                       label="macro", color="black", lw=2.2, linestyle="--")
    if "is_best" in history and history["is_best"].any():
        # The best epoch is the LAST improvement, not the first: idxmax() on a
        # boolean column returns the earliest True, which would mislabel the line
        # whenever the model improved more than once.
        best_epoch = int(history.loc[history["is_best"], "epoch"].max())
        panels[1].axvline(best_epoch, color="grey", linestyle=":", lw=1.4,
                          label=f"best (epoch {best_epoch})")
    panels[1].set_xlabel("epoch")
    panels[1].set_ylabel("Dice")
    panels[1].set_title("Validation Dice (development val, n=40)")
    panels[1].set_ylim(0, 1)
    panels[1].grid(alpha=0.25)
    panels[1].legend()

    figure.suptitle("Baseline v1 training", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_overlays(
    model: nn.Module,
    dataset: SegmentationDataset,
    case_ids: Sequence[str],
    out_path: Path,
    dpi: int = 120,
) -> None:
    """GT vs prediction overlays for a few validation cases (sanity check only).

    Contours rather than filled masks so the underlying image stays visible. This
    is a qualitative check; it must not be used to retune this baseline.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colours = {"STN": "#ff3b30", "SN": "#34c759", "RN": "#0a84ff"}
    model.eval()
    device = next(model.parameters()).device

    figure, panels = plt.subplots(len(case_ids), 3,
                                  figsize=(11.5, 3.8 * len(case_ids)), squeeze=False)
    for row, case_id in enumerate(case_ids):
        index = dataset.case_ids.index(case_id)
        sample = dataset[index]
        image = sample["image"][None].to(device)
        with torch.no_grad():
            prediction = model(image).argmax(1)[0].cpu().numpy()
        truth = sample["label"].numpy()

        # Axial slice at the middle of the D (inferior-superior) axis.
        #
        # The tensors are (D, H, W) = (Z, Y, X), so indexing the FIRST axis gives
        # a (H, W) = (Y, X) axial slice directly: rows are Y (posterior-anterior)
        # and columns are X (right-left). Slicing the last axis instead would give
        # a (D, H) sagittal strip, and the contours would still line up with the
        # image, so the mistake would not announce itself.
        z = truth.shape[0] // 2
        background = sample["image"][0].numpy()[z]

        for column, (title, mask) in enumerate(
            (("ground truth", truth), ("prediction", prediction))
        ):
            ax = panels[row][column]
            ax.imshow(background, cmap="gray", origin="lower", aspect="equal")
            for class_id, name in cs.CLASS_NAMES.items():
                layer = (mask == class_id)[z]
                if layer.any():
                    ax.contour(layer.astype(float), levels=[0.5],
                               colors=[colours[name]], linewidths=1.4, origin="lower")
            ax.set_title(f"{case_id} — {title} (z={z})", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        difference = panels[row][2]
        difference.imshow(background, cmap="gray", origin="lower", aspect="equal")
        for class_id, name in cs.CLASS_NAMES.items():
            gt_layer = (truth == class_id)[z]
            pred_layer = (prediction == class_id)[z]
            if gt_layer.any():
                difference.contour(gt_layer.astype(float), levels=[0.5],
                                   colors=[colours[name]], linewidths=1.2, origin="lower")
            if pred_layer.any():
                difference.contour(pred_layer.astype(float), levels=[0.5],
                                   colors=[colours[name]], linewidths=1.2,
                                   linestyles="dashed", origin="lower")
        difference.set_title("GT (solid) vs pred (dashed)", fontsize=9)
        difference.set_xticks([])
        difference.set_yticks([])

    handles = [Line2D([0], [0], color=colours[n], lw=2, label=n)
               for n in cs.CLASS_NAMES.values()]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 0.0))
    figure.suptitle("Validation sanity check — ground truth vs prediction", fontsize=13)
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline v1 training.")
    parser.add_argument("--config", type=Path, default=Path("configs/subject_clean_v1/baseline_v1.yaml"))
    parser.add_argument("--smoke-overfit", type=int, default=None,
                        help="Tiny-set overfit test on N fixed development-train cases.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap optimizer steps (smoke testing).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override config.")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu.")
    parser.add_argument("--tag", type=str, default="", help="Suffix for output names.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Formal run name, e.g. formal_seed42. Outputs go to "
                             "<results_dir>/<run-id> and <checkpoint_dir>/<run-id>. "
                             "Refuses to overwrite an existing run directory.")
    parser.add_argument("--resume", action="store_true",
                        help="Allow writing into an existing --run-id directory.")
    parser.add_argument("--root", type=Path, default=None,
                        help="Project root. Relative paths in the config are anchored "
                             "here (default: the repository root).")
    parser.add_argument("--skip-final-eval", action="store_true",
                        help="Skip the best-checkpoint final validation pass.")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip training curves and prediction overlays.")
    parser.add_argument("--n-overlay-cases", type=int, default=4,
                        help="How many validation cases to draw in the overlay figure.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


@torch.no_grad()
def final_evaluation(
    checkpoint_path: Path,
    val_loader: DataLoader,
    device: torch.device,
    results_dir: Path,
    config: BaselineConfig,
) -> dict[str, Any]:
    """Reload the best checkpoint and evaluate development val once, with HD95.

    Runs only at the end of training, never per epoch: HD95 needs a distance
    transform per class per case and would dominate epoch time for no benefit --
    checkpoint selection is driven by macro foreground Dice alone.
    """
    payload = load_checkpoint(checkpoint_path)
    # Rebuild the architecture from what the checkpoint itself recorded, so
    # loading never depends on the caller happening to pass a matching config.
    stored = payload.get("config", {}) or {}
    model = AnisotropicUNet3D(UNet3DConfig(
        in_channels=int(stored.get("in_channels", config.in_channels)),
        num_classes=int(stored.get("num_classes", config.num_classes)),
        base_channels=int(stored.get("base_channels", config.base_channels)),
    )).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    per_case: list[sm.CaseMetrics] = []
    for batch in val_loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(image)
        for index in range(image.shape[0]):
            per_case.append(sm.evaluate_case(
                logits[index], label[index],
                case_id=batch["case_id"][index], compute_hd95=True,
            ))

    aggregate = sm.aggregate_cases(per_case, include_hd95=True)

    per_case_path = results_dir / "val_per_case_best.csv"
    pd.DataFrame([m.to_per_case_csv_row() for m in per_case]).sort_values(
        "case_id").to_csv(per_case_path, index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_best_metric": payload.get("best_metric"),
        "selection_metric": "validation macro foreground Dice "
                            "(NOT HD95, which is computed only here)",
        "n_cases": len(per_case),
        "hd95_spacing_dhw_mm": list(cs.SPACING_DHW_MM),
        "metrics": aggregate,
        "macro_foreground_dice_mean": sm.macro_dice_from_aggregate(aggregate),
    }
    summary_path = results_dir / "val_summary_best.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    LOGGER.info("Final evaluation written to %s and %s", per_case_path, summary_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config if args.config.is_file() else None)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    seeding = set_seed(config.seed)

    project_root = resolve_config_paths(config, args.root)
    # Derive the input width from the modality selection. Done here, before any
    # run directory is written, so environment.json records the width the run
    # actually used rather than whatever the config file happened to say.
    modalities = resolve_modalities(config)
    prior_mode = resolve_spatial_prior_mode(config.spatial_prior_mode)
    prior_channels = spatial_prior_channel_count(prior_mode)
    config.in_channels = resolve_input_channels(config)
    LOGGER.info("Inputs: modalities %s (%d ch) + spatial prior %r (%d ch) "
                "-> in_channels=%d", "/".join(modalities), len(modalities),
                prior_mode, prior_channels, config.in_channels)

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("  cache_dir     : %s", config.cache_dir)
    LOGGER.info("  manifest_dir  : %s", config.manifest_dir)
    LOGGER.info("  checkpoint_dir: %s", config.checkpoint_dir)
    LOGGER.info("  results_dir   : %s", config.results_dir)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Device: %s", device)
    if device.type == "cuda":
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))

    overfit = args.smoke_overfit is not None

    # ---- run directory ----------------------------------------------------- #
    # A formal run gets its own directory. Writes never land in the shared parent,
    # and an existing run directory is never overwritten silently -- previous
    # smoke/overfit/diagnostic results must survive.
    base_results_dir = Path(config.results_dir)
    base_checkpoint_dir = Path(config.checkpoint_dir)

    if args.run_id:
        results_dir = base_results_dir / args.run_id
        checkpoint_dir = base_checkpoint_dir / args.run_id
        if results_dir.exists() and not args.resume:
            existing = sorted(p.name for p in results_dir.iterdir())
            print(
                f"\nerror: run directory already exists: {results_dir}\n"
                f"       existing contents: {existing}\n\n"
                f"       Refusing to overwrite. Either pass --resume to continue\n"
                f"       that run, or choose a different --run-id.\n",
                file=sys.stderr,
            )
            return 3
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        record_environment(
            results_dir, config,
            args.config if args.config.is_file() else None,
            Path(config.manifest_dir), seeding,
        )
    else:
        results_dir = base_results_dir
        checkpoint_dir = base_checkpoint_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Run results dir   : %s", results_dir)
    LOGGER.info("Run checkpoint dir: %s", checkpoint_dir)

    # ---- data -------------------------------------------------------------- #
    use_augmentation = bool(config.augmentation_enabled) and not overfit
    LOGGER.info("Augmentation on train split: %s", use_augmentation)
    train_dataset, train_loader = build_dataloader(
        config, "train", shuffle=not overfit, augment=use_augmentation)

    if overfit:
        # Fixed first-N dev-train cases, shuffle off: the point is to show the
        # pipeline can memorise a couple of examples, not to generalise.
        subset = Subset(train_dataset, list(range(min(args.smoke_overfit, len(train_dataset)))))
        train_loader = DataLoader(
            subset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers,
        )
        LOGGER.info("Overfit mode: %d case(s) %s", len(subset),
                    [train_dataset.case_ids[i] for i in range(len(subset))])
        val_dataset = val_loader = None
    else:
        val_dataset, val_loader = build_dataloader(config, "val", shuffle=False)

    # ---- model / loss / optimiser ------------------------------------------ #
    model = AnisotropicUNet3D(UNet3DConfig(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        base_channels=config.base_channels,
    )).to(device)
    params = count_parameters(model)
    LOGGER.info("Model parameters: total=%d trainable=%d", params["total"], params["trainable"])

    criterion = build_criterion(config).to(device)
    LOGGER.info("Criterion: %s (loss_type=%s)", type(criterion).__name__,
                config.loss_type)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # NOTE: checkpoint_dir is deliberately NOT re-derived from config here. It was
    # already resolved above with the --run-id suffix applied; re-reading the bare
    # config value would strip the run directory and drop checkpoints into the
    # shared parent, where consecutive runs would silently overwrite each other.
    tag = args.tag or ("overfit" if overfit else "run")
    history: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    total_steps = 0
    peak_memory_gb = 0.0
    total_amp_skips = 0
    total_nonfinite_steps = 0
    start = time.perf_counter()

    # ---- loop -------------------------------------------------------------- #
    # When --max-steps is given, allow enough epochs to actually reach it. With a
    # tiny dataset (1 batch per epoch) the epoch cap would otherwise stop the run
    # long before the requested step count.
    n_epochs = config.max_epochs
    if args.max_steps is not None:
        n_epochs = max(config.max_epochs, args.max_steps)

    halted: str | None = None
    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            train_losses, steps, amp_stats = train_one_epoch(
                model, train_loader, criterion, optimiser, scaler, device, config,
                max_steps=args.max_steps, step_offset=total_steps, epoch=epoch,
            )
        except TrainingHalt as exc:
            halted = str(exc)
            LOGGER.error("SAFETY HALT: %s", exc)
            break
        except torch.cuda.OutOfMemoryError as exc:
            halted = f"CUDA OOM: {exc}"
            LOGGER.error("SAFETY HALT: %s", halted)
            torch.cuda.empty_cache()
            break
        total_steps += steps

        row: dict[str, Any] = {"epoch": epoch, "step": total_steps}
        # Loss components go in first, in the order the criterion reports them, so
        # the Baseline's three columns keep both their names and their positions.
        for key, value in train_losses.items():
            row[COMPONENT_COLUMNS.get(key, f"train_{key}")] = value
        row["learning_rate"] = optimiser.param_groups[0]["lr"]
        # AMP occasionally skips a step by design; record it so a run that
        # skipped a lot is visible in the history rather than silent.
        row["amp_skipped_steps"] = amp_stats["amp_skipped_steps"]
        row["nonfinite_gradient_steps"] = amp_stats["nonfinite_gradient_steps"]
        total_amp_skips += amp_stats["amp_skipped_steps"]
        total_nonfinite_steps += amp_stats["nonfinite_gradient_steps"]

        if overfit:
            # Report training-set Dice during the overfit test: that IS the metric
            # of interest there, since generalisation is explicitly not the goal.
            cases, aggregate = validate(model, train_loader, device)
            row["train_macro_dice"] = sm.macro_dice_from_aggregate(aggregate)
            for name in cs.CLASS_NAMES.values():
                key = f"dice_{name}"
                if key in aggregate:
                    row[f"train_dice_{name}"] = aggregate[key]["mean"]
            # Predicted/GT volume ratio -- the quantity Experiment B exists to
            # move (Baseline v1 STN: 1.134). Computed exactly as
            # scripts/analyze_baseline_errors.py does, so a smoke figure and a
            # later error-analysis figure are the same measurement.
            #
            # Overfit mode only: emitting it on real runs would change the
            # column set of every existing baseline training_history.csv.
            for class_id, name in cs.CLASS_NAMES.items():
                row[f"train_volume_ratio_{name}"] = sm.volume_ratio_from_cases(
                    cases, class_id)
            metric = row["train_macro_dice"]
        else:
            cases, aggregate = validate(model, val_loader, device)
            row["val_macro_Dice"] = sm.macro_dice_from_aggregate(aggregate)
            for name in cs.CLASS_NAMES.values():
                row[f"val_Dice_{name}"] = aggregate.get(f"dice_{name}", {}).get("mean", math.nan)
                row[f"val_Precision_{name}"] = aggregate.get(
                    f"precision_{name}", {}).get("mean", math.nan)
                row[f"val_Recall_{name}"] = aggregate.get(
                    f"recall_{name}", {}).get("mean", math.nan)
            metric = row["val_macro_Dice"]

        row["epoch_time_seconds"] = time.perf_counter() - epoch_start
        if device.type == "cuda":
            epoch_peak = torch.cuda.max_memory_allocated() / 1e9
            row["gpu_peak_memory_gb"] = epoch_peak
            # Peak memory is reset every epoch so the per-epoch column is that
            # epoch's own peak; keep a separate running maximum, otherwise the
            # run-level figure reported at the end would be only the last epoch's.
            peak_memory_gb = max(peak_memory_gb, epoch_peak)

        improved = (isinstance(metric, float) and math.isfinite(metric)
                    and metric > best_metric)
        row["is_best"] = bool(improved)

        if improved:
            best_metric = metric
            best_epoch = epoch
            epochs_without_improvement = 0
            try:
                save_checkpoint(checkpoint_dir / f"{tag}_best.pt", model, optimiser,
                                None, epoch, best_metric, config)
            except OSError as exc:
                # A checkpoint we cannot write means the run's result is not
                # recoverable -- stop rather than continue losing work.
                halted = f"checkpoint write failed: {exc}"
                LOGGER.error("SAFETY HALT: %s", halted)
                history.append(row)
                break
        else:
            epochs_without_improvement += 1

        history.append(row)
        LOGGER.info(
            "epoch %d/%d step %d loss=%.4f %s=%.4f best=%.4f%s (%.1fs)",
            epoch, n_epochs - 1, total_steps, train_losses["loss"],
            "train_macro" if overfit else "val_macro",
            metric if isinstance(metric, float) and math.isfinite(metric) else float("nan"),
            best_metric if math.isfinite(best_metric) else float("nan"),
            " *" if improved else "", row["epoch_time_seconds"],
        )

        if args.max_steps is not None and total_steps >= args.max_steps:
            LOGGER.info("Reached --max-steps=%d; stopping.", args.max_steps)
            break
        if not overfit and epochs_without_improvement >= config.early_stopping_patience:
            LOGGER.info("Early stopping after %d epochs without improvement.",
                        epochs_without_improvement)
            break

    stopped_early = (not overfit
                     and epochs_without_improvement >= config.early_stopping_patience
                     and halted is None)

    try:
        save_checkpoint(checkpoint_dir / f"{tag}_last.pt", model, optimiser, None,
                        len(history) - 1, best_metric, config)
    except OSError as exc:
        LOGGER.error("Could not write last checkpoint: %s", exc)

    history_frame = pd.DataFrame(history)
    history_path = results_dir / f"training_history.csv"
    if overfit or not args.run_id:
        history_path = results_dir / f"training_history_{tag}.csv"
    history_frame.to_csv(history_path, index=False, encoding="utf-8-sig")

    summary = {
        "tag": tag,
        "run_id": args.run_id,
        "mode": "smoke_overfit" if overfit else "train",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameters": params,
        "epochs_run": len(history),
        "max_epochs": config.max_epochs,
        "total_steps": total_steps,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "selection_metric": "validation macro foreground Dice (STN/SN/RN mean)",
        "stopped_early": bool(stopped_early),
        "early_stopping_patience": config.early_stopping_patience,
        "halted": halted,
        "amp_skipped_steps_total": total_amp_skips,
        "nonfinite_gradient_steps_total": total_nonfinite_steps,
        "amp_note": (
            "AMP skips a step when the scaled gradients overflow; this is its "
            "designed behaviour and is not an error. Only 8 consecutive "
            "non-finite gradients abort a run."
        ),
        "final_train_loss": history[-1]["train_total_loss"] if history else None,
        "initial_train_loss": history[0]["train_total_loss"] if history else None,
        "wall_seconds": time.perf_counter() - start,
        "config": asdict(config),
    }
    if halted:
        summary["halt_reason"] = halted
        (results_dir / "training_halted.txt").write_text(
            f"Training halted at epoch {len(history) - 1}, step {total_steps}.\n"
            f"Reason: {halted}\n\n"
            f"No hyper-parameter was changed automatically. Review and decide.\n",
            encoding="utf-8",
        )
    if device.type == "cuda":
        summary["max_memory_allocated_gb"] = peak_memory_gb
        summary["max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9

    # ---- final evaluation of the best checkpoint --------------------------- #
    # HD95 is computed here and only here: it needs a distance transform per
    # class per case, which would dominate per-epoch validation time. Checkpoint
    # selection remains driven by validation macro foreground Dice alone.
    if not args.skip_final_eval:
        best_checkpoint = checkpoint_dir / f"{tag}_best.pt"
        eval_loader = None if overfit else val_loader
        if overfit:
            LOGGER.info("Overfit mode: no validation split, skipping final eval.")
        elif not best_checkpoint.is_file():
            LOGGER.warning("No best checkpoint at %s; skipping final eval.",
                           best_checkpoint)
        else:
            eval_loader = val_loader

        if eval_loader is not None and best_checkpoint.is_file():
            final = final_evaluation(best_checkpoint, eval_loader, device,
                                     results_dir, config)
            summary["final_evaluation"] = {
                "n_cases": final["n_cases"],
                "checkpoint_epoch": final["checkpoint_epoch"],
                "macro_foreground_dice_mean": final["macro_foreground_dice_mean"],
                "per_case_csv": str(results_dir / "val_per_case_best.csv"),
                "summary_json": str(results_dir / "val_summary_best.json"),
            }

    # ---- figures ----------------------------------------------------------- #
    if not args.skip_figures and not history_frame.empty:
        try:
            plot_training_curves(history_frame, results_dir / "training_curves.png")
            LOGGER.info("Wrote training_curves.png")
        except Exception as exc:  # noqa: BLE001 - figures must not fail a run
            LOGGER.warning("Could not render training curves: %s", exc)

        if not overfit and val_dataset is not None and best_checkpoint.is_file():
            try:
                overlay_payload = load_checkpoint(best_checkpoint)
                overlay_model = AnisotropicUNet3D(UNet3DConfig(
                    in_channels=int(overlay_payload.get("config", {}).get(
                        "in_channels", config.in_channels)),
                    num_classes=int(overlay_payload.get("config", {}).get(
                        "num_classes", config.num_classes)),
                    base_channels=int(overlay_payload.get("config", {}).get(
                        "base_channels", config.base_channels)),
                )).to(device)
                overlay_model.load_state_dict(overlay_payload["model_state"])
                case_ids = val_dataset.case_ids[: max(1, args.n_overlay_cases)]
                plot_prediction_overlays(overlay_model, val_dataset, case_ids,
                                         results_dir / "val_predictions_overlay.png")
                LOGGER.info("Wrote val_predictions_overlay.png (%d cases)", len(case_ids))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Could not render prediction overlays: %s", exc)

    summary_path = results_dir / (
        f"training_summary_{tag}.json" if (overfit or not args.run_id)
        else "training_summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 70)
    print(f"Run finished: {tag}  ({summary['mode']})")
    print("=" * 70)
    if halted:
        print(f"*** HALTED: {halted}")
    print(f"epochs run      : {summary['epochs_run']} / {config.max_epochs}"
          f"   steps: {total_steps}")
    print(f"stopped early   : {summary['stopped_early']}")
    print(f"initial loss    : {summary['initial_train_loss']:.4f}"
          if summary["initial_train_loss"] is not None else "initial loss    : n/a")
    print(f"final loss      : {summary['final_train_loss']:.4f}"
          if summary["final_train_loss"] is not None else "final loss      : n/a")
    print(f"best metric     : {best_metric:.4f} at epoch {best_epoch}")
    if history:
        last = history[-1]
        for name in cs.CLASS_NAMES.values():
            key = f"train_dice_{name}" if overfit else f"val_Dice_{name}"
            if key in last and not math.isnan(last[key]):
                print(f"  {name} Dice      : {last[key]:.4f}")
        macro_key = "train_macro_dice" if overfit else "val_macro_Dice"
        if macro_key in last and not math.isnan(last[macro_key]):
            print(f"  macro Dice    : {last[macro_key]:.4f}")
    if device.type == "cuda":
        print(f"peak allocated  : {summary['max_memory_allocated_gb']:.3f} GB")
        print(f"peak reserved   : {summary['max_memory_reserved_gb']:.3f} GB")
    print(f"wall time       : {summary['wall_seconds'] / 60:.1f} min")
    print(f"history         : {history_path}")
    print(f"summary         : {summary_path}")
    print("=" * 70)
    return 3 if halted else 0


if __name__ == "__main__":
    raise SystemExit(main())

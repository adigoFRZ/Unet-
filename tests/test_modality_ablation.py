"""Unit tests for Experiment D -- the modality ablation.

Two failures matter here, and only one of them is loud.

The loud one is a shape mismatch. The quiet one is a **channel-order swap**: if
the T1-only run were accidentally fed QSM, every shape would still be right, the
loss would still go down, and the run would produce a perfectly plausible number
that answers a different question than the one asked. There is no exception and
no warning for that -- so the tests below pin not just how many channels come out
but *which modality is in which slot*, by comparing against the raw cache.

The second theme is that the ablation must not disturb Baseline v1: the default
modality selection has to return bit-identically what it returned before, and the
network has to stay structurally identical apart from the width of its stem.

Run with:  python -m pytest tests/test_modality_ablation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from _local_fixtures import legacy_training_cache_usable  # noqa: E402

#: The legacy 160/40 experiment split and the cache built over it are local
#: artifacts -- neither is published. `cache/baseline_v1` existing is not enough:
#: after `reproduce.py prepare` it exists but holds the subject-clean cohort
#: (161/38), so these tests must skip rather than run against the wrong cases.
LEGACY_FIXTURE_READY = legacy_training_cache_usable()
LEGACY_FIXTURE_REASON = "legacy experiment split and its cache are not both present"

from data import crop_spec as cs  # noqa: E402

torch = pytest.importorskip("torch", reason="PyTorch is required for these tests")

from data.segmentation_dataset import SegmentationDataset  # noqa: E402
from models.anisotropic_unet3d import (  # noqa: E402
    AnisotropicUNet3D,
    UNet3DConfig,
    count_parameters,
)
from training.train_baseline import (  # noqa: E402
    BaselineConfig,
    load_config,
    resolve_modalities,
)

CACHE_DIR = PROJECT_ROOT / "cache" / "baseline_v1"
#: Experiment D configs and the baseline they are compared against. Resolved
#: inside the subject-clean set, which is the configuration generation the
#: reported results come from.
REPORTED_CONFIGS = PROJECT_ROOT / "configs" / "subject_clean_v1"
BASELINE_CONFIG = REPORTED_CONFIGS / "baseline_v1.yaml"
CONFIG_DIR = REPORTED_CONFIGS / "modality_ablation"

#: The six ablations, and the three-modality baseline for contrast.
COMBOS: dict[str, list[str]] = {
    "t1_only": ["T1"],
    "qsm_only": ["QSM"],
    "nm_only": ["NM"],
    "t1_qsm": ["T1", "QSM"],
    "t1_nm": ["T1", "NM"],
    "qsm_nm": ["QSM", "NM"],
}

needs_cache = pytest.mark.skipif(not LEGACY_FIXTURE_READY,
                                  reason=LEGACY_FIXTURE_REASON)


def _dataset(modalities=None, split: str = "val") -> SegmentationDataset:
    return SegmentationDataset(
        root=PROJECT_ROOT, split=split,
        cache_dir=CACHE_DIR, manifest_dir=PROJECT_ROOT / "manifests" / "experiment",
        modalities=modalities,
    )


# --------------------------------------------------------------------------- #
# 1-6. shapes per combination
# --------------------------------------------------------------------------- #


@needs_cache
@pytest.mark.parametrize("name,modalities", sorted(COMBOS.items()))
def test_ablation_returns_only_the_selected_modalities(name: str, modalities: list[str]) -> None:
    """Each combination yields exactly len(modalities) channels, in DHW crop shape."""
    sample = _dataset(modalities)[0]
    assert sample["image"].shape == (len(modalities), *cs.CROP_SHAPE_DHW), name
    assert sample["image"].dtype == torch.float32
    assert sample["label"].shape == cs.CROP_SHAPE_DHW


@needs_cache
def test_single_modality_combinations_are_one_channel() -> None:
    for name in ("t1_only", "qsm_only", "nm_only"):
        sample = _dataset(COMBOS[name])[0]
        assert sample["image"].shape == (1, 32, 96, 96), name


@needs_cache
def test_pair_combinations_are_two_channels() -> None:
    for name in ("t1_qsm", "t1_nm", "qsm_nm"):
        sample = _dataset(COMBOS[name])[0]
        assert sample["image"].shape == (2, 32, 96, 96), name


# --------------------------------------------------------------------------- #
# 7. channel identity and order -- the silent-swap guard
# --------------------------------------------------------------------------- #


@needs_cache
def test_default_selection_is_the_frozen_order_and_bit_identical() -> None:
    """modalities=None must reproduce the pre-ablation output exactly.

    Compared against the raw cache file rather than against another dataset
    object, so a shared bug in the indexing path cannot make the test agree with
    itself.
    """
    dataset = _dataset(None)
    assert dataset.modalities == tuple(cs.CHANNEL_ORDER) == ("T1", "QSM", "NM")
    assert dataset.channel_indices == (0, 1, 2)

    sample = dataset[0]
    raw = np.load(CACHE_DIR / "images" / f"{sample['case_id']}.npy")
    assert sample["image"].shape == (3, 32, 96, 96)
    assert torch.equal(sample["image"], torch.from_numpy(
        np.ascontiguousarray(raw)).float()), (
        "the default three-modality path no longer returns the cached tensor "
        "unchanged -- Baseline v1 output has been altered"
    )


@needs_cache
@pytest.mark.parametrize("name,modalities", sorted(COMBOS.items()))
def test_each_channel_is_the_modality_it_claims_to_be(name: str, modalities: list[str]) -> None:
    """Channel ``c`` must hold ``modalities[c]``, compared against the raw cache.

    This is the test that catches a silent T1/QSM swap. Every other test here
    would still pass if the dataset returned the wrong modality, because the
    shapes are identical either way.
    """
    dataset = _dataset(modalities)
    sample = dataset[0]
    raw = np.load(CACHE_DIR / "images" / f"{sample['case_id']}.npy")

    for column, modality in enumerate(modalities):
        expected = raw[cs.CHANNEL_ORDER.index(modality)]
        assert np.array_equal(sample["image"][column].numpy(), expected), (
            f"{name}: channel {column} is not {modality}"
        )

    # And the selected channels must genuinely DIFFER from one another, or the
    # check above could pass by accident on degenerate data.
    if len(modalities) > 1:
        assert not torch.equal(sample["image"][0], sample["image"][1])


@needs_cache
def test_three_modality_order_is_preserved_by_an_explicit_selection() -> None:
    """Passing all three explicitly must equal passing nothing (frozen order)."""
    implicit = _dataset(None)[0]["image"]
    explicit = _dataset(list(cs.CHANNEL_ORDER))[0]["image"]
    assert torch.equal(implicit, explicit)
    assert _dataset(["NM", "QSM", "T1"]).channel_indices == (2, 1, 0)


# --------------------------------------------------------------------------- #
# 8-9. labels and image values must not change with the modality selection
# --------------------------------------------------------------------------- #


@needs_cache
def test_labels_are_identical_across_every_combination() -> None:
    """The modality selection must not touch the target at all."""
    reference = _dataset(None)[0]["label"]
    for name, modalities in COMBOS.items():
        sample = _dataset(modalities)[0]
        assert torch.equal(sample["label"], reference), f"{name} changed the label"
        assert sample["case_id"] == _dataset(None)[0]["case_id"]


@needs_cache
def test_image_values_are_the_cached_values_not_renormalised() -> None:
    """No rescaling, no renormalisation: the ablation only selects channels.

    Normalisation belongs to the preprocessing stage; if the dataset were to
    rescale per-modality at load time, a single-modality run would be trained on
    differently-scaled inputs than the three-modality baseline and the comparison
    would be confounded.
    """
    sample = _dataset(["QSM"])[0]
    raw = np.load(CACHE_DIR / "images" / f"{sample['case_id']}.npy")[1]
    assert np.array_equal(sample["image"][0].numpy(), raw)


@needs_cache
def test_case_order_is_identical_across_combinations() -> None:
    ids = [tuple(_dataset(m).case_ids) for m in COMBOS.values()]
    assert all(x == ids[0] for x in ids)
    assert ids[0] == tuple(_dataset(None).case_ids)


# --------------------------------------------------------------------------- #
# 10. model output shape and architecture
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_model_output_shape_for_every_input_width(n_channels: int) -> None:
    model = AnisotropicUNet3D(UNet3DConfig(
        in_channels=n_channels, num_classes=4, base_channels=16))
    with torch.no_grad():
        output = model(torch.randn(2, n_channels, 32, 96, 96))
    assert output.shape == (2, 4, 32, 96, 96)


def test_only_the_stem_width_changes_with_in_channels() -> None:
    """Everything except the first conv must be structurally identical.

    Confirms by construction that the ablation varies the input width and nothing
    else -- if a deeper layer shifted, the comparison between combinations would
    no longer isolate the modality effect.
    """
    reference = AnisotropicUNet3D(UNet3DConfig(in_channels=3, base_channels=16))
    ref_state = reference.state_dict()

    for n_channels in (1, 2):
        model = AnisotropicUNet3D(UNet3DConfig(
            in_channels=n_channels, base_channels=16))
        state = model.state_dict()
        assert set(state) == set(ref_state), "parameter set changed"

        for key, tensor in ref_state.items():
            if key == "encoder_blocks.0.block.0.weight":
                # stem: (out, in, kd, kh, kw) -- only the input dim may differ
                assert state[key].shape[0] == tensor.shape[0]
                assert state[key].shape[2:] == tensor.shape[2:]
                assert state[key].shape[1] == n_channels
            else:
                assert state[key].shape == tensor.shape, f"{key} changed shape"

        # The parameter count difference is exactly the stem's input width.
        delta = count_parameters(reference)["total"] - count_parameters(model)["total"]
        kernel_volume = int(np.prod(ref_state["encoder_blocks.0.block.0.weight"].shape[2:]))
        assert delta == (3 - n_channels) * 16 * kernel_volume


# --------------------------------------------------------------------------- #
# configuration guards
# --------------------------------------------------------------------------- #


def test_all_six_configs_differ_from_baseline_only_in_the_declared_keys() -> None:
    """The modality selection must be the single algorithmic difference."""
    from dataclasses import asdict

    baseline = asdict(load_config(BASELINE_CONFIG))
    for name in sorted(COMBOS):
        experiment = asdict(load_config(CONFIG_DIR / f"{name}.yaml"))
        differing = {k for k in set(baseline) | set(experiment)
                     if baseline.get(k) != experiment.get(k)}
        assert differing == {
            "modalities",        # the experimental variable
            "pin_memory",        # environment workaround, not an experimental
                                 # variable (same wording as Experiments A/B/C)
            "checkpoint_dir",    # output routing
            "results_dir",
        }, f"{name}: unexpected config differences: {sorted(differing)}"

        for key, value in (("seed", 42), ("batch_size", 2), ("max_epochs", 200),
                           ("early_stopping_patience", 40), ("ce_weight", 1.0),
                           ("dice_weight", 1.0), ("base_channels", 16),
                           ("num_classes", 4)):
            assert baseline[key] == experiment[key] == value, f"{name}:{key}"
        assert experiment["learning_rate"] == 3.0e-4
        assert experiment["weight_decay"] == 1.0e-5
        assert experiment["augmentation_enabled"] is False
        assert experiment["loss_type"] == "dice_ce"
        assert experiment["boundary_cache_dir"] is None
        # in_channels must NOT be declared: it is derived from `modalities`.
        assert experiment["in_channels"] == baseline["in_channels"] == 3, (
            f"{name}: in_channels should be left at the default and derived at "
            f"run time, not written into the config"
        )


def test_each_config_selects_its_own_modalities() -> None:
    for name, expected in COMBOS.items():
        config = load_config(CONFIG_DIR / f"{name}.yaml")
        assert list(resolve_modalities(config)) == expected, name


def test_resolve_modalities_defaults_to_all_three() -> None:
    assert resolve_modalities(BaselineConfig()) == ("T1", "QSM", "NM")
    assert len(resolve_modalities(BaselineConfig())) == cs.IN_CHANNELS


def test_resolve_modalities_rejects_bad_selections() -> None:
    with pytest.raises(ValueError, match="unknown modality"):
        resolve_modalities(BaselineConfig(modalities=["T1", "DWI"]))
    with pytest.raises(ValueError, match="at least one"):
        resolve_modalities(BaselineConfig(modalities=[]))
    with pytest.raises(ValueError, match="duplicate"):
        resolve_modalities(BaselineConfig(modalities=["T1", "T1"]))


@needs_cache
def test_dataset_rejects_bad_modality_selections() -> None:
    with pytest.raises(ValueError, match="unknown modality"):
        _dataset(["T1", "SWI"])
    with pytest.raises(ValueError, match="at least one"):
        _dataset([])
    with pytest.raises(ValueError, match="duplicate"):
        _dataset(["QSM", "QSM"])


@needs_cache
def test_in_channels_derivation_matches_the_dataset() -> None:
    """The trainer's derived width must equal what the dataset actually emits."""
    for name, modalities in COMBOS.items():
        config = load_config(CONFIG_DIR / f"{name}.yaml")
        derived = len(resolve_modalities(config))
        sample = _dataset(list(modalities))[0]
        assert sample["image"].shape[0] == derived, name

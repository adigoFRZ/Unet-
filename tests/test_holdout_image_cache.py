"""Tests for the image-only holdout inference cache.

Two things are being protected here.

**The holdout cache must be byte-identical in meaning to the training cache.**
The frozen checkpoints were trained on `(3, 32, 96, 96)` crop tensors built by
`build_baseline_cache.py`. If the holdout tensors are produced by even a slightly
different pipeline -- a shifted crop origin, a flipped axis, a different dtype --
every holdout number is garbage in a way that looks completely normal. So the
test re-runs the holdout pipeline over frozen train/val cases and requires
`np.array_equal`, plus explicit checks of crop bounds, axis mapping and dtype.

**The holdout pipeline must be structurally incapable of reading GT.** The
freeze discipline is worth nothing if the evaluation script quietly reads the
answer key. So the strongest available check is applied: `nib.load` is
monkeypatched to raise if it is ever handed a path containing "label" or "mask",
and the pipeline is then run for real. A future edit that starts reading GT fails
the test rather than silently contaminating the result.

Run with:  python -m pytest tests/test_holdout_image_cache.py -v
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data import crop_spec as cs  # noqa: E402

nib = pytest.importorskip("nibabel", reason="nibabel is required")

import build_holdout_image_cache as holdout  # noqa: E402

FROZEN_CACHE = PROJECT_ROOT / "cache" / "baseline_v1"
HOLDOUT_CACHE = PROJECT_ROOT / "cache" / "holdout_frozen_v1"
MANIFEST_DIR = PROJECT_ROOT / "manifests" / "experiment"
IMAGE_DIR = PROJECT_ROOT / "processed" / "images"
FREEZE_RECORD = (PROJECT_ROOT / "results" / "experiments" / "baseline_deep_ensemble"
                 / "FREEZE_RECORD.json")

needs_frozen = pytest.mark.skipif(not FROZEN_CACHE.is_dir(), reason="frozen cache absent")
needs_holdout = pytest.mark.skipif(not HOLDOUT_CACHE.is_dir(),
                                   reason="holdout cache not built")
# The split manifests name real participant cases and are excluded from the
# public repository, so anything that reads them directly has to skip when they
# are not present rather than fail.
needs_manifest = pytest.mark.skipif(not (MANIFEST_DIR / "internal_test.csv").is_file(),
                                    reason="split manifests absent")


def _case_ids(split: str) -> list[str]:
    import pandas as pd

    return sorted(str(c) for c in pd.read_csv(MANIFEST_DIR / f"{split}.csv")["case_id"])


# --------------------------------------------------------------------------- #
# 1-3. dataset identity and split separation
# --------------------------------------------------------------------------- #

@needs_manifest
def test_internal_test_manifest_has_100_unique_cases() -> None:
    """1 + 2."""
    ids = _case_ids("internal_test")
    assert len(ids) == 100
    assert len(set(ids)) == 100


@needs_holdout
def test_holdout_cache_covers_exactly_the_manifest_cases() -> None:
    """The cache must be exactly the manifest's 100 cases -- no more, no fewer."""
    cached = sorted(p.stem for p in (HOLDOUT_CACHE / "images").glob("*.npy"))
    assert cached == _case_ids("internal_test")


@needs_holdout
def test_holdout_cache_does_not_overlap_any_other_split() -> None:
    """3. A stray train case here would turn a holdout into a resubstitution."""
    cached = {p.stem for p in (HOLDOUT_CACHE / "images").glob("*.npy")}
    for other in ("train", "val", "challenge_test"):
        overlap = cached & set(_case_ids(other))
        assert not overlap, f"holdout cache overlaps {other}: {sorted(overlap)[:5]}"
    assert len(cached) == 100


# --------------------------------------------------------------------------- #
# 4-9. tensor properties, crop and axis mapping
# --------------------------------------------------------------------------- #

@needs_holdout
def test_holdout_tensors_have_the_frozen_shape_dtype_and_channels() -> None:
    """4 + 5. Modality order is checked by rebuilding each case's channel 0..2
    from the per-modality processed volumes and comparing -- a permuted channel
    order has the right shape and dtype and would otherwise pass."""
    frame_ids = _case_ids("internal_test")[:5]
    for case_id in frame_ids:
        tensor = np.load(HOLDOUT_CACHE / "images" / f"{case_id}.npy")
        assert tensor.shape == (3, 32, 96, 96), case_id
        assert tensor.dtype == np.float32, case_id
        for column, modality in enumerate(cs.CHANNEL_ORDER):
            volume = np.asanyarray(nib.load(
                str(IMAGE_DIR / f"{case_id}_{modality}.nii.gz")).dataobj)
            expected = cs.to_tensor_layout(volume).astype(np.float32)
            assert np.array_equal(tensor[column], expected), f"{case_id}:{modality}"


@needs_holdout
def test_holdout_tensors_are_finite() -> None:
    """6."""
    for path in (HOLDOUT_CACHE / "images").glob("*.npy"):
        assert np.isfinite(np.load(path)).all(), path.name


def test_channel_order_is_the_frozen_one() -> None:
    """7."""
    assert tuple(cs.CHANNEL_ORDER) == ("T1", "QSM", "NM")


def test_fixed_crop_and_axis_mapping_are_the_frozen_ones() -> None:
    """8 + 9. Crop is half-open and the transform is a permutation, not a flip."""
    assert cs.CROP_X == (103, 199) and cs.CROP_Y == (103, 199) and cs.CROP_Z == (8, 40)
    assert cs.CROP_SHAPE_XYZ == (96, 96, 32)
    assert cs.CROP_SHAPE_DHW == (32, 96, 96)

    probe = np.zeros(cs.RAW_SHAPE_XYZ, dtype=np.float32)
    probe[103, 104, 9] = 1.0                     # a single voxel inside the crop
    out = cs.to_tensor_layout(probe)
    # crop index (0, 1, 1) in XYZ -> DHW (D,H,W) = (Z,Y,X) = (1, 1, 0)
    assert out.shape == cs.CROP_SHAPE_DHW
    assert out[1, 1, 0] == 1.0
    assert out.sum() == 1.0, "the transform must not duplicate or drop voxels"


# --------------------------------------------------------------------------- #
# 10-11. equivalence with the frozen cache
# --------------------------------------------------------------------------- #

@needs_frozen
@pytest.mark.parametrize("split", ["train", "val"])
def test_holdout_pipeline_reproduces_the_frozen_cache_bitwise(split: str) -> None:
    """10 + 11. Same pipeline, same bytes.

    Compared with ``np.array_equal``, not a tolerance: a float tolerance would
    hide a wrong crop origin or a reinterpolated volume.
    """
    for case_id in _case_ids(split)[:5]:
        cached = np.load(FROZEN_CACHE / "images" / f"{case_id}.npy")
        rebuilt = holdout.build_image_tensor(
            case_id, holdout.case_image_paths(IMAGE_DIR, case_id))
        assert rebuilt.dtype == cached.dtype
        assert rebuilt.shape == cached.shape
        assert np.array_equal(rebuilt, cached), f"{case_id}: pipeline differs ({split})"


@needs_frozen
def test_equivalence_check_function_reports_success() -> None:
    """The gate the build step depends on actually runs and passes on the cache."""
    report = holdout.verify_equivalence(FROZEN_CACHE, IMAGE_DIR, MANIFEST_DIR)
    assert report["all_equal"], report["mismatches"][:3]
    assert report["n_compared"] == 200, (
        "equivalence must be checked over the ENTIRE frozen cache, not a sample")


# --------------------------------------------------------------------------- #
# 12-13. the GT guard
# --------------------------------------------------------------------------- #

def test_image_pipeline_takes_no_label_argument() -> None:
    """12. Structural, not conventional: there is no label parameter to pass."""
    parameters = set(inspect.signature(holdout.build_image_tensor).parameters)
    assert parameters == {"case_id", "image_paths"}, parameters


@needs_holdout
def test_pipeline_cannot_open_a_label_file(monkeypatch) -> None:
    """13. The decisive guard.

    ``nib.load`` is replaced by a wrapper that raises the moment it is handed a
    path containing "label" or "mask". The pipeline is then run for real over
    holdout cases. An edit that starts reading GT fails here instead of quietly
    contaminating the evaluation.
    """
    real_load = nib.load
    opened: list[str] = []

    def guarded_load(path, *args, **kwargs):
        text = str(path).lower()
        opened.append(str(path))
        if "label" in text or "mask" in text:
            raise AssertionError(f"the holdout pipeline tried to open GT: {path}")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(holdout.nib, "load", guarded_load)

    for case_id in _case_ids("internal_test")[:3]:
        tensor = holdout.build_image_tensor(
            case_id, holdout.case_image_paths(IMAGE_DIR, case_id))
        assert tensor.shape == (3, 32, 96, 96)

    assert opened, "the guard never fired, so it was not actually exercised"
    assert all("label" not in p.lower() and "mask" not in p.lower() for p in opened)


def test_builder_script_never_references_a_label_source() -> None:
    """13, second angle: no label path is constructed anywhere in the script.

    Complements the runtime guard -- together they cover both "it does not read
    GT today" and "the code contains no way to".
    """
    source = (PROJECT_ROOT / "scripts" / "build_holdout_image_cache.py").read_text(
        encoding="utf-8")
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))
    for forbidden in ("label_path", "_label.nii.gz", "labels/", "QSM_mask"):
        # the docstring discusses labels; only executable references matter
        assert f"= {forbidden}" not in code, forbidden
        assert f"/ {forbidden}" not in code, forbidden


@needs_holdout
def test_holdout_cache_stores_no_labels_and_no_gt_derived_metadata() -> None:
    """12. Nothing in the cache directory can support model selection."""
    stray = [p.name for p in HOLDOUT_CACHE.rglob("*") if "label" in p.name.lower()]
    assert not stray, f"label-like files in the holdout cache: {stray}"

    import json

    summary = json.loads((HOLDOUT_CACHE / "cache_summary.json").read_text(
        encoding="utf-8"))
    assert summary["GT_pixel_read"] is False
    assert summary["labels_stored"] is False
    for key in ("dice", "hd95", "precision", "recall", "voxels", "volume"):
        assert key not in json.dumps(summary).lower(), key

    import pandas as pd

    index = pd.read_csv(HOLDOUT_CACHE / "cache_index.csv")
    assert list(index.columns) == [
        "case_id", "image_tensor_path", "source_images", "shape", "dtype",
        "channel_order", "image_sha256"]
    assert (index["shape"] == "3x32x96x96").all()
    assert (index["dtype"] == "float32").all()
    assert (index["channel_order"] == "T1|QSM|NM").all()
    assert index["image_sha256"].notna().all()


# --------------------------------------------------------------------------- #
# 14. the freeze must not have moved
# --------------------------------------------------------------------------- #

def test_frozen_artifacts_are_unchanged() -> None:
    """14. The holdout cache work must not have disturbed anything frozen."""
    import json

    if not FREEZE_RECORD.is_file():
        pytest.skip("freeze record absent")
    record = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))

    def sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for artifact in record["frozen_artifacts"]:
        path = Path(artifact["abs_path"])
        if not path.is_file():
            pytest.skip(f"{artifact['rel_path']} unavailable")
        assert sha(path) == artifact["sha256"], artifact["rel_path"]


def test_frozen_cache_builder_was_not_modified() -> None:
    """The split restriction that forced this separate script must still stand."""
    source = (PROJECT_ROOT / "scripts" / "build_baseline_cache.py").read_text(
        encoding="utf-8")
    assert 'CACHEABLE_SPLITS: tuple[str, ...] = ("train", "val")' in source, (
        "build_baseline_cache.py was unfrozen -- the holdout split must not have "
        "been added to the training cache builder")

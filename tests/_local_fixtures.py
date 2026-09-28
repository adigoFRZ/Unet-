"""Which local artifacts this checkout actually has.

The published repository ships exactly one split: the frozen subject-clean
manifests in ``manifests/subject_clean_v1/``. Everything else a data-dependent
test might want is a local artifact that is deliberately **not** published:

* the legacy 160/40 experiment split (``manifests/experiment/``),
* a training cache built over that split,
* the frozen holdout image cache and the historical checkpoints.

Why this module exists: "``cache/baseline_v1`` exists" is not the same question
as "the legacy fixture is available". After ``python scripts/reproduce.py
prepare`` the directory *does* exist -- but it holds the subject-clean cohort
(161 train / 38 val), whose case set differs from the legacy split's (160/40).
Tests written against the legacy cohort must skip in that state, not fail with a
``FileNotFoundError`` for a case the subject-clean regrouping excluded, or
for ``manifests/experiment/val.csv``.

A gate here answers the question the test is really asking: is the fixture this
test needs on disk, and does the cache actually cover the cases it names?
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Training cache the legacy tests were written against.
BASELINE_CACHE = PROJECT_ROOT / "cache" / "baseline_v1"
#: The legacy (pre-subject-clean) split. Local-only: never published.
LEGACY_MANIFEST_DIR = PROJECT_ROOT / "manifests" / "experiment"
#: Image-only cache for the frozen holdout evaluation. Local-only.
HOLDOUT_IMAGE_CACHE = PROJECT_ROOT / "cache" / "holdout_frozen_v1"


def manifest_case_ids(manifest: Path) -> list[str]:
    """Case ids from a manifest, or ``[]`` when it is missing or unreadable."""
    if not manifest.is_file():
        return []
    try:
        import pandas as pd

        return [str(c) for c in pd.read_csv(manifest, encoding="utf-8-sig")["case_id"]]
    except Exception:  # pragma: no cover - a broken manifest is "not available"
        return []


def cache_covers(cache_dir: Path, manifest: Path) -> bool:
    """True when every case named by ``manifest`` has a cached image tensor."""
    case_ids = manifest_case_ids(manifest)
    if not case_ids or not cache_dir.is_dir():
        return False
    return all((cache_dir / "images" / f"{case_id}.npy").is_file()
               for case_id in case_ids)


def legacy_split_present(manifest_dir: Path | None = None) -> bool:
    """The legacy 160/40 split is on disk (it never is in a fresh clone)."""
    directory = manifest_dir or LEGACY_MANIFEST_DIR
    return bool(manifest_case_ids(directory / "train.csv")
                and manifest_case_ids(directory / "val.csv"))


def legacy_training_cache_usable(manifest_dir: Path | None = None,
                                 cache_dir: Path | None = None) -> bool:
    """The legacy split is here *and* the cache was built over exactly its cases.

    Both halves matter. With the split present but the cache built over the
    subject-clean cohort, the legacy tests would run against the wrong case set --
    which is precisely the failure this gate prevents.
    """
    directory = manifest_dir or LEGACY_MANIFEST_DIR
    cache = cache_dir or BASELINE_CACHE
    if not legacy_split_present(directory):
        return False
    return (cache_covers(cache, directory / "train.csv")
            and cache_covers(cache, directory / "val.csv"))

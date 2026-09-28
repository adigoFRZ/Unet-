"""Tests for the deep-ensemble driver's public surface.

The ensemble recipe itself -- softmax over the class axis, arithmetic mean of the
three members' probabilities, one argmax at the end -- is covered in
``tests/test_frozen_holdout_evaluator.py``. What is checked here is the part that
decides *which cases get looked at*, and the release constraint that no
participant case id is baked into the script.

Run with:  ./.venv/Scripts/python.exe -m pytest tests/test_deep_ensemble.py -v
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

pytest.importorskip("torch", reason="PyTorch is required")

import evaluate_deep_ensemble as ev  # noqa: E402


# --------------------------------------------------------------------------- #
# the frozen member registry
# --------------------------------------------------------------------------- #

def test_member_order_is_seed42_seed123_seed2026() -> None:
    """Member order is part of the frozen ensemble: the mean is order-independent,
    but every stored per-member label and figure is not."""
    assert ev.SEEDS == ("seed42", "seed123", "seed2026")
    assert set(ev.RUNS) == set(ev.SEEDS)


# --------------------------------------------------------------------------- #
# case selection is an argument, not a constant
# --------------------------------------------------------------------------- #

def test_no_case_id_is_hard_coded_in_the_script() -> None:
    """Participant case ids must never be literals in published source.

    They are pseudonymised, which is not the same as cleared for redistribution,
    and a hard-coded cohort also freezes an analysis choice into the code.
    """
    source = (PROJECT_ROOT / "scripts" / "evaluate_deep_ensemble.py").read_text(
        encoding="utf-8")
    assert not re.search(r"\bRJPD_\d+\b", source), "a real case id is hard-coded"
    assert "BASELINE_WORST5" not in source, "the fixed worst-case cohort is back"


def test_highlight_defaults_to_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    args = ev.parse_args([])
    assert args.highlight_case_ids == []
    assert args.highlight_case_ids_file is None
    assert ev.resolve_highlight_case_ids(args) == []


def test_highlight_case_ids_come_from_the_command_line() -> None:
    args = ev.parse_args(["--highlight-case-ids", "SYN_001", "SYN_002"])
    assert ev.resolve_highlight_case_ids(args) == ["SYN_001", "SYN_002"]


def test_highlight_case_ids_keep_order_and_drop_duplicates() -> None:
    args = ev.parse_args(["--highlight-case-ids", "SYN_002", "SYN_001", "SYN_002"])
    assert ev.resolve_highlight_case_ids(args) == ["SYN_002", "SYN_001"]


def test_highlight_case_ids_file_is_merged_and_comments_are_ignored(tmp_path) -> None:
    """A file is the practical way to pass a longer cohort."""
    listing = tmp_path / "cases.txt"
    listing.write_text("# the cases I want to look at\nSYN_001\n\nSYN_003  # trailing\n",
                       encoding="utf-8")

    args = ev.parse_args(["--highlight-case-ids", "SYN_000",
                          "--highlight-case-ids-file", str(listing)])

    assert ev.resolve_highlight_case_ids(args) == ["SYN_000", "SYN_001", "SYN_003"]


def test_highlight_file_alone_is_enough(tmp_path) -> None:
    listing = tmp_path / "cases.txt"
    listing.write_text("SYN_000\nSYN_001\n", encoding="utf-8")

    args = ev.parse_args(["--highlight-case-ids-file", str(listing)])

    assert ev.resolve_highlight_case_ids(args) == ["SYN_000", "SYN_001"]

#!/usr/bin/env python
"""Validate the PUBLIC subject-clean manifests before they go to GitHub.

Two independent jobs:

1. **Consistency** - the frozen split must still be exactly
   train 161 / val 38 / internal_test 100 / challenge_test 199 / excluded 2,
   and every pair of splits must share ZERO ``subject_group_id``.

2. **Disclosure scan** - the published CSVs must contain only public dataset
   case ids, the subject grouping derived from the approved image-based audit,
   the split label, and repo-relative file paths. No machine paths, no user
   directories, no tokens, no identifiers beyond the pseudonymised case id.

This script reads the manifests only; it never rewrites them. Exit code is 0
only when both jobs pass, so it can gate a commit.

Usage
-----
    python scripts/check_public_manifests.py
    python scripts/check_public_manifests.py --json results/public_manifest_check.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "manifests" / "subject_clean_v1"

SPLIT_FILES: dict[str, str] = {
    "train": "train.csv",
    "val": "val.csv",
    "internal_test": "internal_test.csv",
    "challenge_test": "challenge_test.csv",
    "excluded": "EXCLUDED_CASES.csv",
}

#: the frozen, non-negotiable case counts of the subject-clean split
EXPECTED_COUNTS: dict[str, int] = {
    "train": 161,
    "val": 38,
    "internal_test": 100,
    "challenge_test": 199,
    "excluded": 2,
}

#: case ids come from the public PDCADxFoundation release and look like RJPD_###
CASE_ID_RE = re.compile(r"^RJPD_\d{3}$")

#: absolute paths, home directories, and Windows drive roots
PATH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("windows_drive", re.compile(r"(?i)\b[a-z]:[\\/]")),
    ("unc_path", re.compile(r"\\\\[^\\\s]+\\")),
    ("posix_home", re.compile(r"(?<![\w.])/(?:home|Users|root|mnt|media)/")),
    ("user_profile", re.compile(r"(?i)[\\/](Users|Documents and Settings)[\\/]")),
    ("tilde_home", re.compile(r"(?<![\w])~[\\/]")),
    ("file_url", re.compile(r"(?i)\bfile://")),
]

#: credential / identifier shapes that must never appear
SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    ("dicom_uid", re.compile(r"\b\d+(?:\.\d+){3,}\b")),
    ("long_digit_id", re.compile(r"(?<!\d)\d{9,}(?!\d)")),
    ("mrn_label", re.compile(r"(?i)\b(mrn|patient_?id|hospital_?id|ssn|nhs_?number)\b")),
    ("phi_name_key", re.compile(r"(?i)\b(patient_?name|first_?name|last_?name|dob)\b")),
    ("date_like", re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")),
]

#: columns the public manifests are allowed to carry
ALLOWED_COLUMNS: dict[str, set[str]] = {
    "split": {"case_id", "split", "group", "T1_path", "QSM_path", "NM_path", "label_path"},
    "excluded": {"case_id", "original_split", "reason", "paired_case",
                 "subject_group_id", "evidence"},
    "subject_groups": {"case_id", "subject_group_id", "split"},
}


def load_manifests() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in SPLIT_FILES.items():
        path = MANIFEST_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing public manifest: {path}")
        frames[name] = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    groups_path = MANIFEST_DIR / "subject_groups.csv"
    if not groups_path.is_file():
        raise FileNotFoundError(f"missing public manifest: {groups_path}")
    frames["subject_groups"] = pd.read_csv(
        groups_path, encoding="utf-8-sig", dtype=str).fillna("")
    return frames


def check_counts(frames: dict[str, pd.DataFrame]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for name, expected in EXPECTED_COUNTS.items():
        actual = len(frames[name])
        if actual != expected:
            problems.append(f"{name}: {actual} cases, expected {expected}")

    total = sum(len(frames[n]) for n in SPLIT_FILES)
    if total != 500:
        problems.append(f"split files cover {total} cases, expected 500")

    sg = frames["subject_groups"]
    if len(sg) != total:
        problems.append(
            f"subject_groups.csv has {len(sg)} rows, expected {total}")

    # subject_groups.csv must agree with the per-split files, case for case.
    derived = {
        row["case_id"]: split
        for split, frame in frames.items() if split in SPLIT_FILES
        for row in frame.to_dict("records")
        for row in [row]  # keep the comprehension flat
    }
    published = dict(zip(sg["case_id"], sg["split"]))
    if derived != published:
        only_derived = sorted(set(derived) - set(published))[:5]
        only_published = sorted(set(published) - set(derived))[:5]
        mismatched = sorted(
            c for c in set(derived) & set(published) if derived[c] != published[c]
        )[:5]
        problems.append(
            "subject_groups.csv disagrees with the per-split files "
            f"(missing={only_derived}, extra={only_published}, "
            f"split_mismatch={mismatched})")
    return not problems, problems


#: the four splits the zero-overlap requirement applies to
OVERLAP_SPLITS: tuple[str, ...] = (
    "train", "val", "internal_test", "challenge_test")


def check_subject_overlap(
    frames: dict[str, pd.DataFrame]
) -> tuple[bool, list[str], dict[str, int]]:
    """Zero shared subject_group_id between any two of the four splits.

    ``excluded`` is deliberately NOT part of this check. An excluded case is by
    construction the partner of a case that stays in train or internal_test --
    removing that partner is what makes the four splits subject-disjoint. Its
    own linkage is validated separately by ``check_excluded_linkage``.
    """
    problems: list[str] = []
    sg = frames["subject_groups"]
    groups_of = {
        split: set(sg.loc[sg.split == split, "subject_group_id"])
        for split in OVERLAP_SPLITS
    }
    overlaps: dict[str, int] = {}
    for a, b in combinations(OVERLAP_SPLITS, 2):
        shared = groups_of[a] & groups_of[b]
        overlaps[f"{a}<->{b}"] = len(shared)
        if shared:
            problems.append(
                f"{a}<->{b}: {len(shared)} shared subject group(s) {sorted(shared)[:5]}")
    return not problems, problems, overlaps


def check_excluded_linkage(
    frames: dict[str, pd.DataFrame]
) -> tuple[bool, list[str], list[str]]:
    """The excluded cases must be exactly the documented duplicate partners.

    Every excluded case has to (a) be absent from all four splits, (b) share its
    subject group with exactly the ``paired_case`` recorded in
    EXCLUDED_CASES.csv, and (c) that partner must actually live in a split.
    """
    problems: list[str] = []
    notes: list[str] = []
    excluded = frames["excluded"]
    sg = frames["subject_groups"]
    split_of = dict(zip(sg["case_id"], sg["split"]))

    for row in excluded.to_dict("records"):
        case_id = row["case_id"]
        if split_of.get(case_id) != "excluded":
            problems.append(f"{case_id}: excluded but not marked 'excluded'")
        partner = row.get("paired_case", "")
        if not partner:
            problems.append(f"{case_id}: no paired_case recorded")
            continue
        if split_of.get(partner) not in OVERLAP_SPLITS:
            problems.append(
                f"{case_id}: paired_case {partner} is in "
                f"{split_of.get(partner)!r}, expected one of {OVERLAP_SPLITS}")
        if row.get("subject_group_id") != sg.loc[
                sg.case_id == partner, "subject_group_id"].iat[0]:
            problems.append(
                f"{case_id}: subject_group_id disagrees with paired_case {partner}")
        notes.append(f"{case_id} removed; partner {partner} stays in "
                     f"{split_of.get(partner)}")
    return not problems, problems, notes


def scan_text(label: str, text: str, patterns) -> list[str]:
    hits: list[str] = []
    for name, pattern in patterns:
        for match in pattern.finditer(text):
            snippet = match.group(0)
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            hits.append(f"{label}: {name} -> {snippet!r}")
    return hits


def check_disclosure(
    frames: dict[str, pd.DataFrame]
) -> tuple[bool, list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []

    for name, frame in frames.items():
        kind = "excluded" if name == "excluded" else (
            "subject_groups" if name == "subject_groups" else "split")
        allowed = ALLOWED_COLUMNS[kind]
        unexpected = sorted(set(frame.columns) - allowed)
        if unexpected:
            problems.append(f"{name}: unexpected column(s) {unexpected}")

        for column in frame.columns:
            for value in frame[column].tolist():
                if not isinstance(value, str) or not value:
                    continue
                where = f"{name}.csv[{column}]"
                for hit in scan_text(where, value, PATH_PATTERNS):
                    problems.append(hit)
                for hit in scan_text(where, value, SENSITIVE_PATTERNS):
                    problems.append(hit)

    # cross-check: every case id must be a public-scheme id
    for name, frame in frames.items():
        if "case_id" not in frame.columns:
            continue
        bad = sorted(c for c in frame["case_id"] if not CASE_ID_RE.match(c))
        if bad:
            problems.append(f"{name}: non-conforming case ids {bad[:5]}")

    # challenge_test must carry no labels and must not be a training source
    challenge = frames["challenge_test"]
    if "label_path" in challenge.columns:
        labelled = [c for c, p in zip(challenge.case_id, challenge.label_path) if p]
        if labelled:
            problems.append(
                f"challenge_test carries {len(labelled)} label path(s); "
                "it must stay label-less")
    else:
        warnings.append("challenge_test has no label_path column")

    # every referenced file path must be repo-relative
    for name in ("train", "val", "internal_test", "challenge_test"):
        frame = frames[name]
        for column in (c for c in frame.columns if c.endswith("_path")):
            for value in frame[column]:
                if not value:
                    continue
                p = Path(value)
                if p.is_absolute():
                    problems.append(f"{name}.csv[{column}]: absolute path {value!r}")
                elif ".." in p.parts:
                    problems.append(f"{name}.csv[{column}]: escaping path {value!r}")
    return not problems, problems, warnings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the machine-readable report here.")
    args = parser.parse_args(argv)

    frames = load_manifests()

    counts_ok, counts_problems = check_counts(frames)
    overlap_ok, overlap_problems, overlaps = check_subject_overlap(frames)
    excl_ok, excl_problems, excl_notes = check_excluded_linkage(frames)
    disc_ok, disc_problems, disc_warnings = check_disclosure(frames)

    line = "=" * 70
    print(line)
    print("PUBLIC MANIFEST CHECK - manifests/subject_clean_v1/")
    print(line)

    print("\n[1] Split sizes")
    for name, expected in EXPECTED_COUNTS.items():
        actual = len(frames[name])
        flag = "ok " if actual == expected else "BAD"
        print(f"    {flag} {name:<16} {actual:>4}  (expected {expected})")
    print(f"    total cases across splits: {sum(len(frames[n]) for n in SPLIT_FILES)}")

    print("\n[2] Pairwise subject_group overlap among the four splits (must all be 0)")
    for pair, n in sorted(overlaps.items()):
        flag = "ok " if n == 0 else "BAD"
        print(f"    {flag} {pair:<40} {n}")

    print("\n[3] Excluded-case linkage")
    for note in excl_notes:
        print(f"    ok  {note}")

    print("\n[4] Disclosure scan")
    problems = counts_problems + overlap_problems + excl_problems + disc_problems
    if problems:
        for p in problems:
            print(f"    BAD {p}")
    else:
        print("    ok  no machine paths, no absolute paths, no credentials,")
        print("        no DICOM UIDs, no email addresses, no dates")
    for w in disc_warnings:
        print(f"    note {w}")

    print("\n[5] Multi-case subject groups (must not straddle two SPLITS)")
    sg = frames["subject_groups"]
    multi = {g: sorted(set(sg.loc[sg.subject_group_id == g, "split"]))
             for g, n in Counter(sg.subject_group_id).items() if n > 1}
    straddling = {
        g: [s for s in splits if s in OVERLAP_SPLITS]
        for g, splits in multi.items()
        if len([s for s in splits if s in OVERLAP_SPLITS]) > 1
    }
    for group, splits in sorted(multi.items()):
        print(f"    ok  {group}: {splits}")
    for group, splits in sorted(straddling.items()):
        print(f"    BAD {group} straddles {splits}")

    ok = counts_ok and overlap_ok and excl_ok and disc_ok and not straddling
    print()
    print(line)
    print(f"PUBLIC_MANIFEST_CHECK = {'PASS' if ok else 'FAIL'}")
    print(line)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump({
                "result": "PASS" if ok else "FAIL",
                "counts": {n: len(frames[n]) for n in SPLIT_FILES},
                "expected_counts": EXPECTED_COUNTS,
                "subject_group_overlap": overlaps,
                "multi_case_groups": {g: sorted(map(str, s)) for g, s in multi.items()},
                "problems": problems,
                "warnings": disc_warnings,
            }, handle, indent=2, ensure_ascii=False)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""SUBJECT-CLEAN TIER 2 —— 21 个正式 training run 的顺序驱动器。

严格边界
--------
* 运行清单 = configs/subject_clean_v1/**/*.yaml 中声明的那 21 个正式 run
* registry（results/subject_clean_rerun/TIER2_RUN_REGISTRY.csv）只是运行状态，
  **首次运行会自动建立**，不是必须先存在的前提条件
* 不增删 run、不改 seed、不改任何科学超参数
* 只写 checkpoints/subject_clean_v1/ 与 results/experiments_subject_clean_v1/
* **不执行任何 internal_test / challenge_test evaluation**
* registry 采用「临时文件 → fsync → atomic rename」安全写入
* 遇到 OOM / NaN / halt / config 错误 → 标记 FAILED 并**停止整个序列**
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(".").resolve()
RERUN = ROOT / "results/subject_clean_rerun"
REG = RERUN / "TIER2_RUN_REGISTRY.csv"
LOGDIR = RERUN / "run_logs"
PROGRESS = RERUN / "TIER2_PROGRESS.json"

#: Interpreter for the child training processes.
#:
#: ``sys.executable`` is the only portable choice: it is whatever interpreter is
#: running THIS script, so it follows the active virtualenv on Windows, Linux and
#: macOS alike. A hard-coded ``.venv/Scripts/python.exe`` works on exactly one
#: platform and only when the venv sits at the repo root under that exact name.
PY = Path(sys.executable)

TRAIN = ROOT / "src/training/train_baseline.py"

# 遇到这些即 STOP（不自动调参、不跳过）
FATAL_PATTERNS = [
    ("OOM", re.compile(r"(?i)(out of memory|CUDA out of memory|OutOfMemoryError)")),
    ("NONFINITE", re.compile(r"(?i)(non-?finite|nan detected|inf detected)")),
    ("HALT", re.compile(r"(?i)training halted")),
]


def sha256_of(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(4 << 20), b""):
            h.update(c)
    return h.hexdigest()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """临时文件 → flush+fsync → atomic rename。

    注意：fsync 必须作用在**写入句柄**上。在 Windows 上对以 "rb" 打开的只读句柄
    调用 os.fsync 会抛 OSError(Errno 9, Bad file descriptor)。
    """
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
        df.to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


#: Formal experiment list is derived from the configs, never hand-maintained.
#: `configs/subject_clean_v1/` holds exactly the 21 formal runs: the baseline,
#: the four single-factor experiments (A/B/C), the six modality ablations (D),
#: the eight multi-seed runs (D2) and the three spatial-prior variants (E).
CONFIG_DIR = ROOT / "configs" / "subject_clean_v1"

#: The frozen execution order of the formal campaign.
#:
#: Membership is still DERIVED from the configs -- this list does not decide
#: which runs exist, it only fixes the sequence, because the original sequence
#: is a hand-chosen one that no sort over the config paths reproduces. Keeping
#: it exactly means a fresh reproduction walks the runs in the same order as the
#: historical campaign, which matters because the driver stops at the first
#: failure. If the derived set and this list ever disagree, the driver refuses
#: to run rather than silently changing the formal experiment list.
CANONICAL_RUN_ORDER: tuple[str, ...] = (
    "baseline_v1",
    "A_augmentation",
    "B_tversky",
    "C_boundary",
    "D_t1",
    "D_qsm",
    "D_nm",
    "D_t1_qsm",
    "D_t1_nm",
    "D_qsm_nm",
    "D2_trimodal_seed123",
    "D2_trimodal_seed2026",
    "D2_qsm_seed123",
    "D2_qsm_seed2026",
    "D2_t1_qsm_seed123",
    "D2_t1_qsm_seed2026",
    "D2_qsm_nm_seed123",
    "D2_qsm_nm_seed2026",
    "E_e1_coords",
    "E_e2_occupancy",
    "E_e3_both",
)

REGISTRY_COLUMNS: tuple[str, ...] = (
    "run_id", "experiment_family", "seed", "original_config", "clean_config",
    "original_checkpoint", "clean_checkpoint_expected", "train_manifest",
    "val_manifest", "status", "start_time", "end_time", "elapsed_seconds",
    "best_epoch", "best_val_macro_Dice", "checkpoint_path", "checkpoint_sha256",
    "attempt_history", "epochs_run", "halted", "amp_skipped_steps_total",
    "nonfinite_gradient_steps_total",
)


def build_registry() -> pd.DataFrame:
    """Derive the formal run list from the configs.

    A run's identity is the last path component of its ``results_dir``, which
    must equal the last component of its ``checkpoint_dir`` -- the same
    convention ``train_baseline.py`` enforces. Nothing here is a scientific
    parameter: the config file itself carries those.
    """
    configs = sorted(CONFIG_DIR.rglob("*.yaml"))
    if not configs:
        raise SystemExit(f"error: no configs found under {CONFIG_DIR}")

    rows: list[dict[str, object]] = []
    for path in configs:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        results_dir = raw.get("results_dir")
        checkpoint_dir = raw.get("checkpoint_dir")
        if not results_dir or not checkpoint_dir:
            raise SystemExit(
                f"error: {path.relative_to(ROOT)} does not declare both "
                f"results_dir and checkpoint_dir."
            )
        run_id = Path(str(results_dir)).name
        if Path(str(checkpoint_dir)).name != run_id:
            raise SystemExit(
                f"error: {path.relative_to(ROOT)} names results_dir "
                f"{results_dir!r} but checkpoint_dir {checkpoint_dir!r}; the two "
                f"must share a run name."
            )
        rows.append({
            "run_id": run_id,
            "experiment_family": run_id.split("_", 1)[0],
            "seed": raw.get("seed"),
            "clean_config": path.relative_to(ROOT).as_posix(),
        })

    frame = pd.DataFrame(rows)
    duplicated = sorted(frame.run_id[frame.run_id.duplicated()].tolist())
    if duplicated:
        raise SystemExit(f"error: duplicate run_id in the configs: {duplicated}")

    derived = set(frame.run_id)
    canonical = set(CANONICAL_RUN_ORDER)
    if derived != canonical:
        extra = sorted(derived - canonical)
        gone = sorted(canonical - derived)
        raise SystemExit(
            "error: the configs no longer describe the frozen formal experiment "
            "list.\n"
            f"       configs present but not in CANONICAL_RUN_ORDER: {extra}\n"
            f"       CANONICAL_RUN_ORDER entries without a config:      {gone}\n"
            "       Adding or removing a formal run is a scientific decision; "
            "update the list deliberately."
        )

    order = {run_id: i for i, run_id in enumerate(CANONICAL_RUN_ORDER)}
    frame["_order"] = frame.run_id.map(order)
    frame = (frame.sort_values("_order")
                  .drop(columns="_order")
                  .reset_index(drop=True))

    for column in REGISTRY_COLUMNS:
        if column not in frame.columns:
            frame[column] = "" if column != "status" else "PLANNED_NOT_STARTED"
    return frame[list(REGISTRY_COLUMNS)]


def run_output_dirs(config_path: Path) -> tuple[Path, Path, str | None]:
    """Resolve where a run writes, from its own config.

    Returns ``(checkpoint_dir, results_dir, problem)``. ``problem`` is a message
    when the config's two directories disagree about the run name, which is the
    one way this convention can be broken.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    ckpt_rel = raw.get("checkpoint_dir")
    res_rel = raw.get("results_dir")
    if not ckpt_rel or not res_rel:
        return Path(), Path(), f"{config_path.name} lacks checkpoint_dir/results_dir"
    ckpt_dir = ROOT / str(ckpt_rel)
    res_dir = ROOT / str(res_rel)
    if ckpt_dir.name != res_dir.name:
        return ckpt_dir, res_dir, (
            f"{config_path.name} names checkpoint_dir {ckpt_rel!r} but "
            f"results_dir {res_rel!r}; they must share the run name")
    return ckpt_dir, res_dir, None


def ensure_registry(reset: bool) -> pd.DataFrame:
    """Load the registry, creating it from the configs when it is absent."""
    if REG.is_file() and not reset:
        reg = pd.read_csv(REG)
        expected = build_registry()
        if sorted(reg.run_id) != sorted(expected.run_id):
            print("*** STOP: registry 与 configs 不一致")
            print("    registry 多出:", sorted(set(reg.run_id) - set(expected.run_id)))
            print("    registry 缺少:", sorted(set(expected.run_id) - set(reg.run_id)))
            raise SystemExit(2)
        return reg

    reg = build_registry()
    REG.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(reg, REG)
    try:
        shown = REG.relative_to(ROOT).as_posix()
    except ValueError:  # registry redirected outside the repo (tests)
        shown = str(REG)
    print(f"已建立 registry: {shown}  ({len(reg)} runs)")
    return reg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subject-clean Tier 2 driver: 21 formal training runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run list and exit without training.")
    parser.add_argument("--reset-registry", action="store_true",
                        help="Recreate the registry from the configs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    reg = ensure_registry(args.reset_registry)
    assert len(reg) == 21, f"registry 行数 {len(reg)} != 21"

    if args.dry_run:
        print()
        print("=" * 88)
        print(f"DRY RUN —— {len(reg)} 个正式 run（未执行任何训练）")
        print("=" * 88)
        for i, row in reg.iterrows():
            print(f"  [{i+1:2d}/21] {row.run_id:<22} family={row.experiment_family:<9} "
                  f"seed={row.seed}  config={row.clean_config}")
        print("=" * 88)
        return 0

    t0 = time.perf_counter()
    progress = {"total": 21, "completed": 0, "failed": 0,
                "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "runs": []}

    for i, row in reg.iterrows():
        run_id = row.run_id
        cfg = ROOT / row.clean_config
        # The output directories are READ FROM THE CONFIG, never assumed here.
        # The config is the single source of truth for where a run writes; the
        # driver and the aggregator both follow it, so the three can never
        # disagree about a run's location.
        ckpt_dir, res_dir, dir_problem = run_output_dirs(cfg)
        if dir_problem:
            print(f"*** STOP: {run_id}: {dir_problem}")
            reg.loc[i, "status"] = "FAILED"
            atomic_write_csv(reg, REG)
            return 2

        # §7 输出保护：目标目录必须不存在（否则 STOP，不静默覆盖）
        if res_dir.exists() or ckpt_dir.exists():
            print(f"*** STOP: {run_id} 的目标目录已存在（{res_dir} / {ckpt_dir}）")
            reg.loc[i, "status"] = "FAILED"
            atomic_write_csv(reg, REG)
            return 2

        reg.loc[i, "status"] = "RUNNING"
        atomic_write_csv(reg, REG)

        log = LOGDIR / f"{run_id}.log"
        start = time.perf_counter()
        print(f"\n{'='*88}\n[{i+1}/21] {run_id}\n  config: {row.clean_config}\n"
              f"  start : {datetime.now():%H:%M:%S}\n{'='*88}", flush=True)

        with log.open("w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(
                [str(PY), str(TRAIN), "--config", str(cfg)],
                cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - start
        tail = log.read_text(encoding="utf-8", errors="replace")

        # 失败分类
        fatal = None
        for name, pat in FATAL_PATTERNS:
            if pat.search(tail):
                fatal = name
                break
        if proc.returncode == 0 and fatal is None:
            status = "COMPLETE"
        else:
            status = "FAILED"

        meta = {"run_id": run_id, "status": status, "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 1),
                "log": log.relative_to(ROOT).as_posix()}
        # 注意：train_baseline.py 写的是 `training_summary_<tag>.json`
        # （默认 tag="run"），不是 `training_summary.json`。用 glob 兼容两种命名。
        sums = sorted(res_dir.glob("training_summary*.json")) if res_dir.is_dir() else []
        summary_p = sums[0] if sums else res_dir / "training_summary.json"
        if summary_p.is_file():
            s = json.loads(summary_p.read_text(encoding="utf-8"))
            meta.update({"best_epoch": s.get("best_epoch"),
                         "best_val_macro_Dice": s.get("best_metric"),
                         "epochs_run": s.get("epochs_run"),
                         "halted": s.get("halted"),
                         "amp_skipped_steps_total": s.get("amp_skipped_steps_total"),
                         "nonfinite_gradient_steps_total":
                             s.get("nonfinite_gradient_steps_total")})
        ck = ckpt_dir / "run_best.pt"
        meta["checkpoint_path"] = ck.relative_to(ROOT).as_posix() if ck.is_file() else None
        meta["checkpoint_sha256"] = sha256_of(ck)

        reg.loc[i, "status"] = status
        reg.loc[i, "start_time"] = datetime.fromtimestamp(
            time.time() - elapsed).isoformat(timespec="seconds")
        reg.loc[i, "end_time"] = datetime.now().isoformat(timespec="seconds")
        reg.loc[i, "elapsed_seconds"] = round(elapsed, 1)
        reg.loc[i, "best_epoch"] = meta.get("best_epoch")
        reg.loc[i, "best_val_macro_Dice"] = meta.get("best_val_macro_Dice")
        reg.loc[i, "checkpoint_path"] = meta.get("checkpoint_path")
        reg.loc[i, "checkpoint_sha256"] = meta.get("checkpoint_sha256")
        atomic_write_csv(reg, REG)

        progress["completed"] = int((reg.status == "COMPLETE").sum())
        progress["failed"] = int((reg.status == "FAILED").sum())
        progress["runs"].append(meta)
        progress["elapsed_total_seconds"] = round(time.perf_counter() - t0, 1)
        atomic_write_json(progress, PROGRESS)

        print(f"  status: {status}  ({elapsed/60:.1f} 分钟)")
        print(f"  best_epoch={meta.get('best_epoch')}  "
              f"best_val_macro_Dice={meta.get('best_val_macro_Dice')}")

        if status != "COMPLETE":
            print(f"\n*** RUN FAILED ({run_id}) fatal={fatal} rc={proc.returncode} — "
                  f"STOP 整个序列，等待人工裁决 ***")
            return 3

    print(f"\n{'='*88}\nTIER 2 完成: {progress['completed']}/21，"
          f"总耗时 {(time.perf_counter()-t0)/3600:.2f} 小时\n{'='*88}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

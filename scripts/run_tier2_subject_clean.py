#!/usr/bin/env python
"""SUBJECT-CLEAN TIER 2 —— 21 个正式 training run 的顺序驱动器。

严格边界
--------
* 唯一运行清单 = results/subject_clean_rerun/TIER2_RUN_REGISTRY.csv
* 不增删 run、不改 seed、不改任何科学超参数
* 只写 checkpoints/subject_clean_v1/ 与 results/experiments_subject_clean_v1/
* **不执行任何 internal_test / challenge_test evaluation**
* registry 采用「临时文件 → fsync → atomic rename」安全写入
* 遇到 OOM / NaN / halt / config 错误 → 标记 FAILED 并**停止整个序列**
"""

from __future__ import annotations

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

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(".").resolve()
RERUN = ROOT / "results/subject_clean_rerun"
REG = RERUN / "TIER2_RUN_REGISTRY.csv"
LOGDIR = RERUN / "run_logs"
PROGRESS = RERUN / "TIER2_PROGRESS.json"
PY = ROOT / ".venv/Scripts/python.exe"
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


def main() -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(REG)
    assert len(reg) == 21, f"registry 行数 {len(reg)} != 21"

    t0 = time.perf_counter()
    progress = {"total": 21, "completed": 0, "failed": 0,
                "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "runs": []}

    for i, row in reg.iterrows():
        run_id = row.run_id
        cfg = ROOT / row.clean_config
        ckpt_dir = ROOT / "checkpoints/subject_clean_v1" / run_id
        res_dir = ROOT / "results/experiments_subject_clean_v1" / run_id

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

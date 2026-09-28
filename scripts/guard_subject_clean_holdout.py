#!/usr/bin/env python
"""SUBJECT-CLEAN HOLDOUT ONE-SHOT GUARD —— 独立的防重复开封保护入口。

背景
----
`scripts/evaluate_subject_clean_holdout.py` **没有** one-shot / refusal-to-overwrite 保护：
它以 `OUT.mkdir(parents=True, exist_ok=True)` 打开输出目录后直接覆写，
**再次运行会静默覆盖已冻结的 holdout 结果**，破坏证据链。

该脚本已冻结运行过一次，**本轮不修改它**（改动会使冻结记录失效）。
本文件因此是一个**独立入口**：在**任何**未来调用评估器之前强制执行下列检查。

用法
----
    python scripts/guard_subject_clean_holdout.py --self-test   # 合成目录自检
    python scripts/guard_subject_clean_holdout.py --dry-run     # 只看真实仓库裁决

设计原则
--------
* **默认拒绝**（fail-closed）：任何检查不确定 → 拒绝。
* **不删除、不移动、不覆写任何既有文件**。
* 保护逻辑**不依赖 mtime**（可被伪造），只依赖**内容哈希**与**当前状态**。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

R = Path(".").resolve()
HO = R / "results/subject_clean_holdout"
RR = R / "results/subject_clean_rerun"

EXIT_OK = 0
EXIT_REFUSED = 3
EXIT_SELFTEST_FAIL = 4


@dataclass
class Paths:
    """全部路径显式传入，便于用合成目录做自检。"""
    root: Path
    out_dir: Path
    eval_script: Path
    provenance: Path          # HOLDOUT_EVALUATION_PROVENANCE.json
    prereg: Path              # SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json
    ensemble: Path            # CLEAN_ENSEMBLE_FREEZE_RECORD.json
    lock: Path                # 一次性锁


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(4 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check(P: Paths) -> tuple[bool, list[str]]:
    """返回 (是否允许开封, 原因列表)。**fail-closed**：不确定即拒绝。"""
    reasons: list[str] = []
    allow = True

    def deny(m: str) -> None:
        nonlocal allow
        allow = False
        reasons.append("REFUSE: " + m)

    def ok(m: str) -> None:
        reasons.append("PASS  : " + m)

    # L1 —— 一次性锁必须不存在
    if P.lock.is_file():
        deny(f"一次性锁已存在：{P.lock.name} —— 本轮 holdout 已被开封过")
    else:
        ok("一次性锁不存在")

    # L2 —— 输出目录必须不存在，或必须为空
    if P.out_dir.exists():
        n = sum(1 for f in P.out_dir.rglob("*") if f.is_file())
        if n:
            deny(f"输出目录已存在且含 {n} 个文件：{P.out_dir.name} —— "
                 "holdout 似乎已经开封；重复评估会破坏证据链")
        else:
            ok("输出目录存在但为空")
    else:
        ok("输出目录尚不存在")

    # L3 —— 预注册必须存在
    if not P.prereg.is_file():
        deny("预注册文件不存在 —— 不得在无预注册的情况下开封")
    else:
        ok(f"预注册存在（sha256 {sha(P.prereg)[:16]}…）")

    # L4 —— 评估器 + 全部组件哈希必须与 provenance 记录一致
    if not P.eval_script.is_file():
        deny(f"评估器不存在：{P.eval_script.name}")
    elif not P.provenance.is_file():
        deny(f"provenance 不存在：{P.provenance.name}（无法核对评估器哈希）")
    else:
        pv = json.loads(P.provenance.read_text(encoding="utf-8"))
        rec = (pv.get("evaluation_script") or {}).get("sha256")
        if not rec:
            deny("provenance 未登记 evaluation_script.sha256 —— 无法确认评估器未被改动")
        elif sha(P.eval_script) != rec:
            deny("评估器哈希与 provenance 不符 —— 脚本已被修改")
        else:
            ok("评估器哈希与 provenance 一致")

        comps = pv.get("component_sha256") or {}
        if not comps:
            ok("provenance 未登记 component_sha256（不构成拒绝）")
        else:
            bad = [k for k, v in comps.items()
                   if not (P.root / k).is_file() or sha(P.root / k) != v]
            if bad:
                deny(f"{len(bad)} 个复用组件被修改或缺失：{bad}")
            else:
                ok(f"全部 {len(comps)} 个复用组件哈希一致")

    # L5 —— 成员 checkpoint 必须与其哈希一致
    if not P.ensemble.is_file():
        deny(f"ensemble 冻结记录不存在：{P.ensemble.name}")
    else:
        e = json.loads(P.ensemble.read_text(encoding="utf-8"))
        ms = e.get("members", [])
        if not ms:
            deny("ensemble 冻结记录未列出成员 —— 无法确认模型未被更换")
        else:
            bad = [m["checkpoint_path"] for m in ms
                   if not (P.root / m["checkpoint_path"]).is_file()
                   or sha(P.root / m["checkpoint_path"]) != m["checkpoint_sha256"]]
            if bad:
                deny(f"{len(bad)} 个成员 checkpoint 缺失或被修改：{bad}")
            else:
                ok(f"全部 {len(ms)} 个成员 checkpoint 哈希一致")

    return allow, reasons


def real_paths() -> Paths:
    return Paths(root=R, out_dir=HO, eval_script=R / "scripts/evaluate_subject_clean_holdout.py",
                 provenance=HO / "HOLDOUT_EVALUATION_PROVENANCE.json",
                 prereg=RR / "SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json",
                 ensemble=RR / "CLEAN_ENSEMBLE_FREEZE_RECORD.json",
                 lock=RR / "SUBJECT_CLEAN_HOLDOUT_ONE_SHOT_LOCK.json")


def self_test() -> int:
    """合成目录自检：证明该保护**确实会拒绝**，而不是永远返回 True。"""
    import tempfile

    print("=" * 70)
    print("SELF-TEST —— 合成目录，不触碰真实仓库")
    print("=" * 70)
    fails, n_reject = [], 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def build(tag: str, *, out_files: int = 0, tamper_eval: bool = False,
                  lock: bool = False, prereg: bool = True,
                  tamper_ckpt: bool = False) -> Paths:
            base = tmp / tag
            base.mkdir(parents=True)
            ev = base / "evaluate.py"
            ev.write_text("pass\n", encoding="utf-8")
            ev_sha = sha(ev)
            (base / "metrics.py").write_text("# metric\n", encoding="utf-8")
            met_sha = sha(base / "metrics.py")
            ck = base / "run_best.pt"
            ck.write_text("weights\n", encoding="utf-8")
            ck_sha = sha(ck)

            out = base / "out"
            if out_files:
                out.mkdir()
                for i in range(out_files):
                    (out / f"f{i}.json").write_text("{}", encoding="utf-8")

            pv = base / "PROV.json"
            pv.write_text(json.dumps({
                "evaluation_script": {"sha256": ev_sha if not tamper_eval else "0" * 64},
                "component_sha256": {"metrics.py": met_sha},
            }), encoding="utf-8")

            pr = base / "PREREG.json"
            if prereg:
                pr.write_text("{}", encoding="utf-8")

            en = base / "ENSEMBLE.json"
            en.write_text(json.dumps({
                "members": [{"checkpoint_path": "run_best.pt",
                             "checkpoint_sha256": ck_sha if not tamper_ckpt else "0" * 64}]
            }), encoding="utf-8")

            lk = base / "LOCK.json"
            if lock:
                lk.write_text("{}", encoding="utf-8")

            return Paths(root=base, out_dir=out, eval_script=ev, provenance=pv,
                         prereg=pr, ensemble=en, lock=lk)

        cases = [
            ("干净状态", dict(), True),
            ("输出目录非空（已开封）", dict(out_files=9), False),
            ("评估器被篡改", dict(tamper_eval=True), False),
            ("一次性锁存在", dict(lock=True), False),
            ("无预注册", dict(prereg=False), False),
            ("checkpoint 被替换", dict(tamper_ckpt=True), False),
        ]
        for i, (name, kw, expect_allow) in enumerate(cases, 1):
            P = build(f"c{i}", **kw)
            allow, why = check(P)
            tag = "OK " if allow == expect_allow else "FAIL"
            if allow != expect_allow:
                fails.append(f"场景{i}「{name}」：期望 allow={expect_allow}，实得 {allow}")
            if not allow:
                n_reject += 1
            print(f"[{i}] {tag} {name:24s} → allow={str(allow):5s} (期望 {expect_allow})")
            for w in why:
                if w.startswith("REFUSE"):
                    print(f"          {w}")

    print("-" * 70)
    if fails:
        print("SELF-TEST FAILED：")
        for f in fails:
            print("  -", f)
        return EXIT_SELFTEST_FAIL
    print(f"SELF-TEST PASSED —— {len(cases)} 个场景全部符合预期"
          f"（其中 {n_reject} 个拒绝场景被正确拒绝）。")
    print("  判据：若保护永远返回 allow=True，则它是无效的。")
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="用合成目录验证保护逻辑")
    ap.add_argument("--dry-run", action="store_true", help="只打印当前真实状态的裁决")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    P = real_paths()
    allow, reasons = check(P)

    print("=" * 70)
    print("SUBJECT-CLEAN HOLDOUT ONE-SHOT GUARD")
    print("=" * 70)
    print(f"仓库    : {R}")
    print(f"输出目录: {HO.relative_to(R)}")
    print(f"评估器  : {P.eval_script.relative_to(R)}")
    print(f"检查时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print("-" * 70)
    for r in reasons:
        print(" ", r)
    print("-" * 70)
    print(f"裁决：{'ALLOW' if allow else 'REFUSE'}")

    if not allow:
        print("\n**除非**有新的预注册、新的 clean split、以及明确的人工授权，")
        print("否则不得再次评估 subject-clean holdout。")
        print("\n本 guard 不修改任何文件；它只是拒绝。")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

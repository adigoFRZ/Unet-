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

冻结后代码改动（L4 的两种放行结论）
----------------------------------
被 provenance 登记的代码在冻结之后可能发生**维护性**改动。这类改动既不能假装
不存在，也不该永远堵死公开仓库，因此 L4 有两种放行结论：

* ``PASS_FROZEN_EXACT``      —— 当前文件哈希与冻结记录逐位一致。
* ``PASS_AUTHORIZED_SUCCESSOR`` —— 哈希不同，但
  ``POST_FREEZE_CODE_AMENDMENTS.json`` 中存在一条**完全匹配**的登记：
  原哈希必须与冻结记录字段逐字符一致（由 JSON pointer 程序化核对，不能凭空写）、
  当前哈希必须与磁盘文件一致、change_type 必须在允许集合内，且
  ``scientific_semantics_changed`` / ``historical_result_changed`` 必须为 false、
  ``authorized_for_future_reproduction`` 必须为 true。

除此之外**一律 DENY**。这里没有“忽略哈希”的开关：未登记的改动、伪造的当前哈希、
与冻结记录不符的原哈希、被标记为影响科学语义的改动，全部拒绝。
冻结记录中的原始 SHA256 **永不被改写**。
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
    amendments: Path          # POST_FREEZE_CODE_AMENDMENTS.json


#: L4 的两种放行结论
V_FROZEN = "PASS_FROZEN_EXACT"
V_SUCCESSOR = "PASS_AUTHORIZED_SUCCESSOR"
V_DENY = "DENY"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(4 << 20), b""):
            h.update(c)
    return h.hexdigest()


def json_pointer(doc, pointer: str):
    """RFC 6901 JSON pointer。找不到时抛 KeyError/IndexError。

    ``~1`` -> ``/``，``~0`` -> ``~``，因此含斜杠的键（如文件路径）可以寻址。
    """
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise KeyError(f"pointer 必须以 / 开头: {pointer!r}")
    node = doc
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(token)]
        else:
            node = node[token]
    return node


def load_amendments(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def classify(
    rel_path: str,
    actual_sha: str,
    frozen_sha: str,
    frozen_source: tuple[Path, str] | None,
    amendments: dict,
    root: Path,
) -> tuple[str, list[str]]:
    """判定一个被登记文件的现状。

    返回 ``(verdict, reasons)``，verdict ∈ {V_FROZEN, V_SUCCESSOR, V_DENY}。

    ``frozen_source`` 是 ``(记录文件, JSON pointer)``：登记条目声称的原哈希必须
    与该指针指向的值逐字符一致。这样原哈希无法被凭空写进来。
    """
    if actual_sha == frozen_sha:
        return V_FROZEN, [f"PASS  : {rel_path} 哈希与冻结记录逐位一致"]

    entries = [e for e in (amendments.get("amendments") or [])
               if e.get("file") == rel_path]
    if not entries:
        return V_DENY, [
            f"REFUSE: {rel_path} 哈希与冻结记录不符，且未在 "
            f"POST_FREEZE_CODE_AMENDMENTS.json 中登记 —— 未授权的修改"
        ]
    if len(entries) > 1:
        return V_DENY, [
            f"REFUSE: {rel_path} 在 amendment 表中有 {len(entries)} 条登记，"
            f"无法唯一确定 —— fail-closed"
        ]

    e = entries[0]
    problems: list[str] = []

    # 1) 当前哈希必须与磁盘一致
    if e.get("current_sha256") != actual_sha:
        problems.append(
            f"登记的 current_sha256 与磁盘文件不一致"
            f"（登记 {str(e.get('current_sha256'))[:16]}… ≠ 实际 {actual_sha[:16]}…）")

    # 2) 原哈希必须与冻结记录字段一致
    if e.get("original_frozen_sha256") != frozen_sha:
        problems.append("登记的 original_frozen_sha256 与冻结记录不符")
    src = e.get("frozen_sha256_source") or {}
    if not src.get("record") or not src.get("pointer"):
        problems.append("缺少 frozen_sha256_source（record + pointer）")
    elif frozen_source is None:
        problems.append("调用方未提供冻结记录的来源，无法核对原哈希")
    else:
        rec_path, pointer = frozen_source
        # 登记表里的 record 写作仓库相对路径；按 root 解析后再比。
        recorded = Path(src["record"])
        if not recorded.is_absolute():
            recorded = root / recorded
        if recorded.resolve() != rec_path.resolve():
            problems.append(
                f"登记的 frozen_sha256_source.record 与 provenance 实际来源不符"
                f"（{src['record']} → {recorded} ≠ {rec_path}）")
        elif pointer != src["pointer"]:
            problems.append(
                f"登记的 frozen_sha256_source.pointer 与 provenance 实际位置不符"
                f"（{src['pointer']} ≠ {pointer}）")
        else:
            try:
                on_record = json_pointer(
                    json.loads(rec_path.read_text(encoding="utf-8")), pointer)
            except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
                problems.append(f"无法从冻结记录解析 pointer {pointer!r}：{exc}")
            else:
                if on_record != frozen_sha:
                    problems.append("冻结记录中该字段与调用方给出的哈希不符")
                elif e.get("original_frozen_sha256") != on_record:
                    problems.append("登记的 original_frozen_sha256 与冻结记录字段不一致")

    # 3) change_type 必须在允许集合内
    allowed = amendments.get("allowed_change_types") or []
    if e.get("change_type") not in allowed:
        problems.append(
            f"change_type {e.get('change_type')!r} 不在允许集合 {allowed} 内")

    # 4) 语义与历史影响必须显式为 false，且必须授权用于未来复现
    if e.get("scientific_semantics_changed") is not False:
        problems.append("scientific_semantics_changed 不是显式的 false")
    if e.get("historical_result_changed") is not False:
        problems.append("historical_result_changed 不是显式的 false")
    if e.get("authorized_for_future_reproduction") is not True:
        problems.append("authorized_for_future_reproduction 不是显式的 true")

    if problems:
        return V_DENY, [f"REFUSE: {rel_path} 的 amendment 登记未通过核验：{p}"
                        for p in problems]

    return V_SUCCESSOR, [
        f"PASS  : {rel_path} 哈希与冻结记录不同，但存在已授权的维护性登记",
        f"        change_type={e.get('change_type')}  "
        f"原哈希与冻结记录字段一致，当前哈希与磁盘一致",
    ]


def check(P: Paths) -> tuple[bool, list[str], str, str]:
    """返回 (是否允许开封, 原因列表, 整体裁决, 代码溯源裁决)。

    **fail-closed**：不确定即拒绝。

    * 整体裁决 —— 是否允许**再次开封** holdout。任何一层不过就是 V_DENY。
    * 代码溯源裁决 —— 只看 L4：被 provenance 登记的文件处于冻结原样
      (V_FROZEN)、或处于已授权的维护性后继 (V_SUCCESSOR)、或有问题 (V_DENY)。

    两者分开报告，是因为「代码溯源是否可信」与「现在能不能再开封 holdout」
    是两个独立的问题：一个已经评估过的 holdout 永远不该再开封（L2 拒绝），
    但那并不意味着当前代码不可信。
    """
    reasons: list[str] = []
    allow = True
    verdict = V_FROZEN
    code_verdict_local: list[str] = [V_FROZEN]

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
        amendments = load_amendments(P.amendments)
        if not amendments:
            reasons.append(
                f"note  : amendment 表不存在或为空（{P.amendments.name}）—— "
                "哈希不符时将一律拒绝")

        def l4(rel_path: str, frozen_sha: str, frozen_source: tuple[Path, str] | None,
               label: str) -> None:
            """对一个被登记文件跑 classify，并把结论并入总裁决。"""
            nonlocal verdict
            target = P.root / rel_path
            if not target.is_file():
                deny(f"{label}缺失：{rel_path}")
                return
            got = sha(target)
            v, why = classify(rel_path, got, frozen_sha, frozen_source, amendments,
                              P.root)
            if v == V_DENY:
                # classify 的 REFUSE 行由 deny() 负责落到 reasons，避免重复打印
                for line in why:
                    if line.startswith("REFUSE: "):
                        deny(line[len("REFUSE: "):])
                    else:
                        reasons.append(line)
                code_verdict_local[0] = V_DENY
            else:
                reasons.extend(why)
                if v == V_SUCCESSOR and code_verdict_local[0] != V_DENY:
                    code_verdict_local[0] = V_SUCCESSOR

        rec = (pv.get("evaluation_script") or {}).get("sha256")
        ev_rel = (pv.get("evaluation_script") or {}).get("path") \
            or P.eval_script.relative_to(P.root).as_posix()
        if not rec:
            deny("provenance 未登记 evaluation_script.sha256 —— 无法确认评估器未被改动")
        else:
            l4(ev_rel, rec, (P.provenance, "/evaluation_script/sha256"), "评估器")

        comps = pv.get("component_sha256") or {}
        if not comps:
            ok("provenance 未登记 component_sha256（不构成拒绝）")
        else:
            for comp_path, comp_sha in comps.items():
                l4(comp_path, comp_sha,
                   (P.provenance, f"/component_sha256/{comp_path.replace('~', '~0').replace('/', '~1')}"),
                   "复用组件")

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

    if not allow:
        verdict = V_DENY
    return allow, reasons, verdict, code_verdict_local[0]


def check_amendments_only(P: Paths) -> int:
    """只核对 amendment 链本身，不管 holdout 是否已开封。

    这是回答“当前代码相对冻结记录处于什么状态”的独立入口，与“现在能不能再
    开封 holdout”无关。
    """
    print("=" * 70)
    print("AMENDMENT CHAIN CHECK")
    print("=" * 70)

    if not P.provenance.is_file():
        print(f"REFUSE: provenance 不存在：{P.provenance}")
        print("\nAMENDMENT_CHAIN_CHECK = FAIL")
        return EXIT_REFUSED

    amendments = load_amendments(P.amendments)
    if not amendments:
        print(f"REFUSE: amendment 表不存在或为空：{P.amendments}")
        print("\nAMENDMENT_CHAIN_CHECK = FAIL")
        return EXIT_REFUSED

    pv = json.loads(P.provenance.read_text(encoding="utf-8"))
    registry: dict[str, tuple[str, tuple[Path, str]]] = {}
    script = pv.get("evaluation_script") or {}
    if script.get("sha256"):
        registry[script.get("path") or "scripts/evaluate_subject_clean_holdout.py"] = (
            script["sha256"], (P.provenance, "/evaluation_script/sha256"))
    for rel, want in (pv.get("component_sha256") or {}).items():
        escaped = rel.replace("~", "~0").replace("/", "~1")
        registry[rel] = (want, (P.provenance, f"/component_sha256/{escaped}"))

    counts = {V_FROZEN: 0, V_SUCCESSOR: 0, V_DENY: 0}
    print(f"{'文件':<52}{'状态'}")
    print("-" * 70)
    for rel, (frozen, source) in registry.items():
        target = P.root / rel
        if not target.is_file():
            counts[V_DENY] += 1
            print(f"{rel:<52}DENY  (缺失)")
            continue
        v, _ = classify(rel, sha(target), frozen, source, amendments, P.root)
        counts[v] += 1
        print(f"{rel:<52}{v}")

    # 反向检查：登记表里是否有多余/失效条目
    extra_problems: list[str] = []
    registered = {e["file"] for e in amendments["amendments"]}
    for rel in sorted(registered - set(registry)):
        extra_problems.append(f"登记了未被 provenance 引用的文件：{rel}")
    for e in amendments["amendments"]:
        path = P.root / e["file"]
        if not path.is_file():
            extra_problems.append(f"登记的文件不存在：{e['file']}")

    print("-" * 70)
    print(f"冻结原样 {counts[V_FROZEN]}  |  已授权后继 {counts[V_SUCCESSOR]}  |  "
          f"拒绝 {counts[V_DENY]}")
    for p in extra_problems:
        print(f"  note: {p}")

    ok = counts[V_DENY] == 0
    print()
    print(f"AMENDMENT_CHAIN_CHECK = {'PASS' if ok else 'FAIL'}")
    if ok:
        print(f"代码溯源裁决 = "
              f"{V_SUCCESSOR if counts[V_SUCCESSOR] else V_FROZEN}")
    return EXIT_OK if ok else EXIT_REFUSED


def real_paths() -> Paths:
    return Paths(root=R, out_dir=HO, eval_script=R / "scripts/evaluate_subject_clean_holdout.py",
                 provenance=HO / "HOLDOUT_EVALUATION_PROVENANCE.json",
                 prereg=RR / "SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json",
                 ensemble=RR / "CLEAN_ENSEMBLE_FREEZE_RECORD.json",
                 lock=RR / "SUBJECT_CLEAN_HOLDOUT_ONE_SHOT_LOCK.json",
                 amendments=HO / "POST_FREEZE_CODE_AMENDMENTS.json")


def self_test() -> int:
    """合成目录自检：证明该保护**确实会拒绝**，而不是永远返回 True。"""
    import tempfile

    print("=" * 70)
    print("SELF-TEST —— 合成目录，不触碰真实仓库")
    print("=" * 70)
    fails, n_reject, n_successor = [], 0, 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def build(tag: str, *, out_files: int = 0, tamper_eval: bool = False,
                  lock: bool = False, prereg: bool = True,
                  tamper_ckpt: bool = False, post_freeze_edit: bool = False,
                  amend_over: dict | None = None,
                  amendments_file: bool = True,
                  amendments_doc: dict | None = None) -> Paths:
            """搭一个合成 holdout 目录。

            ``post_freeze_edit`` 模拟“冻结之后脚本被改过”：先在冻结态记下哈希，
            再把文件改写。``amend_over`` 用来故意破坏某一条登记字段。
            """
            base = tmp / tag
            base.mkdir(parents=True)
            ev = base / "evaluate.py"
            ev.write_text("pass\n", encoding="utf-8")
            frozen_sha = sha(ev)          # 冻结那一刻的内容
            if post_freeze_edit:
                ev.write_text("pass\n# post-freeze maintenance edit\n",
                              encoding="utf-8")

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
                "evaluation_script": {
                    "path": "evaluate.py",
                    "sha256": frozen_sha if not tamper_eval else "0" * 64,
                },
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

            am = base / "AMEND.json"
            if amendments_file:
                if amendments_doc is not None:
                    doc = amendments_doc
                else:
                    doc = {
                        "allowed_change_types": ["MAINTENANCE_REPRODUCIBILITY",
                                                 "COSMETIC_NON_SEMANTIC"],
                        "amendments": [],
                    }
                    if post_freeze_edit:
                        entry = {
                            "file": "evaluate.py",
                            "original_frozen_sha256": frozen_sha,
                            "current_sha256": sha(ev),
                            "frozen_sha256_source": {
                                "record": "PROV.json",
                                "pointer": "/evaluation_script/sha256",
                            },
                            "change_type": "MAINTENANCE_REPRODUCIBILITY",
                            "scientific_semantics_changed": False,
                            "historical_result_changed": False,
                            "authorized_for_future_reproduction": True,
                        }
                        entry.update(amend_over or {})
                        doc["amendments"].append(entry)
                am.write_text(json.dumps(doc), encoding="utf-8")

            return Paths(root=base, out_dir=out, eval_script=ev, provenance=pv,
                         prereg=pr, ensemble=en, lock=lk, amendments=am)

        # (名称, kwargs, 期望 allow, 期望代码溯源裁决)
        cases: list[tuple[str, dict, bool, str]] = [
            # --- 冻结原样 / 非代码层拒绝（回归） -----------------------------
            ("干净状态（冻结态未改）", dict(), True, V_FROZEN),
            ("输出目录非空（已开封）", dict(out_files=9), False, V_FROZEN),
            ("一次性锁存在", dict(lock=True), False, V_FROZEN),
            ("无预注册", dict(prereg=False), False, V_FROZEN),
            ("checkpoint 被替换", dict(tamper_ckpt=True), False, V_FROZEN),

            # --- amendment 链：放行 ------------------------------------------
            ("授权 successor", dict(post_freeze_edit=True), True, V_SUCCESSOR),

            # --- amendment 链：必须仍然拒绝 ----------------------------------
            ("provenance 哈希归零",
             dict(tamper_eval=True, post_freeze_edit=True), False, V_DENY),

            ("amendment 当前哈希伪造",
             dict(post_freeze_edit=True, amend_over={"current_sha256": "f" * 64}),
             False, V_DENY),

            ("amendment 原哈希与冻结记录不符",
             dict(post_freeze_edit=True, amend_over={"original_frozen_sha256": "a" * 64}),
             False, V_DENY),

            ("标记影响科学语义",
             dict(post_freeze_edit=True, amend_over={"scientific_semantics_changed": True}),
             False, V_DENY),

            ("标记影响历史结果",
             dict(post_freeze_edit=True, amend_over={"historical_result_changed": True}),
             False, V_DENY),

            ("未授权用于未来复现",
             dict(post_freeze_edit=True,
                  amend_over={"authorized_for_future_reproduction": False}),
             False, V_DENY),

            ("change_type 非法",
             dict(post_freeze_edit=True, amend_over={"change_type": "SCIENCE_CHANGE"}),
             False, V_DENY),

            ("未登记的新修改",
             dict(post_freeze_edit=True,
                  amendments_doc={"allowed_change_types": ["MAINTENANCE_REPRODUCIBILITY"],
                                  "amendments": []}),
             False, V_DENY),

            ("amendment 表缺失",
             dict(post_freeze_edit=True, amendments_file=False), False, V_DENY),

            ("source 指向错误记录",
             dict(post_freeze_edit=True,
                  amend_over={"frozen_sha256_source": {
                      "record": "OTHER.json",
                      "pointer": "/evaluation_script/sha256"}}),
             False, V_DENY),

            ("source pointer 指向错误位置",
             dict(post_freeze_edit=True,
                  amend_over={"frozen_sha256_source": {
                      "record": "PROV.json",
                      "pointer": "/component_sha256/metrics.py"}}),
             False, V_DENY),
        ]

        for i, (name, kw, exp_allow, exp_code) in enumerate(cases, 1):
            P = build(f"c{i}", **kw)
            allow, why, overall, code = check(P)
            good = (allow == exp_allow and code == exp_code)
            tag = "OK " if good else "FAIL"
            if not good:
                fails.append(
                    f"场景{i}「{name}」：期望 allow={exp_allow}/代码={exp_code}，"
                    f"实得 allow={allow}/代码={code}")
            if not allow:
                n_reject += 1
            if code == V_SUCCESSOR:
                n_successor += 1
            print(f"[{i:2d}] {tag} {name:32s} → allow={str(allow):5s} "
                  f"代码={code}")
            if not good:
                for w in why:
                    print(f"          {w}")

    print("-" * 70)
    if fails:
        print("SELF-TEST FAILED：")
        for f in fails:
            print("  -", f)
        return EXIT_SELFTEST_FAIL
    print(f"SELF-TEST PASSED —— {len(cases)} 个场景全部符合预期"
          f"（{n_reject} 个正确拒绝，{n_successor} 个正确放行为 authorized successor）。")
    print("  判据：若保护永远返回 allow=True，或把未授权改动也放行，则它是无效的。")
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="用合成目录验证保护逻辑")
    ap.add_argument("--dry-run", action="store_true", help="只打印当前真实状态的裁决")
    ap.add_argument("--check-amendments", action="store_true",
                    help="只核对 amendment 链（冻结原样 / 已授权后继 / 拒绝）")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    if a.check_amendments:
        return check_amendments_only(real_paths())

    P = real_paths()
    allow, reasons, verdict, code_verdict = check(P)

    print("=" * 70)
    print("SUBJECT-CLEAN HOLDOUT ONE-SHOT GUARD")
    print("=" * 70)
    print(f"仓库      : {R}")
    print(f"输出目录  : {HO.relative_to(R)}")
    print(f"评估器    : {P.eval_script.relative_to(R)}")
    print(f"amendment : {P.amendments.relative_to(R)}"
          f"{'' if P.amendments.is_file() else '  （不存在）'}")
    print(f"检查时间  : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print("-" * 70)
    for r in reasons:
        print(" ", r)
    print("-" * 70)
    print(f"代码溯源裁决：{code_verdict}")
    print(f"整体裁决    ：{'ALLOW' if allow else 'REFUSE'}  ({verdict})")
    print(f"              （{V_FROZEN} / {V_SUCCESSOR} / {V_DENY}）")

    if not allow:
        print("\n**除非**有新的预注册、新的 clean split、以及明确的人工授权，")
        print("否则不得再次评估 subject-clean holdout。")
        print("\n本 guard 不修改任何文件；它只是拒绝。")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

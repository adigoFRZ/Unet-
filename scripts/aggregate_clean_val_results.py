#!/usr/bin/env python
"""TIER 2 汇总 —— clean-val-38 结果、comparator 选择、ensemble freeze、holdout 预注册。

只读 train / val。**不做任何 internal_test inference，不读 internal_test GT。**
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(".").resolve()
RERUN = ROOT / "results/subject_clean_rerun"
RES = ROOT / "results/experiments_subject_clean_v1"
CKPT = ROOT / "checkpoints/subject_clean_v1"
FAIL: list[str] = []


def sha256_of(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(4 << 20), b""):
            h.update(c)
    return h.hexdigest()


def setsha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()


def _md_table(df: pd.DataFrame) -> str:
    """手写 markdown 表格，避免依赖未安装的 tabulate。"""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def metrics(df: pd.DataFrame) -> dict:
    m = {"n": int(len(df))}
    for c in ("STN", "SN", "RN"):
        m[f"{c}_Dice"] = float(df[f"Dice_{c}"].mean())
    m["macro_Dice"] = float(np.mean([m[f"{c}_Dice"] for c in ("STN", "SN", "RN")]))
    m["STN_Precision"] = float(df["Precision_STN"].mean())
    m["STN_Recall"] = float(df["Recall_STN"].mean())
    m["STN_Pred_GT"] = float((df["PredVoxels_STN"] / df["GTVoxels_STN"]).mean())
    m["STN_HD95_mm"] = float(df["HD95_STN_mm"].mean())
    m["STN_FP"] = int(df["FP_STN"].sum())
    m["STN_FN"] = int(df["FN_STN"].sum())
    return m


def main() -> int:
    reg = pd.read_csv(RERUN / "TIER2_RUN_REGISTRY.csv")
    nc, nf = int((reg.status == "COMPLETE").sum()), int((reg.status == "FAILED").sum())
    print(f"registry: COMPLETE={nc} FAILED={nf} NOT_STARTED={int((reg.status=='PLANNED_NOT_STARTED').sum())}")
    if nc != 21 or nf != 0:
        print("*** STOP: 未达 21/21 ***")
        return 1

    rows = []
    for _, r in reg.iterrows():
        rid = r.run_id
        vp = RES / rid / "val_per_case_best.csv"
        if not vp.is_file():
            FAIL.append(f"{rid}: 缺 val_per_case_best.csv")
            continue
        df = pd.read_csv(vp)
        if len(df) != 38:
            FAIL.append(f"{rid}: val n={len(df)} != 38")
        m = metrics(df)
        s = json.loads(sorted((RES / rid).glob("training_summary*.json"))[0]
                       .read_text(encoding="utf-8"))
        if s.get("selection_metric") != "validation macro foreground Dice (STN/SN/RN mean)":
            FAIL.append(f"{rid}: selection_metric 异常")
        if s.get("halted"):
            FAIL.append(f"{rid}: halted={s.get('halted')}")
        ck = CKPT / rid / "run_best.pt"
        rows.append({
            "run_id": rid, "experiment_family": r.experiment_family, "seed": r.seed,
            "best_epoch": s.get("best_epoch"), "epochs_run": s.get("epochs_run"),
            **m,
            "amp_skipped_steps_total": s.get("amp_skipped_steps_total"),
            "nonfinite_gradient_steps_total": s.get("nonfinite_gradient_steps_total"),
            "checkpoint_path": ck.relative_to(ROOT).as_posix() if ck.is_file() else None,
            "checkpoint_sha256": sha256_of(ck),
            "clean_config": r.clean_config,
            "clean_config_sha256": sha256_of(ROOT / r.clean_config),
        })
    cv = pd.DataFrame(rows).sort_values("macro_Dice", ascending=False).reset_index(drop=True)
    cv.to_csv(RERUN / "CLEAN_VAL_RESULTS.csv", index=False, encoding="utf-8-sig")

    print()
    print("=" * 118)
    print("CLEAN-VAL-38 结果（n=38，development validation，**不是** test/holdout）")
    print("=" * 118)
    show = cv[["run_id", "experiment_family", "seed", "best_epoch", "macro_Dice",
               "STN_Dice", "SN_Dice", "RN_Dice", "STN_Precision", "STN_Pred_GT"]].copy()
    for c in show.columns[4:]:
        show[c] = show[c].round(5)
    print(show.to_string(index=False))

    # ---------------- §12 baseline seed selection ----------------
    base = cv[cv.run_id.isin(["baseline_v1", "D2_trimodal_seed123", "D2_trimodal_seed2026"])].copy()
    base["seed"] = base["seed"].astype(int)
    base = base.sort_values("macro_Dice", ascending=False).reset_index(drop=True)
    base["rank"] = base.index + 1
    base[["seed", "run_id", "macro_Dice", "STN_Dice", "rank"]].to_csv(
        RERUN / "BASELINE_SEED_SELECTION.csv", index=False, encoding="utf-8-sig")
    print()
    print("=" * 118)
    print("§12 BASELINE SEED SELECTION（仅按 clean-val-38 macro foreground Dice）")
    print("=" * 118)
    print(base[["seed", "run_id", "macro_Dice", "rank"]].round(5).to_string(index=False))
    sel = int(base.iloc[0].seed)
    print(f"\n  SELECTED_COMPARATOR: seed{sel}")

    # ---------------- §11 checkpoint selection rule ----------------
    (RERUN / "CHECKPOINT_SELECTION_RULE.md").write_text(
        "# CHECKPOINT SELECTION RULE\n\n"
        "## selection metric（与原实验逐字相同）\n\n"
        "```\nvalidation macro foreground Dice = mean(STN Dice, SN Dice, RN Dice)\n```\n\n"
        "证据：`src/training/train_baseline.py:1429`\n"
        "`\"selection_metric\": \"validation macro foreground Dice (STN/SN/RN mean)\"`\n\n"
        "实现：`:1363-1364` —— `metric > best_metric` 时置 `is_best=True`。\n\n"
        "## tie rule\n\n"
        "**严格大于**（`>`）才更新 best。因此遇到完全相同值时，**保留最早出现的那个 epoch**。\n"
        "这是原项目既有的 deterministic 规则，本轮未新发明、未修改。\n\n"
        "## 每个 run 的 best epoch 与 checkpoint SHA256\n\n"
        + _md_table(cv[["run_id", "best_epoch", "macro_Dice", "checkpoint_sha256"]]) + "\n\n"
        "## 禁止\n\n"
        "不得用 STN Dice / precision / Pred-GT / holdout 表现选择 checkpoint。\n",
        encoding="utf-8")

    # ---------------- §13 ensemble freeze ----------------
    members = ["baseline_v1", "D2_trimodal_seed123", "D2_trimodal_seed2026"]
    mem_rows = []
    for m in members:
        rr = cv[cv.run_id == m].iloc[0]
        mem_rows.append({"member_run_id": m, "seed": int(rr.seed),
                         "best_epoch": int(rr.best_epoch),
                         "checkpoint_path": rr.checkpoint_path,
                         "checkpoint_sha256": rr.checkpoint_sha256,
                         "clean_val_macro_Dice": round(float(rr.macro_Dice), 6)})
    ens = {
        "record_type": "CLEAN_ENSEMBLE_FREEZE_RECORD",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "subject-clean Experiment F 的集成成员冻结（在任何 internal_test 访问之前）",
        "members_selection_rule": "三个 trimodal baseline seed（42/123/2026），"
                                  "与原 Experiment F 的成员定义一致；"
                                  "comparator 由 clean-val-38 macro foreground Dice 决定",
        "members": mem_rows,
        "ensemble_recipe": {
            "per_model": "p_i = softmax(logits_i, dim=1)",
            "combine": "p_ensemble = mean(p_i over members)",
            "decision": "argmax(p_ensemble, dim=class)",
            "forbidden": ["logit averaging", "majority voting", "new weights",
                          "threshold tuning", "calibration", "temperature scaling",
                          "post-processing", "connected component filtering",
                          "morphology"],
            "identical_to_original_experiment_f": True,
            "source": "results/experiments/baseline_deep_ensemble/HOLDOUT_PREREGISTRATION.json "
                      "-> prediction_systems.D_ensemble / ensemble_recipe_frozen",
        },
        "no_test_access": {"internal_test_inference": False,
                           "internal_test_gt_read": False,
                           "challenge_test_gt_read": False},
    }
    (RERUN / "CLEAN_ENSEMBLE_FREEZE_RECORD.json").write_text(
        json.dumps(ens, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("CLEAN_ENSEMBLE_FREEZE_RECORD 已写出；成员：")
    for m in mem_rows:
        print(f"  seed{m['seed']:<6} ep={m['best_epoch']:<4} val={m['clean_val_macro_Dice']:.5f} "
              f"sha={m['checkpoint_sha256'][:16]}…")

    # ---------------- §16 holdout preregistration ----------------
    it = pd.read_csv(ROOT / "manifests/subject_clean_v1/internal_test.csv")
    pre = {
        "preregistration": "SUBJECT-CLEAN Experiment F — one-shot frozen holdout evaluation",
        "status": "PREREGISTERED — holdout NOT yet opened",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "written_before_any_internal_test_access": True,
        "dataset_identity": {
            "name": "SUBJECT-CLEAN FROZEN HOLDOUT (subject-clean internal_test)",
            "path": "manifests/subject_clean_v1/internal_test.csv",
            "manifest_sha256": sha256_of(ROOT / "manifests/subject_clean_v1/internal_test.csv"),
            "n_cases": int(len(it)),
            "case_set_sha256": setsha(it.case_id),
            "case_set_identical_to_original_internal_test": True,
            "subject_overlap_with_train_or_val": 0,
        },
        "prediction_systems": {
            "A_seed42": mem_rows[0]["checkpoint_path"],
            "B_seed123": mem_rows[1]["checkpoint_path"],
            "C_seed2026": mem_rows[2]["checkpoint_path"],
            "D_ensemble": "mean of A/B/C softmax probabilities, argmaxed once",
            "ensemble_recipe_frozen": ens["ensemble_recipe"],
        },
        "member_checkpoint_sha256": {f"seed{m['seed']}": m["checkpoint_sha256"]
                                     for m in mem_rows},
        "primary_comparator": {
            "id": f"seed{sel}",
            "checkpoint": [m for m in mem_rows if m["seed"] == sel][0]["checkpoint_path"],
            "rationale": "clean-val-38 macro foreground Dice 最高（唯一的 seed 选择依据）",
            "selection_metric": "validation macro foreground Dice (STN/SN/RN mean) on n=38",
            "locked": True},
        "primary_endpoint": {
            "name": "case-wise paired macro foreground Dice difference",
            "per_case": "MacroDice_i = mean(Dice_STN_i, Dice_SN_i, Dice_RN_i)",
            "estimand": "mean over the 100 paired cases of "
                        "(MacroDice_ensemble,i − MacroDice_comparator,i)",
            "n_paired_cases": 100,
            "pairing": "same case_id, paired",
            "forbidden_substitutes": ["difference of aggregate means",
                                      "pooled voxel Dice", "test-set re-selection"],
        },
        "primary_success_rule": {
            "all_three_required": ["mean Δ > 0",
                                   "bootstrap 95% CI lower bound > 0",
                                   "two-sided permutation p < 0.05"],
            "otherwise": "primary endpoint NOT confirmed; report as-is",
            "cannot_be_overturned_by": ["secondary metrics", "post-hoc subgroups"]},
        "statistical_analysis": {
            "bootstrap": {"type": "paired case resampling", "n_resamples": 10000,
                          "ci": 95, "seed": 20260927},
            "permutation": {"type": "paired sign-flip", "n_draws": 100000,
                            "two_sided": True, "seed": 20260927,
                            "include_observed": True},
            "no_test_substitution": True},
        "secondary_endpoints": {
            "metric_list": ["STN Dice", "SN Dice", "RN Dice", "STN Precision",
                            "STN Recall", "STN Pred/GT", "STN HD95 (mm)",
                            "STN FP", "STN FN"],
            "all_same_100_case_ids_paired": True,
            "use": "descriptive; cannot redefine primary success",
            "p_value_policy": "uncorrected, reported as descriptive only"},
        "one_shot_policy": {
            "state": "CLOSED -> PREDICTIONS_FROZEN -> OPENED -> COMPLETE",
            "after_first_complete_result": "no further evaluation, no re-tuning",
            "then_forbidden": ["re-running inference", "changing members",
                               "changing comparator", "changing metrics"]},
        "relationship_to_original_experiment_f": {
            "original": "results/experiments/baseline_deep_ensemble/secondary_frozen_holdout/ "
                        "（已冻结的原始预注册评估，保留为 historical result）",
            "this": "subject-clean 的**另一次**一次性评估，写入独立目录",
            "must_not_be_merged": True,
            "reason": "provider 侧发现受试者级重复，原评估的独立性主张不成立"},
        "what_this_phase_did_not_do": ["no internal_test inference",
                                       "no internal_test GT read",
                                       "no challenge_test access",
                                       "no threshold or hyper-parameter tuning"],
    }
    (RERUN / "SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json").write_text(
        json.dumps(pre, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json 已写出")
    print(f"  internal_test n={len(it)}  case_set_sha256={setsha(it.case_id)[:16]}…")
    print(f"  comparator = seed{sel}")

    print()
    print("=" * 118)
    print(f"FAIL 项: {len(FAIL)}")
    for f in FAIL:
        print(f"   - {f}")
    print("=" * 118)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

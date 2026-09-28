#!/usr/bin/env python
"""SUBJECT-CLEAN ONE-SHOT HOLDOUT EVALUATION.

严格复用官方评估器的数学：
  * ensemble  : evaluate_frozen_holdout.run_blind_inference  （softmax -> 概率均值 -> argmax）
  * metrics   : evaluate_frozen_holdout.evaluate_predictions （调用 src/evaluation/segmentation_metrics）
  * statistics: evaluate_frozen_holdout.paired_bootstrap_ci / paired_signflip_pvalue
                （10000 / 100000 / RNG seed 20260927，与预注册一致）

与官方评估器的唯一差别（全部为保证 subject-clean 与不覆盖历史）：
  * 使用 clean 的三 member checkpoint（SHA256 已冻结）
  * 使用 clean internal_test manifest（case-set 与预注册一致）
  * 输出写入**新 namespace** results/subject_clean_holdout/
  * 绝不写回 results/experiments/baseline_deep_ensemble/

一次性原则：本脚本只运行一次。
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_frozen_holdout as EFH  # noqa: E402
from models.anisotropic_unet3d import AnisotropicUNet3D, UNet3DConfig  # noqa: E402
from training.train_baseline import load_checkpoint  # noqa: E402
from data import crop_spec as cs  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RERUN = ROOT / "results/subject_clean_rerun"
OUT = ROOT / "results/subject_clean_holdout"
HOLDOUT_IMG = ROOT / "cache/holdout_frozen_v1/images"
LABELS = ROOT / "processed/labels"
MANIFEST = ROOT / "manifests/subject_clean_v1/internal_test.csv"
PRIMARY_COMPARATOR = "seed123"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(4 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    freeze = json.loads((RERUN / "CLEAN_ENSEMBLE_FREEZE_RECORD.json").read_text(encoding="utf-8"))
    pre = json.loads((RERUN / "SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json").read_text(encoding="utf-8"))
    members = freeze["members"]
    seeds = [f"seed{m['seed']}" for m in members]
    print(f"members: {seeds}   comparator: {PRIMARY_COMPARATOR}")

    manifest_sha = sha256_file(MANIFEST)
    assert manifest_sha == pre["dataset_identity"]["manifest_sha256"], "manifest 与预注册不一致"
    case_ids = [str(c) for c in pd.read_csv(MANIFEST)["case_id"]]
    assert len(case_ids) == 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- PHASE A: 载入冻结 checkpoint 并盲推理（不接触 GT） ----
    models = {}
    for m in members:
        p = ROOT / m["checkpoint_path"]
        assert sha256_file(p) == m["checkpoint_sha256"], f"checkpoint hash mismatch: {p}"
        payload = load_checkpoint(p)
        stored = payload.get("config", {}) or {}
        model = AnisotropicUNet3D(UNet3DConfig(
            in_channels=int(stored.get("in_channels", cs.IN_CHANNELS)),
            num_classes=int(stored.get("num_classes", cs.NUM_CLASSES)),
            base_channels=int(stored.get("base_channels", 16)))).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        models[f"seed{m['seed']}"] = model
        print(f"  loaded seed{m['seed']}: epoch {payload.get('epoch')} "
              f"sha={m['checkpoint_sha256'][:16]}…")

    cache = EFH.FrozenImageCache(HOLDOUT_IMG, case_ids)
    assert sorted(cache.case_ids) == sorted(case_ids)
    infer_start = datetime.now(timezone.utc)
    ids, predictions = EFH.run_blind_inference(cache, models, device)
    infer_end = datetime.now(timezone.utc)
    print(f"PHASE A done: predictions {predictions.shape} ({predictions.dtype}) "
          f"{(infer_end-infer_start).total_seconds():.1f}s")

    np.save(OUT / "predictions_subject_clean.npy", predictions)

    # ---- PHASE B: 第一次读取 GT（holdout 正式开封） ----
    opened_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"\nINTERNAL_TEST_OPENED_AT: {opened_at}")
    print("INTERNAL_TEST_GT_ACCESS_PURPOSE: ONE-SHOT FROZEN EVALUATION ONLY\n")
    label_source = EFH.ProcessedLabelSource(LABELS)
    rows = EFH.evaluate_predictions(ids, predictions, label_source)
    per_case = pd.DataFrame(rows)
    per_case.to_csv(OUT / "HOLDOUT_PER_CASE.csv", index=False, encoding="utf-8-sig")
    print(f"per-case: {per_case.shape}")

    # ---- aggregates ----
    agg = {name: EFH.aggregate(per_case, name) for name in ["ensemble"] + seeds}
    comp = agg[PRIMARY_COMPARATOR]

    # 直接使用官方 aggregate 的 macro_foreground_dice，避免任何定义分叉
    def macro(member):
        return float(agg[member]["macro_foreground_dice"])

    d_ens = per_case[[f"Dice_{c}__ensemble" for c in ("STN", "SN", "RN")]].mean(axis=1).to_numpy()
    d_cmp = per_case[[f"Dice_{c}__{PRIMARY_COMPARATOR}" for c in ("STN", "SN", "RN")]].mean(axis=1).to_numpy()
    deltas = d_ens - d_cmp

    ci_low, ci_high = EFH.paired_bootstrap_ci(deltas)
    p_value = EFH.paired_signflip_pvalue(deltas)
    mean_d = float(deltas.mean())
    median_d = float(np.median(deltas))
    decision = EFH.primary_decision(mean_d, ci_low, p_value)

    primary = {
        "preregistration": "results/subject_clean_rerun/SUBJECT_CLEAN_HOLDOUT_PREREGISTRATION.json",
        "primary_endpoint": "case-wise paired macro foreground Dice difference",
        "comparison": f"ensemble vs {PRIMARY_COMPARATOR}",
        "n_paired": int(len(deltas)),
        "ensemble_macro_Dice": round(macro("ensemble"), 6),
        "comparator_macro_Dice": round(macro(PRIMARY_COMPARATOR), 6),
        "mean_delta": mean_d,
        "median_delta": median_d,
        "bootstrap_ci_low": float(ci_low), "bootstrap_ci_high": float(ci_high),
        "bootstrap_n_resamples": EFH.BOOTSTRAP_RESAMPLES,
        "permutation_p": float(p_value),
        "permutation_n_draws": EFH.PERMUTATION_DRAWS,
        "rng_seed": EFH.RNG_SEED,
        "primary_success_rule": pre["primary_success_rule"],
        "decision": decision,
    }
    (OUT / "HOLDOUT_PRIMARY_ENDPOINT.json").write_text(
        json.dumps(primary, indent=2, ensure_ascii=False), encoding="utf-8")

    sec_rows = []
    for member in ["ensemble", PRIMARY_COMPARATOR] + [s for s in seeds if s != PRIMARY_COMPARATOR]:
        a = agg[member]
        sec_rows.append({"member": member, "n_cases": a["n_cases"],
                         **{k: v for k, v in a.items() if k != "n_cases"}})
    sec = pd.DataFrame(sec_rows)
    sec.to_csv(OUT / "HOLDOUT_SECONDARY_ENDPOINTS.csv", index=False, encoding="utf-8-sig")

    summary = {"evaluation": "SUBJECT-CLEAN ONE-SHOT FROZEN HOLDOUT",
               "started_utc": infer_start.isoformat(timespec="seconds"),
               "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "internal_test_opened_utc": opened_at,
               "n_cases": len(ids), "members": seeds,
               "primary_comparator": PRIMARY_COMPARATOR,
               "primary": primary, "aggregates": agg}
    (OUT / "FINAL_HOLDOUT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 92)
    print("SUBJECT-CLEAN FROZEN HOLDOUT (n=100)")
    print("=" * 92)
    print(f"  ensemble macro Dice   : {macro('ensemble'):.4f}")
    print(f"  comparator macro Dice : {macro(PRIMARY_COMPARATOR):.4f}  ({PRIMARY_COMPARATOR})")
    print(f"  mean delta            : {mean_d:+.6f}")
    print(f"  median delta          : {median_d:+.6f}")
    print(f"  95% bootstrap CI      : [{ci_low:+.6f}, {ci_high:+.6f}]")
    print(f"  permutation p         : {p_value:.6f}")
    print(f"  PRIMARY SUCCESS       : {decision}")
    print("=" * 92)
    el = time.perf_counter() - t0
    print(f"elapsed: {el/60:.2f} 分钟")
    (OUT / "_elapsed.json").write_text(json.dumps({"seconds": el}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

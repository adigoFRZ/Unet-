# scripts/

每个脚本只做一件事，并且都从项目根读取/写入相对路径。

> **运行位置**：大部分脚本接受 `--root`，并借助 `src/utils/paths.py` 把相对路径重新
> 锚定到项目根，因此可以从任意目录启动。
> 4 个脚本（`run_tier2_subject_clean.py`、`aggregate_clean_val_results.py`、
> `evaluate_subject_clean_holdout.py`、`guard_subject_clean_holdout.py`）内部写死了
> `Path(".").resolve()`，**必须**在项目根目录执行。这样保留是因为它们的 sha256 被
> `results/` 下的冻结记录引用，改动会使那些记录失效。

---

## 数据准备与缓存

| 脚本 | 作用 |
|---|---|
| `make_synthetic_data.py` | 生成合成团块数据，用于无数据时端到端验证流程（**非医学数据、非结果**） |
| `prepare_labels.py` | `QSM_mask` → 0/1/2/3（STN/SN/RN）标签，写入 `processed/labels/` |
| `normalize_images.py` | T1/QSM/NM 前景内归一化，写入 `processed/images/` |
| `build_manifests.py` | 生成 `manifests/{train,val,test}.csv`（官方 200/100/200 划分） |
| `build_experiment_split.py` | 由官方划分派生开发划分 `manifests/experiment/`（160/40） |
| `build_baseline_cache.py` | 固定 crop 张量缓存 `cache/baseline_v1/`（训练与验证共用） |
| `build_boundary_cache.py` | 实验 C 的 STN 符号距离图缓存 |
| `build_spatial_prior_cache.py` | 实验 E 的训练集 occupancy 先验缓存 |
| `build_holdout_image_cache.py` | 留出集评估用的**只读图像**缓存（结构上不读 GT） |

## 训练

| 脚本 | 作用 |
|---|---|
| `run_tier2_subject_clean.py` | **根部运行**。按 `TIER2_RUN_REGISTRY.csv` 顺序驱动论文的 21 个正式 run（子进程调用 `src/training/train_baseline.py`） |

单个 run 的入口是 `src/training/train_baseline.py` 本身：

```bash
$PY src/training/train_baseline.py \
    --config configs/subject_clean_v1/baseline_v1.yaml --run-id baseline_v1
```

## 验证与评估

| 脚本 | 作用 |
|---|---|
| `aggregate_clean_val_results.py` | **根部运行**。汇总 21 个 run 的验证集结果、按**验证集前景 macro Dice** 选出对照 seed123、冻结集成、写留出集预注册 |
| `evaluate_deep_ensemble.py` | 集成配方与冻结 seed 注册表（`RUNS` / `SEEDS` 被下游 import）；`prediction = argmax(mean_i softmax(logits_i))`。逐病例 STN 表是**可选**的，用 `--highlight-case-ids` / `--highlight-case-ids-file` 指定，默认不输出——源码里不含任何病例编号 |
| `evaluate_frozen_holdout.py` | 盲推理 → 指标 → 配对 bootstrap / 符号翻转置换的完整机制（被下游 import）。统计量本身来自 `src/evaluation/statistics.py`，这里保留的是**预注册常数**并逐个显式传入 |
| `evaluate_subject_clean_holdout.py` | **根部运行**。留出集**一次性**评估；**已执行完毕，不要重跑** |
| `guard_subject_clean_holdout.py` | **根部运行**。防止误重跑留出集评估的保护入口，`--self-test` 可自检 |

## 分析与复算

| 脚本 | 作用 |
|---|---|
| `analyze_baseline_errors.py` | 基线的逐例误差分析：逐 (病例, 类别) 体积比 / TP-FP-FN / Dice / HD95、左右 STN 拆分、最差病例、相关性。它定义的 `volume_ratio_<class>` 是 `src/` 里同名量的口径来源 |
| `analyze_stn_failure_mechanism.py` | STN 过分割（未解决的核心失效）的机制分析，只用开发验证集，只读推理 |
| `reproduce_headline_numbers.py` | 从**本地**逐病例表重新汇总主比较数字，核对论文数值。只读已落盘结果，不做推理 |

> `analyze_*` 与 `reproduce_headline_numbers.py` 依赖 `results/` 下的逐病例产物。
> 那些文件含真实病例编号，已被 `.gitignore` 排除，只在保留有这些文件的机器上可运行。

---

## 常用命令

```bash
PY=./.venv/Scripts/python.exe

# 单元测试
$PY -m pytest tests -q

# 无数据快速验证
$PY scripts/make_synthetic_data.py --out tests/_synthetic_data --n-train 3 --n-val 1
$PY src/training/train_baseline.py --config configs/synthetic_smoke.yaml \
    --smoke-overfit 2 --max-steps 40

# 从原始数据走到可训练的缓存
$PY scripts/prepare_labels.py    --root .
$PY scripts/normalize_images.py  --root . --splits train,val
$PY scripts/build_manifests.py   --root .
$PY scripts/build_experiment_split.py --root .
$PY scripts/build_baseline_cache.py   --root .
```

查看任一脚本的完整参数：`$PY scripts/<name>.py --help`。

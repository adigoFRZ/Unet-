# scripts/

每个脚本只做一件事，并且都从项目根读取/写入相对路径。

> **运行位置**：大部分脚本接受 `--root`，并借助 `src/utils/paths.py` 把相对路径重新
> 锚定到项目根，因此可以从任意目录启动。
> 4 个脚本（`run_tier2_subject_clean.py`、`aggregate_clean_val_results.py`、
> `evaluate_subject_clean_holdout.py`、`guard_subject_clean_holdout.py`）内部写死了
> `Path(".").resolve()`，**必须**在项目根目录执行。

---

## 首选入口：`reproduce.py`

```bash
python scripts/reproduce.py smoke      # 无数据，一条命令跑通全链路
python scripts/reproduce.py prepare --data-root <PDCADxFoundation 路径>
python scripts/reproduce.py preflight  # 21 个 run 的输入与 cache 是否齐备
python scripts/reproduce.py train      # 21 个正式 run；registry 自动建立
python scripts/reproduce.py summarize --yes   # 汇总验证结果并写出冻结记录
python scripts/reproduce.py evaluate --data-root <PDCADxFoundation 路径>
```

所有子进程都用 `sys.executable` 启动，Windows / Linux / macOS 以及任意虚拟环境均可运行。

- `prepare` 会构建 **21 个 run 需要的全部缓存**（baseline 训练缓存、`C_boundary` 的
  STN 符号距离图、`E_e2_occupancy` / `E_e3_both` 的 subject-clean occupancy 先验），
  然后自动跑 preflight；只有全部通过才输出 `PREPARE = PASS`。
- occupancy 先验**严格**基于 `manifests/subject_clean_v1/train.csv` 的 161 例构建，
  并把病例清单写入 `occupancy_meta.json`。preflight 对照该清单，旧 160 例缓存会被
  拒绝，**不会退回旧缓存**。
- `evaluate` 是**独立复现评估**入口：用组员自己训练的成员 checkpoint，在冻结的
  100 例 `internal_test` 上评估，结果写入 `results/reproduction_eval_v1/`。
  它拒绝写入作者的任何命名空间，也拒绝把 `internal_test` 说成新鲜未见测试集。

### 历史结果 vs 当前代码（重要）

历史 Experiment F / subject-clean holdout 的数值是**冻结的历史结果**；本仓库当前
公开的代码是经过维护性重构的 **reproduction implementation**。两者**不必**哈希一致，
也不应该假装一致——冻结时的脚本版本没有被保存在任何提交里。

可追溯性由「冻结哈希 + 已授权的修改链」共同保证：

- `results/subject_clean_holdout/HOLDOUT_EVALUATION_PROVENANCE.json`
  保留冻结当时记录的原始 SHA256，**永不被改写**。
- `results/subject_clean_holdout/POST_FREEZE_CODE_AMENDMENTS.json`
  逐条登记其后发生的改动，记录原哈希、当前哈希、原因、改动类型，以及
  `scientific_semantics_changed` / `historical_result_changed` /
  `authorized_for_future_reproduction` 三个布尔裁定。原哈希必须与该条引用的
  冻结记录字段（经 JSON pointer 定位）逐字符一致，因此无法凭空编造。

`guard_subject_clean_holdout.py` 据此给出结论：

| 结论 | 含义 |
|---|---|
| `PASS_FROZEN_EXACT` | 文件与冻结记录逐位一致 |
| `PASS_AUTHORIZED_SUCCESSOR` | 哈希不同，但存在完全匹配的已授权维护性登记 |
| `DENY` | 其余一切：未登记的改动、伪造的当前哈希、与冻结记录不符的原哈希、被标记为影响科学语义或历史结果的改动 |

没有“忽略哈希”的开关。改动类型限定为 `MAINTENANCE_REPRODUCIBILITY` 与
`COSMETIC_NON_SEMANTIC`。

```bash
python scripts/guard_subject_clean_holdout.py --self-test         # 合成目录自检，任何环境可跑
python scripts/guard_subject_clean_holdout.py --check-amendments  # 代码溯源裁决（需本地冻结记录）
python scripts/guard_subject_clean_holdout.py --dry-run           # 含“能否再开封 holdout”（需本地冻结记录）
```

**运行前提**：后两条需要本地的冻结记录（provenance / ensemble freeze record /
预注册）。这些文件含逐病例信息，**不随仓库公开**，所以在一个全新 clone 上它们会报
“provenance 不存在”并 REFUSE——这是预期行为。`--self-test` 只在合成目录上工作，
任何环境都能跑。

注意「代码溯源裁决」与「能否再开封 holdout」是两件事：一个已经评估过的 holdout
永远不该再开封（`--dry-run` 会因此 REFUSE），但这不代表当前代码不可信。

`run_tier2_subject_clean.py` 与 `aggregate_clean_val_results.py` 未出现在任何冻结
哈希登记中，可以自由修改。

---

## 数据准备与缓存

| 脚本 | 作用 |
|---|---|
| `make_synthetic_data.py` | 生成合成团块数据，用于无数据时端到端验证流程（**非医学数据、非结果**） |
| `prepare_labels.py` | `QSM_mask` → 0/1/2/3（STN/SN/RN）标签，写入 `processed/labels/` |
| `normalize_images.py` | T1/QSM/NM 前景内归一化，写入 `processed/images/` |
| `reproduce.py` | **统一复现入口**：`smoke` / `prepare` / `preflight` / `train` / `summarize` / `evaluate` |
| `check_public_manifests.py` | 公开 manifest 的数量、subject overlap 与敏感性信息扫描（提交前必跑） |
| `build_manifests.py` | 生成 `manifests/{train,val,test}.csv`（官方 200/100/200 划分） |
| `build_experiment_split.py` | 由官方划分派生开发划分 `manifests/experiment/`（160/40） |
| `build_baseline_cache.py` | 固定 crop 张量缓存 `cache/baseline_v1/`（训练与验证共用） |
| `build_boundary_cache.py` | 实验 C 的 STN 符号距离图缓存（`--manifest-dir manifests/subject_clean_v1`） |
| `build_spatial_prior_cache.py` | 实验 E 的训练集 occupancy 先验缓存。队列大小来自**给定的 manifest**（`--expect-n-train` 显式传 161），不依赖任何硬编码默认值；建好后会把缓存里的病例清单与 manifest 逐条对照 |
| `build_holdout_image_cache.py` | 留出集评估用的**只读图像**缓存（结构上不读 GT） |

## 训练

| 脚本 | 作用 |
|---|---|
| `run_tier2_subject_clean.py` | **根部运行**。按 `TIER2_RUN_REGISTRY.csv` 顺序驱动论文的 21 个正式 run（子进程调用 `src/training/train_baseline.py`） |

单个 run 的入口是 `src/training/train_baseline.py` 本身。**config 已经决定了 run 的
名字与输出目录，不要再传 `--run-id`**（与 config 里的目录名重复会被新版代码拒绝）：

```bash
$PY src/training/train_baseline.py --config configs/subject_clean_v1/baseline_v1.yaml
```

## 验证与评估

| 脚本 | 作用 |
|---|---|
| `aggregate_clean_val_results.py` | **根部运行**。汇总 21 个 run 的验证集结果、按**验证集前景 macro Dice** 选出对照 seed123、冻结集成、写留出集预注册 |
| `evaluate_deep_ensemble.py` | 集成配方与冻结 seed 注册表（`RUNS` / `SEEDS` 被下游 import）；`prediction = argmax(mean_i softmax(logits_i))`。逐病例 STN 表是**可选**的，用 `--highlight-case-ids` / `--highlight-case-ids-file` 指定，默认不输出——源码里不含任何病例编号 |
| `evaluate_frozen_holdout.py` | 盲推理 → 指标 → 配对 bootstrap / 符号翻转置换的完整机制（被下游 import）。统计量本身来自 `src/evaluation/statistics.py`，这里保留的是**预注册常数**并逐个显式传入 |
| `evaluate_subject_clean_holdout.py` | **根部运行**。作者侧留出集**一次性**评估；**已执行完毕，不要重跑**。它读仓库内 `processed/labels`，因为作者那一轮的数据就放在仓库内——**组员复现请用下面的 `evaluate_reproduction.py`** |
| `evaluate_reproduction.py` | **组员侧的独立复现评估**（`reproduce.py evaluate` 调用的就是它）。GT 与影像一律从 `--data-root` 定位，不假定仓库内路径；结果写入 `results/reproduction_eval_v1/`，拒绝写入作者命名空间 |
| `guard_subject_clean_holdout.py` | **根部运行**。防止误重跑留出集评估的保护入口，`--self-test` 可自检 |

`evaluate_reproduction.py` 的固定内容（不可配置、不可关闭）：

- comparator **从本地冻结的预注册记录读取**，必须是已冻结成员且标记 `locked`；
  不硬编码、也不在本脚本里选择；
- ensemble 配方沿用冻结实现：`softmax(logits, dim=1)` → 成员概率图取算术平均 →
  一次 `argmax`（由 `evaluate_frozen_holdout.run_blind_inference` 执行）；
- 指标与配对统计沿用冻结实现，且统计常数必须与预注册记录一致，不一致直接拒绝；
- `internal_test` 图像张量用**与训练缓存相同**的管线构建，并先与训练缓存**逐位
  比对**通过后才开始推理；
- `internal_test` 病例集不得与 `train` / `val` / `challenge_test` 重叠；
- 报告里明确写：这不是作者的历史评估，`internal_test` 也**不是**新鲜未见测试集，
  可主张的只有权重层面的独立性。

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

`python` 指当前虚拟环境里的解释器（Windows 上是 `.venv\Scripts\python.exe`，
Linux/macOS 上是 `.venv/bin/python`）。下面是分步版本，`reproduce.py` 只是把它们串起来。

```bash
# 单元测试（缺数据/缓存的用例自动跳过；`prepare` 之后仍会跳过的那些，
# 是因为它们需要**未随仓库公开**的本地产物：旧的 manifests/experiment/ 划分
# 或历史 checkpoint。判定逻辑见 tests/_local_fixtures.py）
python -m pytest tests -q

# 1) 无数据快速验证（等价于 python scripts/reproduce.py smoke）
python scripts/make_synthetic_data.py --out tests/_synthetic_data --n-train 3 --n-val 1 --overwrite
python scripts/prepare_labels.py   --root tests/_synthetic_data
python scripts/normalize_images.py --root tests/_synthetic_data
python scripts/build_manifests.py  --root tests/_synthetic_data
python scripts/build_baseline_cache.py --root tests/_synthetic_data \
    --split-dir tests/_synthetic_data/manifests
python src/training/train_baseline.py --config configs/synthetic_smoke.yaml --max-epochs 1

# 2) 官方数据（等价于 python scripts/reproduce.py prepare --data-root <PATH>）
python scripts/prepare_labels.py   --root <DATA_ROOT>
python scripts/normalize_images.py --root <DATA_ROOT>
python scripts/build_baseline_cache.py --root <DATA_ROOT> \
    --split-dir manifests/subject_clean_v1 \
    --out-dir cache/baseline_v1 \
    --image-dir <DATA_ROOT>/processed/images \
    --label-dir <DATA_ROOT>/processed/labels
# 实验 C 的边界监督
python scripts/build_boundary_cache.py --root . \
    --manifest-dir manifests/subject_clean_v1 \
    --label-dir cache/baseline_v1/labels \
    --out-dir cache/boundary_v1/stn_signed_distance --splits train,val
# 实验 E 的 occupancy 先验（161 例，来自冻结划分）
python scripts/build_spatial_prior_cache.py --root . \
    --manifest-dir manifests/subject_clean_v1 \
    --label-dir cache/baseline_v1/labels \
    --out-dir cache/spatial_prior_v1_subject_clean --expect-n-train 161

# 3) 开工前检查（等价于 python scripts/reproduce.py preflight）
python scripts/reproduce.py preflight

# 4) 独立复现评估（等价于 python scripts/reproduce.py evaluate --data-root <DATA_ROOT>）
python scripts/evaluate_reproduction.py --data-root <DATA_ROOT>
```

> 正式实验**不再需要** `build_experiment_split.py`：论文用的划分已经冻结在
> `manifests/subject_clean_v1/` 并随仓库提供。该脚本保留仅为复现历史划分。

查看任一脚本的完整参数：`python scripts/<name>.py --help`。

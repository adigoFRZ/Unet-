# 脑深部核团自动分割（DBS 靶点定位）

基于多模态 MRI 的脑深部核团自动分割研究，目标结构为丘脑底核（STN）、黑质（SN）
与红核（RN）。项目以可复现评价为中心，并明确报告了一个负面结果。

## 研究背景

脑深部电刺激（DBS）通过植入脑深部毫米级靶点的电极实施治疗。涉及的核团——主要是 STN，也包括 SN 与 RN——体积小、在单一 MRI 对比下对比度低，目前仍主要依靠术前影像上的人工勾画。人工勾画耗时长、个体间差异大，且每次手术规划都要重新进行，因此能够给出一致核团边界的方法，对手术规划和大规模解剖研究都有实际价值。
本项目的重点研究评价规范、固定划分、逐病例指标、预注册比较等环节而非架构创新。

## 方法

- **模型**：各向异性 3D U-Net。最细的两级编码器只在平面内做卷积与下采样。体素在
  层厚方向为 2 mm、平面内为 0.67 mm，这两级若用完整 3D 卷积核，会把某一轴上的
  6 mm 解剖结构与 2 mm 混在一起。
- **输入**：每位受试者的三个配准模态（T1、QSM、NM）在颅内做强度归一化后，裁
  剪到目标核团周围的固定区域。
- **目标函数**：使用交叉熵与只覆盖 STN/SN/RN 的前景 macro soft Dice 项。背景占裁剪
  区域的 99% 以上的被排除在外。
- **模型选择**：只用验证集**前景 macro Dice**。HD95 在选定权重后计算一次，不参
  与模型选择。
- **多种子训练**：同一配置在 seed 42 / 123 / 2026 下各训练一次，以区分配置效应与
  种子噪声。
- **集成**：报告中，三成员取 softmax 概率图的平均后只做一次 argmax：
  `prediction = argmax( mean_i softmax(logits_i) )`。
- **指标**：Dice、HD95（毫米，使用真实各向异性 spacing）、Precision、Recall，
  均先按病例计算、再对病例取平均。

另有实验评估数据增强、损失函数、边界相关目标、模态组合与空间先验。

## 数据

本项目基于 **PDCADxFoundation（MICCAI 2025）** 数据集，使用其配准的多模态 MRI
（T1 / QSM / NM）与核团标签。原始数据请从官方来源获取，其使用遵循原数据集的
许可条款。

本仓库**不包含**任何原始影像、NIfTI/DICOM 文件、模型权重或大型缓存；也**不包含**
官方未公开的 test ground truth。

仓库**包含**的是为复现本项目实验而提供的**衍生元数据**：冻结的 subject-clean
划分 `manifests/subject_clean_v1/`。其中只有公开数据集原有的病例编号（`RJPD_###`）、
基于影像证据推导的 subject 分组、划分标签与仓库内相对路径。该元数据的公开**不
等同于**对原始数据的分发授权，也**不改变**原数据集许可对数据本身的约束。

代码可脱离真实数据运行：`scripts/make_synthetic_data.py` 会写出一份同样目录
布局的合成团块数据（非医学数据、非结果），用于端到端自检。

## 结果

评价使用整改后的 subject-clean 划分：训练 161 例、验证 38 例，100 例留出受试者用于报告的主比较，另有 199 例未使用。

| 模型 | Macro Dice |
| --- | ---: |
| 对照（seed123） | 0.750928 |
| 集成 | 0.761433 |
| seed2026 | 0.764735 |

- 平均配对差值 Δ = +0.010504
- 95% bootstrap 区间 = [+0.007519, +0.013582]
- 配对置换检验 p ≈ 0.00001

集成优于预先指定的对照 seed123。单成员 seed2026 的描述性 macro Dice 高于集成。
训练与评价工具均在本仓库中提供。

## 限制

1. **subject-clean 是分组整改，不等于已确认受试者身份。** 重复与近似重复的采集是
   依据**已识别影像之间的关联**做 subject-level grouping 后重新分组的；它**不能**
   表述为经过医院身份信息确认的 biological identity ground truth。
2. **`internal_test` 是整改后的固定 100 例复评集，不是从未使用过的新鲜独立 test
   set。** 同一批 100 例在划分整改前已被评估过一次。权重层面的独立性成立——它们
   从未参与任何模型选择——但数据层面的新颖性不成立。
3. **STN 过分割仍然存在。** 集成预测的 STN 体积约为真值的 1.68 倍，precision 远
   低于 recall。这是本项目未解决的核心失效。

## 复现

### 0. 安装

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

`torch` 不写在 `requirements.txt` 里，因为 CUDA 构建要与驱动匹配；CPU 机器去掉
`--index-url` 即可。详见 `requirements.txt` 内的注释。

### 1. 无医学数据的程序验证

不需要任何数据，一条命令跑通全链路（生成合成数据 → 标签 → 归一化 → manifest →
缓存 → 1 epoch 训练 → 保存 checkpoint → 重新载入 → 验证）：

```bash
python scripts/reproduce.py smoke
```

它使用随机团块，**不是医学数据，输出不是结果**，只用于确认代码接线完好。

### 2. 使用官方数据复现训练

```bash
python scripts/reproduce.py prepare --data-root <PDCADxFoundation 路径>
python scripts/reproduce.py train
python scripts/reproduce.py summarize
```

**数据怎么放**：`--data-root` 指向**包含官方数据集目录**的那一层，例如
`--data-root /data`，其中 `/data/PDCADxFoundation_Train_Images_Masks_20250331/...`
下是每例一个文件夹、内含 `T1/QSM/NM` 与 `QSM_mask/NM_mask` 的 NIfTI。病例目录会被
递归发现，嵌套层级与目录命名不影响识别。`prepare` 会把 `processed/` 写到
`<data-root>` 下，把 cache 写到本仓库的 `cache/baseline_v1/`。

- **最终 subject-clean split 已随仓库提供**（`manifests/subject_clean_v1/`），
  无需重新生成，也不会被重新生成。
- `prepare` 会把磁盘上的病例与冻结 manifest 逐一比对；病例数或病例集合不一致时
  **直接报错并列出 missing / unexpected cases**，不会自动重新划分。
- `train` 的 21 个 run 由 `configs/subject_clean_v1/` 自动派生，运行记录
  （registry）在首次运行时自动建立，无需任何预置文件。
- 原始医学影像**不随仓库提供**，请从官方来源获取。

### 3. 评估

- `python scripts/reproduce.py summarize` 汇总 38 例 development validation 的结果，
  并写出后续评估所需的冻结记录（clean-val 表、seed 选择、ensemble 冻结、holdout 预注册）。
- 论文报告的 100 例 `internal_test` 主比较是**一次性**的，**已经消耗**，不得重跑；
  `scripts/guard_subject_clean_holdout.py` 会拒绝再次开封。
- 提交任何 manifest 改动前，先跑一次公开性检查：

```bash
python scripts/check_public_manifests.py     # 期望 PUBLIC_MANIFEST_CHECK = PASS
```

### 关于「结果」一节的数字

**那些数字是冻结的历史结果，clone 之后不会自动重现。** 它们来自已经完成并封存的
一次性评估，运行 `smoke` 或 `prepare` 都不会产生它们。要得到可比的数字，需要按上面
的流程自行训练，并使用同一份冻结划分（`manifests/subject_clean_v1/`）。

当前仓库里的代码是经过维护性重构的 **reproduction implementation**，与产生历史
结果时的代码不保证文件哈希一致，也不假装一致。

可追溯性由**冻结哈希 + 已授权的修改链**维护：冻结记录保留原始 SHA256 且永不被
改写；任何其后发生的改动都登记在
`results/subject_clean_holdout/POST_FREEZE_CODE_AMENDMENTS.json`（**该文件随仓库
公开**），说明原因、改动类型，并明确声明是否影响科学语义与历史结果。

例如，subject-clean holdout 评估器原先把 comparator（`seed123`）硬编码在代码里，
现在改为从冻结的预注册记录读取——解析结果仍是 `seed123`，实验定义未变，只是消除了
未来复现时的硬编码。

> `guard_subject_clean_holdout.py --check-amendments` 是**维护者侧的检查**：它要把
> amendment 与冻结记录逐条对照，而冻结记录（provenance、ensemble freeze record、
> 预注册）含逐病例信息，**不随仓库公开**。因此这条命令在全新 clone 上会报
> “provenance 不存在”，这是预期行为，不是故障。任何环境下都可以运行的是自检：
>
> ```bash
> python scripts/guard_subject_clean_holdout.py --self-test   # 合成目录，17 个场景
> ```

## 目录

| | |
| --- | --- |
| `src/` | 模型、数据读取、损失、指标、训练与评估机制 |
| `configs/` | 每个实验一个 YAML；论文结果用的配置在 `configs/subject_clean_v1/` |
| `scripts/` | 数据准备、缓存构建、训练驱动、评估、误差分析 |
| `manifests/subject_clean_v1/` | 冻结划分与 subject 分组元数据（**公开**） |
| `tests/` | 单元测试；依赖数据或缓存的用例在缺数据时自动跳过 |
| `requirements.txt` | 依赖；`torch` 需按 CUDA 版本单独安装（见文件内注释） |

运行方式见 `scripts/README.md` 与 `configs/README.md`。

## 关于本仓库

公开版本经过隐私与可维护性整理：不含影像、NIfTI/DICOM、模型权重与大型缓存，
逐病例**结果**记录一律留在本地。公开的只有冻结划分所需的病例编号、subject 分组
与划分标签。论文报告的数值来自冻结的实验记录；核心模型、损失、集成规则与指标
定义与实验执行时一致，未作改动。

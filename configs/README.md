# configs/

每个实验一个 YAML，**写全所有影响结果的键**（约 20 个）。
这样单个文件就能完整说明一次实验是怎么跑的，不用去别处拼默认值；
代价只是几个文件之间看起来有重复，那些重复正是"与基线完全相同"的证据。

共享默认值定义在 `src/training/train_baseline.py::BaselineConfig`：
YAML 里没写的键取该 dataclass 的值，改默认值等于改所有实验，不要改。

`load_config()` 遇到未知键会直接报错，所以拼错键名不会静默失效。

---

## `subject_clean_v1/` —— 论文结果用的配置

最终划分（train 161 / val 38）上 21 个正式 run 的配置，也是仓库里**唯一**
能对上论文数字的一组配置。

> 这些文件的 sha256 记录在 `results/subject_clean_rerun/CLEAN_VAL_RESULTS.csv`
> 与 `results/subject_clean_evidence_closure/EFFECTIVE_CONFIG_AUDIT.csv` 中
> （两份记录含病例级信息，只在本地保留）。**请勿修改内容**，否则本机的溯源
> 记录会对不上。

```
subject_clean_v1/
├── baseline_v1.yaml                     基线：T1+QSM+NM，CE + 前景 macro Dice
├── augmentation_v1.yaml                 实验 A：保守仿射 + 逐模态强度增强
├── stn_tversky_v1.yaml                  实验 B：STN 非对称 Tversky（α=0.6, β=0.4）
├── stn_boundary_v1.yaml                 实验 C：STN 边界距离项
├── modality_ablation/                   实验 D：单模态与两模态组合（6 个）
├── modality_multiseed/                  实验 D2：跨 seed 稳健性（8 个）
└── spatial_prior/                       实验 E：坐标 / occupancy 先验（3 个）
```

大多数配置的注释里写着"本文件是 `configs/baseline_v1.yaml` 的改动版"：
指的就是同一目录下的 `subject_clean_v1/baseline_v1.yaml`，
即这批配置的公共基线。

基线全部超参数（未写进 YAML 的即取此值）：

| | |
|---|---|
| crop / 通道 | `(32, 96, 96)`，通道序 T1, QSM, NM |
| 模型 | 各向异性 3D U-Net，`base_channels=16`，5,240,420 参数 |
| 损失 | CrossEntropy + 前景 macro soft Dice（不含背景） |
| 优化 | AdamW，lr 3e-4，weight decay 1e-5，batch 2，AMP，grad clip 1.0 |
| 轮数 | 最多 200 epoch，40 epoch 无提升早停 |
| 选择 | 只用验证集 macro foreground Dice，不用 HD95 |
| 划分 | `manifests/subject_clean_v1/` |

路由键（`manifest_dir` / `checkpoint_dir` / `results_dir`）指向的目录含病例级
记录，不在公开仓库中；用自有数据复现时改成自己的路径即可。

## `stn_interface_soft_v1.yaml` / `stn_interface_soft_mc_v1.yaml`

G1 / G1-MC「STN 界面软目标」机制实验的配置。这条线**不是**论文报告的消融，
而是一个负结果：软目标把 STN 的目标质量偏置消掉之后，STN 仍然向外漂移。

保留这两份配置是因为对应实现在 `src/data/interface_soft_target*.py` 与
`src/losses/stn_interface_soft_ce.py`，被 `src/training/train_baseline.py`
直接 import 并注册进损失表；删掉配置会让这份实现变成没有入口的孤儿代码。
它们跑在**整改前**的开发划分上，引用 `manifests/experiment/`。

## `synthetic_smoke.yaml`

合成数据冒烟测试专用，不是实验配置。见 `scripts/README.md`。

---

## 复现某个实验

```bash
PY=./.venv/Scripts/python.exe

# 训练（run 目录已存在时会拒绝覆盖）
$PY src/training/train_baseline.py \
    --config configs/subject_clean_v1/modality_ablation/t1_only.yaml \
    --run-id t1_only_seed42
```

输出写到配置里的 `results_dir` / `checkpoint_dir`。运行开始前会在该 run 的
结果目录下记录一份 `environment.json`（torch / CUDA / GPU、种子、各 manifest
与配置文件的 sha256）和一份 `run_config.yaml`（本次实际使用的配置副本），
用于事后确认某个数字是哪个配置、哪份划分跑出来的。

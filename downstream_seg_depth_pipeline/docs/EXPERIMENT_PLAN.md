# SynthUrbanSAT 下游实验计划 (OKR)

> 目标：用一套**冻结骨干探针 (frozen-backbone probing)** 协议，证明我们用
> OSM + 生成式流水线造出的合成数据集 **SynthUrbanSAT**，在城市遥感稠密预测任务上
> 比真实数据集 **US3D (DFC2019)** *更好 (better)* 且 *更大/可扩展 (larger)*。
>
> 投稿目标：ICLR。本文件用 OKR 方式组织——每个 **Objective (O)** 是一个科学命题，
> 每个 **Key Result (KR)** 都对应一个**人能直接看懂的交付物**（PNG 图 / CSV 表）。
> 先不跑大实验，所有命令将在服务器 `conda activate flux_train` 下执行。

---

## 0. North Star（北极星指标）

> **"在永远只用真实 US3D 测试集评测的前提下，把合成数据加入训练能否稳定提升
> 分割 mIoU 和高度 RMSE，并随合成数据量持续变好？"**

只要这句话成立，"更好 + 更大"的故事就成立。

---

## 1. 为什么这样设计（ICLR 顶会常见做法）

合成数据类论文（如 DINOv2 评测协议、各类 *Sim2Real* / *TSTR* 工作）通常用以下四种证据：

| 证据类型 | 含义 | 我们对应的 KR |
|---|---|---|
| **TSTR** (Train on Synthetic, Test on Real) | 只用合成训练，测真实，逼近真实上限 = 合成"够真" | KR1.1 / KR2.1 |
| **Augmentation gain** | 真实 + 合成 > 纯真实 = 合成"有增量" | KR1.2 / KR2.2 |
| **Low-data regime** | 真实标注稀缺时合成增益最大 = 合成"能替代标注" | KR1.3 / KR2.3 |
| **Scaling law** | 性能随合成数据量单调上升 = 合成"可扩展/更大" | KR1.4 / KR2.4 |
| **Baseline ablation**（加分） | 击败朴素增广 / 无控制生成 = 是"生成流水线"在起作用 | O3 |

**关键控制变量原则**：骨干 (DINOv2) 冻结，只训练任务头 (head)。这样实验中**唯一变量
就是训练数据的配比**，性能任何变化都能干净地归因到"数据"，而非"模型/调参"。这是
让因果结论可信的核心，也是 reviewer 最看重的一点。

---

## 2. 模型与超参（搜文献后的选型）

DINOv2 官方在稠密任务（深度估计、语义分割）上的标准评测就是
**「冻结 DINOv2 骨干 + linear / DPT 头」**，这正是我们采用的配置。

### 2.1 骨干 (backbone) — DINOv2，冻结
| 变体 | 参数量 | 特征维度 | block 数 | 用途 |
|---|---|---|---|---|
| ViT-S/14 | 21 M | 384 | 12 | 快速消融/调试 |
| ViT-B/14 | 86 M | 768 | 12 | 可选骨干尺度消融 |
| **ViT-L/14** | **300 M** | **1024** | **24** | **主实验**（性能/算力平衡） |
| ViT-g/14 | 1100 M | 1536 | 40 | 仅在算力充足时做上限对照 |

- patch_size = 14 → 输入 `518×518`（= 14×37）得到 `37×37` patch grid。
- 取 4 个中间层做多尺度特征：ViT-L 用 `out_indices=[4,11,17,23]`（喂给 DPT）。
- `freeze=True`，`eval()` 模式，不更新 BN/统计量。

### 2.2 任务头 (head)
| 头 | 结构 | 适用 | 可训练参数量(约) | 角色 |
|---|---|---|---|---|
| **DPT** | 4 路特征 → reassemble → FPN 式融合 → 稠密输出 | 分割 + 高度 | ViT-L ≈ 6–8 M | **主头**（稠密预测标准头） |
| **Linear** | 单/多层 1×1 卷积 | 分割（控制对照） | < 0.1 M | 控制实验，证明结论不靠强 head |

> 之前的疑问澄清：**DINOv2 是骨干，DPT 是头，二者不是"两个模型"**，而是
> "骨干 + 头"的一套。分割和高度共用同一冻结骨干、同一种 DPT 结构，只是输出通道
> 不同（分割 = 6 类 logits；高度 = 1 通道，经 `sigmoid × ndsm_max_m` 映射到米）。

### 2.3 训练协议（两任务一致）
| 项 | 值 |
|---|---|
| 输入尺寸 | 518×518 |
| 优化器 | AdamW, lr=1e-3, wd=1e-4 |
| batch / epoch | 8 / 50 |
| 精度 | AMP (fp16) |
| 分割 loss | CrossEntropy (ignore_index=255) |
| 高度 loss | 仅在有效像素上的 masked L1 (米) |
| 随机种子 | 3 个 (0,1,2)，报告 mean ± std |
| 测试集 | **永远是真实 US3D 留出城市**（地理不重叠，防泄漏） |

---

## 3. 数据三态 —— 整个对比的核心

你只需要提供**原始 US3D 数据集**（rgb + seg + depth）。其余合成数据由 OSM +
生成流水线产生。我们把训练数据分成三态：

| 代号 | 名称 | RGB 来源 | seg/高度(条件)来源 | 布局 | 回答的问题 |
|---|---|---|---|---|---|
| **R** | 真实 US3D | 真实卫星图 | 真实 US3D 标注 | US3D 真实城市 | 基线 |
| **Sₚ** | US3D-配对合成 | **生成**（FLUX+ControlNet，以 US3D 自身 seg+nDSM 为条件） | US3D 真实标注 | **与 R 相同** | **生成的 RGB 是否够真，能否替代真实 RGB？**（隔离 RGB 质量） |
| **Sₒ** | OSM 合成 | **生成**（以 OSM 派生 seg+nDSM 为条件） | OSM 派生标注 | **全新、可无限扩展** | 新布局 + 规模化能否带来增量？（"更大"） |

> 两步走，正是你描述的思路：
> 1. **先"重生成 US3D"(Sₚ)**：用 US3D 的 seg 和 depth 当条件生成 RGB，layout 与真实
>    完全配对。把 Sₚ 与 R 对比，**干净地测量生成器的保真度**（label→pixel 关系是否守住）。
>    这是合成数据的"近上限"对照。
> 2. **再用 OSM 海量 seg/depth (S₀)** 替换或叠加：布局全新、数量不设上限，测试
>    "规模化 + 多样性"带来的真实增益。

由此组合出全部训练条件：

```
R                 (real_fraction∈{0.1,0.25,0.5,1.0}, synth=0)
Sₚ                (real=0, US3D-paired 全量)              ← TSTR-paired
S₀                (real=0, OSM 全量)                      ← TSTR-osm
R + Sₚ            (full real + US3D-paired)               ← 保真度增益上限
R + S₀            (full real + OSM, count 扫描)           ← 主增益 + scaling
```

---

## 4. Objectives & Key Results

### 🎯 O1 — 语义分割：合成数据提升并可扩展真实 US3D 上的地物/建筑分割
- **主指标**：6 类 **mIoU**（真实测试集）。**次指标**：per-class IoU、OA、F1。

| KR | 命题（成功判据） | 交付物（人能看懂） |
|---|---|---|
| **KR1.1** 生成保真 / TSTR | 仅用 **Sₚ** 训练，mIoU ≥ **90% × R** | `kr1_1_tstr_bars.png`（R / Sₚ / S₀ 的 mIoU 柱状图带误差棒）+ `kr1_1_seg_qual.png`（RGB｜GT｜pred 三联图） |
| **KR1.2** 增广增益 | **R+S₀ > R**，3 seed 显著 | `kr1_2_condition_bars.png` + `kr1_2_perclass_iou.csv` |
| **KR1.3** 低数据增益 | 在 real_fraction=10% 时 R+S₀ 相对 R 增益最大 | `kr1_3_lowdata_curve.png`（两条线：R vs R+S₀） |
| **KR1.4** 合成 scaling law | mIoU 随 S₀ 数量 {0,1k,5k,10k,25k,50k} 单调上升 | `kr1_4_scaling_curve.png` |

### 🎯 O2 — 高度/深度 (AGL nDSM)：合成数据提升并可扩展建筑高度估计
- **主指标**：**RMSE (米)**。**次指标**：MAE(米)、δ<1.25、组合指标 **DFC2019 mIoU-3**。

| KR | 命题（成功判据） | 交付物 |
|---|---|---|
| **KR2.1** TSTR | 仅 **Sₚ** 训练，RMSE ≤ R + 0.5 m（或 ≤ 110% × R） | `kr2_1_tstr_bars.png` + `kr2_1_height_qual.png`（RGB｜GT 高度｜pred｜误差图，共享色标） |
| **KR2.2** 增广增益 | **R+S₀** 的 RMSE < R | `kr2_2_condition_bars.png` |
| **KR2.3** 低数据增益 | real=10% 时 RMSE 下降最多 | `kr2_3_lowdata_curve.png` |
| **KR2.4** scaling law | RMSE 随 S₀ 数量单调下降 | `kr2_4_scaling_curve.png` |
| **KR2.5** 组合指标 | **mIoU-3**（类别对 *且* \|Δh\|<1m）随 R+S 上升 | `kr2_5_miou3_bars.png` |

### 🎯 O3 (stretch) — 证明是"生成流水线"在起作用，而非"只是更多像素"
- 基线：① 经典增广（copy-paste / 颜色抖动）② raw-FLUX（无 ControlNet 控制）
  ③ 仅 Blender-RGB（无扩散精修）。
- **KR3.1**：SynthUrbanSAT 的 R+S 在 mIoU 和 RMSE 上**全面优于**上述基线
  → `kr3_baselines_bars.png`。

---

## 5. "更好 + 更大" 如何被证明

- **更好（保真）**：KR1.1/KR2.1 显示 **Sₚ ≈ R**（TSTR 接近上限），且 R+S>R。
- **更大（可扩展）**：KR1.4/KR2.4 的 scaling 曲线在 US3D ~480 tile 天花板之上**继续上升**；
  KR1.3/KR2.3 的低数据曲线显示合成能**替代稀缺真实标注**。

---

## 6. 交付物清单（全部落在 `output/`，全部可视化）

```
output/
├── scaling_results/results.csv            # 全部数值（任务×条件×seed）
├── figures/
│   ├── kr1_1_tstr_bars.png   kr1_1_seg_qual.png
│   ├── kr1_2_condition_bars.png           kr1_2_perclass_iou.csv
│   ├── kr1_3_lowdata_curve.png
│   ├── kr1_4_scaling_curve.png
│   ├── kr2_1_tstr_bars.png   kr2_1_height_qual.png
│   ├── kr2_2_condition_bars.png
│   ├── kr2_3_lowdata_curve.png
│   ├── kr2_4_scaling_curve.png
│   └── kr2_5_miou3_bars.png
└── summary_figure.png                     # 论文主图（多子图拼版）
```

每张 KR 图都能独立看懂：柱状图带 ±std、曲线带图例、定性三联图带标题。

---

## 7. 执行步骤（服务器上，按顺序）

```bash
conda activate flux_train
cd SynthUrbanSAT/downstream_pipeline
pip install -r requirements.txt

# 0) 离线 sanity（无需数据/GPU）
python -m tests.test_smoke

# 1) 地理不泄漏切分（留出一个城市做测试，例如 OMA）
python scripts/make_splits.py --src /path/us3d_flat --out ../train_pipeline/dataset \
    --test-prefixes OMA --val-fraction 0.1

# 2) 准备两类合成数据目录（与真实同构: rgb/ seg/ depth/）
#    Sp -> dataset_synth_us3d/  (条件=US3D seg+nDSM)
#    So -> dataset_synth_osm/   (条件=OSM seg+nDSM)

# 3) 单条件冒烟（小步验证）
python scripts/train_probe.py --task segmentation --real-fraction 1.0 --synth-count 0

# 4) 全实验矩阵（两任务×3 seed×两曲线×两合成源）→ CSV + 全部 KR 图
python experiments/run_scaling.py --tasks segmentation height --seeds 0 1 2 \
    --synth-source osm
python experiments/run_scaling.py --tasks segmentation height --seeds 0 1 2 \
    --synth-source us3d_paired
```

---

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 合成 RGB 与真实存在域差 (domain gap) | Sₚ 配对实验隔离 RGB 质量；必要时做轻量颜色对齐 |
| 测试集泄漏（同城训练/测试） | `make_splits.py --test-prefixes` 按城市留出，地理隔离 |
| 标签空间不一致 | US3D-Enhanced 与 OSM 已统一为同一 6 类调色板，高度均为 AGL nDSM(米) |
| scaling 曲线早饱和 | 报告饱和点本身即结论；并在低数据区强调增益 |
| 单一指标偶然性 | 3 seed + 误差棒 + 组合指标 mIoU-3 交叉印证 |

# RUNBOOK — Phase 1: Real vs Synthetic (服务器一键运行)

> **给 AI / 给未来的我**：在一台干净的服务器上 `git clone` 本仓库后，**只需要**
> 提供两样东西：
> 1. 一个 **HuggingFace token**（`HF_TOKEN`，用于下载模型权重）；
> 2. **US3D 数据集的路径**（一个包含 `rgb/ seg/ depth/` 三个子目录的扁平文件夹）。
>
> 然后按本文件从上到下执行命令即可。整条链路（安装依赖 → 下载权重 → 切分数据 →
> 生成合成数据 → 训练探针 → 出图）会自动跑完。**Phase 1 只对比 Real vs Synthetic，
> 不做 scaling / low-data 曲线**（那些留到 Phase 2，见 `experiments/run_scaling.py`）。

实验设计的完整 OKR 见 [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)。

---

## 0. 这个实验在做什么（一句话）

冻结 DINOv2 主干（frozen backbone），只训练任务头（DPT head）。**唯一的自变量是
训练数据**：

| 条件 | 训练数据 | 含义 |
|------|----------|------|
| **R** | 真实 US3D | baseline |
| **S** | 合成（`us3d_paired` / S_p）| 用我们的生成器在 US3D 自己的 seg+nDSM 上重绘 RGB，布局与真实完全一致 → TSTR，检验“生成 RGB 是否能替代真实 RGB” |

两者都在**同一份真实 US3D 测试集**（按城市留出，geographic holdout）上评测。
分割（segmentation）与高度（height）**分开各训一个头**（详见第 6 节）。

---

## 1. 环境前置（一次）

```bash
# 进入下游评测目录
cd SynthUrbanSAT/downstream_seg_depth_pipeline

# 激活生成所需的 conda 环境（服务器上已存在，含 torch + FLUX 依赖）
conda activate flux_train

# 提供 HF token（必填）
export HF_TOKEN=<your_hf_token>

# 权重默认从 JasonXF/SynthUrbanSAT_bestmodel 下载，通常不用额外设置。
# 如需覆盖到别的 repo，再设置：
# export WEIGHTS_REPO=your-hf-name/your-weights-repo
```

> 4 个基础权重文件：`flux2_dev_fp8mixed.safetensors`、`flux2-vae.safetensors`、
> `mistral_3_small_flux2_fp8.safetensors`、`FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors`。
> LoRA / HDC²A checkpoint 和 tokenizer 已一起打包在 `JasonXF/SynthUrbanSAT_bestmodel`，
> 用 `HF_TOKEN` 下载，无需手动处理。

## 2. 安装 + 下权重（一条命令）

```bash
bash setup.sh
```

它会：安装下游探针与生成管线的 Python 依赖 → 从
`JasonXF/SynthUrbanSAT_bestmodel` 拉取 `generation_pipeline/weights` 所需的
`base/`、`lora/`、`tokenizer/` → 验证权重齐全。结束时打印 `Setup complete.`。

## 3. 跑完整 Phase 1（一条命令）

```bash
bash run_phase1.sh --us3d-dir /path/to/US3D
```

参数（一般用默认即可）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--us3d-dir` | （必填）| 扁平 US3D 根目录，含 `rgb/ seg/ depth/` |
| `--test-prefixes` | `JAX` | 作为测试集留出的城市前缀（US3D 有 JAX/OMA）|
| `--gen-seeds` | `0` | 每个真实瓦片生成几张合成 RGB（1 个 seed = 合成与真实等量）|
| `--train-seeds` | `0 1 2` | 训练重复的随机种子数（出均值±方差）|
| `--gen-limit` | `0` | >0 时只生成前 N 个瓦片，用于快速冒烟测试 |

`run_phase1.sh` 依次执行：
1. `scripts/make_splits.py` —— 按城市留出，切成 `train/val/test`（无泄漏）；
2. `scripts/gen_synth_from_real.py` —— 用 US3D 训练集的 seg+depth 生成合成 RGB，
   **标签直接复制真实的 seg/depth**（只有 RGB 是合成的）→ 写入 `dataset_synth_us3d/train/`；
3. `python -m experiments.run_phase1` —— 对 seg 和 height 各训 **R** 与 **S** 两个探针，
   在真实测试集上评测，出柱状图与定性面板。

### 先冒烟一遍（强烈建议）

第一次先用小样本确认链路通：

```bash
bash run_phase1.sh --us3d-dir /path/to/US3D --gen-limit 8 --train-seeds 0
```

通过后再跑完整版（去掉 `--gen-limit`，恢复 `--train-seeds 0 1 2`）。

## 4. 结果在哪、怎么看

```
output/phase1/
├── results.csv                         # R vs S 数字（mean ± std）
├── results.json
└── figures/
    ├── segmentation_R_vs_S_bars.png    # 头条图：分割 mIoU，R vs S
    ├── height_R_vs_S_bars.png          # 头条图：高度 RMSE(m)，R vs S
    ├── qual_segmentation_R/*.png       # RGB | GT | 预测 三联图
    ├── qual_segmentation_S/*.png
    ├── qual_height_R/*.png             # RGB | GT | 预测 | 误差图
    └── qual_height_S/*.png
```

**怎么解读**：
- 分割看 **mIoU 越高越好**；高度看 **RMSE(m) 越低越好**。
- 若 **S 接近 R**（差距小）→ 我们的合成 RGB 足以替代真实 RGB（核心卖点成立），
  下一步可进入 Phase 2 的 scaling（用 OSM 无限布局把 S 越堆越大）。
- 若 S 明显差 → 检查定性面板，定位是 RGB 保真度问题还是某些类别系统性失败。

## 5. Phase 2（暂不跑，先看 Phase 1 结果再决定）

scaling / low-data 曲线、`osm` 合成源、联合 mIoU-3 等都在
`experiments/run_scaling.py`。**Phase 1 结果确认后**再启动。

## 6. 一个关键事实：seg 和 depth 是分开训的

当前实现里，分割与高度是**两个独立的单任务训练**：各自一个 `ProbeModel`
（独立的 DPT head、独立的优化与训练循环）。它们**共享同一套冻结的 DINOv2 主干权重**
（同一份预训练参数，前向时不更新），但两个头**分别训练、互不影响**。

> DFC 那种把两者合在一起的 **联合 mIoU-3**（高度阈值参与的三类指标）需要一个
> “一个主干 + 两个头”的联合模型，目前**尚未实现**，留待需要时再加。

---

### 常见问题排查

| 现象 | 处理 |
|------|------|
| `base FLUX.2 weights still missing` | `HF_TOKEN` 对 `JasonXF/SynthUrbanSAT_bestmodel` 没有读权限，或 `WEIGHTS_REPO` 指错了 |
| 生成步骤报显存 | Pro 6000 96GB 足够；若 OOM 降低生成并发或 `--gen-seeds 0` |
| `No checkpoints in weights/lora` | `setup.sh` 没拿到 LoRA，确认 `HF_TOKEN` 有该私有仓库读权限 |
| DINOv2 下载失败 | 主干通过 `torch.hub` 从 GitHub 拉取，确认服务器可访问外网 |

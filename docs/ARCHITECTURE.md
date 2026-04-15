# HDC²A + Flux2 ControlNet — Technical Architecture Guide

> 本文档详细描述项目的完整训练流程、每一步的张量维度变化、模型加载顺序、VRAM 占用，以及超参数设置。  
> 所有描述均配有代码引用（文件 + 行号），方便对照查看。

---

## 目录

1. [总览：训练流程](#1-总览训练流程)
2. [模型加载顺序与 VRAM 占用](#2-模型加载顺序与-vram-占用)
3. [数据流：从图片到 latent tokens](#3-数据流从图片到-latent-tokens)
4. [HDC²A Adapter 详解](#4-hdc2a-adapter-详解)
5. [文本编码（Mistral-3.1）](#5-文本编码mistral-31)
6. [Transformer 前向传播](#6-transformer-前向传播)
7. [损失函数：Flow Matching](#7-损失函数flow-matching)
8. [训练/冻结参数统计](#8-训练冻结参数统计)
9. [VRAM 实测数据](#9-vram-实测数据)
10. [超参数一览与调整建议](#10-超参数一览与调整建议)
11. [Checkpoint 与恢复](#11-checkpoint-与恢复)

---

## 1. 总览：训练流程

`train_script.py` 的 `main()` 函数（第 109 行）按以下顺序执行：

```
Step 1  → 文本 embedding 预计算（独占 VRAM，完成后释放）
Step 2  → 加载 VAE（0.17 GB）
Step 3  → 加载 Transformer（FP8解压→key转换→ControlNet合并→FP8压缩 = 40.7 GB）
Step 4  → 创建 HDC²A Adapter（0.1 GB）
Step 5  → 创建 Optimizer（AdamW，惰性分配，首次 step 后约 +15 GB）
Step 6  → 可选 Resume from checkpoint
Step 7  → 快速前向测试（随机数据，验证 pipeline）
Step 8  → 训练循环（train → validate → save checkpoint）
```

> **关键设计**：文本编码器（~17 GB）和 Transformer（~41 GB）无法同时装入 80 GB 显存，
> 因此文本 embedding **在加载 Transformer 之前**预计算并保存到磁盘，然后释放编码器。  
> 参考：`train_script.py` 第 130–155 行。

---

## 2. 模型加载顺序与 VRAM 占用

以下数据在 **RTX PRO 6000 (96 GB)、512×512 图像、batch_size=1** 下实测。

| 阶段 | 操作 | 累积 VRAM | 代码位置 |
|------|------|-----------|----------|
| Pre-flight | 空 | 0.0 GB | `train_script.py:120` |
| Step 1 | 文本编码器加载+编码+卸载 | 峰值 ~17 GB → 0 GB | `train_script.py:130–155` |
| Step 2 | 加载 VAE | 0.17 GB | `train_script.py:158–161`, `utility.py:89–120` |
| Step 3 | 加载 Transformer | 40.7 GB | `train_script.py:165–173`, `utility.py:122–227` |
| Step 4 | 创建 HDC²A | 40.8 GB | `train_script.py:177–186` |
| Step 5+7 | Optimizer + 前向测试 | **峰值 73.5 GB** | `train_script.py:205–221` |
| 训练中 | 每个 batch 稳态 | **56.1 GB** | `train.py:57–161` |

### Transformer 加载细节（`utility.py:122–227`）

```
A. dequant_fp8_state_dict()     → 读 FP8 safetensors，乘以 scale 还原 BF16
                                   （utility.py:58–87）
B. convert_flux2_transformer_checkpoint_to_diffusers()
                                → ComfyUI 键名 → diffusers 键名
                                   （utility.py:137–141）
C. 加载 ControlNet 权重          → 合并到 diffusers state_dict
                                   （utility.py:143–153）
D. 创建 meta device 模型         → Flux2ControlTransformer2DModel(control_in_dim=3072)
                                   （utility.py:155–169）
E. 合并权重，跳过 control_img_in → 维度不匹配（预训练 260 → 我们 3072）
   未加载的参数用 kaiming_uniform 初始化
                                   （utility.py:172–197）
F. 冻结 backbone                → 只有 name 含 'control' 的参数设 requires_grad=True
                                   （utility.py:205–206）
G. FP8 压缩冻结层              → 203 个 frozen Linear → FP8FrozenLinear
                                → 72.9 GB → 40.7 GB，节省 32.2 GB
                                   （utility.py:208–213）
```

---

## 3. 数据流：从图片到 latent tokens

### 3.1 RGB → Latent Tokens

```
RGB 图像 [B, 3, 512, 512]   float [0, 1]
    ↓  rgb * 2 - 1 → [-1, 1]                        utility.py:266
    ↓
VAE encode → latent  [B, 32, 64, 64]                utility.py:268
    ↓
patchify 2×2 → [B, 128, 32, 32]                     utility.py:229–235
    ↓  (将每 2×2 spatial block 的 32 channel 合并为 128 channel)
BN normalize: (x - bn_mean) / bn_std                utility.py:270
    ↓
pack → [B, 1024, 128]                               utility.py:237–240
```

**解释**：512×512 图经 VAE (in_channels=3, latent_channels=32) 编码为 64×64 latent。
然后 2×2 patchify 将 spatial 缩小一半（32×32），channel 扩大 4 倍（32×4=128）。
最终 flatten 为 32×32=1024 个 token，每个 128 维。

### 3.2 数据集格式

```
dataset/train/
├── rgb/tile_001.png         [512×512 RGB]
├── seg/tile_001.png         [512×512 RGB, 5 类颜色编码]
├── depth/tile_001.png       [512×512 灰度]
└── captions.json            {"tile_001.png": "A satellite view of ..."}
```

数据集类 `HDC2ADataset`（`dataprep.py:32–117`）返回：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `rgb` | `[3, H, W]` | float [0,1] | 目标 RGB |
| `seg` | `[H, W]` | long [0..4] | 类别索引（从 color_map.json 解码） |
| `depth` | `[1, H, W]` | float [0,1] | 深度图 |
| `prompt_embeds` | `[512, 15360]` | float | 预计算文本 embedding（如有） |
| `filename` | str | - | 用于 logging |

---

## 4. HDC²A Adapter 详解

完整类定义：`models.py:207–264`

### 4.1 SemanticEncoder（`models.py:54–96`）

处理离散的分割图：

```
seg_map [B, 512, 512]  long (5 classes)
    ↓
nn.Embedding(5, 64)  → [B, 512, 512, 64]            models.py:66, 80
    ↓  permute → [B, 64, 512, 512]
边界检测：3×3 均值卷积 → |x - x_smooth| > threshold  models.py:84–88
    ↓  boundary_conv + mask 加权                       models.py:89–90
ConvStem (3 层, 每层 stride=2):
    Conv2d(64→128)  → [B, 128, 256, 256]             models.py:70
    Conv2d(128→256) → [B, 256, 128, 128]             models.py:71
    Conv2d(256→512) → [B, 512, 64, 64]               models.py:72
    ↓  flatten(2).transpose(1,2) → [B, 4096, 512]
Linear(512, 768) → [B, 4096, 768]                    models.py:78, 94
```

### 4.2 DepthEncoder（`models.py:98–134`）

处理连续深度图：

```
depth_map [B, 1, 512, 512]  float [0, 1]
    ↓
Fourier encoding (32 bands):
    sin(2π · depth · freq_bands) + cos(...)           models.py:123–127
    → [B, 64, 512, 512]
    ↓  cat(depth, fourier) → [B, 65, 512, 512]       models.py:128
ConvStem (同上，input=65) → [B, 512, 64, 64]         models.py:112–119
    ↓  flatten → [B, 4096, 512]
Linear(512, 768) → [B, 4096, 768]                    models.py:121, 132
```

**Fourier bands**：`freq_bands = 2^[0, 1, ..., 31]`，将深度值映射到 64 维正弦/余弦编码，
捕捉不同尺度的深度变化（`models.py:109–110`）。

### 4.3 DoubleStreamFusion（`models.py:136–203`）× 3 blocks

两路 token 的联合注意力机制：

```
输入：T_s [B, 4096, 768],  T_d [B, 4096, 768]

  T_s → LayerNorm → QKV投影 → q_s, k_s, v_s  [B, 4096, 12, 64]
  T_d → LayerNorm → QKV投影 → q_d, k_d, v_d  [B, 4096, 12, 64]
                                                          models.py:173–176
  ↓  RoPE 旋转位置编码                                      models.py:178–181
  ↓
  K_joint = cat(k_s, k_d) → [B, 8192, 12, 64]            models.py:183
  V_joint = cat(v_s, v_d) → [B, 8192, 12, 64]            models.py:184
  ↓
  out_s = SDPA(q_s, K_joint, V_joint) → [B, 4096, 768]   models.py:189
  out_d = SDPA(q_d, K_joint, V_joint) → [B, 4096, 768]   models.py:190
  ↓
  T_s = T_s + out_s;  T_s = T_s + FFN_s(LN(T_s))        models.py:195–198
  T_d = T_d + out_d;  T_d = T_d + FFN_d(LN(T_d))

输出：T_s [B, 4096, 768],  T_d [B, 4096, 768]
```

**要点**：每个流的 Q 独立，但 K 和 V 是两路拼接的——这样每个 token 能同时看到自身流和对方流的信息。

### 4.4 Gated Merge + 下采样（`models.py:248–264`）

```
T_concat = cat(T_s, T_d, dim=-1) → [B, 4096, 1536]      models.py:249
    ↓
gate_net: Linear(1536→768) → GELU → Linear(768→1) → Sigmoid
    g → [B, 4096, 1]                                      models.py:234–236, 251
    ↓
T_s_proj = W_s(T_s) → [B, 4096, 3072]                    models.py:237, 253
T_d_proj = W_d(T_d) → [B, 4096, 3072]                    models.py:238, 254
T_merged = g * T_s_proj + (1 - g) * T_d_proj              models.py:255
    ↓  × sigmoid(output_scale)                             models.py:240, 257
    ↓  （output_scale 初始值 -5.0 → sigmoid ≈ 0.0067，训练初期接近零）
reshape → [B, 3072, 64, 64]                               models.py:259–260
AvgPool2d(2,2) → [B, 3072, 32, 32]                        models.py:241, 262
flatten → [B, 1024, 3072]                                 models.py:263

输出：control_context [B, 1024, 3072]
```

---

## 5. 文本编码（Mistral-3.1）

代码：`text_encoder.py`

### 5.1 架构

Mistral-3.1-Small-24B (24B params, ~17 GB FP8)。  
从 **第 10、20、30 层** 的 hidden states 提取特征。

```
文本 prompt → PixtralProcessor tokenize → input_ids [B, 512]
    ↓
Mistral3ForConditionalGeneration forward
    output_hidden_states=True                          text_encoder.py:136
    ↓
提取 layer 10, 20, 30 的 hidden states:
    各 [B, 512, 5120]                                  text_encoder.py:142
    ↓  stack → [B, 3, 512, 5120]
    ↓  permute+reshape → [B, 512, 15360]              text_encoder.py:148

输出：prompt_embeds [B, 512, 15360]
```

### 5.2 预计算流程（推荐）

**为什么要预计算？** 文本编码器 ~17 GB + Transformer ~41 GB = 58 GB 权重，加上前向传播的 activation 远超 80 GB。

```
precompute_and_save_embeddings()                       text_encoder.py:164–237
    1. 加载 text encoder + tokenizer
    2. 读取 dataset/{train,val}/captions.json
    3. 分 batch 编码所有 caption → [512, 15360] per sample
    4. 保存到 output/text_embeddings.pt
       格式: {"train/tile_001.png": tensor[512,15360], ...}
    5. 卸载 text encoder，释放 VRAM
```

在训练时，`HDC2ADataset.__getitem__` 通过文件名从 `embeddings_dict` 取出预计算的 embedding：  
`dataprep.py:115–116`

如果没有配置文本编码器（`text_encoder_path=None, precomputed_embeddings=None`），
训练时会用 **随机 embeddings** 作为占位，仅用于架构测试：  
`train.py:107–108`

---

## 6. Transformer 前向传播

模型：`Flux2ControlTransformer2DModel`  
定义：`models/videox_fun/models/flux2_transformer2d_control.py`  
配置：8 个 double-stream layers, 48 个 single-stream layers, 48 heads × 128 dim = 6144 hidden  
`utility.py:157–168`

### 输入

| 参数 | 形状 | 来源 |
|------|------|------|
| `hidden_states` | `[B, 1024, 128]` | 加噪后的 latent tokens |
| `encoder_hidden_states` | `[B, 512, 15360]` | 文本 embedding |
| `control_context` | `[B, 1024, 3072]` | HDC²A 输出 |
| `img_ids` | `[B, 1024, 4]` | latent 位置编码 |
| `txt_ids` | `[B, 512, 4]` | 文本位置编码 |
| `timestep` | `[B]` | flow matching 时间步 |
| `guidance` | `[B]` | guidance scale (默认 3.5) |

### 内部处理

```
1. 文本 token [B, 512, 15360] → linear proj → [B, 512, 6144]
2. 图像 token [B, 1024, 128]  → linear proj → [B, 1024, 6144]
3. control_context [B, 1024, 3072] → control_img_in → [B, 1024, 6144]
   （control_img_in 是我们初始化的层，维度 3072→6144）
4. 双流 attention (8 层)：文本流 + 图像流交替交互
   其中 control_layers=[0,2,4,6] 注入 control signal
5. 单流 attention (48 层)：拼接的文本+图像 token 做自注意力
6. 输出：[B, 1024, 128]  预测的速度场（velocity）
```

调用代码：`train.py:116–128`

### 输出

`noise_pred [B, 1024, 128]` — 与输入 `hidden_states` 同维度的预测速度。

---

## 7. 损失函数：Flow Matching

代码：`train.py:31–53`

```python
# 采样时间步 t ∈ [0, 1]，均匀分布
t = sample_timestep(B, device)                         # train.py:31–33

# 构造加噪 latent 和目标
noise = randn_like(target_packed)
noisy_latent = (1 - t) * target + t * noise            # train.py:49
flow_target = noise - target                           # train.py:50

# 前向传播后计算损失
loss = F.mse_loss(noise_pred, flow_target)             # train.py:129
```

**直觉**：模型学习预测 "从 clean latent 到 noise 的速度"。  
- `t=0` 时 `noisy_latent = target`（完全干净）, 目标 = `noise - target`  
- `t=1` 时 `noisy_latent = noise`（完全噪声）, 目标不变  
- 推理时从 `t=1`（纯噪声）沿预测的速度场积分到 `t=0`，得到 clean latent

---

## 8. 训练/冻结参数统计

实测输出（`train_script.py:188–193`）：

| 组件 | 参数量 | 可训练? | 存储格式 |
|------|--------|---------|----------|
| VAE (AutoencoderKLFlux2) | ~100M | ❌ 冻结 | BF16 |
| Transformer backbone | ~7.8B | ❌ 冻结 | **FP8** (省 50% VRAM) |
| ControlNet control blocks | **4133.5M** | ✅ 训练 | BF16 |
| **HDC²A Adapter** | **52.4M** | ✅ 训练 | BF16 |
| **总计可训练** | **4185.9M** | - | - |

FP8 冻结压缩（`models.py:287–324`）：
- 将冻结的 203 个 `nn.Linear` 替换为 `FP8FrozenLinear`
- 前向：FP8 权重 × scale → BF16，反向：梯度只通过输入流回
- VRAM: 72.9 GB → 40.7 GB，节省 32.2 GB（`utility.py:208–213`）

---

## 9. VRAM 实测数据

测试环境：NVIDIA RTX PRO 6000 (96 GB), 512×512 图像, BF16

### Batch Size = 1

| 指标 | 数值 |
|------|------|
| 权重加载后 | 40.8 GB |
| 前向测试峰值 | **73.5 GB** |
| 训练稳态 (每 step) | **56.1 GB** |
| 训练 epoch 时间 (8 samples) | ~10.8 s |

### Batch Size = 2

| 指标 | 数值 |
|------|------|
| 第一个 step | 56.1 GB |
| 第二个 step | **OOM (>93 GB)** |

### 推荐配置

| GPU | 建议 batch_size | grad_accum_steps | 有效 batch size |
|-----|-----------------|------------------|-----------------|
| A100 80 GB | 1 | 4–8 | 4–8 |
| H100 80 GB | 1 | 4–8 | 4–8 |
| RTX PRO 6000 96 GB | 1 | 4–8 | 4–8 |

> **注意**：batch_size=2 在 96 GB GPU 上也会 OOM。  
> 原因：Transformer 的 double-stream attention (token 数 = 1024 img + 512 txt = 1536)  
> 的 activation 随 batch size 线性增长，加上 AdamW optimizer states 占用大量显存。

---

## 10. 超参数一览与调整建议

定义位置：`train_script.py:52–107` (`CONFIG` dict)

### 模型超参数

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `image_size` | 512 | 输入分辨率 | 改大需更多 VRAM |
| `num_classes` | 5 | 分割类别数 | 按数据集类别调整 |
| `control_in_dim` | 3072 | HDC²A 输出维度 | 供给 Transformer 的 control_img_in |
| `fusion_dim` | 768 | 融合层内部维度 | 越大表达力越强，VRAM 越多 |
| `num_fusion_blocks` | 3 | DoubleStreamFusion 层数 | 1–5，更多层学习更深的交叉特征 |
| `num_heads` | 12 | 注意力头数 | 需整除 fusion_dim |
| `num_fourier_bands` | 32 | 深度 Fourier 编码维度 | 16–64 |
| `boundary_threshold` | 0.1 | 边界检测阈值 | 较小→更多边界增强 |

### 训练超参数

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `batch_size` | 1 | 单卡 batch size | A100 80GB 只能用 1 |
| `grad_accum_steps` | 4 | 梯度累积步数 | 有效 batch = 1×4 = 4 |
| `learning_rate` | 1e-5 | AdamW 学习率 | 1e-6 ~ 5e-5 |
| `weight_decay` | 0.01 | L2 正则 | 0.001–0.1 |
| `max_grad_norm` | 1.0 | 梯度裁剪 | 防止爆炸 |
| `num_epochs` | 100 | 总 epoch 数 | 视数据量和 loss 收敛调整 |
| `guidance_scale` | 3.5 | CFG scale | 训练时固定值 |

### 文本编码超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `text_seq_len` | 512 | 文本 token 最大长度 |
| `text_dim` | 15360 | 文本 embedding 维度 (3 layers × 5120) |
| `text_encoder_path` | None | HuggingFace 模型路径或 ID |
| `precomputed_embeddings` | None | 预计算 .pt 文件路径 |

### 日志与保存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `log_interval` | 10 | 每 N step 打印 loss 和 VRAM |
| `save_every_n_epochs` | 1 | 每 N epoch 保存 checkpoint |
| `val_every_n_epochs` | 1 | 每 N epoch 做验证 |

---

## 11. Checkpoint 与恢复

### 保存（`train.py:230–256`）

```
output/checkpoint_epoch_0010/
├── hdc2a.pt              HDC²A 完整 state_dict
├── control_params.pt     Transformer 中 name 含 'control' 的参数
├── optimizer.pt          AdamW 状态（momentum + variance）
└── meta.pt               {epoch, loss, config}
```

### 恢复（`train.py:260–281`）

在 `train_script.py` 中设置 `resume_from`（第 102 行）：

```python
'resume_from': 'output/checkpoint_epoch_0010',
```

加载流程：
1. `hdc2a.load_state_dict(hdc2a.pt)`
2. `transformer.load_state_dict(control_params.pt, strict=False)` — 只加载 control 参数
3. `optimizer.load_state_dict(optimizer.pt)` — 恢复 Adam 状态
4. 从 `meta.pt` 读取 epoch 号，从下一 epoch 继续

---

## 附录：一次完整训练 step 的数据流

```
                    ┌── seg [B,512,512] ──→ SemanticEncoder ──→ [B,4096,768] ──┐
                    │                                                           │
dataset ──→ batch ──┤── depth [B,1,512,512] → DepthEncoder ──→ [B,4096,768] ──┼──→ DoubleStreamFusion ×3
                    │                                                           │         ↓
                    │                                                           │    GatedMerge + Pool
                    │                                                           │         ↓
                    │                                                        control_context [B,1024,3072]
                    │                                                                      ↓
                    ├── rgb [B,3,512,512] → VAE enc → patchify → pack ─→ [B,1024,128]     │
                    │         ↓                                                             │
                    │     target_packed                                                     │
                    │         ↓                                                             │
                    │   t ~ U[0,1]  noise ~ N(0,1)                                         │
                    │         ↓                                                             │
                    │   noisy = (1-t)·target + t·noise ──────────────→ hidden_states        │
                    │                                                      ↓               │
                    └── prompt_embeds [B,512,15360] ──→ encoder_hidden_states               │
                              (precomputed or random)         ↓                             │
                                                      ┌──────┴──────────────────────────────┘
                                                      ↓
                                            Flux2ControlTransformer
                                                      ↓
                                            noise_pred [B,1024,128]
                                                      ↓
                                      loss = MSE(noise_pred, noise - target)
                                                      ↓
                                            loss.backward() → optimizer.step()
```

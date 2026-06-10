# Generation Pipeline 使用手册

`generation_pipeline/` 是 SynthUrbanSAT 的最后一段：输入 OSM pipeline 生成的 segmentation 和 depth，加载训练好的 HDC2A + FLUX.2 ControlNet/LoRA checkpoint，输出合成的 satellite RGB。

当前保留两个主要入口：

- `app.py`：单图 WebUI 推理，也带一个不开 WebUI 的论文对比图 compare mode。
- `generation_pipeline.py`：面向 OSM output folder 的批量生成脚本，用来大批量产出 synth RGB、HDC2A feature 和 metadata。

旧的 `batch_eval.py` 已经删除；不要再用旧 train/val sanity-check 入口。

## 快速开始

```bash
cd generation_pipeline

# 第一次使用：准备权重。可选 COMFY_MODELS 指向已有 ComfyUI models 目录。
export COMFY_MODELS=/path/to/ComfyUI/models
bash setup.sh

# 进入训练/推理环境后启动 WebUI
conda activate flux_train
python app.py
```

默认 WebUI 地址是 `http://<host>:7860`。如果在服务器上跑：

```bash
python app.py --host 0.0.0.0 --port 7860
```

如果不想启动时预加载 checkpoint：

```bash
python app.py --no-preload
```

## 权重目录

`setup.sh` 会把权重放到 ignored 的 `generation_pipeline/weights/`：

```text
weights/
├── base/
│   ├── flux2_dev_fp8mixed.safetensors
│   ├── flux2-vae.safetensors
│   ├── mistral_3_small_flux2_fp8.safetensors
│   └── FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors
├── tokenizer/
└── lora/<checkpoint>/
    ├── meta.pt
    ├── hdc2a.pt
    └── control_params.pt
```

常用 setup 命令：

```bash
bash setup.sh              # 默认/latest checkpoint
bash setup.sh --copy       # 不用 symlink，直接复制 base weights
bash setup.sh --all-ckpts  # 下载所有配置过的 LoRA/HDC2A checkpoints
```

token 来源优先级：`HF_TOKEN`、`generation_pipeline/.env`、`../train_pipeline/.env`。

## OSM Pipeline 输入结构

批量生成和 compare mode 默认吃 OSM pipeline output folder。典型结构：

```text
../osm_pipeline/output/<city-or-run>/
└── tile_XXXX/
    ├── 2_rgb.png              # OSM/satellite reference RGB，只做参考/对比，不作为 conditioning
    ├── 4_seg.png              # root/top view segmentation
    ├── 5_depth.png            # root/top view PNG depth
    ├── 5_depth.exr            # root/top view EXR depth，可选
    ├── near-nadir-1/
    │   ├── 1_seg.png
    │   └── 2_depth.png
    ├── near-nadir-2/
    │   ├── 1_seg.png
    │   └── 2_depth.png
    └── near-nadir-*/
        ├── 1_seg.png
        └── 2_depth.png
```

默认 root/top view 使用 `4_seg.png + 5_depth.png`。固定 near-nadir 使用 `near-nadir-N/1_seg.png + 2_depth.png`。

如果传 `--depth-exr` 或 `--compare-depth-exr`，脚本只是把 depth 文件后缀从 `.png` 换成 `.exr`：

- root/top view：`5_depth.png` -> `5_depth.exr`
- near-nadir view：`2_depth.png` -> `2_depth.exr`

对应的 EXR 文件必须真实存在，否则该 item 会报 missing/skip。

## 入口 1：WebUI 单图推理

启动：

```bash
python app.py
```

WebUI 适合做交互式单图实验。主要功能：

- 选择并加载 `weights/lora/<checkpoint>`。
- 输入 prompt JSON，并 encode prompt。
- 上传单张 `seg`、`depth`、可选 `rgb reference`。
- 也可以扫描一个包含 `seg/ depth/ rgb/` 子目录的普通 dataset folder。
- 生成多个 seed 的 HDC2A + LoRA 结果。
- 可选生成 vanilla baseline：
  - `seg-only`：原版 Flux2 + Union ControlNet，只用 segmentation colorized control。
  - `depth-only`：原版 Flux2 + Union ControlNet，只用 depth colorized control。
- 显示并保存：seg preview、depth preview、HDC2A feature heatmap、ours per-seed grid、baseline、summary grid。

WebUI 里的 `Save run` 会写到：

```text
generation_pipeline/output/<run_name>/
├── inputs/
│   ├── seg_original.*
│   ├── depth_original.*
│   ├── rgb_reference.*
│   ├── seg_colorized.png
│   ├── depth_colorized.png
│   └── rgb_preview.png
├── ours/
│   ├── hdc2a_feature.png
│   └── seed_XXX.png
├── baseline_seg_only/
├── baseline_depth_only/
├── summary_grid.png
├── prompts/
│   ├── prompt.json
│   └── flat.txt
└── metadata.json
```

注意：WebUI 当前已有 `ours`、`HDC2A feature`、`seg-only baseline`、`depth-only baseline`。如果要论文里的 `without LoRA`，用下面的 compare mode。

## 入口 2：批量生成 OSM Output

脚本：

```bash
python generation_pipeline.py --input-dir ../osm_pipeline/output/omaha-984
```

这个入口用于真正批量产出 dataset-style 的 synth RGB。它每个 selected tile 只选一个 view：

- 不加 near-nadir 参数：使用 root/top view。
- `--near-nadir 2`：固定使用 `near-nadir-2`。
- `--near-nadir-random`：每个 tile 随机选择一个已发现的 `near-nadir-*`。

### 常用命令

默认 root/top view，所有 tile，两个 seed：

```bash
python generation_pipeline.py \
  --input-dir ../osm_pipeline/output/omaha-984 \
  --seed 0,42
```

固定第二个 near-nadir，只跑前 10 个 tile，用两个 GPU 并行：

```bash
python generation_pipeline.py \
  --input-dir ../osm_pipeline/output/omaha-984 \
  --near-nadir 2 \
  --batch 0,10 \
  --gpus 0,1 \
  --seed 1,2,42
```

随机 near-nadir，每个 tile 选择一个 view，选择可复现：

```bash
python generation_pipeline.py \
  --input-dir ../osm_pipeline/output/omaha-984 \
  --near-nadir-random \
  --random-seed 7 \
  --seed 0,42
```

使用 EXR depth：

```bash
python generation_pipeline.py \
  --input-dir ../osm_pipeline/output/omaha-984 \
  --depth-exr \
  --seed 42
```

先检查会跑哪些 item，不加载模型：

```bash
python generation_pipeline.py \
  --input-dir ../osm_pipeline/output/omaha-984 \
  --near-nadir 2 \
  --batch 0,10 \
  --dry-run
```

断点续跑：

```bash
python generation_pipeline.py \
  --input-dir ../osm_pipeline/output/omaha-984 \
  --near-nadir 2 \
  --batch 0,100 \
  --gpus 0,1 \
  --seed 1,2,42 \
  --skip-existing
```

### Seed 参数

推荐用逗号格式：

```bash
--seed 1,2,42
```

兼容旧格式：

```bash
--seeds 1 2 42
```

批量脚本会对每个 selected tile/view 生成所有 seed。多个 seed 是在同一次 `sample_ours` 调用里 batch 生成，不是每个 seed 单独加载模型。

### Tile 选择参数

```bash
--batch 0,10
```

表示按 tile folder 排序后选择半开区间 `[0, 10)`，也就是前 10 个 tile。

也可以指定 tile 名：

```bash
--tile-names tile_0001 tile_0002 tile_0100
```

`--tile-names` 和 `--batch` 同时出现时，代码先按 tile names 过滤，再做 batch range。

### 多 GPU 行为

```bash
--gpus 0,1
```

会启动两个 worker process，每个 worker 独占一个 `CUDA_VISIBLE_DEVICES`。tile 分配是取模分片：

```text
GPU 0 worker: selected tile index 0, 2, 4, 6, ...
GPU 1 worker: selected tile index 1, 3, 5, 7, ...
```

例如：

```bash
python generation_pipeline.py \
  --input-dir ../osm_pipeline/output/omaha-984 \
  --batch 0,10 \
  --gpus 0,1 \
  --near-nadir 2 \
  --seed 1,2,42
```

会让 GPU0 跑第 1/3/5/7/9 个 selected tile，GPU1 跑第 2/4/6/8/10 个 selected tile。每个 tile 内的 `1,2,42` 三个 seed 一次性 batch 生成。

### 批量输出结构

默认输出目录：

```text
generation_pipeline/output/osm_batch__<input-name>__<view>__depth-<png|exr>__<ckpt-name>/
```

例如：

```text
output/osm_batch__omaha-984__near-nadir-2__depth-png__checkpoint_epoch_0315/
```

每个 conditioning item 镜像成 `tile/view/depth_format/`，保证生成 RGB 和 OSM seg/depth 一一对应：

```text
output/.../
├── tile_0001/
│   └── near-nadir-2/
│       └── depth_png/
│           ├── rgb_seed_0001.png
│           ├── rgb_seed_0002.png
│           ├── rgb_seed_0042.png
│           ├── hdc2a_feature.png
│           ├── grid.png
│           └── metadata.json
├── tile_0002/
│   └── near-nadir-2/
│       └── depth_png/
│           └── ...
├── manifest.json               # 单 GPU/单 worker 时
├── manifest_shard00.json       # 多 GPU worker manifest
├── manifest_shard01.json
└── manifest_controller.json    # 多 GPU controller manifest
```

每个 `metadata.json` 包含：

```json
{
  "tile": "tile_0001",
  "view": "near-nadir-2",
  "depth_tag": "png",
  "seg_path": ".../tile_0001/near-nadir-2/1_seg.png",
  "depth_path": ".../tile_0001/near-nadir-2/2_depth.png",
  "rgb_ref_path": ".../tile_0001/2_rgb.png",
  "seeds": [1, 2, 42],
  "rgb_outputs": {
    "1": ".../rgb_seed_0001.png",
    "2": ".../rgb_seed_0002.png",
    "42": ".../rgb_seed_0042.png"
  },
  "feature_output": ".../hdc2a_feature.png",
  "grid_output": ".../grid.png",
  "num_steps": 28,
  "cfg": 3.5
}
```

Downstream 如果要严格对齐，建议读每个 sample 的 `metadata.json` 或顶层 manifest 里的 `items`。

`grid.png` 包含：

```text
Seg | Depth | HDC2A feature | seed=<seed1> | seed=<seed2> | ... | RGB ref
```

## 入口 3：论文展示图 Compare Mode

`app.py` 的 compare mode 用来做论文/汇报展示图。它不打开 WebUI，只处理少量你选中的 tile。为了防止误跑 984 个 tile，它要求必须显式选择 tile：

- `--compare-tile-names ...` 指定 tile。
- `--compare-random-tiles N` 随机抽 N 个 tile。
- `--compare-batch START,END` 选择排序后的一个小范围。

默认输出目录：

```text
generation_pipeline/output/<osm-folder-name>-compare/
```

例如输入 `../osm_pipeline/output/omaha-984`，默认输出就是：

```text
generation_pipeline/output/omaha-984-compare/
```

### Compare 常用命令

指定 tile，固定 near-nadir-2，单个 seed：

```bash
python app.py \
  --compare-osm-dir ../osm_pipeline/output/omaha-984 \
  --compare-tile-names tile_0001 tile_0002 \
  --compare-near-nadir 2 \
  --compare-seed 42
```

随机选 6 个 tile，root/top view，seed 42：

```bash
python app.py \
  --compare-osm-dir ../osm_pipeline/output/omaha-984 \
  --compare-random-tiles 6 \
  --compare-random-seed 7 \
  --compare-seed 42
```

选择前 10 个 tile 中的一段，用 EXR depth：

```bash
python app.py \
  --compare-osm-dir ../osm_pipeline/output/omaha-984 \
  --compare-batch 0,10 \
  --compare-depth-exr \
  --compare-seed 42
```

覆盖输出目录：

```bash
python app.py \
  --compare-osm-dir ../osm_pipeline/output/omaha-984 \
  --compare-random-tiles 4 \
  --compare-seed 42 \
  --compare-out output/my_paper_figures
```

### Compare 会输出哪些图

每个 tile/view/depth/seed 输出：

```text
output/omaha-984-compare/<tile>/<view>/depth_<png|exr>/seed_0042/
├── osm_rgb.png          # OSM pipeline 的 2_rgb.png reference
├── seg.png              # colorized segmentation
├── depth.png            # normalized/colorized depth
├── hdc2a_feature.png    # HDC2A control context heatmap
├── synth_rgb.png        # 正常 HDC2A + LoRA 结果
├── without_lora.png     # HDC2A 仍在，但 LoRA disabled 的结果
├── seg_only.png         # vanilla Flux2 + Union ControlNet, seg-only control
├── depth_only.png       # vanilla Flux2 + Union ControlNet, depth-only control
├── grid.png             # 横向对比图
└── metadata.json
```

`grid.png` 顺序：

```text
OSM sat RGB | Seg | Depth | HDC2A feature | Synth RGB | Without LoRA | Seg only | Depth only
```

### Compare 和批量生成的区别

`generation_pipeline.py` 适合大规模产出 synth dataset：

- 支持多个 seed。
- 支持多 GPU 分片。
- 输出每个 tile/view/depth 的 synth RGB、feature、grid、metadata。
- 不跑 expensive 的 baseline 对比。

`app.py --compare-osm-dir` 适合少量论文图：

- 只用一个 seed：`--compare-seed 42`。
- 会跑多个对比：normal synth、without LoRA、seg-only baseline、depth-only baseline。
- 不建议一次跑很多 tile，因为 baseline 会占显存，而且需要在 ours/baseline 之间切换模型。

## 常用参数速查

### `generation_pipeline.py`

```text
--input-dir PATH             OSM output folder，必填
--near-nadir N               固定使用 near-nadir-N
--near-nadir-random          每个 tile 随机选一个 near-nadir-*
--random-seed N              near-nadir random 的随机种子
--depth-exr                  使用 .exr depth，否则默认 .png
--batch START,END            选择排序后的 tile 半开区间
--tile-names ...             指定 tile folders
--gpus 0,1                   多 GPU worker 分片
--seed 1,2,42                逗号分隔 seeds，推荐
--seeds 1 2 42               空格分隔 seeds，兼容旧格式
--num-steps 28               Euler steps
--cfg 3.5                    guidance scale
--ckpt NAME                  weights/lora/ 下的 checkpoint 目录名
--prompt-json PATH           prompt JSON
--out PATH                   覆盖输出目录
--dry-run                    只打印任务，不加载模型
--skip-existing              输出已存在则跳过，断点续跑
--no-grids                   不保存 grid.png
--no-tiles                   不保存 rgb_seed_XXXX.png，只保存 feature/metadata/grid
```

### `app.py` compare mode

```text
--compare-osm-dir PATH             OSM output folder，启用 compare mode
--compare-tile-names ...           指定展示 tile
--compare-random-tiles N           随机抽 N 个展示 tile
--compare-batch START,END          选择排序后的 tile 范围
--compare-near-nadir N             固定 near-nadir-N
--compare-near-nadir-random        每个 tile 随机一个 near-nadir-*
--compare-random-seed N            tile/view random seed
--compare-depth-exr                使用 .exr depth
--compare-seed N                   单个展示 seed
--compare-steps 28                 steps
--compare-cfg 3.5                  guidance scale
--compare-ckpt NAME                checkpoint 目录名
--compare-prompt-json PATH         prompt JSON
--compare-out PATH                 覆盖输出目录
```

## Smoke Test

权重准备好后可以跑 smoke test：

```bash
SYNTHURBANSAT_GOLDEN_SET=../train_pipeline/dataset/val python _smoke_test.py
```

它会加载 checkpoint、encode prompt、生成一个 seed、跑两个 baseline，并保存到 `output/smoke_test/`。

## 注意事项

- `2_rgb.png` 只是 reference/展示，不作为生成 conditioning。
- 真正 conditioning 是 segmentation + depth。
- `without_lora.png` 不是“没有 HDC2A”，而是 HDC2A control 仍然存在、LoRA disabled。
- `seg_only.png` 和 `depth_only.png` 是 vanilla Flux2 + Union ControlNet baseline，不使用 HDC2A。
- baseline 模型很占显存；compare mode 内部会在 ours 和 baseline 之间切换模型，所以适合少量 tile。
- 多 GPU 批量生成时每个 worker 都会加载自己的模型副本；确保每张 GPU 显存足够。
- 如果需要最终 dataset 对齐，优先使用 `generation_pipeline.py` 的 per-sample `metadata.json` 或顶层 manifest。
- 如果只是挑论文展示图，优先使用 `app.py --compare-osm-dir`。

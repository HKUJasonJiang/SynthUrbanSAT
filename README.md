# SynthUrbanSAT

**中文摘要**：SynthUrbanSAT 面向遥感基础模型训练中配对数据稀缺、地域差异大、多样性不足的问题。项目将真实城市布局、程序化三维几何和生成式卫星影像连接起来：先从 OpenStreetMap 生成像素对齐的语义图、深度图和 3D 城市场景，再用 US3D 数据训练 FLUX.2-dev + ControlNet/LoRA，最后把 OSM 生成的 seg/depth 转换为伪真实卫星 RGB。

**English summary**: SynthUrbanSAT is a three-stage pipeline for scalable synthetic urban remote-sensing data. It decouples real-world layout fidelity from image realism: OSM provides geographically grounded urban layouts, the procedural pipeline produces aligned segmentation/depth/3D products, and the generation pipeline uses a trained FLUX.2-dev ControlNet/LoRA model to synthesize realistic satellite RGB images.

## Motivation

Large remote-sensing models need paired, diverse, and geographically varied data. Existing dataset construction has two bottlenecks:

- Manual annotation of RGB satellite images is expensive and slow.
- Pure procedural rendering with Blender or UE can generate labels, but the rendered RGB is often unrealistic and the procedural city layouts may not follow real-world urban distributions.

OpenStreetMap gives realistic city layout priors, but OSM vectors do not precisely correspond to real RGB satellite pixels. SynthUrbanSAT uses OSM for geometry and semantic/depth supervision, then learns the missing photorealistic RGB appearance with a conditional generative model.

## Repository Layout

```text
SynthUrbanSAT/
├── train_pipeline/       # FLUX.2-dev + ControlNet/LoRA training on US3D seg/depth/RGB
├── osm_pipeline/         # OSM bbox -> aligned 3D mesh, segmentation, depth, metadata
├── generation_pipeline/  # OSM seg/depth + trained checkpoint -> synthetic satellite RGB
├── README.md             # this overview
├── setup.sh              # thin dispatcher; each pipeline also has its own setup
└── .gitignore            # ignores weights, datasets, cache, outputs, secrets
```

## Pipeline Story

```text
Real city layout from OSM
        |
        v
osm_pipeline
  bbox -> OSM vectors -> 3D mesh / .glb / .blend
       -> aligned segment map + depth map + metadata
        |
        v
train_pipeline
  US3D RGB + segment + depth pairs
       -> train HDC2A + FLUX.2-dev ControlNet/LoRA
        |
        v
generation_pipeline
  OSM segment + depth + trained checkpoint
       -> pseudo-realistic satellite RGB
```

The final dataset keeps semantic and height correspondence from the procedural OSM branch while improving visual realism through conditional image generation.

## Quick Start

Each pipeline is intentionally independent because users may only need one stage.

```bash
# 1. Train FLUX.2-dev ControlNet/LoRA on US3D
cd train_pipeline
cp .env.example .env
bash setup.sh --test-both 0,1,2,3
bash run.sh

# 2. Generate OSM-derived seg/depth/3D products
cd ../osm_pipeline
bash setup.sh
python auto_pipeline.py --city omaha_test --bbox -96.135 41.260 -96.130 41.265 --vlm-mode skip --clean

# 3. Generate synthetic satellite RGB from seg/depth
cd ../generation_pipeline
bash setup.sh
python app.py
```

For a thin root-level dispatcher:

```bash
bash setup.sh train
bash setup.sh osm
bash setup.sh generation
```

## Data and Weights Policy

This repository is designed to stay lightweight on GitHub. The following are intentionally ignored and should be downloaded or generated locally:

- `**/weights/`: FLUX.2, VAE, text encoder, ControlNet, LoRA/HDC2A checkpoints.
- `**/dataset/`: US3D and other training datasets.
- `**/output/`: generated tiles, checkpoints, inference outputs, logs.
- `**/cache/`: intermediate OSM/geospatial products.
- `.env*`: HuggingFace and WandB tokens.

`generation_pipeline/setup.sh` can symlink base FLUX.2 weights from a local ComfyUI installation via `COMFY_MODELS`, while downloading LoRA/HDC2A checkpoints into the ignored `generation_pipeline/weights/` directory.

## Pipeline READMEs

- [train_pipeline/README.md](train_pipeline/README.md): training data, HDC2A architecture, LoRA/ControlNet training, multi-GPU commands.
- [osm_pipeline/README.md](osm_pipeline/README.md): OSM-to-3D generation, Blender requirements, tile outputs, CLI and WebUI.
- [generation_pipeline/README.md](generation_pipeline/README.md): weight setup, Gradio app, batch inference, smoke tests.

## Hardware Notes

- Training and generation are GPU-heavy. The default experiments target A100/H100/H200-class GPUs.
- OSM processing can run on CPU, but Blender 4.0+ must be available on `PATH` for 3D assembly and rendering.
- The generation WebUI needs local model weights and enough VRAM to load FLUX.2-dev and ControlNet components.

## Citation

Citation information will be added after the paper is released.

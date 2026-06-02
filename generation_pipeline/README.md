# Generation Pipeline: Seg/Depth to Synthetic Satellite RGB

This pipeline is the final stage of SynthUrbanSAT. It loads OSM-derived segmentation and depth maps, applies the trained HDC2A + FLUX.2-dev ControlNet/LoRA checkpoint from the training pipeline, and generates pseudo-realistic satellite RGB images.

It can run as an interactive Gradio WebUI or as a batch folder inference script.

## Quick Start

```bash
cd generation_pipeline

# Optional: point to an existing ComfyUI model folder for base FLUX.2 weights.
export COMFY_MODELS=/path/to/ComfyUI/models

# Downloads/symlinks ignored local weights into generation_pipeline/weights/.
bash setup.sh

conda activate flux_train
python app.py
```

The app opens on `http://<host>:7860` by default.

## Weight Layout

`setup.sh` populates the ignored `weights/` directory:

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

Base weights are symlinked by default from `COMFY_MODELS` when present. Pass `--copy` if you need a physical copy. LoRA/HDC2A checkpoints are downloaded from HuggingFace into `weights/lora/`.

```bash
bash setup.sh              # latest/default checkpoint only
bash setup.sh --copy       # copy base weights instead of symlinking
bash setup.sh --all-ckpts  # download all configured LoRA checkpoints
```

Tokens are read from `HF_TOKEN`, `generation_pipeline/.env`, or `../train_pipeline/.env`.

## Inputs

The pipeline accepts a segment map, a depth map, and an optional RGB reference. The WebUI can scan a parent folder with `seg/`, `depth/`, and `rgb/` subfolders, or you can upload files manually.

Typical upstream input comes from:

```text
../osm_pipeline/output/<city>/tile_XXXX/
├── 3_seg.png
├── 4_depth.png
├── 5_depth.exr
└── 2_rgb.png      # optional reference, not used as conditioning
```

## Batch Inference

```bash
python generation_pipeline.py \
  --input-dir ../osm_pipeline/output/omaha_test \
  --depth-ext both \
  --seeds 0 42
```

Outputs are written to `output/`, which is ignored by Git.

## Smoke Test

After weights are available:

```bash
SYNTHURBANSAT_GOLDEN_SET=../train_pipeline/dataset/val python _smoke_test.py
```

The smoke test loads a checkpoint, encodes a prompt, generates one seed, runs both baselines, and saves `output/smoke_test/`.

## Implementation Notes

- Path constants live in [pipeline/__init__.py](pipeline/__init__.py).
- The pipeline reuses training modules from `../train_pipeline` through `sys.path`.
- Base and trained checkpoints are never committed to Git.
- `app.py --no-preload` starts the UI without immediately loading the first checkpoint.

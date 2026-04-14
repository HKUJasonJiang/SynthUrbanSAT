# HDC²A + Flux2 ControlNet Training

Fine-tune a Flux2 ControlNet with a custom **Heterogeneous Dual-Condition Adapter (HDC²A)** that generates RGB satellite images from segmentation + depth map pairs, conditioned on Mistral-3.1 text embeddings.

---

```bash
```

## How to Run

### First Time (New Server)

```bash
git clone https://github.com/<YOUR_USERNAME>/HDC2A_training.git
cd HDC2A_training
bash setup.sh
```

This will:
1. Install Miniconda (if not present)
2. Create the `flux_train` conda environment (from `environment.yml`)
3. Download dataset from HuggingFace (`JasonXF/US3D-Enhanced`)
4. Download model weights from HuggingFace (`JasonXF/SynthUrbanSAT`)
5. Verify all dependencies (PyTorch, CUDA, diffusers, wandb…)
6. Run a smoke test (load models → forward → backward → gradient check)

Then start training:

```bash
bash run_train.sh
```

### Subsequent Runs

Edit the variables at the top of **`run_train.sh`** (learning rate, epochs, batch size, etc.), then:

```bash
bash run_train.sh
```

That's it. All configuration lives in `run_train.sh`.

### Testing

```bash
# Smoke test — random data, no dataset needed
bash run_train.sh --test --no-wandb

# Test with real dataset — loads data, runs 1 training step
bash run_train.sh --test-data --no-wandb
```

Test output is auto-saved to `output/test_<timestamp>.log`.

### Resume Training

```bash
# Option A: set RESUME in run_train.sh
RESUME="output/checkpoint_epoch_0010"

# Option B: pass via CLI
bash run_train.sh --resume output/checkpoint_epoch_0010
```

### Override Any Parameter from CLI

```bash
bash run_train.sh --lr 5e-6 --batch-size 2 --num-epochs 50
```

---

## File Structure

```
HDC2A_training/
├── run_train.sh          ← Edit this for hyperparameters
├── setup.sh              ← First-time setup (env + deps + test)
├── train_script.py       ← Python entry point (called by run_train.sh)
├── environment.yml       ← Conda env definition
├── requirements.txt      ← pip fallback
│
├── scripts/              ← All Python modules
│   ├── models.py             HDC²A adapter + FP8 utilities
│   ├── train.py              Train / validate / test / checkpoint
│   ├── dataprep.py           Dataset & dataloaders
│   ├── utility.py            Memory helpers, model loading
│   ├── text_encoder.py       Mistral-3.1 precompute pipeline
│   └── colors.py             ANSI terminal colors
│
├── models/videox_fun/    ← VideoX-Fun model files (local copy)
├── configs/color_map.json
├── weights/              ← Model weights (symlinks)
├── dataset/{train,val}/  ← Your data
└── output/               ← Checkpoints & logs
```

### Which Files Do I Edit?

| File | Purpose |
|------|---------|
| **`run_train.sh`** | All hyperparameters — learning rate, epochs, batch size, paths |
| **`setup.sh`** | Only for first-time setup or after changing `environment.yml` |

Everything else is internal — you don't need to touch `train_script.py` or anything in `scripts/`.

---

## Architecture

### HDC²A Adapter

```
seg  [B, H, W]     → SemanticEncoder → T_s [B, 4096, 768] ─┐
                                                              ├→ 3× DoubleStreamFusion
depth [B, 1, H, W] → DepthEncoder    → T_d [B, 4096, 768] ─┘
                                                              │
                                             GatedMerge + AvgPool → [B, 1024, 3072]
```

### Training Pipeline

| Component | Status | Params |
|-----------|--------|--------|
| Flux2 Transformer backbone | Frozen (FP8) | ~12B |
| ControlNet control blocks | **Trainable** | 4133.5M |
| HDC²A Adapter | **Trainable** | 52.4M |

- Loss: flow matching (velocity prediction)
- Optimizer: AdamW, effective BS = `batch_size × grad_accum_steps`
- VRAM: ~56 GB (bs=1), fits A100 80 GB
- Metrics → **WandB** with real-time progress %

---

## Dataset Format

Place data in `dataset/train/` and `dataset/val/` with this layout:

```
train/
├── rgb/         *.png  (target RGB, 512×512)
├── seg/         *.png  (color-coded segmentation)
├── depth/       *.png  (grayscale depth, normalised to [0,1])
└── captions.json       {"filename.png": "A satellite image of ...", ...}
```

Filenames must match across `rgb/`, `seg/`, `depth/`.

### Segmentation Classes

| ID | Class    | RGB |
|----|----------|-----|
| 0  | Building | (128, 0, 0) |
| 1  | Road     | (128, 128, 128) |
| 2  | Water    | (0, 0, 128) |
| 3  | Foliage  | (0, 128, 0) |
| 4  | Grass    | (128, 128, 0) |

---

## Configuration Reference

All variables in `run_train.sh` (also overridable via `--flag value`):

| Variable | Default | CLI Flag |
|----------|---------|----------|
| `TRANSFORMER_PATH` | `weights/flux2_dev_fp8mixed.safetensors` | `--transformer-path` |
| `VAE_PATH` | `weights/flux2-vae.safetensors` | `--vae-path` |
| `CONTROLNET_PATH` | `weights/FLUX.2-dev-Fun-...` | `--controlnet-path` |
| `DATASET_DIR` | `dataset` | `--dataset-dir` |
| `OUTPUT_DIR` | `output` | `--output-dir` |
| `TEXT_ENCODER_PATH` | *(empty)* | `--text-encoder-path` |
| `PRECOMPUTED_EMBEDDINGS` | *(empty)* | `--precomputed-embeddings` |
| `IMAGE_SIZE` | `512` | `--image-size` |
| `NUM_EPOCHS` | `100` | `--num-epochs` |
| `BATCH_SIZE` | `1` | `--batch-size` |
| `LEARNING_RATE` | `1e-5` | `--lr` |
| `GRAD_ACCUM_STEPS` | `4` | `--grad-accum-steps` |
| `MAX_GRAD_NORM` | `1.0` | `--max-grad-norm` |
| `SEED` | `42` | `--seed` |
| `RESUME` | *(empty)* | `--resume` |

## Checkpoint Layout

```
output/checkpoint_epoch_XXXX/
├── hdc2a.pt           # HDC²A adapter weights
├── control_params.pt  # ControlNet trainable params
├── optimizer.pt       # Optimizer state
└── meta.pt            # {epoch, loss, config}
```

## Deploy to Another Machine

```bash
scp -r HDC2A_training/ user@host:/path/
# On the new machine:
bash setup.sh        # creates env, checks deps, tests
bash run_train.sh    # start training
```

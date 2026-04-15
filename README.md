# HDC²A + Flux.2 ControlNet Training

Fine-tune a Flux.2 ControlNet with a custom **Heterogeneous Dual-Condition Adapter (HDC²A)** that generates RGB satellite images from segmentation + depth map pairs, conditioned on text embeddings.

<p align="center">
  <img src="docs/architecture.png" width="90%"/>
</p>

## Quick Start

```bash
# 1. Clone & setup (one-click: installs env, downloads data & weights, runs smoke test)
git clone https://github.com/HKUJasonJiang/SynthUrbanSAT.git
cd SynthUrbanSAT
bash setup.sh

# 2. Train (single GPU)
bash run_train.sh

# 3. Train (multi-GPU DDP)
GPUS=0,1,2,3 bash run_train.sh
```

### Test Without Training

```bash
bash run_train.sh --test --no-wandb        # smoke test (random data)
bash run_train.sh --test-data --no-wandb   # 1 epoch with real data
```

### Override Hyperparameters

```bash
bash run_train.sh --lr 5e-6 --batch-size 3 --num-epochs 200 --lora-rank 256
```

### Resume from Checkpoint

```bash
bash run_train.sh --resume output/checkpoint_epoch_0010
```

---

## Overview

| Component | Status | Params |
|-----------|--------|--------|
| Flux.2 Transformer backbone | Frozen (FP8) | ~12B |
| ControlNet control blocks + LoRA | **Trainable** | ~4.1B + 9.8M |
| HDC²A Adapter | **Trainable** | 52.4M |

- **Input**: segmentation map + depth map + text prompt
- **Output**: photorealistic satellite RGB image (512×512)
- **Loss**: flow matching (velocity prediction) with min-SNR weighting
- **LoRA**: ON by default (rank=32); disable with `--no-lora`

---

## Project Structure

```
├── run_train.sh        ← Hyperparameters & launch config (edit this)
├── setup.sh            ← One-click environment setup
├── train_script.py     ← Training entry point
├── upload.sh           ← Push results to HuggingFace
├── scripts/            ← Training modules (models, data, utils)
├── docs/
│   ├── setup_server.md     ← Server deployment & multi-GPU guide
│   └── ARCHITECTURE.md     ← Detailed architecture & tensor shapes
└── output/             ← Checkpoints & logs
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| **[Server Setup & Multi-GPU Guide](docs/setup_server.md)** | Full server deployment, DDP experiments, tmux workflow, HF upload |
| **[Architecture Details](docs/ARCHITECTURE.md)** | HDC²A internals, tensor dimensions, VRAM profiling, model loading sequence |

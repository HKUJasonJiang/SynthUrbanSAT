# Server Setup Guide

## 1. Install Git & tmux

Make sure you create a user in the linux by `sudo adduser [username]` and add sudo permission `sudo usermod -aG sudo <user_name>`.

```bash
sudo apt update && sudo apt install -y git tmux
```

## 2. Clone Repo

```bash
git clone https://github.com/HKUJasonJiang/SynthUrbanSAT.git
cd SynthUrbanSAT
```

## 3. Configure HF Tokens

create a `.env` file OR replace with the given one:
```bash
HF_TOKEN_READ=<your_hf_read_token>
HF_TOKEN_WRITE=<your_hf_write_token>
WANDB_API_KEY=<your_wandb_api_key>
```

## 4. Run Setup (one-click)

Simply run this to test a multi-gpu machine, if it works, it means all things are ready to go:

```bash
tmux new-session -s setup 'bash setup.sh --test-both 0,1,2,3'
```

Other test code (if --test-both works, skip)
```bash
tmux new-session -s setup 'bash setup.sh'                          # full setup + single-GPU smoke test
tmux new-session -s setup 'bash setup.sh --skip-test'              # skip smoke test, just running setup
tmux new-session -s setup 'bash setup.sh --test-single-gpu 0'      # single-GPU test on GPU 0
tmux new-session -s setup 'bash setup.sh --test-multi-gpu 0,1,2,3' # multi-GPU (DDP) test on 4 GPUs
```

> Note: or you can run the old-fasion way to install dependencies via `conda init && conda create -n flux_train python=3.12` and `pip install -r requirements.txt`

## 5. Train (Single GPU)

default running (setup the config inside):

```bash
bash run_train.sh
```

OR (different config):
```bash
tmux new-session -s train 'python train_script.py --seed 42 --name lora_baseline_2 --num_epochs 500'
```

### GPU Usage

| VRAM (GiB) | Recommended Batch_Size | Notes |
|---|---:|---|
| 80 (A100) | 3 | common production setting |
| 96 (Pro6000/h100) | 4 | usually needs gradient checkpointing + no extra overhead |
| 140 (h200) | 8 |  126244MiB / 143771MiB |


> Assumption: image_size=1024, dtype=bfloat16, grad_accum_steps=4, freeze_controlnet_backbone=True.
> If OOM happens, reduce `--batch-size` first; keep effective batch size via higher `--grad-accum-steps`.


## 6. Multi-GPU Training (DDP)

Training supports multi-GPU via PyTorch DDP (`torchrun`).

<!-- - **Single GPU**: `python train_script.py --name ... `
- **Multi-GPU**: `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_script.py --name ...`
- Concurrent multi-GPU experiments on the same node must use **different `--master_port`** values (default: 29500)
- Single-GPU experiments don't need `torchrun` or `--master_port`

| GPUs | Launcher | Effective batch size |
|---|---|---|
| 1 GPU | `python train_script.py` | `batch_size × grad_accum` |
| N GPUs | `torchrun --nproc_per_node=N` | `batch_size × grad_accum × N` |

### Ablation CLI Flags

| Flag | Effect |
|---|---|
| `--disable-depth` | Zero out depth input; HDC²A runs seg-only |
| `--no-minsnr` | Disable min-SNR loss weighting (uniform w=1) |
| `--use-lora` | Inject LoRA into ControlNet control blocks |
| `--lora-rank R` | LoRA rank (default 32) |
| `--lora-alpha A` | LoRA alpha (default 32.0) | -->

### Experiment Plan (8×A100 80GB)

| # | Name | GPUs | batch_size | Effective BS | Special Args |
|---|---|---|---|---|---|
| 1 | `lora_baseline_4A100_main` | 0,1,2,3 | 3 | 3×4×4=48 | adapter_lr=1.2e-3 (3e-4 × 4 linear scaling) |
| 2 | `abl_seg_only_2A100` | 4,5 | 3 | 3×4×2=24 | `--disable-depth` |
| 3 | `lora_rank_256_1A100` | 6 | 3 | 3×4=12 | `--lora-rank 256` |
| 4 | `abl_time_1A100` | 7 | 3 | 3×4=12 | `--no-minsnr` |

### Launch Commands

All 4 experiments run concurrently in separate tmux sessions. Copy-paste these commands:

```bash
# ── Exp 1: Main baseline, 4×A100 ─────────────────────────────────────────────
tmux new-session -d -s exp1 '
cd ~/SynthUrbanSAT &&
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ~/miniconda/envs/flux_train/bin/torchrun \
    --nproc_per_node=4 --master_port=29500 \
  train_script.py \
    --name lora_baseline_4A100_main \
    --batch-size 3 \
    --adapter-lr 8e-4 \
    --seed 42
'

# ── Exp 2: Seg-only ablation, 2×A100 ─────────────────────────────────────────
tmux new-session -d -s exp2 '
cd ~/SynthUrbanSAT &&
CUDA_VISIBLE_DEVICES=4,5 \
  ~/miniconda/envs/flux_train/bin/torchrun \
    --nproc_per_node=2 --master_port=29501 \
  train_script.py \
    --name abl_seg_only_2A100 \
    --batch-size 3 \
    --disable-depth \
    --seed 42
'

# ── Exp 3: LoRA rank 256, 1×A100 ─────────────────────────────────────────────
tmux new-session -d -s exp3 '
cd ~/SynthUrbanSAT &&
CUDA_VISIBLE_DEVICES=6 python train_script.py \
    --name lora_rank_256_1A100 \
    --batch-size 3 \
    --lora-rank 256 \
    --seed 42
'

# ── Exp 4: Uniform timestep weight, 1×A100 ───────────────────────────────────
tmux new-session -d -s exp4 '
cd ~/SynthUrbanSAT &&
CUDA_VISIBLE_DEVICES=7 python train_script.py \
    --name abl_time_1A100 \
    --batch-size 3 \
    --no-minsnr \
    --seed 42
'
```

### *Monitoring & tmux Basics*

```bash
tmux ls                       # list all running sessions
tmux attach -t exp1           # attach to experiment 1
# Ctrl-B D                   — detach (session keeps running)

watch -n 1 nvidia-smi         # continuously monitor GPU utilization
tail -f output/lora_baseline_4A100_main/train.log   # follow log
```

### *Cleanup tmux Sessions*

```bash
tmux kill-session -t setup    # kill a specific session (e.g. setup)
tmux kill-session -t exp1     # kill experiment 1
tmux kill-session -t exp2     # kill experiment 2

tmux kill-server              # kill ALL tmux sessions at once (nuclear option)
```

> **Notes:**
> - A100 80GB → `--batch-size 3` is safe; default `grad_accum_steps=4`.
> - Checkpoints and logs go to `output/<NAME>/`, WandB project defaults to `<NAME>`.
> - Only rank 0 saves checkpoints and logs to WandB.
> - LR linear scaling: when using N GPUs, multiply base LR by N (e.g. 3e-4 × 4 = 1.2e-3).

## 7. (Optional) Push Results to HuggingFace

Upload a specific experiment's output folder:

```bash
bash upload.sh --name lora_baseline_4A100_main       # upload output/lora_baseline_4A100_main/
bash upload.sh --name abl_seg_only_2A100             # upload output/abl_seg_only_2A100/
bash upload.sh --name lora_rank_256_1A100            # upload output/lora_rank_256_1A100/
bash upload.sh --name abl_time_1A100                 # upload output/abl_time_1A100/
```

> Token is read from `.env` (`HF_TOKEN_WRITE`). Run without args to see available experiments.

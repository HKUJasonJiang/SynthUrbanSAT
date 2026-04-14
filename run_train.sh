#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# HDC²A + Flux2 ControlNet  —  Training Launch Script
#
# Usage:
#   bash run_train.sh                    # train with settings below
#   bash run_train.sh --test             # smoke test (forward only)
#   bash run_train.sh --test-data        # 1-step data test
#   bash run_train.sh --overfit          # overfit sanity check
#   bash run_train.sh --no-wandb         # disable WandB
#
# All defaults below match CONFIG in train_script.py.
# --name is REQUIRED: output goes to output/<NAME>/, WandB project = <NAME>.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Conda environment ────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda}"
ENV_NAME="flux_train"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Run: bash setup_and_run.sh   to create the '$ENV_NAME' environment first."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce VRAM fragmentation
export NO_COLOR=1  # prevent rich/wandb service from sending terminal probe sequences (OSC11, DA1)

# ── Load secrets from .env (WANDB_API_KEY, HF tokens, etc.) ─────────────────
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a  # auto-export
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ══════════════════════════  EDIT BELOW  ══════════════════════════════════════

# ── Run name (required, maps to output dir + WandB project) ──────────────────
NAME="hdc2a_run"                 # output/<NAME>/  ;  --wandb-project defaults to this

# ── Paths ────────────────────────────────────────────────────────────────────
TRANSFORMER_PATH="weights/flux2_dev_fp8mixed.safetensors"
VAE_PATH="weights/flux2-vae.safetensors"
CONTROLNET_PATH="weights/FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors"
DATASET_DIR="dataset"
COLOR_MAP_PATH="configs/color_map.json"
TEXT_ENCODER_PATH="weights/mistral_3_small_flux2_fp8.safetensors"
PRECOMPUTED_EMBEDDINGS="output/text_embeddings_global.pt"

# ── Model ────────────────────────────────────────────────────────────────────
IMAGE_SIZE=1024
NUM_CLASSES=6
CONTROL_IN_DIM=3072
FUSION_DIM=768
NUM_FUSION_BLOCKS=3
NUM_HEADS=12
NUM_FOURIER_BANDS=32
BOUNDARY_THRESHOLD=0.1

# ── Training ─────────────────────────────────────────────────────────────────
NUM_EPOCHS=500
BATCH_SIZE=4
ADAPTER_LR=3e-4                  # HDC²A adapter learning rate
BACKBONE_LR=0.0                  # ControlNet backbone LR (0 = frozen)
FREEZE_BACKBONE=true             # true = only train 52M adapter
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
GRAD_ACCUM_STEPS=4               # effective_bs = BATCH_SIZE × GRAD_ACCUM_STEPS
GUIDANCE_SCALE=3.5
NUM_WORKERS=0
SEED=42                         # empty = no seed; set e.g. 42 for reproducibility

# ── Text ─────────────────────────────────────────────────────────────────────
TEXT_SEQ_LEN=512
TEXT_DIM=15360

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_INTERVAL=10
SAVE_EVERY_N_EPOCHS=5
VAL_EVERY_N_EPOCHS=1

# ── WandB ────────────────────────────────────────────────────────────────────
WANDB_ENTITY=""                 # empty = personal namespace of the logged-in user (safest default)
WANDB_PROJECT=""                 # empty = use NAME as project

# ── Resume ───────────────────────────────────────────────────────────────────
RESUME=""                        # e.g. "output/hdc2a_run/checkpoint_epoch_0027"

# ── Data augmentation ────────────────────────────────────────────────────────
AUGMENT=false                    # true = --augment, false = --no-augment

# ══════════════════════════  END CONFIG  ══════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════════

CMD=(
    "$PYTHON" -u train_script.py
    --name "$NAME"
    --transformer-path "$TRANSFORMER_PATH"
    --vae-path "$VAE_PATH"
    --controlnet-path "$CONTROLNET_PATH"
    --dataset-dir "$DATASET_DIR"
    --color-map-path "$COLOR_MAP_PATH"
    --image-size "$IMAGE_SIZE"
    --num-classes "$NUM_CLASSES"
    --control-in-dim "$CONTROL_IN_DIM"
    --fusion-dim "$FUSION_DIM"
    --num-fusion-blocks "$NUM_FUSION_BLOCKS"
    --num-heads "$NUM_HEADS"
    --num-fourier-bands "$NUM_FOURIER_BANDS"
    --boundary-threshold "$BOUNDARY_THRESHOLD"
    --num-epochs "$NUM_EPOCHS"
    --batch-size "$BATCH_SIZE"
    --adapter-lr "$ADAPTER_LR"
    --weight-decay "$WEIGHT_DECAY"
    --max-grad-norm "$MAX_GRAD_NORM"
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
    --guidance-scale "$GUIDANCE_SCALE"
    --num-workers "$NUM_WORKERS"
    --text-seq-len "$TEXT_SEQ_LEN"
    --text-dim "$TEXT_DIM"
    --log-interval "$LOG_INTERVAL"
    --save-every-n-epochs "$SAVE_EVERY_N_EPOCHS"
    --val-every-n-epochs "$VAL_EVERY_N_EPOCHS"
)

# Only pass --wandb-entity if non-empty (empty = use logged-in user's personal namespace)
[[ -n "$WANDB_ENTITY" ]] && CMD+=(--wandb-entity "$WANDB_ENTITY")

# Backbone LR (only pass when > 0, which also implies unfreeze)
if [[ "$FREEZE_BACKBONE" == "true" ]]; then
    CMD+=(--freeze-backbone)
else
    CMD+=(--unfreeze-backbone)
    [[ -n "$BACKBONE_LR" && "$BACKBONE_LR" != "0" && "$BACKBONE_LR" != "0.0" ]] \
        && CMD+=(--backbone-lr "$BACKBONE_LR")
fi

# Data augmentation
if [[ "$AUGMENT" == "true" ]]; then
    CMD+=(--augment)
fi

# Optional args (only add if non-empty)
[[ -n "$TEXT_ENCODER_PATH" ]]       && CMD+=(--text-encoder-path "$TEXT_ENCODER_PATH")
[[ -n "$PRECOMPUTED_EMBEDDINGS" ]]  && CMD+=(--precomputed-embeddings "$PRECOMPUTED_EMBEDDINGS")
[[ -n "$RESUME" ]]                  && CMD+=(--resume "$RESUME")
[[ -n "$SEED" ]]                    && CMD+=(--seed "$SEED")
[[ -n "$WANDB_PROJECT" ]]           && CMD+=(--wandb-project "$WANDB_PROJECT")

# Pass through any extra CLI args (e.g. --test, --no-wandb, --overfit)
CMD+=("$@")

# ── Detect test mode → auto-tee to log ──────────────────────────────────────
IS_TEST=false
for a in "$@"; do
    [[ "$a" == "--test" || "$a" == "--test-data" ]] && IS_TEST=true
done

OUTPUT_DIR="output/$NAME"

echo "═══════════════════════════════════════════════════════════════"
echo "  HDC²A Training  |  $(date)"
echo "  Python: $PYTHON"
echo "  Name:   $NAME"
echo "  Mode:   $(if $IS_TEST; then echo 'TEST'; else echo 'TRAIN'; fi)"
echo "═══════════════════════════════════════════════════════════════"

if $IS_TEST; then
    # Test mode: tee output to timestamped log
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="$OUTPUT_DIR/test_$(date +%Y%m%d_%H%M%S).log"
    echo "  Log:    $LOG_FILE"
    echo "═══════════════════════════════════════════════════════════════"
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    # Drain any terminal query responses (OSC11/DA1) buffered in TTY stdin
    "$PYTHON" -c "import sys,termios; termios.tcflush(0,termios.TCIFLUSH)" 2>/dev/null || true
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo -e "  \033[1;32m✓ TEST PASSED\033[0m"
    else
        echo -e "  \033[1;31m✗ TEST FAILED (exit code $EXIT_CODE)\033[0m"
    fi
    echo "  Log saved: $LOG_FILE"
    echo "═══════════════════════════════════════════════════════════════"
    exit $EXIT_CODE
else
    echo "═══════════════════════════════════════════════════════════════"
    "${CMD[@]}"
    # Drain any terminal query responses buffered in TTY stdin
    "$PYTHON" -c "import sys,termios; termios.tcflush(0,termios.TCIFLUSH)" 2>/dev/null || true
fi

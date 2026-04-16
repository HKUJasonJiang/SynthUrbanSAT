#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# HDC²A + Flux2 ControlNet  —  Launch All Experiments
#
# Launches 4 ablation experiments concurrently in tmux sessions.
# Designed for an 8×A100 (80GB) machine.
#
# Prerequisites:
#   1. bash setup.sh --test-both 0,1,2,3   (all green ✓)
#   2. .env file with HF_TOKEN_READ, HF_TOKEN_WRITE, WANDB_API_KEY
#
# Usage:
#   bash run.sh                # launch all 4 experiments
#   bash run.sh --dry-run      # print commands without launching
#
# Monitor:
#   tmux ls                    # list running sessions
#   tmux attach -t exp1        # attach to experiment 1 (Ctrl-B D to detach)
#   watch -n 1 nvidia-smi      # GPU utilization
#
# Stop:
#   tmux kill-session -t exp1  # stop one experiment
#   tmux kill-server           # stop ALL experiments
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── CUDA library path (needed when conda env doesn't expose libcuda.so) ──────
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# ── Config ───────────────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda}"
ENV_NAME="flux_train"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
TORCHRUN="$CONDA_BASE/envs/$ENV_NAME/bin/torchrun"
HF_REPO="JasonXF/SynthUrbanSAT-Output"
SEED=42

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${CYAN}[INFO]${RESET} $*"; }
ok()    { echo -e "${GREEN}[  OK]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET} $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET} $*"; }

# ── Parse args ───────────────────────────────────────────────────────────────
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        *)         shift ;;
    esac
done

# ── Pre-flight checks ───────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  HDC²A Experiment Launcher${RESET}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"

# Check python
if [[ ! -x "$PYTHON" ]]; then
    fail "Python not found at $PYTHON"
    fail "Run: bash setup.sh  first."
    exit 1
fi

# Check tmux
if ! command -v tmux &>/dev/null; then
    fail "tmux is required. Install with: sudo apt install -y tmux"
    exit 1
fi
ok "tmux available"

# Check torchrun
if [[ ! -x "$TORCHRUN" ]]; then
    fail "torchrun not found at $TORCHRUN"
    exit 1
fi
ok "torchrun available"

# Check GPU count
NUM_GPUS=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [[ "$NUM_GPUS" -lt 8 ]]; then
    warn "This machine has $NUM_GPUS GPU(s). The experiment plan requires 8 GPUs."
    warn "Experiments will be adjusted to fit available GPUs."
fi

# Show GPU info
"$PYTHON" -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    vram = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'    GPU {i}: {name}  ({vram:.1f} GB)')
"

# Check .env
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    fail ".env not found. Copy .env.example to .env and fill in tokens."
    exit 1
fi
ok ".env found"

# Check WandB key
if grep -q 'WANDB_API_KEY=.\+' "$SCRIPT_DIR/.env" 2>/dev/null; then
    ok "WANDB_API_KEY set in .env"
else
    warn "WANDB_API_KEY not set in .env — WandB logging may not work"
fi

# Check HF write token
if grep -q 'HF_TOKEN_WRITE=.\+' "$SCRIPT_DIR/.env" 2>/dev/null; then
    ok "HF_TOKEN_WRITE set in .env"
else
    warn "HF_TOKEN_WRITE not set — auto-upload to HuggingFace will fail"
fi

echo ""

# ── Load .env for this shell ─────────────────────────────────────────────────
set -a
source "$SCRIPT_DIR/.env"
set +a

# ── Experiment definitions ───────────────────────────────────────────────────
# Format: session_name | gpus | nproc | master_port | train_args
# Note: keep these 4 experiments aligned with README / setup_server plan.
declare -a EXPERIMENTS=(
    # Exp 1: Main baseline, 4×A100
    "exp1|0,1,2,3|4|29500|--name lora_baseline_4A100_main --batch-size 12 --adapter-lr 8e-4 --hf-repo ${HF_REPO} --seed ${SEED}"

    # Exp 2: Seg-only ablation, 2×A100
    "exp2|4,5|2|29501|--name abl_seg_only_2A100 --batch-size 6 --disable-depth --hf-repo ${HF_REPO} --seed ${SEED}"

    # Exp 3: LoRA rank 256, 1×A100
    "exp3|6|1|0|--name lora_rank_256_1A100 --batch-size 3 --lora-rank 256 --hf-repo ${HF_REPO} --seed ${SEED}"

    # Exp 4: Uniform timestep weight, 1×A100
    "exp4|7|1|0|--name abl_time_1A100 --batch-size 3 --no-minsnr --hf-repo ${HF_REPO} --seed ${SEED}"
)

# ── Launch ───────────────────────────────────────────────────────────────────
LAUNCHED=0
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r SESSION GPUS NPROC PORT TRAIN_ARGS <<< "$exp"

    # Parse --name value for readable logs / tmux-exists checks
    NAME=$(echo "$TRAIN_ARGS" | sed -n 's/.*--name \([^ ]*\).*/\1/p')
    [[ -z "$NAME" ]] && NAME="$SESSION"

    # Check if session already exists
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        warn "tmux session '$SESSION' already running — skipping $NAME"
        warn "  Kill it first: tmux kill-session -t $SESSION"
        continue
    fi

    # Check if requested GPUs are available
    MAX_GPU=$(echo "$GPUS" | tr ',' '\n' | sort -n | tail -1)
    SKIP_DUE_TO_GPU=false
    if [[ "$MAX_GPU" -ge "$NUM_GPUS" ]]; then
        warn "Skipping $NAME — needs GPU $MAX_GPU but only $NUM_GPUS GPU(s) available"
        SKIP_DUE_TO_GPU=true
    fi

    # Build command (equivalent to the 4 explicit tmux commands in docs)
    if [[ "$NPROC" -gt 1 ]]; then
        CMD="cd $SCRIPT_DIR && source $SCRIPT_DIR/.env && CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --nproc_per_node=$NPROC --master_port=$PORT train_script.py $TRAIN_ARGS"
    else
        CMD="cd $SCRIPT_DIR && source $SCRIPT_DIR/.env && CUDA_VISIBLE_DEVICES=$GPUS $PYTHON train_script.py $TRAIN_ARGS"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${CYAN}[$SESSION]${RESET} $NAME (GPUs: $GPUS)"
        echo "  $CMD"
        if [[ "$SKIP_DUE_TO_GPU" == "true" ]]; then
            echo "  [dry-run note] this experiment would be skipped on this machine"
        fi
        echo ""
    else
        if [[ "$SKIP_DUE_TO_GPU" == "true" ]]; then
            continue
        fi
        info "Launching $SESSION: $NAME (GPUs: $GPUS, ${NPROC} proc)"
        tmux new-session -d -s "$SESSION" "$CMD"
        ok "$SESSION launched in tmux"
        LAUNCHED=$((LAUNCHED + 1))
    fi
done

echo ""
if [[ "$DRY_RUN" == "true" ]]; then
    info "Dry run — no sessions launched. Remove --dry-run to launch."
else
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
    ok "$LAUNCHED experiment(s) launched"
    echo ""
    echo "  tmux ls                       # list sessions"
    echo "  tmux attach -t exp1           # attach to exp1 (Ctrl-B D to detach)"
    echo "  watch -n 1 nvidia-smi         # monitor GPU usage"
    echo "  tmux kill-server              # stop ALL experiments"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
fi

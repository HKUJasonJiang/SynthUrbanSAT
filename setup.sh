#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# HDC²A + Flux2 ControlNet  —  One-Click Environment Setup
#
# This script performs:
#   1. Install Miniconda (if not present)
#   2. Create conda environment (flux_train)
#   3. Download dataset from HuggingFace (JasonXF/US3D-Enhanced)
#   4. Download model weights from HuggingFace (JasonXF/SynthUrbanSAT)
#   5. Verify all dependencies
#   6. Run smoke test
#
# Tokens:
#   Create a .env file (see .env.example) with HF_TOKEN_READ and HF_TOKEN_WRITE.
#   Alternatively, pass via environment:  HF_TOKEN_READ=hf_xxx bash setup.sh
#
# Usage:
#   bash setup.sh                          # full setup + test
#   bash setup.sh --skip-test              # skip smoke test
#
# After this passes, run training with:
#   bash run_train.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ───────────────────────────────────────────────────────────────────
ENV_NAME="flux_train"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda}"
CONDA_BIN="$CONDA_BASE/bin/conda"
ENV_YML="environment.yml"
LOG_DIR="output"

# Load .env if present (tokens, overrides)
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# HuggingFace repos
HF_DATASET_REPO="JasonXF/US3D-Enhanced"
HF_MODEL_REPO="JasonXF/SynthUrbanSAT"
HF_TOKEN_READ="${HF_TOKEN_READ:-}"
HF_TOKEN_WRITE="${HF_TOKEN_WRITE:-}"

# Miniconda installer URL
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

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
header(){ echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"; \
          echo -e "${BOLD}  $*${RESET}"; \
          echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"; }

# ── Parse args ───────────────────────────────────────────────────────────────
SKIP_TEST=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-test)  SKIP_TEST=true; shift ;;
        *)            shift ;;
    esac
done

mkdir -p "$LOG_DIR"

# Check HF read token
if [[ -z "$HF_TOKEN_READ" ]]; then
    fail "HF_TOKEN_READ is required for downloading dataset & weights."
    fail "Create a .env file with:  HF_TOKEN_READ=hf_xxxxx"
    fail "Or run:  HF_TOKEN_READ=hf_xxxxx bash setup.sh"
    exit 1
fi

if [[ -n "$HF_TOKEN_WRITE" ]]; then
    ok "HF_TOKEN_WRITE found — results can be pushed to HuggingFace"
else
    warn "HF_TOKEN_WRITE not set — you won't be able to push results to HF"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1/5 — Miniconda
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 1/5 — Miniconda"

if [[ -x "$CONDA_BIN" ]]; then
    ok "Miniconda already installed at $CONDA_BASE"
else
    info "Miniconda not found at $CONDA_BASE — installing..."
    INSTALLER="/tmp/miniconda_installer.sh"
    if command -v wget &>/dev/null; then
        wget -q -O "$INSTALLER" "$MINICONDA_URL"
    elif command -v curl &>/dev/null; then
        curl -sSL -o "$INSTALLER" "$MINICONDA_URL"
    else
        fail "Neither wget nor curl found. Cannot download Miniconda."
        exit 1
    fi
    bash "$INSTALLER" -b -p "$CONDA_BASE"
    rm -f "$INSTALLER"
    ok "Miniconda installed at $CONDA_BASE"
fi

# Initialize conda for this shell
eval "$("$CONDA_BIN" shell.bash hook 2>/dev/null)"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2/5 — Conda Environment
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 2/5 — Conda Environment ($ENV_NAME)"

if conda env list | grep -qw "$ENV_NAME"; then
    ok "Environment '$ENV_NAME' exists"
else
    warn "Environment '$ENV_NAME' not found — creating from $ENV_YML ..."
    info "This may take a few minutes (downloading PyTorch + CUDA)."
    if "$CONDA_BIN" env create -f "$ENV_YML"; then
        ok "Environment '$ENV_NAME' created"
    else
        fail "conda env create failed. Check output above."
        fail "Hint: try manually: $CONDA_BIN env create -f $ENV_YML"
        exit 1
    fi
fi

conda activate "$ENV_NAME"
PYTHON="$(which python)"
ok "Activated: $ENV_NAME  (python: $PYTHON)"

# Ensure huggingface_hub is installed (needed for downloads)
if ! "$PYTHON" -c "import huggingface_hub" 2>/dev/null; then
    info "Installing huggingface_hub..."
    "$PYTHON" -m pip install -q huggingface_hub
    ok "huggingface_hub installed"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3/5 — Download Dataset from HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 3/5 — Download Dataset ($HF_DATASET_REPO)"

DATASET_DIR="$SCRIPT_DIR/dataset"
if [[ -d "$DATASET_DIR/train" && -d "$DATASET_DIR/val" && -d "$DATASET_DIR/test" ]]; then
    TRAIN_COUNT=$(find "$DATASET_DIR/train" -type f | wc -l)
    if [[ "$TRAIN_COUNT" -gt 0 ]]; then
        ok "Dataset already exists ($TRAIN_COUNT files in train/). Skipping download."
    else
        info "Dataset dirs exist but appear empty — re-downloading..."
        NEED_DATASET=true
    fi
else
    NEED_DATASET=true
fi

if [[ "${NEED_DATASET:-false}" == "true" ]]; then
    info "Downloading dataset from $HF_DATASET_REPO ..."
    "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_DATASET_REPO}',
    repo_type='dataset',
    local_dir='${DATASET_DIR}',
    token='${HF_TOKEN_READ}',
)
print('Download complete.')
"
    ok "Dataset downloaded to $DATASET_DIR"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4/5 — Download Weights from HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 4/5 — Download Weights ($HF_MODEL_REPO)"

WEIGHTS_DIR="$SCRIPT_DIR/weights"
if [[ -d "$WEIGHTS_DIR" ]] && ls "$WEIGHTS_DIR"/*.safetensors &>/dev/null 2>&1; then
    WEIGHT_COUNT=$(ls "$WEIGHTS_DIR"/*.safetensors 2>/dev/null | wc -l)
    ok "Weights already present ($WEIGHT_COUNT .safetensors files). Skipping download."
else
    info "Downloading weights from $HF_MODEL_REPO ..."
    mkdir -p "$WEIGHTS_DIR"
    "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_MODEL_REPO}',
    repo_type='model',
    local_dir='${WEIGHTS_DIR}',
    token='${HF_TOKEN_READ}',
)
print('Download complete.')
"
    ok "Weights downloaded to $WEIGHTS_DIR"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5/5 — Dependency Check
# ═══════════════════════════════════════════════════════════════════════════════
header "Step 5/5 — Dependency Check"

CHECKS_PASSED=true

# Python version
PY_VER=$("$PYTHON" --version 2>&1)
ok "Python: $PY_VER"

# Torch + CUDA
TORCH_INFO=$("$PYTHON" -c "
import torch
cuda_ok = torch.cuda.is_available()
gpu = torch.cuda.get_device_name() if cuda_ok else 'N/A'
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda_ok else 0
print(f'torch={torch.__version__} cuda={torch.version.cuda} gpu={gpu} vram={vram:.1f}GB cuda_ok={cuda_ok}')
" 2>&1) || { fail "torch import failed"; CHECKS_PASSED=false; }

if [[ "$TORCH_INFO" == *"cuda_ok=True"* ]]; then
    ok "PyTorch: $TORCH_INFO"
else
    fail "CUDA not available: $TORCH_INFO"
    CHECKS_PASSED=false
fi

# Key packages
for pkg in diffusers transformers accelerate safetensors wandb einops psutil sentencepiece; do
    if "$PYTHON" -c "import $pkg" 2>/dev/null; then
        VER=$("$PYTHON" -c "import $pkg; print(getattr($pkg, '__version__', 'ok'))" 2>/dev/null)
        ok "$pkg=$VER"
    else
        fail "$pkg not installed"
        CHECKS_PASSED=false
    fi
done

# WandB login check
if "$PYTHON" -c "import wandb; assert wandb.api.api_key, 'no key'" 2>/dev/null; then
    ok "wandb: logged in"
else
    warn "wandb: not logged in — run 'wandb login' to enable cloud logging"
    warn "       (training will still work in offline mode)"
fi

if [[ "$CHECKS_PASSED" != "true" ]]; then
    fail "Dependency check failed. Fix the issues above and re-run."
    fail "Hint: pip install -r requirements.txt"
    exit 1
fi

ok "All dependencies OK"

# ═══════════════════════════════════════════════════════════════════════════════
# Smoke Test (optional)
# ═══════════════════════════════════════════════════════════════════════════════
header "Smoke Test"

if [[ "$SKIP_TEST" == "true" ]]; then
    warn "Skipping smoke test (--skip-test)"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TEST_LOG="$LOG_DIR/test_${TIMESTAMP}.log"
    info "Running: python train_script.py --test"
    info "Log: $TEST_LOG"
    echo ""

    # Force colors in piped output
    if "$PYTHON" -u train_script.py --test --no-wandb --name _smoke_test 2>&1 | tee "$TEST_LOG"; then
        echo ""
        ok "Smoke test PASSED"
        ok "Log: $TEST_LOG"
    else
        echo ""
        fail "Smoke test FAILED (see log above)"
        fail "Log: $TEST_LOG"
        fail ""
        fail "Common issues:"
        fail "  • OOM: reduce --image-size in run_train.sh"
        fail "  • Missing weights: check weights/ symlinks"
        fail "  • Import errors: pip install -r requirements.txt"
        exit 1
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
header "Setup Complete"
ok "Environment ready. Start training with:"
echo ""
echo "    bash run_train.sh"
echo ""

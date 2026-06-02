#!/usr/bin/env bash
# downstream_seg_depth_pipeline/setup.sh — one-shot server setup.
#
# Goal: on a fresh `git clone`, the ONLY thing you must provide is an HF token.
# This installs Python deps and pulls every model weight needed by Phase 1.
#
# Inputs (env vars or .env in this folder / ../train_pipeline/.env):
#   HF_TOKEN            (REQUIRED)  HuggingFace token with read access.
#   WEIGHTS_REPO        (optional)  full weights repo; default:
#                                   JasonXF/SynthUrbanSAT_bestmodel
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash setup.sh                 # deps + all Phase-1 weights
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$HERE/../generation_pipeline"
WEIGHTS_REPO="${WEIGHTS_REPO:-JasonXF/SynthUrbanSAT_bestmodel}"

# ─── HF token ────────────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    for envf in "$HERE/.env" "$GEN/.env" "$HERE/../train_pipeline/.env"; do
        if [[ -f "$envf" ]]; then
            tok="$(grep -E '^HF_TOKEN(_READ)?=' "$envf" | tail -n1 | cut -d= -f2- || true)"
            [[ -n "$tok" ]] && { export HF_TOKEN="$tok"; break; }
        fi
    done
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set. Run:  export HF_TOKEN=hf_xxx" >&2
    exit 1
fi
export HF_TOKEN

echo "[1/4] Python deps (downstream probe) ..."
pip install -q -r "$HERE/requirements.txt"
echo "[2/4] Python deps (generation pipeline) ..."
pip install -q -r "$GEN/requirements.txt" || true

echo "[3/4] Full generation weights from $WEIGHTS_REPO ..."
python - "$WEIGHTS_REPO" "$GEN/weights" <<'PY'
import os, sys
from huggingface_hub import snapshot_download

repo, dst = sys.argv[1:]
snapshot_download(
    repo_id=repo,
    repo_type="model",
    token=os.environ.get("HF_TOKEN") or None,
    local_dir=dst,
    allow_patterns=["base/*", "lora/*", "tokenizer/*"],
)
print("  weights placed in", dst)
PY

echo "[4/4] Verifying generation weights ..."

# ─── Verify required generation weights ─────────────────────────────────────
W="$GEN/weights/base"
need_base=0
for f in flux2_dev_fp8mixed.safetensors flux2-vae.safetensors \
         mistral_3_small_flux2_fp8.safetensors \
         FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors; do
    [[ -e "$W/$f" || -L "$W/$f" ]] || need_base=1
done
if (( need_base )); then
    echo "ERROR: base FLUX.2 weights still missing after downloading $WEIGHTS_REPO" >&2
    exit 1
fi
[[ -f "$GEN/weights/lora/checkpoint_epoch_0315/meta.pt" ]] || { echo "ERROR: LoRA checkpoint missing" >&2; exit 1; }
[[ -f "$GEN/weights/tokenizer/tokenizer.json" ]] || { echo "ERROR: tokenizer missing" >&2; exit 1; }
echo "  all required weights present."

echo "Setup complete. Next: bash run_phase1.sh --us3d-dir /path/to/US3D"

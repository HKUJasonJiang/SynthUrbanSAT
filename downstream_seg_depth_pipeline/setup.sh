#!/usr/bin/env bash
# downstream_seg_depth_pipeline/setup.sh — one-shot server setup.
#
# Goal: on a fresh `git clone`, the ONLY thing you must provide is an HF token
# (and, for synthetic generation, a way to obtain the base FLUX.2 weights).
# This installs Python deps and pulls every model weight needed by Phase 1.
#
# Inputs (env vars or .env in this folder / ../train_pipeline/.env):
#   HF_TOKEN            (REQUIRED)  HuggingFace token with read access.
#   BASE_WEIGHTS_REPO   (optional)  HF repo holding the 4 base FLUX.2 fp8 files
#                                   (flux2_dev_fp8mixed.safetensors, flux2-vae,
#                                   mistral_3_small_flux2_fp8, FLUX.2-dev-Fun-
#                                   Controlnet-Union-2602). If your machine has
#                                   ComfyUI, set COMFY_MODELS instead and skip.
#   COMFY_MODELS        (optional)  path to a local ComfyUI models/ dir.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash setup.sh                 # deps + all Phase-1 weights
#   bash setup.sh --copy          # physically copy base weights (no symlinks)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$HERE/../generation_pipeline"
COPY_FLAG=""
for a in "$@"; do case "$a" in --copy) COPY_FLAG="--copy";; esac; done

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

echo "[3/4] Generation weights (tokenizer + LoRA/HDC2A + base via COMFY) ..."
bash "$GEN/setup.sh" $COPY_FLAG || true

# ─── Optional: pull base weights from HF if still missing ────────────────────
W="$GEN/weights/base"
need_base=0
for f in flux2_dev_fp8mixed.safetensors flux2-vae.safetensors \
         mistral_3_small_flux2_fp8.safetensors \
         FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors; do
    [[ -e "$W/$f" || -L "$W/$f" ]] || need_base=1
done
if (( need_base )); then
    if [[ -n "${BASE_WEIGHTS_REPO:-}" ]]; then
        echo "[4/4] Base weights missing -> downloading from $BASE_WEIGHTS_REPO ..."
        python - "$BASE_WEIGHTS_REPO" "$W" <<'PY'
import os, sys, glob, shutil
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1:]
out = snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None,
                        local_dir=dst + "/_hf_tmp")
os.makedirs(dst, exist_ok=True)
for p in glob.glob(out + "/**/*.safetensors", recursive=True):
    tgt = os.path.join(dst, os.path.basename(p))
    if not os.path.exists(tgt):
        shutil.move(p, tgt)
shutil.rmtree(dst + "/_hf_tmp", ignore_errors=True)
print("  base weights placed in", dst)
PY
    else
        echo "[4/4] WARNING: base FLUX.2 weights still missing." >&2
        echo "      Set COMFY_MODELS (local ComfyUI) or BASE_WEIGHTS_REPO (HF repo)" >&2
        echo "      and re-run, otherwise synthetic generation will fail." >&2
    fi
else
    echo "[4/4] Base weights present."
fi

echo "Setup complete. Next: bash run_phase1.sh --us3d-dir /path/to/US3D"

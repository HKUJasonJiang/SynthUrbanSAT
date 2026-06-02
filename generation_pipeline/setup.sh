#!/usr/bin/env bash
# generation_pipeline/setup.sh — one-shot weights setup for the self-contained webui.
#
# Populates ./weights/ so the pipeline can run with NO references to paths
# outside this directory. Two modes per file:
#
#   1. SYMLINK (default, 0 disk overhead) to an existing local copy if found
#      on this machine. Set COMFY_MODELS to point at a ComfyUI models folder.
#   2. Fall back to downloading from HuggingFace using ${HF_TOKEN}.
#
# LoRA / HDC2A checkpoint is always pulled from the HF repo
#   JasonXF/Flux2-dev-controlnet-lora-weights (folder lora_512_dim_512_H200).
# By default only checkpoint_epoch_0315 is downloaded; pass --all-ckpts for all.
#
# Usage:
#   bash setup.sh                  # symlink base + download latest LoRA ckpt
#   bash setup.sh --copy           # physically copy base weights (slower, ~62GB)
#   bash setup.sh --all-ckpts      # also fetch the other 3 LoRA checkpoints
#   HF_TOKEN=hf_xxx bash setup.sh  # override token (otherwise reads .env / env)
#
set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$HERE/weights"
COMFY="${COMFY_MODELS:-$HERE/../../AIGC/ComfyUI/models}"
HF_CACHE_DEFAULT="$HOME/.cache/huggingface/hub"
TOKENIZER_SNAP="$HF_CACHE_DEFAULT/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c/tokenizer"

LORA_REPO="JasonXF/Flux2-dev-controlnet-lora-weights"
LORA_REMOTE_DIR="lora_512_dim_512_H200"
DEFAULT_CKPT="checkpoint_epoch_0315"
ALL_CKPTS=(checkpoint_epoch_0152 checkpoint_epoch_0313 checkpoint_epoch_0314 checkpoint_epoch_0315)

MODE="symlink"   # or "copy"
GET_ALL_CKPTS=0
for arg in "$@"; do
    case "$arg" in
        --copy)       MODE="copy" ;;
        --all-ckpts)  GET_ALL_CKPTS=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "[setup.sh] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$W/base" "$W/lora" "$W/tokenizer" "$HERE/output"

# ─── Load HF token ──────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    for envf in "$HERE/.env" "$HERE/../train_pipeline/.env"; do
        if [[ -f "$envf" ]]; then
            tok="$(grep -E '^HF_TOKEN(_READ)?=' "$envf" | tail -n1 | cut -d= -f2- || true)"
            if [[ -n "$tok" ]]; then HF_TOKEN="$tok"; break; fi
        fi
    done
fi
export HF_TOKEN="${HF_TOKEN:-}"

# ─── Place one base weight ──────────────────────────────────────────────────
# usage: place <src_path> <dst_path>
place() {
    local src="$1" dst="$2"
    if [[ -e "$dst" || -L "$dst" ]]; then
        echo "  ✓ $(basename "$dst") (already present)"
        return 0
    fi
    if [[ -e "$src" ]]; then
        if [[ "$MODE" == "copy" ]]; then
            echo "  → cp $src → $dst"; cp -f "$src" "$dst"
        else
            echo "  → ln -s $src → $dst"; ln -s "$src" "$dst"
        fi
    else
        echo "  ✗ source missing: $src" >&2
        echo "    (will need to be downloaded manually; see README.md)" >&2
        return 1
    fi
}

echo "[1/3] Linking base weights into $W/base ..."
place "$COMFY/diffusion_models/flux2_dev_fp8mixed.safetensors"               "$W/base/flux2_dev_fp8mixed.safetensors"  || true
place "$COMFY/vae/flux2-vae.safetensors"                                     "$W/base/flux2-vae.safetensors"           || true
place "$COMFY/text_encoders/mistral_3_small_flux2_fp8.safetensors"           "$W/base/mistral_3_small_flux2_fp8.safetensors" || true
place "$COMFY/controlnet/FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors"   "$W/base/FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors" || true

# ─── Tokenizer ──────────────────────────────────────────────────────────────
echo "[2/3] Tokenizer ..."
if [[ -d "$W/tokenizer" ]] && [[ -f "$W/tokenizer/tokenizer_config.json" || -L "$W/tokenizer/tokenizer_config.json" ]]; then
    echo "  ✓ tokenizer already present"
elif [[ -d "$TOKENIZER_SNAP" ]]; then
    rmdir "$W/tokenizer" 2>/dev/null || true
    if [[ "$MODE" == "copy" ]]; then
        echo "  → cp -r $TOKENIZER_SNAP → $W/tokenizer"
        cp -r "$TOKENIZER_SNAP" "$W/tokenizer"
    else
        echo "  → ln -s $TOKENIZER_SNAP → $W/tokenizer"
        ln -sfn "$TOKENIZER_SNAP" "$W/tokenizer"
    fi
else
    echo "  → snapshot not found locally, downloading FLUX.2-dev tokenizer from HF ..."
    python - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='black-forest-labs/FLUX.2-dev',
    allow_patterns=['tokenizer/*'],
    local_dir=os.environ['HOME'] + '/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/local',
    token=os.environ.get('HF_TOKEN') or None,
)
PY
    rmdir "$W/tokenizer" 2>/dev/null || true
    ln -sfn "$HOME/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/local/tokenizer" "$W/tokenizer"
fi

# ─── LoRA / HDC2A checkpoint(s) ─────────────────────────────────────────────
echo "[3/3] Downloading LoRA + HDC2A checkpoint(s) from $LORA_REPO ..."
if [[ -z "$HF_TOKEN" ]]; then
    echo "  WARN: no HF_TOKEN — repo may be private. Set HF_TOKEN env var or .env" >&2
fi

if (( GET_ALL_CKPTS )); then
    WANT_CKPTS=( "${ALL_CKPTS[@]}" )
else
    WANT_CKPTS=( "$DEFAULT_CKPT" )
fi

for ck in "${WANT_CKPTS[@]}"; do
    if [[ -f "$W/lora/$ck/meta.pt" ]]; then
        echo "  ✓ $ck (already present)"
        continue
    fi
    echo "  → downloading $LORA_REMOTE_DIR/$ck ..."
    python - "$LORA_REPO" "$LORA_REMOTE_DIR" "$ck" "$W/lora" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo, remote_dir, ck, dst_root = sys.argv[1:]
out = snapshot_download(
    repo_id=repo,
    allow_patterns=[f'{remote_dir}/{ck}/*'],
    token=os.environ.get('HF_TOKEN') or None,
    local_dir=dst_root + '/_hf_tmp',
)
import shutil
src = os.path.join(out, remote_dir, ck)
dst = os.path.join(dst_root, ck)
if os.path.isdir(dst):
    shutil.rmtree(dst)
shutil.move(src, dst)
shutil.rmtree(os.path.join(dst_root, '_hf_tmp'), ignore_errors=True)
print(f'    ✓ saved to {dst}')
PY
done

echo
echo "Done. Weights layout:"
ls -lh "$W/base/" 2>/dev/null | sed 's/^/  /'
echo "  lora:"
ls -d "$W/lora"/*/ 2>/dev/null | sed 's/^/    /'
echo
echo "Next:  python app.py"

#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Upload experiment results to HuggingFace
#
# Usage:
#   bash upload.sh --name lora_baseline_4A100_main
#   bash upload.sh --name abl_seg_only_2A100
#   bash upload.sh --name lora_rank_256_1A100 --repo JasonXF/SynthUrbanSAT
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect python
if [[ -x "${CONDA_BASE:-$HOME/miniconda}/envs/flux_train/bin/python" ]]; then
    PYTHON="${CONDA_BASE:-$HOME/miniconda}/envs/flux_train/bin/python"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "ERROR: python not found. Activate flux_train env first."
    exit 1
fi

"$PYTHON" upload.py "$@"

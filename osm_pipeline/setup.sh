#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

echo "[1/3] Installing Python dependencies from requirements.txt"
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[2/3] Checking Blender availability"
if command -v blender >/dev/null 2>&1; then
    blender --version | head -n 1
else
    echo "WARN: blender was not found on PATH. Install Blender >= 4.0 before running Stage F." >&2
fi

echo "[3/3] Creating runtime directories"
mkdir -p output cache

echo "Done. Try: python auto_pipeline.py --help"

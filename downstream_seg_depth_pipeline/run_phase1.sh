#!/usr/bin/env bash
# downstream_seg_depth_pipeline/run_phase1.sh
#
# PHASE 1 — Real vs Synthetic (no scaling). One command end-to-end:
#   1. split US3D into leakage-free train/val/test (geographic city holdout),
#   2. generate the synthetic counterpart S_p (our RGB on US3D seg+depth),
#   3. train two probes per task (R = real-only, S = synthetic-only),
#   4. evaluate both on the SAME real test set + emit bar charts and panels.
#
# Prereq: `bash setup.sh` already ran (deps + weights). You only provide the
# US3D dataset path here.
#
# Usage:
#   bash run_phase1.sh --us3d-dir /data/US3D            # flat rgb/seg/depth root
#   bash run_phase1.sh --us3d-dir /data/US3D --test-prefixes JAX --gen-seeds 0
#
# Results -> output/phase1/{results.csv, figures/}
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

US3D_DIR=""
TEST_PREFIXES="JAX"            # hold out Jacksonville for test (US3D has JAX/OMA)
GEN_SEEDS="0"                  # synthetic RGBs per tile (1 seed == same count as real)
TRAIN_SEEDS="0 1 2"
GEN_LIMIT="0"                  # >0 = only generate first N tiles (smoke test)
REAL_ROOT="../train_pipeline/dataset"
SYNTH_ROOT="./dataset_synth_us3d"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --us3d-dir)       US3D_DIR="$2"; shift 2;;
        --test-prefixes)  TEST_PREFIXES="$2"; shift 2;;
        --gen-seeds)      GEN_SEEDS="$2"; shift 2;;
        --train-seeds)    TRAIN_SEEDS="$2"; shift 2;;
        --gen-limit)      GEN_LIMIT="$2"; shift 2;;
        --real-root)      REAL_ROOT="$2"; shift 2;;
        --synth-root)     SYNTH_ROOT="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done
[[ -n "$US3D_DIR" ]] || { echo "ERROR: --us3d-dir is required" >&2; exit 1; }

echo "==[1/3] Splitting US3D (test holdout: $TEST_PREFIXES) =="
python scripts/make_splits.py \
    --src "$US3D_DIR" --out "$REAL_ROOT" \
    --test-prefixes $TEST_PREFIXES --val-fraction 0.1 --seed 0

echo "==[2/3] Generating synthetic S_p from real train seg+depth =="
GEN_ARGS=(--real-split "$REAL_ROOT/train" --out-split "$SYNTH_ROOT/train" --seeds $GEN_SEEDS)
[[ "$GEN_LIMIT" != "0" ]] && GEN_ARGS+=(--limit "$GEN_LIMIT")
python scripts/gen_synth_from_real.py "${GEN_ARGS[@]}"

echo "==[3/3] Training R vs S probes (seg + height) and plotting =="
python -m experiments.run_phase1 \
    --tasks segmentation height \
    --seeds $TRAIN_SEEDS \
    --synth-source us3d_paired

echo "Phase 1 done. See output/phase1/figures/ and output/phase1/results.csv"

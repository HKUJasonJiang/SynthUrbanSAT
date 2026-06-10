#!/usr/bin/env bash
set -euo pipefail

cd /home/jason/SynthUrbanSAT/downstream_seg_depth_pipeline
source /home/jason/miniconda3/etc/profile.d/conda.sh
conda activate flux

out=output/phase1_fair_trainval_100ep
config=output/phase1/config.fair_trainval_100ep.yaml

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_fair_worker.py \
  --config "$config" --condition R --tasks height --seed 0 \
  --synth-source us3d_paired --synth-count 100000 --out "$out" \
  > "$out/logs/R_height_rerun.log" 2>&1 &
pid_r=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/run_fair_worker.py \
  --config "$config" --condition S --tasks height --seed 0 \
  --synth-source us3d_paired --synth-count 100000 --out "$out" \
  > "$out/logs/S_height_rerun.log" 2>&1 &
pid_s=$!

status=0
wait "$pid_r" || status=$?
wait "$pid_s" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "height rerun failed with status ${status}"
  exit "$status"
fi

CUDA_VISIBLE_DEVICES=0 python scripts/make_fair_report.py \
  --config "$config" --out "$out" --samples 12 \
  > "$out/logs/report.log" 2>&1

echo "[height rerun done] $out"
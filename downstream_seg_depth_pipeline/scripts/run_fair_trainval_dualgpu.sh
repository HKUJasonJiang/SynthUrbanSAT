#!/usr/bin/env bash
set -euo pipefail

cd /home/jason/SynthUrbanSAT/downstream_seg_depth_pipeline
source /home/jason/miniconda3/etc/profile.d/conda.sh
conda activate flux

out=output/phase1_fair_trainval_100ep
config=output/phase1/config.fair_trainval_100ep.yaml

rm -rf output/phase1_fair_100ep output/phase1 "$out"
mkdir -p output/phase1 "$out/logs"

python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
cfg['data']['real_root'] = '/home/jason/data/US3D-Enhanced-png'
cfg['data']['train_splits'] = ['train', 'val']
cfg['data'].setdefault('synth_sources', {})['us3d_paired'] = '/home/jason/data/US3D-Synthetic'
cfg['data']['synth_source'] = 'us3d_paired'
cfg['train']['epochs'] = 100
cfg['train']['height_lr'] = 1.0e-4
cfg['train']['num_workers'] = 4
out = Path('output/phase1/config.fair_trainval_100ep.yaml')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print('[config]', out)
PY

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_fair_worker.py \
  --config "$config" --condition R --tasks segmentation height --seed 0 \
  --synth-source us3d_paired --synth-count 100000 --out "$out" \
  > "$out/logs/R.log" 2>&1 &
pid_r=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/run_fair_worker.py \
  --config "$config" --condition S --tasks segmentation height --seed 0 \
  --synth-source us3d_paired --synth-count 100000 --out "$out" \
  > "$out/logs/S.log" 2>&1 &
pid_s=$!

status=0
wait "$pid_r" || status=$?
wait "$pid_s" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "fair trainval run failed with status ${status}"
  exit "$status"
fi

CUDA_VISIBLE_DEVICES=0 python scripts/make_fair_report.py \
  --config "$config" --out "$out" --samples 6 \
  > "$out/logs/report.log" 2>&1

echo "[done] $out"
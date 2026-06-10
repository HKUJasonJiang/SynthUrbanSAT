#!/usr/bin/env bash
set -euo pipefail

cd /home/jason/SynthUrbanSAT/downstream_seg_depth_pipeline
source /home/jason/miniconda3/etc/profile.d/conda.sh
conda activate flux

export HF_TOKEN="$(awk -F= '/export HF_TOKEN=/{print $2; exit}' RUNBOOK.md)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

real_root=/home/jason/data/US3D-Enhanced-png
synth_root=/home/jason/data/US3D-Synthetic
prompt_embed=output/phase1/prompt_embed.pt

cleanup_children() {
  jobs -pr | xargs -r kill || true
}
trap cleanup_children INT TERM

rm -rf "$synth_root" output/phase1_fair_100ep
mkdir -p "$synth_root" output/phase1
cp "$real_root/depth_encoding.json" "$synth_root/depth_encoding.json"
cp "$real_root/prompt.json" "$synth_root/prompt.json"

python scripts/precompute_prompt_embed.py \
  --ckpt checkpoint_epoch_0152 \
  --prompt-json "$real_root/prompt.json" \
  --out "$prompt_embed"

run_split() {
  local split="$1"
  echo "== generate ${split} on GPU0+GPU1 =="
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python scripts/gen_synth_from_real.py \
    --real-split "$real_root/$split" \
    --out-split "$synth_root/$split" \
    --seeds 0 \
    --num-steps 30 \
    --ckpt checkpoint_epoch_0152 \
    --prompt-json "$real_root/prompt.json" \
    --prompt-embed "$prompt_embed" \
    --num-shards 2 \
    --shard-index 0 &
  local pid0=$!

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python scripts/gen_synth_from_real.py \
    --real-split "$real_root/$split" \
    --out-split "$synth_root/$split" \
    --seeds 0 \
    --num-steps 30 \
    --ckpt checkpoint_epoch_0152 \
    --prompt-json "$real_root/prompt.json" \
    --prompt-embed "$prompt_embed" \
    --num-shards 2 \
    --shard-index 1 &
  local pid1=$!

  local status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "generation failed for ${split} with status ${status}"
    return "$status"
  fi
}

run_split train
run_split val
run_split test

python - <<'PY'
from pathlib import Path
from scripts.labels import LabelSpace
from scripts.data import TileDataset

label_space = LabelSpace()
root = Path('/home/jason/data/US3D-Synthetic')
expected = {'train': 400, 'val': 80, 'test': 30}
for split, want in expected.items():
    dataset = TileDataset(str(root / split), label_space, 518, 'segmentation')
    print('[verify synthetic]', split, len(dataset))
    if len(dataset) != want:
        raise SystemExit(f'{split}: expected {want}, got {len(dataset)}')
PY

python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
cfg['data']['real_root'] = '/home/jason/data/US3D-Enhanced-png'
cfg['data'].setdefault('synth_sources', {})['us3d_paired'] = '/home/jason/data/US3D-Synthetic'
cfg['train']['epochs'] = 100
cfg['train']['height_lr'] = 1.0e-4
out = Path('output/phase1/config.fair_100ep.yaml')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print('[config]', out)
PY

python -m experiments.run_phase1 \
  --config output/phase1/config.fair_100ep.yaml \
  --tasks segmentation height \
  --seeds 0 \
  --synth-source us3d_paired \
  --out output/phase1_fair_100ep
# Dataset20 Two-Server Run Plan

This plan prepares the 20-city OSM dataset run for two external servers. The source tile plans are copied under `osm_pipeline/plans/dataset20/`, so the servers only need to pull the repository and run the launcher.

## Global Settings

| item | value |
|---|---:|
| OSM seed | `64` for building-height seed, tree scatter seed, and tree-height seed |
| Generation seed | `64` |
| View | `near-nadir-1` only |
| Depth input for generation | PNG (`near-nadir-1/2_depth.png`) |
| Generation resume | `--skip-existing` |
| OOM guard | `--seed-chunk-size 1` |
| OSM plan source | `osm_pipeline/plans/dataset20/<city>/tile_plan.json` |
| OSM output | `osm_pipeline/output/<city>/` |
| Default manifest | `osm_pipeline/plans/dataset20/manifest.json` now points to the remaining 16 cities |
| Generation output | `generation_pipeline/output/osm_batch__<city>__near-nadir-1__depth-png__<ckpt>/` |
| HF repo | `JasonXF/SynthUrbanSAT-5k` by default, override with `--hf-repo` |
| HF layout | `<city>/osm/` and `<city>/generation/near-nadir-1_seed64/` |

## Completed / Deprecated Cities

These 4 cities are already generated/uploaded and are excluded from the default remaining manifest. The old Tampa upload is marked as deprecated, but the new `tampa-1638` plan remains in the remaining run.

| city | status |
|---|---|
| `des-moines-450` | completed |
| `jacksonville-1980` | completed |
| `omaha-984` | completed |
| `wichita-300` | completed |
| `tampa-1080` | old external run; mark/delete externally, not used as final Tampa |

The original 20-city manifest is preserved at `osm_pipeline/plans/dataset20/manifest_full20.json`. The default `manifest.json` now contains the remaining 16 final cities, including the new `tampa-1638`.

## Remaining Server Split

The remaining split keeps both machines almost exactly balanced.

| server | cities | tiles |
|---|---|---:|
| H200 | houston-2632, tampa-1638, chicago-1395, austin-784, lincoln-513, topeka-480, tallahassee-409, washington-dc-352 | 8,200 |
| H100 | new-york-city-2738, philadelphia-1722, oklahoma-city-1316, little-rock-1140, columbia-sc-450, vienna-391, potsdam-336, vaihingen-90 | 8,183 |
| total remaining | 16 cities | 16,383 |

## Completion Definition

A city is **not complete** after OSM alone or generation alone. For this dataset run, a city is complete only when all three stages finish successfully:

1. `osm`: generate OSM-derived labels, depth, polygons, point cloud, mesh, and near-nadir folders under `osm_pipeline/output/<city>/`.
2. `generation`: generate pseudo-RGB from `near-nadir-1` with seed `64` under `generation_pipeline/output/osm_batch__<city>__near-nadir-1__depth-png__<ckpt>/`.
3. `upload`: upload both city artifacts to Hugging Face under `<city>/osm/` and `<city>/generation/near-nadir-1_seed64/`.

The practical status labels should be `planned -> osm_done -> generation_done -> hf_uploaded`. Only `hf_uploaded` counts as final done.

## Recommended Execution

Run OSM first on both machines. OSM is mostly CPU/network/Blender work and is safest as one city at a time; the pipeline still uses internal worker pools for IO, OSM, and canopy preprocessing.

```bash
python osm_pipeline/scripts/run_osm_generation_batch.py \
  --machine h200 \
  --stage osm

python osm_pipeline/scripts/run_osm_generation_batch.py \
  --machine h100 \
  --stage osm
```

Then run generation. If a server has one visible GPU, use `--gpus 0`; if it has multiple GPUs, pass them as `--gpus 0,1` and the generation script will shard tiles across workers.

```bash
python osm_pipeline/scripts/run_osm_generation_batch.py \
  --machine h200 \
  --stage generation \
  --gpus 0 \
  --continue-on-error

python osm_pipeline/scripts/run_osm_generation_batch.py \
  --machine h100 \
  --stage generation \
  --gpus 0 \
  --continue-on-error
```

Finally upload completed cities to Hugging Face. Upload is separate on purpose: network failures should not force rerunning OSM or generation.

```bash
export HF_TOKEN_WRITE=hf_xxx

python osm_pipeline/scripts/run_osm_generation_batch.py \
  --machine h200 \
  --stage upload \
  --hf-repo JasonXF/SynthUrbanSAT-5k \
  --continue-on-error

python osm_pipeline/scripts/run_osm_generation_batch.py \
  --machine h100 \
  --stage upload \
  --hf-repo JasonXF/SynthUrbanSAT-5k \
  --continue-on-error
```

For a single command per machine, `--stage all` runs OSM, generation, and upload for one city before moving to the next city. This is convenient for a closed loop, but `--stage osm`, then `--stage generation`, then `--stage upload` is easier to monitor and recover.

## Dry Run / Validation

Before the real run:

```bash
python osm_pipeline/scripts/run_osm_generation_batch.py \
  --machine h200 \
  --stage all \
  --dry-run

python osm_pipeline/scripts/run_osm_generation_batch.py \
  --machine h100 \
  --stage all \
  --dry-run
```

The launcher dry run validates that all plan files exist and prints the exact OSM/generation/upload commands without starting any stage. After OSM outputs exist, you can run `generation_pipeline/generation_pipeline.py --dry-run` directly for one city if you want to validate selected `near-nadir-1` files.

## Resume and OOM Notes

- OSM logs go to `logs/dataset20/<machine>/<city>/osm.log` and OSM itself also writes `osm_pipeline/output/<city>/metadata/run_live.log` plus `run_progress_latest.json`.
- Generation logs go to `logs/dataset20/<machine>/<city>/generation.log`; generation also writes `events.jsonl` or `events_shardXX.jsonl` inside its output folder.
- Upload logs go to `logs/dataset20/<machine>/<city>/upload.log`; upload can be safely rerun because `huggingface_hub.upload_folder` syncs folder contents to the same repo path.
- `--skip-existing` means generation can be restarted safely after interruption.
- `--seed-chunk-size 1` keeps peak VRAM lower. Since this plan uses only seed `64`, it also gives the cleanest resume behavior.
- Avoid compare mode for the full run. Compare mode loads baselines and is for small paper figures, not dataset generation.
- If a city fails, rerun with `--only-city <city> --stage <osm|generation>` after checking the corresponding log.

## Recommended Planning

The safest operating mode is stage-gated, not one giant fire-and-forget run:

1. Run `--stage osm` on both servers and watch `run_progress_latest.json` plus `osm.log`.
2. After a city has all expected tile folders and near-nadir outputs, run `--stage generation` with `--skip-existing` and `--seed-chunk-size 1`.
3. After generation completes for that city, run `--stage upload`.
4. Keep a simple external checklist with one row per city and columns `osm_done`, `generation_done`, `hf_uploaded`, `notes`.

This makes failures cheap: if OSM fails, do not start generation for that city; if generation OOMs, rerun generation only; if upload dies, rerun upload only.

## Example Single-City Recovery

```bash
python osm_pipeline/scripts/run_osm_generation_batch.py \
  --machine h200 \
  --only-city houston-2632 \
  --stage generation \
  --gpus 0 \
  --continue-on-error
```

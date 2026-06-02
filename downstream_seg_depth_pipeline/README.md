# downstream_pipeline

Downstream-task evaluation for **SynthUrbanSAT** (ICLR experiments). It answers
one question: *does our synthetic dataset make remote-sensing models better than
the real US3D data alone?* — on two tasks, **semantic segmentation** and **AGL
height estimation**.

> Full OKR plan (objectives, key results, deliverables, hyper-params, schedule):
> see [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).

## Design

* **Frozen probing.** A single frozen **DINOv2** backbone is shared by both
  tasks; only a lightweight head is trained (linear or **DPT**). The *only*
  variable across experiments is the **training-data mixture**, so any change in
  test performance is attributable to data, not architecture.
* **Always test on real US3D.** Synthetic data may appear in *training* only.
* **Unified targets.** Segmentation uses the shared 6-class palette
  (`road, water, foliage, building, grass, ground`); height is **AGL nDSM in
  metres** — US3D `depth/*.tif` and OSM `depth/*.exr` already agree on this.

### Conditions
| code | train data | purpose |
|------|------------|---------|
| **R**   | real only            | baseline |
| **S**   | synthetic only       | train-on-synthetic, test-on-real (TSTR) realism check |
| **R+S** | real + synthetic     | augmentation gain |

### Two synthetic sources
| source | conditioning | layouts | isolates |
|--------|--------------|---------|----------|
| **us3d_paired** (Sₚ) | US3D's own seg + nDSM | same as real | generated-RGB fidelity (TSTR upper bound) |
| **osm** (Sₒ) | OSM-derived seg + nDSM | novel, unlimited | scaling + diversity ("larger") |

Select with `--synth-source {us3d_paired,osm}` (configured in `data.synth_sources`).

### Curves
* **Low-data:** real fraction ∈ {0.1, 0.25, 0.5, 1.0}, with/without fixed synthetic.
  Real US3D labels are scarce (~480 tiles) — the gain should be largest here.
* **Synth-scale:** synthetic count ∈ {0, 1k, 5k, 10k, 25k, 50k} at full real.
  Demonstrates the scalability advantage of generated data.

## Layout
```
downstream_pipeline/
├── configs/{default.yaml, label_map.json}
├── docs/EXPERIMENT_PLAN.md       # OKR plan (objectives, KRs, deliverables)
├── scripts/
│   ├── labels.py      # 6-class decode (palette + RGB)
│   ├── data.py        # TileDataset + build_mixture + resolve_synth_root
│   ├── backbone.py    # frozen DINOv2 (torch.hub)
│   ├── heads.py       # LinearHead, DPTHead
│   ├── model.py       # ProbeModel = backbone + head
│   ├── metrics.py     # mIoU/IoU/OA/F1, RMSE/MAE/δ, DFC mIoU-3
│   ├── visualize.py   # KR figures: seg triptych, height panel, curves, bars
│   ├── train_probe.py # one condition end-to-end
│   ├── evaluate.py    # metrics on the real test set (+ qualitative dump)
│   └── make_splits.py # leakage-free geographic train/val/test split
├── experiments/run_scaling.py   # R / S / R+S + both curves -> JSON/CSV + PNGs
└── tests/test_smoke.py          # offline sanity checks (no GPU/network/data)
└── tests/test_smoke.py          # offline sanity checks (no GPU/network/data)
```

## Quick start
```bash
cd SynthUrbanSAT/downstream_pipeline
pip install -r requirements.txt

# 0. offline sanity (no data needed)
python -m tests.test_smoke

# 1. build a leakage-free split (test = a held-out city, here OMA tiles)
python scripts/make_splits.py --src /path/us3d_flat --out ../train_pipeline/dataset \
    --test-prefixes OMA --val-fraction 0.1

# 2. single condition (segmentation, real-only)
python scripts/train_probe.py --task segmentation --real-fraction 1.0 --synth-count 0

# 3. full experiment matrix (both tasks, 3 seeds, both curves) -> CSV + KR PNGs
python experiments/run_scaling.py --tasks segmentation height --seeds 0 1 2 \
    --synth-source osm
```

## Data expected
```
<root>/<split>/rgb/*.tif|png   seg/*.png   depth/*.tif|exr   # stems aligned
```
`real_root` defaults to `../train_pipeline/dataset`; point `synth_root` at the
generated RGB + OSM seg/AGL pairs.

## Success criteria
1. **R+S > R** on the real test set for both tasks, stable across ≥3 seeds.
2. **Largest gain in the low-data regime** (smallest real fraction).
3. **S (TSTR)** reaches a high fraction of **R** — synthetic realism + valid labels.
4. **Monotone improvement** with synthetic scale until saturation.
5. (later) Beats classic augmentation / raw-FLUX / Blender-RGB baselines.

Metrics: segmentation → mIoU (primary), per-class IoU, OA, F1; height → RMSE (m,
primary), MAE, δ<1.25; combined → DFC2019 **mIoU-3** (label correct *and* height
error < 1 m).

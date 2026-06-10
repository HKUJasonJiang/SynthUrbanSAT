# Near-Nadir and Random Seed Selection Experiment

This document defines a proposed experiment for selecting useful near-nadir views
and random seeds before generating a larger synthetic training set from the Omaha
OSM pipeline output.

Status: confirmed for implementation, except for minor threshold tuning after
the first artifact-score dry run.

## Goal

We want to test whether mild near-nadir views and random seed diversity improve
downstream robustness, while avoiding low-quality synthetic images caused by
over-slanted viewpoints or unstable seeds.

The experiment is not meant to prove final downstream performance by itself.
It is an evidence-based pre-screening step:

- choose representative Omaha tiles by segmentation distribution;
- generate a controlled set of view/seed candidates;
- filter obvious artifacts with reproducible image statistics;
- summarize quality/failure patterns by view and seed;
- use downstream training ablations as the final proof.

## Input Dataset

OSM output root:

```bash
osm_pipeline/output/omaha-984
```

Root/top-view files per tile:

```text
tile_XXXX/
  2_rgb.png
  4_seg.png
  5_depth.png
```

Near-nadir files per tile:

```text
tile_XXXX/near-nadir-N/
  1_seg.png
  2_depth.png
```

Generation should use PNG depth by default.

## Segmentation Classes

The current color map is:

| Class ID | Name | RGB |
|---:|---|---|
| 0 | road | `(0, 0, 255)` |
| 1 | water | `(0, 225, 255)` |
| 2 | foliage | `(0, 255, 0)` |
| 3 | building | `(255, 0, 0)` |
| 4 | grass | `(128, 0, 128)` |
| 5 | ground | `(0, 0, 0)` |

For this experiment:

- building-rich means high `building` ratio;
- tree-rich means high `foliage` ratio, not grass;
- water means high `water` ratio.

Grass is tracked as context but should not be used as the tree criterion.

## Preliminary Omaha Segmentation Statistics

I scanned all 984 root segmentation files at full resolution.

| Class | Mean | P50 | P75 | P90 | P95 | P99 | Max | Count > 0 | Count >= 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| building | 0.1152 | 0.1225 | 0.1531 | 0.1766 | 0.1993 | 0.2785 | 0.4739 | 970 | 629 |
| foliage | 0.1254 | 0.0609 | 0.1790 | 0.3391 | 0.4735 | 0.6884 | 0.8971 | 862 | 383 |
| grass | 0.5280 | 0.5502 | 0.5934 | 0.6200 | 0.6362 | 0.6651 | 0.7406 | 984 | 982 |
| foliage + grass | 0.6534 | 0.6413 | 0.6973 | 0.7805 | 0.8379 | 0.9300 | 0.9815 | 984 | 984 |
| water | 0.0069 | 0.0000 | 0.0000 | 0.0035 | 0.0176 | 0.1554 | 0.8380 | 158 | 19 |
| road | 0.1329 | 0.1269 | 0.1671 | 0.1989 | 0.2204 | 0.2589 | 0.3206 | 984 | 725 |
| ground | 0.0915 | 0.0952 | 0.1051 | 0.1153 | 0.1207 | 0.1286 | 0.1337 | 982 | 374 |

Important observation:

- Omaha only has 19 tiles with water ratio `>= 10%`.
- If we need exactly 20 water-focused tiles, the 20th tile must use a fallback
  threshold, such as the highest-water tile below 10%.

## Proposed Tile Selection

Select 60 tile roots:

- 20 building-rich tiles;
- 20 tree-rich tiles;
- 20 water-rich tiles.

The selected groups should be saved to a manifest before generation:

```text
generation_pipeline/output/selection_omaha984_60tiles/
  seg_distribution_all_tiles.csv
  selected_tiles.json
  selected_tiles.csv
  selection_summary.md
  preview_grid_building.png
  preview_grid_tree.png
  preview_grid_water.png
```

### Building-Rich Selection

Primary ranking:

```text
sort by building_ratio descending
```

Proposed inclusion rule:

- select 20 tiles from the upper tail of building ratio;
- avoid water-dominant tiles unless they naturally appear in the top building
  list, because this group should primarily test dense built environments.

Suggested reproducible rule:

```text
building_ratio >= P95_building if enough candidates;
otherwise take top 20 by building_ratio.
exclude tiles already selected for water.
```

From the preliminary scan, top building examples are:

```text
tile_0751 b=0.474
tile_0759 b=0.472
tile_0755 b=0.400
tile_0765 b=0.347
tile_0369 b=0.326
tile_0404 b=0.320
tile_0410 b=0.319
tile_0757 b=0.299
tile_0307 b=0.289
tile_0749 b=0.286
```

The final list should be produced by code and written to the manifest, not typed
manually from this proposal.

### Tree-Rich Selection

Primary ranking:

```text
sort by foliage_ratio descending
```

Proposed inclusion rule:

- select 20 tiles with high foliage ratio;
- use `foliage`, not `foliage + grass`;
- avoid water-dominant tiles when possible so the tree group does not duplicate
  the water group.

Suggested reproducible rule:

```text
foliage_ratio >= P95_foliage if enough candidates;
otherwise take top 20 by foliage_ratio.
exclude selected water tiles.
exclude selected building tiles unless there are fewer than 20 candidates.
```

From the preliminary scan, top foliage examples are:

```text
tile_0931 f=0.897
tile_0553 f=0.838
tile_0933 f=0.816
tile_0261 f=0.808
tile_0890 f=0.779
tile_0421 f=0.734
tile_0461 f=0.733
tile_0974 f=0.717
tile_0054 f=0.700
tile_0487 f=0.690
```

### Water-Rich Selection

Primary ranking:

```text
sort by water_ratio descending
```

Preferred inclusion rule:

```text
water_ratio >= 0.10
```

Preliminary scan found only 19 tiles satisfying that rule. Therefore the
proposed fallback is:

```text
select all tiles with water_ratio >= 0.10;
then add highest remaining water_ratio tile until count == 20.
```

Top water examples:

```text
tile_0741 w=0.838
tile_0742 w=0.803
tile_0740 w=0.482
tile_0698 w=0.350
tile_0743 w=0.328
tile_0419 w=0.314
tile_0300 w=0.230
tile_0700 w=0.226
tile_0428 w=0.211
tile_0699 w=0.193
```

## Overlap Policy

There are two possible policies.

Confirmed policy: disjoint groups.

- A tile can belong to only one group.
- Priority order: water first, then building, then tree.
- Reason: water is rare, so protect the rare group first.
- This gives cleaner per-group statistics.

Alternative policy: allow overlap.

- A tile can be in multiple groups if it is genuinely mixed.
- This is more faithful to scene composition but makes group-level conclusions
  less clean.

Use disjoint groups.

## View and Seed Set

Views:

```text
root
near-nadir-1
near-nadir-2
near-nadir-3
```

Confirmed: do not include `near-nadir-4` in the main selection experiment. It
can be kept as a separate stress-test set because the previous visual check
suggested it is too slanted and may introduce label-image misalignment.

Seeds:

```text
1,2,4,8,16,32,64,128
```

Rationale:

- seed magnitude has no semantic meaning;
- use powers of two for a simple reproducible sweep;
- eight seeds are enough to estimate per-seed failure rate without making the
  screening run too large.

Total candidate images:

```text
60 tiles x 4 views x 8 seeds = 1920 generated RGB images
```

Each generated tile/view should also save:

- `hdc2a_feature.png`;
- `grid.png`;
- `metadata.json`.

Confirmed: keep `grid.png` for this stage. Storage is not the limiting factor
for the screening run, and grids are useful for manual review.

## Generation Output Layout

Use the existing batch generation layout:

```text
generation_pipeline/output/
  osm_batch__omaha-984-selection60__root__depth-png__checkpoint_epoch_0315/
  osm_batch__omaha-984-selection60__near-nadir-1__depth-png__checkpoint_epoch_0315/
  osm_batch__omaha-984-selection60__near-nadir-2__depth-png__checkpoint_epoch_0315/
  osm_batch__omaha-984-selection60__near-nadir-3__depth-png__checkpoint_epoch_0315/
```

Inside each output:

```text
tile_XXXX/<view>/depth_png/
  rgb_seed_0002.png
  rgb_seed_0017.png
  ...
  hdc2a_feature.png
  grid.png
  metadata.json
```

The selection manifest must store the source seg/depth paths so every generated
RGB can be traced back to the OSM condition image.

## Artifact Filtering

The artifact filter should not try to decide which image is scientifically
"best". Its job is to remove obviously broken images and summarize risk by
view/seed/group.

Recommended output:

```text
generation_pipeline/output/selection_omaha984_60tiles/
  artifact_scores.csv
  artifact_summary_by_view.csv
  artifact_summary_by_seed.csv
  artifact_summary_by_group.csv
  failed_candidates.json
  borderline_candidates.json
  accepted_candidates.json
```

### Hard Image Statistics

Compute these per generated RGB:

| Metric | Meaning | Failure direction |
|---|---|---|
| `mean_luma` | average brightness | too low or too high |
| `std_luma` | contrast | too low |
| `p01_luma`, `p99_luma` | black/white clipping | extreme clipping |
| `black_ratio` | pixels with luma near 0 | too high |
| `white_ratio` | pixels with luma near 255 | too high |
| `saturation_mean` | average HSV saturation | too high or too low |
| `saturation_p95` | saturated-color tail | too high |
| `laplacian_var` | sharpness / blur proxy | too low |
| `entropy` | texture/detail proxy | too low |

Initial thresholds should be data-driven:

1. compute metrics for the 60 corresponding real RGB tiles;
2. compute metric distribution for synthetic candidates;
3. mark a candidate as failed if it is far outside the real-RGB range.

Suggested robust threshold:

```text
fail if metric is outside [real_p01 - margin, real_p99 + margin]
```

For metrics where only one direction matters:

```text
std_luma < real_p01(std_luma) * 0.7
laplacian_var < real_p01(laplacian_var) * 0.5
black_ratio > max(real_p99(black_ratio) * 2, 0.05)
white_ratio > max(real_p99(white_ratio) * 2, 0.05)
```

These margins should be printed in the summary and can be adjusted after visual
inspection of failures.

### Condition Consistency Proxy

We cannot fully judge whether the generated RGB is geometrically correct without
training and testing the downstream task. However, we can compute weak proxies:

1. Convert generated RGB to grayscale.
2. Run edge detection or Sobel magnitude on RGB.
3. Compute boundaries from seg class transitions.
4. Compute depth-gradient boundaries from depth.
5. Measure whether strong RGB edges overlap seg/depth boundaries.

Metrics:

```text
edge_alignment_seg
edge_alignment_depth
edge_density_rgb
seg_boundary_density
depth_boundary_density
```

Important: this is a weak warning signal only. Do not reject a candidate solely
because edge alignment is low unless visual inspection confirms the issue.

### Near-Nadir Slant Proxy

Do not infer slant from RGB at first. Use view-level policy:

- root, nn1, nn2: main candidates;
- nn3: candidate but monitored carefully;
- nn4: excluded from main selection experiment, optional stress test.

For nn3, check whether artifact failure rate is significantly higher than root,
nn1, and nn2. If nn3 has high failure rate or downstream performance drops, it
should be excluded from the main training recipe.

### Manual Review Subset

After artifact scoring, create review grids:

```text
review_grids/
  worst_by_artifact_score/
  random_accepted/
  per_view_examples/
  per_seed_examples/
```

Manual review should inspect:

- top 20 worst candidates overall;
- top 10 worst per view;
- random 5 accepted candidates per view;
- all candidates from nn3 that are near the failure threshold.

Manual labels:

```text
pass
borderline
fail
```

The manual labels should be used to tune thresholds once, then freeze them for
the formal experiment.

## Selection After Filtering

The first pass should not select the single "best" image per tile. It should
estimate useful diversity and failure rates.

Recommended accepted-set policy:

- reject hard artifact failures;
- keep all remaining root/nn1/nn2 candidates for downstream ablation;
- keep nn3 candidates only if its failure rate is close to nn1/nn2 and manual
  review does not show systematic slant artifacts.

Optional top-k policy for storage-limited runs:

- per tile/view, keep top 2 seeds by artifact score;
- require the two seeds to be visually distinct enough, measured by image
  embedding distance or simple RGB histogram distance;
- never pick a seed that failed hard checks.

## Analysis Tables

The script should report:

1. Seg distribution summary over all 984 tiles.
2. Selected tile summary by group.
3. Candidate count:

```text
60 tiles x 4 views x 8 seeds = 1920
```

4. Failure rate by view:

```text
root
near-nadir-1
near-nadir-2
near-nadir-3
```

5. Failure rate by seed:

```text
1,2,4,8,16,32,64,128
```

6. Failure rate by scene group:

```text
building-rich
tree-rich
water-rich
```

7. Failure interaction table:

```text
view x seed
view x scene_group
seed x scene_group
```

8. Recommended recipe for downstream ablation.

## Downstream Ablation Plan

Use downstream validation/test performance as the final criterion.

Recommended downstream training sets:

1. `real_only`
2. `real + synth_root_1seed`
3. `real + synth_root_8seed_filtered`
4. `real + synth_root_nn1_nn2_8seed_filtered`
5. `real + synth_root_nn1_nn2_nn3_8seed_filtered`

Optional stress-test:

6. `real + synth_root_nn1_nn2_nn3_nn4_8seed_filtered`

Important split rule:

- split by tile, not by generated image;
- all seeds/views from the same tile must stay in the same split;
- do not use real RGB from validation/test tiles to tune image-quality
  thresholds.

## Compute and Storage Estimate

From the previous 10-tile run:

- 5 views x 10 tiles x 4 seeds used about 422 MB for batch outputs;
- RGB files were about 349 MB;
- grids were about 68 MB.

The proposed 60-tile experiment has:

```text
60 x 4 views x 8 seeds = 1920 RGBs
```

This is 9.6x the number of RGBs in the 10-tile pilot
(`10 x 5 views x 4 seeds = 200 RGBs`).

Rough storage estimate:

- keeping RGB + feature + metadata + grid: about 4 GB;
- keeping RGB + feature + metadata only: about 3.4 GB.

Runtime will depend on whether multi-GPU scheduling is stable. The 10-tile pilot
took longer because one launcher path produced partial shards and required
manual repairs. For this experiment, use explicit per-GPU queues with disjoint
tile lists to avoid same-GPU collisions.


## Implemented Quality Evaluation Scripts

The first two traceable evaluation scripts are now implemented:

```text
generation_pipeline/scripts/selection_stats.py
generation_pipeline/scripts/artifact_score.py
generation_pipeline/scripts/write_generation_plan.py
```

### Step 1: Select Representative Tiles

Run:

```bash
/data/home/jason/miniconda3/envs/flux/bin/python \
  generation_pipeline/scripts/selection_stats.py \
  --input-dir osm_pipeline/output/omaha-984 \
  --out generation_pipeline/output/selection_omaha984_60tiles \
  --group-size 20 \
  --water-threshold 0.10
```

Current verified output:

```text
generation_pipeline/output/selection_omaha984_60tiles/
  seg_distribution_all_tiles.csv
  selected_tiles.csv
  selected_tiles.json
  selection_summary.md
  preview_grid_building.png
  preview_grid_tree.png
  preview_grid_water.png
```

Traceability fields in `selected_tiles.json`:

- script version;
- command line;
- git commit and dirty status;
- input/output paths;
- class map;
- distribution summary;
- selection policy;
- selected tile records;
- first-1MB SHA256 prefix for each seg image.

The current Omaha run scanned 984 tiles and selected:

```text
water:    20 tiles, with 1 fallback below water_ratio=10%
building: 20 tiles
tree:     20 tiles
```

### Step 2: Generate Candidate Images

Use the selected tile list from:

```text
generation_pipeline/output/selection_omaha984_60tiles/selected_tiles.json
```

Main experiment parameters:

```text
views: root, near-nadir-1, near-nadir-2, near-nadir-3
seeds: 1,2,4,8,16,32,64,128
```

Write a traceable generation plan without starting the heavy job:

```bash
/data/home/jason/miniconda3/envs/flux/bin/python \
  generation_pipeline/scripts/write_generation_plan.py \
  --selection generation_pipeline/output/selection_omaha984_60tiles/selected_tiles.json \
  --input-dir osm_pipeline/output/omaha-984 \
  --out-dir generation_pipeline/output/selection_omaha984_60tiles \
  --gpus 0,1 \
  --seeds 1,2,4,8,16,32,64,128
```

This writes:

```text
generation_pipeline/output/selection_omaha984_60tiles/generation_plan.json
generation_pipeline/output/selection_omaha984_60tiles/run_generation_selection.sh
generation_pipeline/output/selection_omaha984_60tiles/generation_run.log  # created when the shell is run
```

To start generation later:

```bash
bash generation_pipeline/output/selection_omaha984_60tiles/run_generation_selection.sh
```

The shell uses explicit per-GPU queues with disjoint tile shards. This avoids
same-GPU model loading collisions and is more traceable than blindly launching
overlapping controller jobs.

### Step 3: Score Artifacts After Generation

After candidate images exist, run:

```bash
/data/home/jason/miniconda3/envs/flux/bin/python \
  generation_pipeline/scripts/artifact_score.py \
  --selection generation_pipeline/output/selection_omaha984_60tiles/selected_tiles.json \
  --generated-root generation_pipeline/output/osm_batch__omaha-984-selection60__root__depth-png__checkpoint_epoch_0315 \
  --generated-root generation_pipeline/output/osm_batch__omaha-984-selection60__near-nadir-1__depth-png__checkpoint_epoch_0315 \
  --generated-root generation_pipeline/output/osm_batch__omaha-984-selection60__near-nadir-2__depth-png__checkpoint_epoch_0315 \
  --generated-root generation_pipeline/output/osm_batch__omaha-984-selection60__near-nadir-3__depth-png__checkpoint_epoch_0315 \
  --out generation_pipeline/output/selection_omaha984_60tiles/artifact_eval
```

Artifact scoring outputs:

```text
artifact_eval/
  real_rgb_metrics.csv
  artifact_scores.csv
  artifact_summary_by_view.csv
  artifact_summary_by_seed.csv
  artifact_summary_by_group.csv
  artifact_summary_by_view_seed.csv
  artifact_summary_by_view_group.csv
  failed_candidates.json
  borderline_candidates.json
  accepted_candidates.json
  artifact_manifest.json
  artifact_summary.md
  review_grids/
    worst_failed.png
    worst_overall.png
    random_accepted.png
```

Traceability fields in `artifact_manifest.json`:

- script version;
- command line;
- git commit and dirty status;
- selection manifest path;
- generated roots;
- real-RGB baseline statistics;
- artifact thresholds;
- accepted/borderline/failed counts.

## Proposed Implementation Steps

After confirmation, implement scripts in this order:

1. `selection_stats.py`
   - scan all seg maps;
   - write `seg_distribution_all_tiles.csv`;
   - select 20 building, 20 tree, 20 water tiles;
   - write selected manifests and preview grids.

2. `run_selection_generation.py` or documented shell commands
   - call existing `generation_pipeline.py`;
   - generate 4 views x 8 seeds for selected tiles;
   - use explicit GPU queues.

3. `artifact_score.py`
   - compute real-RGB metric baselines;
   - score synthetic outputs;
   - write failure/accepted manifests;
   - make review grids.

4. `summarize_selection_experiment.py`
   - aggregate failure rate tables;
   - recommend view/seed recipe.

## Confirmed Choices

1. Use disjoint groups with priority `water -> building -> tree`.
2. For water, allow the 20th tile to be the highest-water tile below 10%,
   because only 19 Omaha tiles have `water_ratio >= 10%`.
3. Exclude `near-nadir-4` from the main experiment and keep it only as optional
   stress test.
4. Use seeds `1,2,4,8,16,32,64,128`.
5. Keep `grid.png`; storage is acceptable at this stage.

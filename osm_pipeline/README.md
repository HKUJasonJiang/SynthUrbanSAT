# OSM Pipeline: Real-World Layout to Segmentation, Depth, and 3D Mesh

This pipeline converts an OpenStreetMap bounding box into aligned urban remote-sensing products for SynthUrbanSAT. It uses real OSM layout priors, procedural geometry, Blender assembly, and canopy-aware foliage generation to produce segmentation maps, depth maps, 3D meshes, and metadata.

中文摘要：该流程负责把真实城市 layout 变成可训练的控制条件。OSM 提供建筑、道路、水体、草地等真实空间分布；Blender 阶段生成三维几何、树木和像素对齐的 seg/depth，为后续 `generation_pipeline` 生成伪真实卫星图提供条件输入。

## Outputs

Each city is written under `output/<city>/`:

```text
output/<city>/
├── <city>_osm.png
├── <city>_rgb.png
├── <city>_seg.png
├── <city>_depth.png
├── metadata/
└── tile_XXXX/
    ├── 1_osm.png
    ├── 2_rgb.png
    ├── 3_seg.png
    ├── 4_depth.png
    ├── 5_depth.exr
    ├── blender/
    │   ├── <tile>.glb
    │   └── <tile>.blend
    └── metadata/
```

`output/` and `cache/` are ignored by Git. Regenerate them locally instead of committing them.

## Setup

Blender >= 4.0 must be installed and available on `PATH` because Stage F invokes Blender in background mode.

```bash
cd osm_pipeline
bash setup.sh
```

On Windows PowerShell, you can also use your existing environment:

```powershell
python -m pip install -r requirements.txt
blender --version
```

## CLI Examples

Single-tile debug run:

```bash
python auto_pipeline.py \
  --city omaha_single \
  --bbox -96.135 41.260 -96.130 41.265 \
  --vlm-mode skip \
  --tree-density 0.00015 \
  --scatter-mode canopy_prob \
  --allow-non-foliage \
  --use-blender-seg \
  --topdown-tree-xy-scale 3.5 \
  --clean
```

Small multi-tile run:

```bash
python auto_pipeline.py \
  --city omaha_rich_test \
  --bbox -96.135 41.255 -96.125 41.265 \
  --vlm-mode skip \
  --tree-density 0.00015 \
  --scatter-mode canopy_prob \
  --allow-non-foliage \
  --use-blender-seg \
  --topdown-tree-xy-scale 3.5 \
  --clean
```


City-scale master plan workflow:

```bash
# 1) Save a stable grid plan without generating tiles.
python auto_pipeline.py \
  --city omaha_full \
  --bbox -96.40 41.05 -95.85 41.45 \
  --plan-only

# 2) Generate the city in batches from that fixed plan.
python auto_pipeline.py \
  --city omaha_full \
  --plan output/omaha_full/metadata/tile_plan.json \
  --tile-range 0001:1000

python auto_pipeline.py \
  --city omaha_full \
  --plan output/omaha_full/metadata/tile_plan.json \
  --tile-range 1001:2000
```

In `osm_app.py`, use Tab 2 to search a city, inspect the Esri satellite
overview, click NW and SE corners, then click **Save Master Plan**. The saved
plan can be used by the **Generate Tile Range from Plan** button or the CLI
commands above. Keep the same plan file for every batch to avoid overlap or
coordinate drift.

## WebUI

```bash
python osm_app.py
```

The app launches on `http://127.0.0.1:8765` when available and exposes the same city/tile generation controls as the CLI.

## Main Files

- [auto_pipeline.py](auto_pipeline.py): batch pipeline controller and city/tile orchestration.
- [osm_app.py](osm_app.py): Gradio interface for interactive generation.
- [configs/default.yaml](configs/default.yaml): default OSM, geometry, rendering, and foliage settings.
- [scripts/3_blender_assemble.py](scripts/3_blender_assemble.py): Blender background assembly, rendering, and export.
- [dataprep/](dataprep/): geospatial data loading, reprojection, rasterization, and canopy utilities.

## Notes

- `3_seg.png`, `4_depth.png`, and `5_depth.exr` are designed to be pixel-aligned conditioning inputs for `../generation_pipeline`.
- `2_rgb.png` is a real satellite reference fetched for inspection and evaluation; it is not assumed to precisely correspond to OSM labels.
- Generated city configs, caches, outputs, and Blender artifacts can become large and should stay local.

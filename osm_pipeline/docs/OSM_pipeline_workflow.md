# OSM Pipeline 工作流程总览

本文总结 `osm_pipeline` 当前的主流程：如何从一个 WGS84 bbox 生成固定分辨率 tile，如何拉取并栅格化 OSM 数据，如何估计建筑高度、生成绿植/树木分布，以及最终如何产出与卫星图对齐的 segmentation、depth、3D 场景和 metadata。

代码主入口：

- CLI / batch: `osm_pipeline/auto_pipeline.py`
- UI: `osm_pipeline/osm_app.py`
- tile 规划: `osm_pipeline/dataprep/tile_grid.py`
- OSM tag 映射: `osm_pipeline/dataprep/osm_tags.py`
- KR2 几何: `osm_pipeline/scripts/2_build_geometry.py`
- KR3 Blender / tree scatter: `osm_pipeline/scripts/3_blender_assemble.py`
- near-nadir preset: `osm_pipeline/scripts/render_sat_depth_tests.py`
- 树木规则细节: `osm_pipeline/docs/tree_generation_rules.md`

## 1. 总体阶段

`auto_pipeline.py` 把一个 city / bbox 拆成 tile，然后按阶段执行：

| 阶段 | 名称 | 主要产物 |
|---|---|---|
| A | `plan_tiles` | `tile_XXXX` 网格、每个 tile 的 WGS84 bbox |
| B | satellite + OSM basemap | Esri RGB、OSM cartographic basemap |
| C0 | city-level union OSM + global grid + road width calibration | 城市级 OSM union、Web Mercator city seg grid、本地道路宽度表 |
| C | OSM 6-class fetch + rasterize | `seg_osm`、class GeoJSON、类别比例 |
| D | canopy height + foliage geojson | canopy `.npz`、`*_foliage_canopy.geojson` |
| E | KR1/KR2 geometry build | per-class GeoJSON、GLB、建筑 metadata |
| F | KR3 Blender assemble | `.blend`、tree instances、topview seg/depth、point cloud |
| G | near-nadir derived views | `near-nadir-1..4/` 中的斜视 seg/depth/shadow preset |
| H | aggregate city | city-level mosaic、tile index、global building metadata |

最终每个 tile 文件夹会整理成标准文件：

```text
output/<city>/tile_XXXX/
├── 1_osm.png
├── 2_rgb.png
├── 3_blender_preview.png
├── 4_seg.png
├── 5_depth.png
├── 5_depth.exr
├── 6_polygon_outline.png
├── 6_polygons.json
├── 7_pointcloud.png
├── 7_pointcloud.ply
├── near-nadir-1/
│   ├── 1_seg.png
│   ├── 2_depth.png
│   ├── 3_shadow.png
│   └── params.json
├── near-nadir-2..4/
├── blender/
│   ├── tile_XXXX.glb
│   ├── tile_XXXX.blend
│   └── tree_instances.json
└── metadata/
    ├── tile_XXXX.json
    ├── tile_XXXX.meta.json
    └── tile_XXXX_osm_buildings.geojson
```

city 级还会输出：

```text
output/<city>/
├── <city>_osm.png
├── <city>_rgb.png
├── <city>_seg.png
├── <city>_depth.png
├── config/
│   ├── run_config_latest.json
│   ├── run_config_<timestamp>_<status>.json
│   └── road_width_calibration_latest.json
└── metadata/
    ├── <city>.json
    ├── tile_index.geojson
    ├── tile_plan.json
    ├── strategy.json
    ├── run_timing_latest.json
    ├── run_live.log
    └── run_progress_latest.json
```

## 2. Tile 分辨率和 GSD 如何确定

当前 batch pipeline 的 tile 分辨率是固定的：

- `gsd = 0.5 m/px`
- `size_px = 1024`
- 因此每个 tile 的地面边长：

```text
tile_m = gsd * size_px = 0.5 * 1024 = 512 m
```

所以一个标准 tile 覆盖约 `512 m x 512 m`，输出图像是 `1024 x 1024 px`，即每个像素代表约 `0.5 m` 地面距离。

### 2.1 规划公式

`plan_tiles(area_bbox_wgs, gsd=0.5, size_px=1024, overlap=0.0)` 的逻辑是：

1. 输入 bbox 为 WGS84 `(W, S, E, N)`。
2. 根据 bbox 自动选择本地 UTM CRS。
3. 把 bbox 的 SW / NE 角投影到 UTM 米制坐标。
4. 计算：

```text
tile_m = gsd * size_px
stride_m = tile_m * (1 - overlap)
```

5. 按 UTM 米制宽高用 `ceil` 计算需要多少行/列，保证覆盖完整输入 bbox。
6. 从西北角开始，row-major 生成 `tile_0001`, `tile_0002`, ...。
7. 每个 tile 的中心和边界再从 UTM 反投影回 WGS84，保存为 `TilePlan.bbox_wgs`。

这意味着 tile 的物理尺寸在 UTM 里严格是 `512m`，但保存到 WGS84 后经纬度跨度会随纬度变化。

### 2.2 batch 和 UI 的差异

batch 的 `auto_pipeline.py` 直接使用 `TilePlan.bbox_wgs`，不再走 UI 的 `bbox_from_center()`。这是为了避免 UI 预览里 `1.05x` padding 导致相邻 tile 有约 5% 重叠，从而在 city mosaic 里出现偏移。

UI 单 tile 预览的 `bbox_from_center()` 会用：

```text
half = gsd * size / 2 * 1.05
```

所以 UI 预览 bbox 稍大；batch 生产则使用 planner 给出的精确 bbox。

### 2.3 影像 zoom 不是 GSD 的来源

卫星图来自 Esri World Imagery XYZ tile。代码会选择最小的 zoom，使 bbox 横向覆盖的 Web Mercator tile 像素数大于等于目标 `out_size=1024`，然后 crop + resize 成 `1024 x 1024`。

也就是说：

- pipeline 的训练/渲染 GSD 来自 `gsd=0.5` 和 `size_px=1024`
- Esri zoom 只是为了下载足够清晰的底图
- 最终 RGB、seg、depth 都被整理到同一个 `1024 x 1024` tile grid

## 3. OSM 数据模型：6 类语义

当前 segmentation 类别固定为 6 类：

| class | id | 来源/含义 |
|---|---:|---|
| road | 0 | OSM road graph / highway line buffer |
| water | 1 | OSM 面状水体：`natural=water/bay/strait`、`waterway=riverbank`、reservoir 等；线状 river/stream/canal 不强行闭合成面 |
| foliage | 2 | 不主要来自 OSM tree tag；由 canopy/SAM3/tree instances 生成 |
| building | 3 | OSM `building=*` |
| grass | 4 | OSM grass / park / garden / pitch / grassland 等 |
| ground | 5 | bbox 内减去其他类别后的隐式背景 |

重叠时按 priority 画：

```text
ground < foliage < grass < water < road < building
```

也就是建筑覆盖道路/水/草等，road 也会覆盖低优先级类别。

### 3.1 OSM 拉取路径

优先级是：

1. 如果 `cache/pbf/<region>/` 有预处理好的 parquet，则用本地 Geofabrik PBF cache，速度最快。
2. 否则使用 combined Overpass fetch。
3. 对 road 会拉 highway line，保留 `highway`、`width`、`lanes` 等 tag，再按 road type 做米制 buffer。

road buffer 的全局 fallback 半宽示例：

| highway | half width |
|---|---:|
| motorway | 9 m |
| trunk | 8 m |
| primary | 7 m |
| secondary | 6 m |
| tertiary | 5 m |
| residential | 4 m |
| service | 3 m |
| default | 4 m |

当前 batch 默认不会只用这张固定表，而是启用本地道路宽度校准：

```text
road_width_mode = local_lanes
road_width_min_samples = 3
road_lane_width_m = 3.4
road_lane_margin_m = 1.2
road_width_blend_weight = 0.65
```

校准逻辑：

1. 若 OSM road edge 有 `width=*`，优先解析真实宽度。
2. 否则若有 `lanes=*`，用：

```text
full_width_m = lanes * 3.4 + 1.2
half_width_m = full_width_m / 2
```

3. 对每种 `highway` 类型取 city-level median。
4. 若该类型样本数不少于 3，用 `0.65 * local_median + 0.35 * global_default` 混合。
5. 样本稀疏时只做弱混合，避免少量异常 OSM tag 把道路宽度拉歪。
6. 校准后的 road buffer 会重新生成 city-level road geometry，并写入 `config/road_width_calibration_latest.json`。

### 3.2 City-level C0 的作用

多 tile 运行时，pipeline 会先尝试在 city union bbox 上做一次 OSM union fetch，并建立全局 Web Mercator raster grid。每个 tile 的 Stage C 再从 city union geometry clip 出自己的 bbox。

这样做的目的：

- 减少每个 tile 单独请求 Overpass 的次数
- 避免跨 tile 的道路/建筑边界在相邻 tile 中拓扑不一致
- city overview 和 per-tile seg 可以从同一个全局 raster/grid 切片，减少接缝
- 在 city 级别统计 `width/lanes` tag，形成更接近本地道路风格的 buffer 宽度

如果 C0 失败，会 fallback 到 per-tile OSM fetch。

### 3.3 water polygon 修正

当前默认：

```text
water_fix_mode = geometry_filter
```

这一步主要处理 OSM 水体面常见的错误：线状河流被强行闭合成巨大 polygon、coastline/边界线造成 tile-filling sliver、极细长水体片段等。实现上：

- OSM tag 侧不再把 `coastline` 当水体面。
- `waterway=river/stream/canal` 作为线状水道处理，不强行闭合成 polygon。
- `waterway=riverbank` 和正常 `natural=water` 等面状水体保留。
- Stage C 后对 water polygon 做几何 sanity filter：删除过小、极细长、低填充率或疑似大面积裁剪 sheet 的水体。

另有 `imagery_filter` 模式会用 Esri RGB 的蓝/暗水体启发式进一步筛水，但当前默认采用更稳定的 `geometry_filter`。

## 4. 建筑模型和建筑高度

建筑来自 OSM `building=*` footprint。KR2 会把每个建筑 footprint 投影到本地 UTM，然后 extrude 成 3D mesh。

### 4.1 高度来源优先级

每栋楼高度按下面顺序确定：

1. OSM `height` 或 `building:height`
2. OSM `building:levels` 或 `levels` 乘 `meters_per_level`
3. 如果都没有，则从配置分布中采样

默认配置：

```text
meters_per_level = 3.0 m
building_height_range_m = [3.0, 30.0]
height_dist = lognormal
height_seed = 42
```

因此没有高度 tag 的建筑会在 `[3,30]m` 内按 lognormal 抽样，且 seed 固定，可复现。

### 4.2 支持的建筑高度分布

KR2 支持：

- `flat`: 总是取 `(hmin + hmax) / 2`
- `uniform`: 在 `[hmin, hmax]` 均匀采样
- `lognormal`: 默认，中心约在中间值附近，带长尾
- `bimodal`: 约 70% 低层 + 30% 较高建筑，用于 suburban + landmark 混合

### 4.3 metadata 如何保存

KR2 会写：

```text
metadata/tile_XXXX.meta.json
```

其中包含：

- `building_id`
- `osm_id`
- local UTM centroid
- `height_m`
- `footprint_area_m2`
- per-class mesh 顶点/面数和建筑高度统计

随后 `_write_tile_metadata()` 会把 local centroid 转回 WGS84，写入：

```text
metadata/tile_XXXX.json
```

里面包含每个 building 的：

- id
- centroid_lon / centroid_lat
- height_m
- footprint_area_m2

并尽量保留原始：

```text
metadata/tile_XXXX_osm_buildings.geojson
```

后续 compositing depth / seg 时，会用这个 GeoJSON + metadata 中的 height，把建筑 footprint rasterize 成 height map。如果缺 metadata，fallback 建筑高度是 `15m`。

## 5. 绿植、树冠和树木分布

这里要区分三层东西：

1. `grass`: OSM 中的草地、公园、球场等地表语义。
2. `foliage substrate`: canopy/SAM3 推出来的“可长树区域”。
3. `tree instances`: Blender 中真实散布出来的一棵棵树，用于最终树冠 seg 和 depth。

### 5.1 foliage 不直接依赖 OSM tree tags

`osm_tags.py` 里明确说明：OSM 的树数据在 suburb 等区域太稀疏，所以 `foliage` 不主要从 OSM fetch。OSM 只稳定提供 building / road / water / grass，foliage 主要由 canopy/SAM3 和 Blender tree instances 生成。

### 5.2 Canopy 高度数据

默认 canopy source 是：

```text
eth_10m
```

即 ETH Global Canopy Height 10m 数据。也支持用户提供 local GeoTIFF。

`build_canopy_npz()` 会：

1. 按 tile 的 lat/lon 找 ETH 3 度 COG tile。
2. 下载并缓存到 `cache/canopy/eth/`。
3. 重新投影并双线性采样到 tile 本地 UTM grid。
4. 输出一个与渲染 frustum 对齐的 `.npz`：

```text
heights: float32, shape=(1024,1024), canopy height in metres
gsd: 0.5
size: 1024
extent: 512
sw_x_utm / sw_y_utm
utm_crs
source
```

这个 canopy grid 的行列定义是：

```text
row 0 = north
col 0 = west
x_local = (col + 0.5) * gsd
y_local = (size - 1 - row + 0.5) * gsd
```

### 5.3 canopy 如何变成 foliage polygon

Stage D 会把 canopy height 栅格转成 `*_foliage_canopy.geojson`：

1. 对 canopy height 做 Gaussian smooth，减少 10m 原始数据的块状感。
2. 默认取 `height >= 2m` 作为 tree canopy mask。
3. 如果设置了 `target_foliage_ratio`，会自动提高 threshold，直到 foliage 覆盖比例不超过目标值。
4. mask vectorize 成 WGS84 polygon。

默认 `AutoPipelineConfig.target_foliage_ratio = 0.25`，意思是 canopy 过密时，会尝试只保留最高的约 25% tile 区域作为 canopy foliage。

### 5.4 KR2 中 foliage substrate 的处理

KR2 构建 GLB 时会把 foliage class 与 canopy/SAM3 foliage geojson union 起来，但会减去：

```text
building, road, water, grass
```

这样 tree scatter 的 substrate 不会覆盖到建筑、道路、水体和明确的 grass 上。

注意：最终可视化 seg 里，pipeline 又会把 canopy substrate 作为 grass 底层处理，只把真正的 tree instance crown 标成 foliage。这是为了让“草地/绿地”和“一棵棵树冠”在 semantic map 里分开。

### 5.5 树木 scatter 默认模式

当前默认 scatter 模式是：

```text
scatter_mode = canopy_prob
tree_density = 0.00015 trees/m^2
gn_tree_amount = 0.25
gn_min_distance = 3.5m
topdown_tree_xy_scale = 1.8
tree_h_dist = lognormal
tree_h_min = 6m
tree_h_max = 10m
```

旧的 Python PCG tree scatter 仍保留参数接口，但当前真正起主要作用的是 Blender Geometry Nodes scatter。`canopy_prob` 的核心逻辑：

1. 遍历 canopy grid 中 `h >= min_canopy_h` 的 cell。
2. 用 canopy height 归一化后作为概率密度：

```text
p_h = h / max(H)
p_cell = base_density * gsd^2 * canopy_prob_scale * p_h
```

3. 乘以类别权重：

```text
foliage: 1.0
grass/ground with allow_non_foliage: 0.3
building/road/water/empty: 0.0
```

4. 若 Bernoulli 命中，则在该 cell 周围展开一个 cluster。
5. cluster 内树高优先使用 canopy height 加少量 jitter。
6. cluster 内允许一定树冠重叠，cluster 太小会整簇回滚，避免零散碎片。

当前 GN 默认把树冠做成更适合遥感 top-view 的宽矮形态：

```text
gn_xy_stretch_min/max = 3.0 / 5.0
gn_z_stretch_min/max  = 0.25 / 0.55
```

解释上可以理解为：XY 尺度放大，让树冠在 top-view 里形成连续斑块；Z 尺度压低，再用 `tree_h_max=10m` 做最终硬上限，避免树木 depth 接近 30m 建筑。

还可以开启：

- `canopy_prob_streets` 或 `enable_street_trees`: 沿 road 边界加行道树。
- `procedural_augment_ratio`: 在真实 canopy 基础上额外加程序化树。
- `allow_non_foliage`: 允许 grass/ground 以较低权重长树。

### 5.6 树木安全距离、碰撞和孤立过滤

GN scatter 先通过 proximity mask 避开障碍物：

```text
gn_safe_building = 2.5m
gn_safe_road = 3.0m
gn_safe_water = 2.0m
```

之后会把 GN tree instances realize 成真实 mesh，再做一次几何过滤。每棵树用最终 top-view 树冠 disk 检查与障碍物的关系：

```text
obstacle_classes = {building, road, water}
collision_radius = tree_r_xy_m * topdown_tree_xy_scale + tree_building_margin
tree_min_overlap_count = 3
```

过滤规则：

1. 若树冠 disk 与 building / road / water mesh 相交，则删除该树。
2. 对剩余树计算树冠 disk 之间的重叠数量。
3. 若某棵树的 overlap count `<= tree_min_overlap_count`，当前默认即 `0..3`，则认为是孤立小树/噪声点并删除。
4. 保留下来的树写入 `GN_Filtered_Trees`，原 GN ground plane 和 hidden tree templates 不参与最终 render。

这一步会改变真实 Blender 几何，因此同时影响 depth、`.blend`、`tree_instances.json` 和最终 segmentation；不是只在 `4_seg.png` 上把 road/water 重新画回来。

### 5.7 树的 3D 资产和高度

KR3 从：

```text
assets/trees/*.blend
```

加载树种模板。每棵树实例共享 template mesh data，只复制 Object，降低 Blender 负担。

每棵树会记录：

- centered local UTM position
- target height `h`
- crown XY radius / scale
- yaw
- species

这些写到：

```text
blender/tree_instances.json
```

最终 `4_seg.png` 和 `5_depth.*` 都会尽量基于这些 tree instances 组合/渲染，而不是只靠粗 canopy polygon。

树高会在 realize、top-view XY inflate、depth render 前、`tree_instances.json` 写出前和保存 `.blend` 前多次 cap。当前默认：

```text
max_tree_height_m = 10.0
```

这个 cap 是 mesh vertex/world-Z 级别的真实几何压缩，所以保存的 `.blend`、depth、seg 合成和 metadata 中的树高保持一致。

## 6. Segmentation 如何生成并对齐

pipeline 内部有两套坐标栅格：

1. UTM grid：用于真实米制距离、KR2 几何、KR3 Blender render、tree scatter。
2. Web Mercator squashed grid：用于和 Esri / OSM XYZ tile 下载来的 `1_osm.png`、`2_rgb.png` 像素对齐。

### 6.1 Stage C 原始 OSM seg

Stage C 会先在 UTM grid 上 rasterize OSM geometry：

```text
seg_osm: uint8, shape=(1024,1024), values 0..5
```

并按 class priority 覆盖。

### 6.2 Final `4_seg.png`

Stage F 后处理会生成 Web Mercator 对齐的 `topview_treeseg.png`，再复制成：

```text
4_seg.png
```

生成方式大致是：

1. 在 Web Mercator grid 上重新 rasterize OSM vectors，和 `2_rgb.png` 对齐。
2. canopy substrate 作为 grass 底层。
3. open ground 会按 RGB green score + smooth noise，把一部分 bare ground 转成 grass。默认范围是 80%。
4. 根据 `topview_tree_mask.png` 或 `tree_instances.json` 把真实树冠画成 foliage。
5. road 和 water 对 tree mask 是硬排除。
6. 在建筑 footprint 内，使用建筑高度和树高做 occlusion：如果建筑更高，则该 pixel 保持 building；否则树冠可覆盖。

最终 `4_seg.png` 是 categorical RGB palette，不是 Blender 抗锯齿 render 的语义图。

## 7. Depth / nDSM 如何生成

KR3 会在 Blender 中输出 top-down nDSM：

```text
topview_depth.exr
topview_depth.png
```

Stage F 后处理整理成：

```text
5_depth.exr
5_depth.png
```

其中：

- `5_depth.exr`: float depth / nDSM，单位米，保留高度信息。
- `5_depth.png`: 8-bit 或可视化版本，用于快速查看和下游控制。

为了和 `2_rgb.png` / `4_seg.png` 对齐，`5_depth.png` 和 `5_depth.exr` 会从 KR3 的 UTM topview warp 到 Web Mercator tile grid。

代码里还有一个 LiDAR-like depth composer，可以直接根据 building footprint height 和 tree_instances crown height 生成 crisp DSM-like depth。当前标准输出仍主要来自 KR3 topview depth，再做 Mercator 对齐。

### 7.1 Near-nadir derived views

每个 tile 还会额外生成 4 组 near-nadir preset：

```text
near-nadir-1/
  1_seg.png
  2_depth.png
  3_shadow.png
  params.json
...
near-nadir-4/
```

这些不是替代标准 `4_seg.png` / `5_depth.png` 的 canonical top-down label，而是额外的近天底视角训练/消融条件。实现方式是读取已经像素对齐的 top-down `4_seg.png` 和 `5_depth.png`，在固定 tile frame 内根据高度做 relief displacement，并用同一个 displacement 同步 warp segmentation 和 depth；同时根据 sun angle 生成 shadow mask。

当前 preset：

| folder | camera off-nadir | camera azimuth | sun elevation | sun azimuth |
|---|---:|---:|---:|---:|
| `near-nadir-1` | 5 deg | 180 deg | 60 deg | 135 deg |
| `near-nadir-2` | 15 deg | 135 deg | 45 deg | 180 deg |
| `near-nadir-3` | 25 deg | 225 deg | 35 deg | 225 deg |
| `near-nadir-4` | 20 deg | 90 deg | 25 deg | 270 deg |

需要注意：当前 near-nadir preset 是 image-space relief displacement，不是完整 Blender/RPC 物理成像模型。它的价值在于提供更接近真实卫星 near-nadir pair 的轻量视角扰动，同时保持 `seg/depth/shadow` 在同一 warp 下对齐。

## 8. 输出与 downstream 的关系

`generation_pipeline` 期望读取：

```text
../osm_pipeline/output/<city>/tile_XXXX/
```

其中最关键的是：

- `2_rgb.png`: 真实卫星参考图，主要用于检查和评估；不假设与 OSM label 完全一致。
- `4_seg.png`: OSM/canopy/tree-instance 生成的 6-class semantic condition。
- `5_depth.png` / `5_depth.exr`: nDSM / height condition。

这些文件目标是像素对齐，供后续图像生成 pipeline 作为控制条件。

## 9. 当前 pipeline 的关键假设

1. 每个 tile 固定 `1024 px`，GSD 固定 `0.5 m/px`，即约 `512m x 512m`。
2. OSM 提供真实城市 layout 的硬结构：建筑、道路、水体、草地。
3. 建筑高度若 OSM 缺失，则用可复现随机分布补足，不是真实测高。
4. `foliage` 不信 OSM tree tag，主要用 ETH canopy height / SAM3 / Blender tree instances 补。
5. canopy polygon 是树木散布 substrate，不等于最终 semantic foliage；最终 foliage 更偏“一棵棵树冠”。
6. 内部建模用 UTM 米制坐标，最终 PNG 为了对齐 Esri/OSM tile，会转到 Web Mercator squashed grid。
7. city-level C0 union fetch 和 global raster grid 是为了减少跨 tile 接缝。
8. 道路宽度默认使用 city-level `width/lanes` 统计校准；它是本地统计近似，不是逐条道路的真实测绘宽度。
9. water 默认使用 OSM 面状水体 + 几何 sanity filter；线状水道不被强行闭合成水面。
10. 最终 `4_seg.png` 是 categorical semantic map，`5_depth.exr` 是更可靠的米制高度图，`5_depth.png` 是可视化/控制图。
11. `near-nadir-*` 是从 top-down label 派生的 aligned view perturbation，用于模拟真实卫星轻微倾斜；标准 top-down 输出仍是 canonical 几何监督。

## 10. 最短可复核命令

生成一个小 bbox：

```bash
cd osm_pipeline
python auto_pipeline.py \
  --city omaha_rich_test \
  --bbox -96.135 41.255 -96.125 41.265 \
  --tree-density 0.00015 \
  --scatter-mode canopy_prob \
  --use-blender-seg \
  --topdown-tree-xy-scale 1.8 \
  --water-fix-mode geometry_filter \
  --road-width-mode local_lanes \
  --clean
```

保存 master plan，只规划不生成：

```bash
cd osm_pipeline
python auto_pipeline.py \
  --city omaha_full \
  --bbox -96.40 41.05 -95.85 41.45 \
  --plan-only
```

按固定 plan 分批生成：

```bash
cd osm_pipeline
python auto_pipeline.py \
  --city omaha_full \
  --plan output/omaha_full/metadata/tile_plan.json \
  --tile-range 0001:1000
```

注意：当前 `auto_pipeline.py` 的 argparse 中没有 `--vlm-mode` 参数；如果参考旧 README 示例运行时报 unknown argument，应删除 `--vlm-mode skip`。

# OSM Pipeline 默认规则与实现细节

本文记录当前 `osm_pipeline` 的默认生成规则，供论文方法部分整理使用。它描述的是 2026-06-08 当前代码默认设置，覆盖 tile/GSD、OSM 语义类别、建筑高度、绿植/树木生成、碰撞过滤、seg/depth 对齐和输出 metadata。

## 1. 空间基准与 tile 设置

当前 batch pipeline 使用固定 tile 尺寸：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `gsd` | `0.5 m/px` | 输出 raster 的标称地面采样距离 |
| `size_px` | `1024` | 每个 tile 的输出图像边长 |
| `tile_m` | `512 m` | 每个 tile 的米制边长，`gsd * size_px` |
| `overlap` | `0.0` | tile 间默认无重叠 |
| CRS | local UTM | 每个 bbox 自动选择对应 UTM zone 后做米制规划 |

因此每个标准 tile 覆盖约：

```text
512 m * 512 m = 1024 px * 1024 px at 0.5 m/px
```

这里的 `0.5 m/px` 是 pipeline 的输出设计分辨率，不是 Esri World Imagery 原生分辨率。卫星图下载时会选择足够高的 XYZ zoom，然后 crop/resize 到 `1024 x 1024`；segmentation、depth、point cloud 和 Blender top-view 都被整理到同一个标称 `0.5 m/px` 网格。

## 2. OSM 语义类别

当前 segmentation 使用 6 类，所有输出统一为 `1024 x 1024`：

| 类别 | class id | 主要来源 |
|---|---:|---|
| road | 0 | OSM `highway=*` line，经道路类型 buffer 成 polygon |
| water | 1 | OSM 面状水体：`natural=water/bay/strait`、`waterway=riverbank`、reservoir 等；线状 river/stream/canal 不强行闭合成面 |
| foliage | 2 | ETH canopy / SAM canopy / Blender tree top-view mask |
| building | 3 | OSM `building=*` footprint |
| grass | 4 | OSM grass / park / garden / pitch / grassland，以及部分 open ground 重分类 |
| ground | 5 | bbox 内未被其他类别覆盖的背景 |

OSM 类别 rasterize 时采用固定优先级：

```text
ground < foliage < grass < water < road < building
```

最终树冠 mask 合成到 segmentation 时会强制排除 road/water，并且通过建筑高度图处理 tree-building occlusion。因此最终 `4_seg.png` 不是简单画图覆盖，而是融合了 OSM 几何、Blender top-view tree mask 和高度遮挡判断。

## 3. OSM 数据获取和几何处理

多 tile batch 默认先做 city-level union OSM fetch：

1. 对整个 city bbox 取一次 union OSM geometry。
2. 将 union geometry 投影到 Web Mercator/city grid。
3. 每个 tile 从 union geometry 中 clip 出自己的 class GeoJSON。
4. 若 city-level union 失败，则 fallback 到 per-tile Overpass。

这样可以降低 Overpass 请求次数，并减少跨 tile 边界处的道路/建筑拓扑不一致。

道路由 OSM centerline buffer 得到 polygon。全局 fallback 半宽按 `highway` 类型分配：

| road type | half width |
|---|---:|
| motorway | 9 m |
| trunk | 8 m |
| primary | 7 m |
| secondary | 6 m |
| tertiary | 5 m |
| residential | 4 m |
| service | 3 m |
| default | 4 m |

当前默认还会做 city-level 道路宽度校准，而不是完全依赖固定表：

| 参数 | 默认值 |
|---|---:|
| `road_width_mode` | `local_lanes` |
| `road_width_min_samples` | `3` |
| `road_lane_width_m` | `3.4 m` |
| `road_lane_margin_m` | `1.2 m` |
| `road_width_blend_weight` | `0.65` |

校准时，OSM `width=*` 优先；若缺失，则使用 `lanes * 3.4 + 1.2` 估计 full road width。每个 `highway` 类型在 city 级别取 median，样本数足够时与全局默认表按 `0.65 / 0.35` 混合。校准结果会用于重新 buffer city-level road geometry，并写入：

```text
output/<city>/config/road_width_calibration_latest.json
```

water 默认使用：

```text
water_fix_mode = geometry_filter
```

具体规则是：不把 `coastline` 当作水体面；不把 `waterway=river/stream/canal` 这类线状水道强行闭合成 polygon；保留 `waterway=riverbank` 和正常面状水体；再删除过小、极细长、低填充率或疑似 tile-filling clipped sheet 的水体 polygon。`imagery_filter` 作为可选模式存在，但当前默认采用更稳定的 `geometry_filter`。

## 4. 建筑生成规则

建筑 footprint 来自 OSM `building=*` polygon。KR2 阶段将 footprint 投影到 UTM，并 extrude 成 3D mesh。

### 4.1 建筑高度来源

每栋建筑高度按以下优先级确定：

1. 若 OSM 中存在 `height` 或 `building:height`，直接解析为米。
2. 否则若存在 `building:levels` 或 `levels`，使用：

```text
height = levels * meters_per_level
meters_per_level = 3.0 m
```

3. 若 OSM 无可用高度/层数字段，则从默认分布采样。

当前默认建筑高度参数：

| 参数 | 默认值 |
|---|---:|
| `height_dist` | `lognormal` |
| `height_seed` | `42` |
| `height_min` | `3.0 m` |
| `height_max` | `30.0 m` |

### 4.2 建筑默认 lognormal 分布

对于缺失高度 tag 的建筑，默认从截断 lognormal 分布采样：

```text
hmin = 3.0
hmax = 30.0
mid = (hmin + hmax) / 2
mu = log(mid)
sigma = max((log(hmax) - mu) / 1.96, 0.15)
h = clip(exp(N(mu, sigma)), hmin, hmax)
```

这给出以中低层为主、带高层长尾的建筑高度分布。高度最大值被硬限制为 `30 m`。采样 seed 固定，因此同一 tile 的建筑高度可复现。

支持的备选分布包括 `flat`、`uniform`、`lognormal`、`bimodal`，但当前默认使用 `lognormal`。

## 5. 绿植区域生成规则

绿植区域由多源合成得到。默认 canopy source 为：

| 参数 | 默认值 |
|---|---:|
| `canopy_source` | `eth_10m` |
| `target_foliage_ratio` | `0.25` |
| `open_ground_to_grass` | `True` |
| `open_ground_to_grass_min/max` | `0.80 / 0.80` |

Stage D 会读取 ETH 10m canopy height grid，构建 canopy `.npz`，并根据目标绿植比例做稀疏化。其逻辑是：在 canopy height map 中寻找阈值，使高于阈值的区域尽量接近 `target_foliage_ratio = 0.25`。再将该二值区域 vectorize 成 `*_foliage_canopy.geojson`，供 KR2/KR3 使用。

此外，pipeline 会把一部分 open bare ground 重分类为 grass，以避免 OSM 空白区域过度黑底化。当前默认固定为将可用 open ground 的 80% 重分类为 grass。

## 6. 树木生成规则

当前真正生效的树木生成路径是 Blender Geometry Nodes scatter，而不是旧的 Python PCG tree scatter。旧的 `tree_h_dist=lognormal` 参数仍保留用于接口兼容，但 GN 路径中最终树高主要由 tree asset mesh、GN scaling、top-view XY inflate 和最终 height cap 决定。

### 6.1 默认树木参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `scatter_mode` | `canopy_prob` | 使用 canopy probability field 控制树分布 |
| `allow_non_foliage` | `True` | 允许 open land 作为候选散布面，但通过低密度、道路/水体/建筑过滤和孤立过滤控制树量 |
| `gn_tree_amount` | `0.25` | GN 树点密度系数；当前默认值用于补回碰撞过滤后过空的住宅 tile |
| `gn_noise_scale` | `0.10` | forest patch noise scale |
| `gn_min_distance` | `3.5 m` | Poisson scatter 最小点间距 |
| `topdown_tree_xy_scale` | `1.8` | top-view 树冠 XY 膨胀倍数 |
| `tree_h_min` | `6.0 m` | legacy 参数，保留接口 |
| `tree_h_max` | `10.0 m` | 最终树高硬上限 |
| `tree_h_seed` | `11` | legacy seed |
| `tree_h_dist` | `lognormal` | legacy 分布名 |

GN 非均匀缩放默认值：

| 参数 | 默认值 |
|---|---:|
| `gn_xy_stretch` | `0.75` |
| `gn_z_stretch` | `0.50` |
| `gn_xy_stretch_min_at_0` | `3.00` |
| `gn_xy_stretch_min_at_1` | `3.00` |
| `gn_xy_stretch_max_at_0` | `5.00` |
| `gn_xy_stretch_max_at_1` | `5.00` |
| `gn_z_stretch_min_at_0` | `0.25` |
| `gn_z_stretch_min_at_1` | `0.25` |
| `gn_z_stretch_max_at_0` | `0.55` |
| `gn_z_stretch_max_at_1` | `0.55` |

解释：

- XY stretch 被设得较大，以形成遥感影像中更连续的树冠斑块。
- Z stretch 被压低，避免树木 depth 像建筑一样冲到 20-30m。
- `topdown_tree_xy_scale=1.8` 会永久作用于 realized tree object，使 segmentation 中的树冠范围和保存的 `.blend` / depth 一致。
- 树高最终用 geometry-level cap 硬限制到 `10 m`。

### 6.2 树木候选位置

GN scatter 在地面 plane 上根据 canopy probability、patch noise 和安全距离生成候选点。当前允许 open land 作为候选面，因此不只限于 OSM `foliage` polygon；这有利于补充 OSM 缺失树木，但后续会通过几何过滤去掉不合理位置。

安全距离默认：

| 参数 | 默认值 |
|---|---:|
| `gn_safe_building` | `2.5 m` |
| `gn_safe_road` | `3.0 m` |
| `gn_safe_water` | `2.0 m` |

这些安全距离参与 GN proximity mask；之后还有 realized mesh 级别的二次过滤。

### 6.3 树木碰撞与孤立树过滤

GN 生成虚拟 tree instances 后，会将实例 realize 为真实 mesh，并执行几何过滤。过滤对象包括：

```text
obstacle_classes = {building, road, water}
```

每棵树用最终 XY 树冠 disk 做碰撞检测：

```text
collision_radius = tree_r_xy_m * topdown_tree_xy_scale + margin_m
```

当前默认：

| 参数 | 默认值 |
|---|---:|
| `tree_building_collision` | `True` |
| `tree_building_margin` | `0.0 m` |
| `tree_min_overlap_count` | `3` |
| `topdown_tree_xy_scale` | `1.8` |

过滤逻辑：

1. 如果树冠 disk 与 building / road / water mesh 相交，则删除该树。
2. 对剩余树计算树冠 disk 的相互重叠数。
3. 若某棵树的 overlap count `<= tree_min_overlap_count`，当前默认即 `0..3`，则认为它是孤立小树/噪声点并删除。
4. 保留的树写入 `GN_Filtered_Trees` collection，原 GN ground plane 和 hidden tree templates 不参与最终渲染。

这一步是几何约束，影响 depth 和 `.blend`，不是单纯在 segmentation 上把 road/water 画回去。

### 6.4 树高上限

树高 cap 在多个阶段执行：

1. realize tree instance 时，根据当前 world-space bbox 初步压缩高度。
2. top-view XY inflate 后，再次检查并 cap。
3. depth 渲染前执行 `pre-depth tree height cap`。
4. 写 `tree_instances.json` 前执行 `pre-dump tree height cap`。
5. 保存 `.blend` 前后再次执行 cap。

最终 cap 采用 mesh vertex 级别的 world-Z 压缩，保证实际渲染几何、depth、`tree_instances.json` 和保存的 `.blend` 一致：

```text
max_tree_height_m = 10.0
```

在最近一次 Omaha 20 tile 默认测试中，树高统计为：

```text
n    = 1728 trees
p50  = 4.69 m
mean = 4.86 m
p90  = 7.68 m
max  = 10.00 m
```

## 7. Segmentation 与 depth 对齐

当前默认：

```text
use_blender_seg = True
```

因此 Stage F 会使用 Blender top-view render 获得 tree-crown mask，再与 Web Mercator 原生 OSM class map 合成。合成时：

- road 和 water 对 tree mask 是硬排除。
- building 与 tree 的重叠按高度判断；如果 tree height 低于 building height，则 tree 被视为被建筑遮挡。
- `seg_6class_notree.png` 会保留未加树的底图，`seg_6class.png` / `4_seg.png` 为最终合成图。

Depth 输出包括：

| 文件 | 含义 |
|---|---|
| `5_depth.exr` | float depth/height 信息 |
| `5_depth.png` | 归一化可视化 depth |

Depth 最大高度归一化默认仍按 `30 m`，与建筑高度上限一致。因此树木最高 `10 m`，建筑最高 `30 m`，两者在 depth 图中会有明显量级差异。

## 8. Near-nadir 派生视角

除 canonical top-down `4_seg.png` / `5_depth.png` 外，每个 tile 还会写出 4 组 near-nadir preset：

```text
near-nadir-1/1_seg.png
near-nadir-1/2_depth.png
near-nadir-1/3_shadow.png
near-nadir-1/params.json
...
near-nadir-4/...
```

这些 preset 由 `scripts/render_sat_depth_tests.py` 从已对齐的 top-down seg/depth 派生：根据 height map、camera off-nadir angle 和 azimuth 在固定 tile frame 内做 relief displacement，并用同一 warp 同步变换 segmentation 和 depth；shadow mask 根据 sun angle 从 warped depth 投影得到。

| folder | camera off-nadir | camera azimuth | sun elevation | sun azimuth |
|---|---:|---:|---:|---:|
| `near-nadir-1` | 5 deg | 180 deg | 60 deg | 135 deg |
| `near-nadir-2` | 15 deg | 135 deg | 45 deg | 180 deg |
| `near-nadir-3` | 25 deg | 225 deg | 35 deg | 225 deg |
| `near-nadir-4` | 20 deg | 90 deg | 25 deg | 270 deg |

Near-nadir 输出的定位是轻量模拟真实卫星影像中常见的 mild off-nadir 成像几何，用于训练或消融更接近 US3D/DFC 这类真实 satellite pair 的输入分布。它不是完整 Blender/RPC 物理相机渲染，也不替代 top-down canonical label。

## 9. 输出 metadata

每个 tile 保存：

```text
metadata/tile_XXXX.json
metadata/tile_XXXX.meta.json
metadata/tile_XXXX_osm_buildings.geojson
blender/tree_instances.json
```

每次 batch run 还会保存项目级配置和进度日志：

```text
output/<city>/config/run_config_latest.json
output/<city>/config/run_config_<timestamp>_<status>.json
output/<city>/config/road_width_calibration_latest.json
output/<city>/metadata/run_live.log
output/<city>/metadata/run_progress_latest.json
output/<city>/metadata/run_timing_latest.json
```

关键字段包括：

- `gsd_m_per_px`
- `image_size_px`
- `tile_extent_m`
- WGS84 bbox
- UTM center
- per-building `height_m`
- per-tree `{x_centered, y_centered, h, r_xy_m}`
- tree collision/filter stats:

```text
tree_building_collision_filter:
  total
  kept
  removed
  removed_by_class: {building, road, water}
  removed_isolated
  min_tree_overlaps
  xy_scale_multiply
  max_tree_height_m
```

这些 metadata 可用于复现实验、统计高度分布、检查树木被 building/road/water 删除的数量，以及后续论文中报告 procedural model 的参数。

## 10. 当前默认参数汇总

| 模块 | 参数 | 默认值 |
|---|---|---:|
| tile | `gsd` | `0.5 m/px` |
| tile | `size_px` | `1024` |
| tile | `tile_m` | `512 m` |
| building | `height_dist` | `lognormal` |
| building | `height_min` | `3.0 m` |
| building | `height_max` | `30.0 m` |
| building | `height_seed` | `42` |
| grass | `open_ground_to_grass` | `True` |
| grass | `open_ground_to_grass_min/max` | `0.80 / 0.80` |
| canopy | `canopy_source` | `eth_10m` |
| canopy | `target_foliage_ratio` | `0.25` |
| tree | `scatter_mode` | `canopy_prob` |
| tree | `allow_non_foliage` | `True` |
| tree | `gn_tree_amount` | `0.25` |
| tree | `gn_min_distance` | `3.5 m` |
| tree | `gn_safe_building` | `2.5 m` |
| tree | `gn_safe_road` | `3.0 m` |
| tree | `gn_safe_water` | `2.0 m` |
| tree | `topdown_tree_xy_scale` | `1.8` |
| tree | `gn_xy_stretch_min/max` | `3.0 / 5.0` |
| tree | `gn_z_stretch_min/max` | `0.25 / 0.55` |
| tree | `tree_h_max` | `10.0 m` |
| tree filter | `tree_min_overlap_count` | `3` |
| water | `water_fix_mode` | `geometry_filter` |
| road | `road_width_mode` | `local_lanes` |
| road | `road_lane_width_m / margin` | `3.4 m / 1.2 m` |
| road | `road_width_blend_weight` | `0.65` |
| rendering | `use_blender_seg` | `True` |
| depth | `depth_max_height_m` | `30.0 m` |
| near-nadir | presets | `4 views: 5/15/25/20 deg off-nadir` |

## 11. 可写入论文的方法描述草稿

We generate each sample as a georeferenced 512 m by 512 m tile rendered at 1024 by 1024 pixels, corresponding to a nominal ground sampling distance of 0.5 m/px. OSM building footprints, roads, water bodies, grass/park areas, and residual ground are projected into a local UTM coordinate system and rasterized into a six-class semantic map. Road widths are calibrated at the city level from OSM width and lane tags, while water polygons are filtered to remove common forced-closure and clipped-sheet artifacts. Building heights are obtained from OSM height or level tags when available; otherwise, they are sampled from a clipped log-normal distribution in [3 m, 30 m] with a fixed random seed.

Vegetation is synthesized from a canopy-height prior and Blender Geometry Nodes. A canopy mask is thinned to a target foliage ratio of 0.25 and used as the foliage substrate for tree scattering; open-land scatter is enabled at low density by default. Tree instances are generated with anisotropic scaling to create broad, connected canopy patches in top view, while vertical scaling is constrained and final tree geometry is capped to 10 m. Candidate trees whose crown disks intersect building, road, or water geometry are removed, and isolated tree fragments with insufficient crown overlap are discarded. The final top-view tree mask is rendered in Blender and merged with the OSM semantic map, while depth is rendered from the same realized 3D geometry to maintain pixel-level consistency between semantic and depth outputs. In addition to the canonical nadir labels, we derive four near-nadir paired views by applying height-driven relief displacement to the aligned semantic and depth maps, producing view-consistent segmentation, depth, and shadow masks for mild off-nadir satellite training conditions.

# osm_pipeline (3D 城市/绿化离模流水线系统)

> 精简版：基于原 `ProcedureOSM/` 重构，仅保留 `osm_app.py` + `auto_pipeline.py` 实际使用的代码路径。
>
> **2026-05-28 重构清单**：移除 VLM 提示生成（Stage G）、SAM3 植被分割入口、UE5 OBJ 导出（Stage 6）、可选 city blend 合并（Stage I），以及 `on_save_and_build` 旧 step-4 处理函数等未挂接到 UI 的 orphan handler。当前 Stage 序列：`B → C → D → E → F → H`。注意：`dataprep/sam3_to_polygons.py` 仍保留，因为 ETH canopy 树冠矢量化复用了其中的 mask→polygon 工具函数，并不代表恢复 SAM3 推理阶段。
>
> 一键将任意经纬度包围圈 (WGS84 Bounding Box) 转换为适用于 Blender 渲染的高逼真 3D 城市瓦片，
> 并生成用于 FLUX2 控制网 (ControlNet) 训练的高空正射 **(1_osm, 2_rgb, 3_seg, 4_depth, 5_depth)** 完美对齐训练集。
>
> 6个分类标签语义层：`ground=0, foliage=1, grass=2, water=3, building=4, road=5`

---

## 1. 核心管线与输出结构

本系统通过一个轻量化的后台多线程批处理系统 [auto_pipeline.py](auto_pipeline.py) 实现高通量并行生成。

### 1.1 瓦片目录输出规范
对于生成的每个瓦片 (`tile_XXXX`)，其输出目录内提供高度标准化、已经过 Web 墨卡托重采样且严格像素对齐的 5 大核心产品：
1. **`1_osm.png`**：OSM Carto 风格的城市矢量特征底图。
2. **`2_rgb.png`**：Esri WorldImagery 高清航空卫星影像。
3. **`3_seg.png`**：结合了 Blender 真实物理散射树冠投影的 6 分类高保真语义分割图 (Native Mercator-aligned with trees)。
4. **`4_depth.png`**：16-bit 线性绝对物理尺度 LiDAR 模拟灰度高度图 (0.0=地面, 1.0=50米物理高度截断)。
5. **`5_depth.exr`**：32-bit Float 单通道物理绝对高度图（不截断，供深度学习/三维重建高保真解析）。

### 1.2 城市与瓦片输出结构规范
为确保训练集环境极致纯净，管线在 Stage F 完成后将**自动擦除全部冗余中间件**（如原始 raw 瓦片、中间 topview png/exr 以及 Blender 运行 stderr 临时日志），最终生成高标准、模块化布局的城市瓦片数据集。

```
output/<city>/
├── <city>_osm.png           # 城市整图拼合 - OSM 底图 (基于瓦片无畸变完美拼接)
├── <city>_rgb.png           # 城市整图拼合 - Esri 卫星影像 (无缝对齐)
├── <city>_seg.png           # 城市整图拼合 - 完美对齐的 6-class 语义分割（带物理树冠）
├── <city>_depth.png         # 城市整图拼合 - 物理 16-bit LiDAR 深度图
├── metadata/                # 城市级全局记录与断点参数配置
│   ├── <city>.json          # 全面记录区块拓扑信息、经纬度、建筑物信息汇总
│   ├── tile_index.geojson   # 区块地理边界 GeoJSON
│   ├── .pipeline_state.json # 管线断点状态进度日志 (Pipeline States)
│   ├── _failures.json       # 记录失败瓦片与重试状态
│   └── strategy.json        # 植被散射策略参数备忘
└── tile_XXXX/               # 单格标准瓦片目录 (NW 坐标顺序编号)
    ├── 1_osm.png            # 1024×1024 标准对齐产品
    ├── 2_rgb.png            # 1024×1024 标准对齐产品
    ├── 3_seg.png            # 1024×1024 标准对齐产品 (物理树冠平铺)
    ├── 4_depth.png          # 1024×1024 标准对齐产品 (16-bit 线性绝对深度)
    ├── 5_depth.exr          # 32-bit Float 原始精度物理深度文件 (重构核心)
    ├── blender/
    │   ├── <tile>.glb       # 阶段E导出的 6-class 三维物理模型
    │   └── <tile>.blend     # 阶段F装配完成的 Blender 工程主场景 (支持重新烘焙)
    └── metadata/
        ├── <tile>.json      # 建筑物 UTM 精定位、建筑几何多边形参数
        ├── <tile>.meta.json # KR2 高度、投影和顶点元数据
        └── <tile>_osm_buildings.geojson # 瓦片内 raw OSM 建筑物矢量集合
```

> **注意：** 所有临时生成的中间缓存文件（如雷达高度 `canopy/*.npz`、矢量分块 `geojson/*.geojson`、诊断 `fig/*` 等）均已重定向到全局 `cache/` 目录中，在 `output/` 文件夹下实现了 100% 的纯净归档。

---

## 2. 快速开始与环境部署

### 2.1 环境安装
```powershell
python -m venv .venv\pcg
.\.venv\pcg\Scripts\Activate.ps1

pip install -r requirements.txt
# 目前支持最精简轻量运行
```
Blender 软件（版本 $\ge 4.0$）必须加入系统 `PATH` 环境变量中，因为流水线会通过无窗口命令行 `blender --background` 驱动装配和渲染程序。

### 2.2 测试流水线
使用命令行运行单瓦片或多瓦片拼接任务（已验证的 omaha 示例）：

```powershell
# (1) 单瓦片（约 5 km × 5 km 内的一个 512 m tile）
python auto_pipeline.py --city omaha_single --bbox -96.135 41.260 -96.130 41.265 `
    --vlm-mode skip --tree-density 0.00015 --scatter-mode canopy_prob `
    --allow-non-foliage --use-blender-seg --topdown-tree-xy-scale 3.5 --clean

# (2) 6 瓦片（3×2 网格，omaha 富树郊区，约 1.3 km × 1 km）
python auto_pipeline.py --city omaha_rich_test --bbox -96.135 41.255 -96.125 41.265 `
    --vlm-mode skip --tree-density 0.00015 --scatter-mode canopy_prob `
    --allow-non-foliage --use-blender-seg --topdown-tree-xy-scale 3.5 --clean
```

> 想边调参边可视化？跑 `python osm_app.py`，浏览器开 http://127.0.0.1:8765 — 单瓦片调试 Tab 用「Generate All」一键出全套 PNG + .blend，参数面板与下方第 3 节速查表完全对应。

---

## 3. 树木与场景参数完整指南（AutoPipelineConfig）

所有可调参数集中在一个 `@dataclass`：[AutoPipelineConfig](ProcedureOSM/auto_pipeline.py#L775)（[ProcedureOSM/auto_pipeline.py](ProcedureOSM/auto_pipeline.py)，从第 775 行开始）。  
**改默认值就改这个类的字段**（永久生效）；**临时调一次**就传同名 CLI flag（一次性覆盖）。Gradio UI（[osm_app.py](ProcedureOSM/osm_app.py)）也是从这个类反射读默认值，所以三处永远同步。

### 3.0 全参数速查表（按功能分组）

> 🔗 = 点击跳到 `auto_pipeline.py` 对应行；CLI flag 留空表示「只能在类里改默认值，无 CLI 覆盖」。

#### 🏗️ 建筑物高度

| 字段 | 默认 | CLI flag | 作用 |
| :--- | :---: | :--- | :--- |
| [height_dist](ProcedureOSM/auto_pipeline.py#L780) | `lognormal` | — | 建筑高度采样分布：`uniform` \| `lognormal` \| `bimodal` |
| [height_seed](ProcedureOSM/auto_pipeline.py#L781) | `42` | — | 高度随机种子 |
| [height_min](ProcedureOSM/auto_pipeline.py#L782) | `3.0` | — | 最低建筑高 (m) |
| [height_max](ProcedureOSM/auto_pipeline.py#L783) | `30.0` | — | 最高建筑高 (m) |

#### 🌳 树木——数量与密度

| 字段 | 默认 | CLI flag | 作用 |
| :--- | :---: | :--- | :--- |
| [tree_density](ProcedureOSM/auto_pipeline.py#L786) | `0.005` | `--tree-density` | 全局树密度 (棵/m²)。**最常调** |
| [scatter_mode](ProcedureOSM/auto_pipeline.py#L826) | `canopy_prob` | `--scatter-mode` | 散射算法：`canopy_prob` \| `canopy_prob_streets` \| `cluster` \| `noise_forest` 等 |
| [allow_non_foliage](ProcedureOSM/auto_pipeline.py#L830) | `True` | `--allow-non-foliage` / `--no-allow-non-foliage` | 是否允许在非 OSM foliage 区（草地/空地）散射 |
| [enable_street_trees](ProcedureOSM/auto_pipeline.py#L832) | `False` | `--enable-street-trees` | 道路中线两侧加街道树 |
| [procedural_augment_ratio](ProcedureOSM/auto_pipeline.py#L837) | `0.0` | `--procedural-augment-ratio` | 在 ETH 真实树之外再补 N% 程序化树 |
| [canopy_prob_scale](ProcedureOSM/auto_pipeline.py#L839) | `1.0` | `--canopy-prob-scale` | 对 ETH 每像元生树概率乘以系数（>1 更密） |
| [scatter_seed](ProcedureOSM/auto_pipeline.py#L785) | `11` | — | 散射随机种子 |

#### 🌳 树木——XY 冠幅（横向大小）

| 字段 | 默认 | CLI flag | 作用 |
| :--- | :---: | :--- | :--- |
| [topdown_tree_xy_scale](ProcedureOSM/auto_pipeline.py#L841) | `1.0` | `--topdown-tree-xy-scale` | **仅 XY 缩放**（Z 不动），渲染 remote-sensing 风大斑块用 `3.5` |
| [uniform_tree_scale](ProcedureOSM/auto_pipeline.py#L820) | `False` | `--uniform-tree-scale` / `--no-uniform-tree-scale` | 是否按高度均匀缩放 |

#### 🌳 树木——Z 高度

| 字段 | 默认 | CLI flag | 作用 |
| :--- | :---: | :--- | :--- |
| [tree_h_dist](ProcedureOSM/auto_pipeline.py#L788) | `lognormal` | `--tree-height-dist` | 树高分布：`flat` \| `uniform` \| `lognormal` \| `bimodal` \| `beta_u` |
| [tree_h_seed](ProcedureOSM/auto_pipeline.py#L789) | `11` | — | 树高随机种子 |
| [tree_h_min](ProcedureOSM/auto_pipeline.py#L790) | `6.0` | `--tree-height-min` | 最低树高 (m) |
| [tree_h_max](ProcedureOSM/auto_pipeline.py#L791) | `20.0` | `--tree-height-max` | 最高树高 (m) |
| [tree_height_low_frac](ProcedureOSM/auto_pipeline.py#L807) | `0.65` | `--tree-height-low-frac` | `bimodal` 时低端峰占比 (0–1) |

#### 🌳 树木——群聚结构

| 字段 | 默认 | CLI flag | 作用 |
| :--- | :---: | :--- | :--- |
| [cluster_size_min](ProcedureOSM/auto_pipeline.py#L796) | `10` | `--cluster-size-min` | 每个树群最少棵数 |
| [cluster_size_max](ProcedureOSM/auto_pipeline.py#L797) | `20` | `--cluster-size-max` | 每个树群最多棵数 |
| [cluster_disk_radius_min](ProcedureOSM/auto_pipeline.py#L798) | `4.0` | `--cluster-disk-radius-min` | 群聚椭圆最小半径 (m) |
| [cluster_disk_radius_max](ProcedureOSM/auto_pipeline.py#L799) | `10.0` | `--cluster-disk-radius-max` | 群聚椭圆最大半径 (m) |
| [cluster_disk_aspect](ProcedureOSM/auto_pipeline.py#L800) | `0.65` | `--cluster-disk-aspect` | 椭圆短/长轴比，<1 拉长，=1 正圆 |
| [cluster_size_dist](ProcedureOSM/auto_pipeline.py#L805) | `uniform` | `--cluster-size-dist` | 群大小分布：`uniform` \| `bimodal` \| `beta_u` |
| [cluster_size_low_frac](ProcedureOSM/auto_pipeline.py#L806) | `0.7` | `--cluster-size-low-frac` | `bimodal` 时小群占比 |
| [cluster_overlap_factor](ProcedureOSM/auto_pipeline.py#L814) | `0.45` | `--cluster-overlap-factor` | 群内树冠允许重叠程度，<1 冠融合 |
| [cluster_min_keep_ratio](ProcedureOSM/auto_pipeline.py#L815) | `0.6` | `--cluster-min-keep-ratio` | 群放不下到这个比例就整组丢弃 |
| [cluster_min_size_abs](ProcedureOSM/auto_pipeline.py#L819) | `10` | `--cluster-min-size-abs` | **绝对最小棵数**，少于则整组删除（=0 关闭） |
| [tree_species](ProcedureOSM/auto_pipeline.py#L787) | `None` | — | 可用树种 .blend 列表（None=自动扫 `assets/trees/`） |

#### 🎨 渲染 / 全局

| 字段 | 默认 | CLI flag | 作用 |
| :--- | :---: | :--- | :--- |
| [use_blender_seg](ProcedureOSM/auto_pipeline.py#L843) | `True` | `--use-blender-seg` / `--no-blender-seg` | 用 Blender 直渲 topview 当 seg（保证与 depth 1:1 对齐） |
| [canopy_source](ProcedureOSM/auto_pipeline.py#L845) | `eth_10m` | — | 冠层高度数据源 |
| [target_foliage_ratio](ProcedureOSM/auto_pipeline.py#L846) | `None` | `--target-foliage-ratio` | 目标植被覆盖比例（自动调高度阈值） |
| [vlm_mode](ProcedureOSM/auto_pipeline.py#L848) | `per_tile` | `--vlm-mode` | VLM prompt 模式：`per_tile` \| `city_only` \| `skip` |
| [overlap](ProcedureOSM/auto_pipeline.py#L778) | `0.0` | `--overlap` | 相邻瓦片重叠比例 |
| [gsd](ProcedureOSM/auto_pipeline.py#L854) | `0.5` | — | 地面采样距离 (m/像素)，锁死 |
| [size_px](ProcedureOSM/auto_pipeline.py#L855) | `1024` | — | 单瓦片像素边长，锁死 |
| [io_workers](ProcedureOSM/auto_pipeline.py#L850) | `8` | `--io-workers` | Stage B 并发数 |
| [osm_workers](ProcedureOSM/auto_pipeline.py#L851) | `4` | `--osm-workers` | Stage C 并发数 |
| [canopy_workers](ProcedureOSM/auto_pipeline.py#L852) | `4` | `--canopy-workers` | Stage D 并发数 |

---

### 3.1 调参指南——数量与密度

- **`tree_density`** 是最常用旋钮。`0.005` = 5 棵/100 m²（密林），`0.00015` = 1.5 棵/10000 m²（稀疏郊区）。城市 omaha 实测 `0.00015` 视觉舒适。
- **`scatter_mode`** 默认 `canopy_prob` 用 ETH 10 m 真实冠层做概率密度场，最贴近航片。
- **`allow_non_foliage=True`** 是 omaha/芝加哥这种 OSM foliage 标注稀缺城市的救命稻草——靠 ETH canopy + 排除 building/road/water 散射，能在住宅院落补出真实绿化。
- **`canopy_prob_scale`** 与 `tree_density` 是乘法关系，想小幅微调密度优先调它（保持群聚形态不变）。

```powershell
# 稀疏城市郊区
--tree-density 0.00015 --scatter-mode canopy_prob --allow-non-foliage
# 密林公园
--tree-density 0.005 --canopy-prob-scale 1.5
```

### 3.2 调参指南——XY 冠幅（横向）

⚠️ **`topdown_tree_xy_scale` 只缩放 XY，Z 不动**（2026-05 修过的 bug）。  
想让 3_seg.png 看起来像 remote-sensing 真实树冠大斑块？传 `3.5`；想看真实物理冠幅？传 `1.0`。该缩放是**永久写入 .blend**，所以 3_seg.png 与三维场景 1:1 对齐。

```powershell
--topdown-tree-xy-scale 3.5    # 大斑块 (用于 ControlNet 训练目标)
--topdown-tree-xy-scale 1.0    # 真实物理冠幅
```

### 3.3 调参指南——Z 高度（独立于 XY）

- `tree_h_min/max` 严格控制单棵树绝对高度区间。
- `tree_h_dist=lognormal` 自然生态分布（少量大树，多数中小灌木）。
- `bimodal` 适合「乔木 + 灌木」双层林，配 `tree_height_low_frac` 调灌木占比。

```powershell
--tree-height-dist lognormal --tree-height-min 6 --tree-height-max 20
```

### 3.4 调参指南——群聚结构

- **想要连绵大林**：`cluster_size_min/max` 加大（30/60），`cluster_disk_radius_max` 加到 20，`cluster_overlap_factor` 调小到 0.35。
- **想要散点绿岛**：`cluster_size_min/max` 减到 3/8，`cluster_min_size_abs=0`。
- **碎片噪声太多**：`cluster_min_size_abs=10`（默认就是 10）能整组删除 <10 棵的零碎树群。

```powershell
# 大斑块连绵森林
--cluster-size-min 30 --cluster-size-max 60 --cluster-disk-radius-max 20 \
  --cluster-overlap-factor 0.35 --cluster-size-dist bimodal
```

### 3.5 调参指南——建筑物高度

OSM 缺高度信息时按分布采样。`lognormal` 适合多数城市；想做 manhattan 高度对比强烈的天际线就 `bimodal`。

```powershell
# 注意：这几个字段当前只能改 AutoPipelineConfig 默认值或在 osm_app UI 改
# （没暴露成 CLI flag，因为多数城市默认 3-30 m 就够）
```

### 3.6 调参指南——渲染开关

- **`use_blender_seg=True`**（默认）：3_seg.png 直接用 Blender 物理散射后的 topview 渲染，与 4_depth.png/5_depth.exr 像素级 1:1 对齐。**强烈推荐保留**。
- **`vlm_mode=skip`**：跳过 Mistral 自然语言描述生成，省下 ~3 s/tile 模型加载。批量训练数据准备建议保留 `per_tile`。
- **`target_foliage_ratio`**：硬性目标植被覆盖率（如 `0.15`=15%）。系统会自动反算 ETH 高度阈值，强迫输出达到目标。

```powershell
--use-blender-seg --vlm-mode skip --target-foliage-ratio 0.15
```

---

## 5. 项目工程布局规范 (Repo Layout)

请参阅以下代码工程的结构：
* [ProcedureOSM/auto_pipeline.py](ProcedureOSM/auto_pipeline.py) — 统一控制流与拼接纠错。
* [ProcedureOSM/osm_app.py](ProcedureOSM/osm_app.py) — Gradio 图形交互控制与多并行中转。
* [ProcedureOSM/scripts/3_blender_assemble.py](ProcedureOSM/scripts/3_blender_assemble.py) — Blender 无窗口自动化几何物化、散射渲染及绝对雷达测距。
* [ProcedureOSM/configs/default.yaml](ProcedureOSM/configs/default.yaml) — 基本网格 GSD、建筑物、层高与道路拓扑半径全局配置参数。
* [ProcedureOSM/dataprep/](ProcedureOSM/dataprep/) — 包含地理几何纠歪、OSM 数据清洗与雷达高度转换的核心工具层。

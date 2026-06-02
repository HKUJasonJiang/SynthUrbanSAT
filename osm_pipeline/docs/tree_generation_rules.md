# 树木生成规则（KR3 Tree Scatter Rules）

> 适用范围：`SynthUrbanSAT/osm_pipeline` KR3 阶段（[scripts/3_blender_assemble.py](../scripts/3_blender_assemble.py)）。
> 树木 = 真实 `.blend` 物种模板 + 在 Blender 场景里按规则散布（scatter）出来的实例。
> 配置文件：[configs/default.yaml](../configs/default.yaml) `tree_assets:` 段。

---

## 0. 总览（先结论）

KR3 在已生成的 GLB 场景里"种树"的核心是两步：

1. **资产准备**：从 `assets/trees/<species>.blend` 里把每个物种的主体 mesh 取出来作为 **template**（隐藏在 `_tree_templates` 集合）。所有实例都 **共享 template 的 mesh 数据块**，只复制 Object —— Blender 端轻量、KR4 渲染快。
2. **散布（scatter）**：按某种 **scatter_mode** 决定 *在哪、放多少、放多高* 的树，再实例化 template、加随机偏航和缩放。

`scatter_mode` 一共 4 种主模式 + 1 种附加模式（街道树），默认 `canopy_prob`。

| 模式 | 数据来源 | 何时用 | 实现函数 |
|------|---------|--------|----------|
| `cluster` | 仅 OSM foliage 多边形 | 无 canopy 数据时的"程序化"散布 | `_scatter_pcg` |
| `poisson_disk` | OSM foliage 多边形 | 想要不重叠树冠的均匀分布 | `_scatter_poisson_disk` |
| `canopy_driven` | ETH/Meta canopy 高度网格 | 想让密度+高度都跟真实树冠数据走 | `_scatter_canopy_driven` |
| `canopy_prob` ⭐默认 | ETH/Meta canopy 高度网格 | 概率式 Bernoulli + 软类别权重 + 簇展开 | `_scatter_canopy_probabilistic` |
| `canopy_prob_streets` | 上者 + 道路边界 | 在 `canopy_prob` 基础上再加街道行道树 | + `_add_street_trees` |

---

## 1. 资产装载：物种 → template

代码：`_load_tree_species_dir()`（[L505 附近](../scripts/3_blender_assemble.py)）

- 遍历 `tree_assets.dir`（默认 `assets/trees/`）下的每个 `.blend` 文件，文件名（不含扩展名）即 **物种名**（如 `oak.blend` → species `oak`）。
- 从每个 .blend 里挑一个代表性 MESH 作 template，记录其 base height（包围盒 Z 高度）和 base Z（底部到原点的距离，用于"把树干放到 z=0"的对齐）。
- 文件名以 `bush` 开头会被打上 `is_bush=True`，后续高度会乘 0.35（变矮树丛）。
- 若 `tree_assets.dir` 没东西，回退到 `foliage_assets.blend_file` 的整包 Nature_Pack。

**关键点：所有树共享 mesh 数据**

```python
new_obj = bpy.data.objects.new(name_hint, template.data)  # 共享 .data
new_obj["class"] = "foliage"; new_obj["class_id"] = 2
new_obj["is_tree_instance"] = True                          # 给后续渲染/计数用
```

`is_tree_instance` 这个标记让 KR3 的 top-view 渲染可以 **隐掉 foliage 多边形底板**，只让真正的树冠贡献绿色像素 —— 这是 KR3 自然鸟瞰图的关键。

---

## 2. 高度采样：`_sample_tree_height`

[L587](../scripts/3_blender_assemble.py)。在 `[hmin, hmax] = height_range_m`（默认 `[3, 14]` m）里按分布采样：

| `height_dist` | 含义 |
|---|---|
| `flat` | 取中点（基本不用） |
| `uniform` | 区间均匀 |
| `lognormal` ⭐默认 | 中位数≈中点，长尾偏小（贴近城市真实分布） |
| `bimodal` | 双峰：`low_frac` 在 hmin 附近（小树/灌木），其余在 hmax 附近（成树）；中间稀少 |
| `beta_u` | Beta(0.5, 0.5) 映射 —— 也是两端集中 |

采样后乘以 `height_scale` 并夹到 `[0.5, 4*hmax]` 防异常。

对每棵树最终的 **缩放比**：

$$
\text{scale} = \frac{\text{target\_h}}{\text{base\_h}}
$$

并把 Z 偏移 `-base_z * scale` 让树干贴 z=0。

---

## 3. 模式 1：`cluster`（`_scatter_pcg`，[L712](../scripts/3_blender_assemble.py)）

UE-PCG 风格的"独立点 + 聚簇"散布。**不依赖** canopy 数据。

**步骤**：

1. 收集 `class==foliage` 的三角面（排除已是 tree/bush 的 mesh），和 `class==ground/grass` 三角面。每张面积权重 = 面积 $A$。
2. 每个 surface 目标数 $N = A_\text{total} \cdot \text{density}$（受 `max_trees` 限）。
3. 拆分：

   $$
   N_\text{indep} = N(1-c), \quad N_\text{clusters} = \frac{N - N_\text{indep}}{K_\text{children}}
   $$

   其中 $c=$`cluster_strength`（0~1），$K=$`cluster_children`。
4. **独立点**：在三角面上做面积加权采样，再用重心坐标随机取点。
5. **簇**：先选种子点；在种子周围放 $K$ 棵子树，XY 偏移服从 $\mathcal{N}(0, \sigma^2)$，$\sigma=$`cluster_radius_m`。
6. 每棵树：随机选 species → 采高度 → 随机 yaw → spawn → 检测 `_pos_blocked_by_building` 防止穿墙。

---

## 4. 模式 2：`poisson_disk`（`_scatter_poisson_disk`）

跟 `cluster` 同样的 surface 三角面采样，但引入 **可变半径泊松盘约束**：

- 每棵树有 $r_\min = 0.4 \cdot h$（粗略树冠半径，与高度成正比）。
- 新候选点 $(x,y)$ 接受当且仅当：与任何已放置树 $i$ 满足 $\lVert \mathbf{p} - \mathbf{p}_i \rVert \geq r_\min + r_{\min,i}$。
- 用空间哈希 `_hash_key`（2m 单元）加速近邻查询。

结果：树冠 **不重叠**、视觉上更"工整"，适合公园 / 商业绿化带。

---

## 5. 模式 3：`canopy_driven`（`_scatter_canopy_driven`，[L1139](../scripts/3_blender_assemble.py)）

需要 ETH/Meta canopy 高度网格 NPZ（由 [dataprep/canopy_height.py](../dataprep/canopy_height.py) 产生）：

```
{ h: (N,N) float32 米, gsd: 米/像素, size: N, extent: 边长(米) }
```

NW-origin、行主序；像元中心局部 UTM：`x = (col+0.5)*g`，`y = ((N-1-row)+0.5)*g`。

**步骤**：

1. **eligible cells**：所有 `h >= min_canopy_h`（默认 2 m）的 (row, col)。
2. **目标总数**：

   $$
   N_\text{target} = \text{base\_density} \cdot A_\text{eligible} \cdot \text{boost}
   $$

   `boost` 历史上是 4，**现在固定为 1**（更可控）。若结果为 0 给个保底 `max(50, n_eligible//4)`。
3. **簇模式**：把目标拆成 `n_clusters = N_target / avg_cluster_size` 个簇。簇尺寸 `_sample_cluster_size(cs_min, cs_max, dist)`，同样支持 `uniform/bimodal/beta_u`。
4. 每簇：
   - 随机选一个 eligible cell 作种子（cell 中心）→ 必须落在 foliage 上（`on_foliage` 射线测试）。
   - 簇形状是 **随机朝向的椭圆**：$r_\max \in U(\text{cr\_min}, \text{cr\_max})$，$r_\min = r_\max \cdot \text{aspect}$，旋转角 $\theta \sim U(0, 2\pi)$。
   - 在单位圆内拒绝采样，再拉伸到椭圆并旋转 $\theta$ 落点。
   - 每棵树的高度优先用 **该点 canopy 高度 × U(0.85, 1.05)**，越界再退到种子的 canopy 高度。
   - 间距：$r_\min = \max(1.2,\ 0.35 h)$，用同一个 hash 防过密。

---

## 6. 模式 4：`canopy_prob` ⭐ 默认（`_scatter_canopy_probabilistic`，[L1437](../scripts/3_blender_assemble.py)）

**逐元胞 Bernoulli + 软类别权重 + 簇展开 + 碎片回滚**。当前最贴近真实 ETH 树冠空间分布的模式。

### 6.1 单元格的"投币"概率

对每个 `h >= min_canopy_h` 的 cell $(r, c)$，其中心 $(c_x, c_y)$，先算：

$$
p_h = \frac{h}{\max(\mathbf{H})}, \qquad
p_\text{cell} = \text{base\_density} \cdot g^2 \cdot s_\text{prob} \cdot p_h
$$

其中 $g$=gsd（米/像素），$s_\text{prob}=$`canopy_prob_scale`。再乘 **类别权重** $w(c_x, c_y)$：

```python
weight = 1.0  if class == "foliage"
       = 0.3  if class in {"grass","ground"} and allow_non_foliage
       = 0.0  otherwise (building / road / water / nothing)
```

—— 这就是 **B2** 改进：放宽硬 `on_foliage` 限制，让住宅区院子、空地也能长树，但权重打 3 折。

若 `rng.random() < p_cell * w` —— **本格中签**，触发一次"簇展开"。

### 6.2 簇展开

中签的 cell 不只放 1 棵，而是在半径 `cluster_radius_m` 的圆盘内尝试放 `cluster_size` 棵：

- 簇大小：`_sample_cluster_size(cs_min, cs_max, dist=cluster_size_dist)`，默认 `[3, 7]`。
- 每候选点：再做一次 weight 检查（避免簇跨过马路）。
- 高度：直接用 cell 的 canopy $h \cdot U(0.85, 1.05)$，bush 类型再 × 0.35。
- 簇内间距：$r_\min^\text{eff} = 0.35 h \cdot \text{cluster\_overlap\_factor}$（默认 0.45）—— **簇内允许树冠重叠**，看起来才像连片树冠。
- 簇间间距：用完整的 $r_\min$ —— 不同簇之间保留正常距离。

### 6.3 碎片回滚

簇最终大小 < `max(cluster_size_min * cluster_min_keep_ratio, cluster_min_size_abs)` 时，**整簇删除**（包括 `bpy.data.objects.remove` 和回退 hash）。这避免了"3 棵孤树漂在路边"那种不自然的碎片。

### 6.4 程序化增广（B4）

`procedural_augment_ratio > 0` 时再追加一波 **风格化散布**：

- 数量 = `round(n_real * ratio)`，最多受 `max_trees - n_total` 限。
- 在整块 tile 矩形内均匀采样 $(x, y)$。
- 用同样的 `weight_fn` 过滤；高度 `U(hmin, hmax)`（不参考 canopy）。
- 用 0.35h 的间距防过密。

作用：单纯的 ETH 概率分布可能稀疏，增广能补一些 stylized 的视觉密度。

---

## 7. 附加：街道树（`_add_street_trees`，[L1718](../scripts/3_blender_assemble.py)）

仅当 `enable_street_trees` 或 `scatter_mode=canopy_prob_streets` 时启用。

- 用 bmesh 提取所有 `class==road` mesh 的 **边界边**（`len(e.link_faces)==1`）。
- 沿每条边按 `spacing_m`（默认 10m）等分；在边的 **法线方向** 偏移 `offset_m`（默认 2m），先试左侧再试右侧，避开建筑包围盒。
- 永远不 spawn bush。高度比普通树小 15%（×0.85），让街道树更瘦高一致。
- 用同样的 hash 防与已有树相撞。

---

## 8. 共享辅助：建筑回避 & 类别射线

| 函数 | 作用 |
|------|------|
| `_init_building_avoidance(r)` | 收集所有 `class==building` 物体的 2D 包围盒，外扩 `r` 米 |
| `_pos_blocked_by_building(x,y)` | O(N_buildings) 判定，散布前先剔除穿墙位置 |
| `_get_foliage_test()` | 从 z=1000 向下射线，命中物体 `class=="foliage"` 才返回 True |
| `_get_class_weight_fn(allow_non_foliage)` | 软类别权重（前述 1.0 / 0.3 / 0.0） |
| `_canopy_at_local(grid, x, y)` | 把局部 UTM 反投影到 canopy 格子，取该 cell 的 $h$ |

---

## 9. 默认参数速查（[configs/default.yaml](../configs/default.yaml) `tree_assets:`）

| 参数 | 默认 | 物理含义 |
|------|------|---------|
| `scatter_mode` | `canopy_prob` | 主模式 |
| `tree_density` | 0.0050 | 等价 base_density，trees / m² |
| `height_range_m` | `[3.0, 14.0]` | 单棵树最终高度范围 |
| `height_dist` | `lognormal` | 高度分布形状 |
| `min_canopy_h` | 2.0 m | canopy 元胞被视作 eligible 的最低高度 |
| `canopy_prob_scale` | 1.0 | 概率全局乘子（B1） |
| `allow_non_foliage` | `true` | 是否允许 grass/ground 长树（B2） |
| `procedural_augment_ratio` | 0.0 | 增广比例（B4） |
| `cluster_size_min/max` | 3 / 7 | 每个 Bernoulli 命中扩展出几棵 |
| `cluster_radius_m` | 5.0 | 簇圆盘半径 |
| `cluster_overlap_factor` | 0.45 | 簇内树冠重叠强度 |
| `enable_street_trees` | `false` | 是否加街道树（B3） |
| `max_trees` | 3000 | 每 tile 全局上限 |

---

## 10. 数据流总图

```
┌──────────────────────────────────────────────────────────────┐
│  KR2 输出 GLB（buildings/road/foliage/grass/ground 多边形）  │
└────────────────────┬─────────────────────────────────────────┘
                     │ import_glb
                     ▼
        ┌────────────────────────────┐
        │  Blender scene (类标签)     │
        └─────┬────────────┬─────────┘
              │            │
   ETH NPZ ──►│  scatter   │◄── tree_assets/*.blend (templates)
   (可选)     │  (mode)    │
              ▼            ▼
        ┌────────────────────────────┐
        │  每棵树：xyz, scale, yaw   │
        │  shares template.data       │
        │  is_tree_instance = True    │
        └────────────┬───────────────┘
                     ▼
         _dump_tree_instances_json   →  metadata/<tile>_trees.json
                     │
                     ▼
            KR4 渲染 / KR5 后处理
```

---

## 11. 调参建议

- **树太稀**：先调 `tree_density`（线性影响 Bernoulli 概率），再调 `canopy_prob_scale`（B1，整体乘子）。
- **树太密 / 结块**：把 `cluster_size_max` 降到 4-5，`cluster_overlap_factor` 加大到 0.55-0.65 让簇松一点。
- **马路两边光秃秃**：打开 `enable_street_trees`，调小 `spacing_m`。
- **住宅院子没绿**：保留 `allow_non_foliage: true`，必要时把 grass/ground 的 0.3 权重在 `_get_class_weight_fn` 里调高。
- **想要"小树丛 + 大成树"的城市分布**：`height_dist: bimodal`，`height_low_frac: 0.7`。
- **想完全脱离 canopy 数据走纯程序化**：`scatter_mode: cluster`（PCG 风格）。

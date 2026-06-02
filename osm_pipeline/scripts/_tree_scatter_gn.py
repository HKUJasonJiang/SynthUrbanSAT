"""Geometry-Nodes-based vegetation scatter for SynthUrbanSat (KR3).

This module replaces the previous Python-loop scatter functions
(`_scatter_pcg`, `_scatter_poisson_disk`, `_scatter_canopy_driven`,
`_scatter_canopy_probabilistic`, `_scatter_extra_on_open_ground`,
`_topup_foliage_until_target`, `_add_street_trees`) with a pure
Geometry Nodes pipeline built programmatically via ``bpy``.

Pipeline (matches the 5-step spec):
  1. Environmental Mask via 3 ``Geometry Proximity`` nodes
     (buildings / roads / water collections joined in-graph).
  2. Patchy distribution via a ``Noise Texture`` driven by world
     ``Position`` and gated through a high-contrast ``Map Range``.
  3. ``Distribute Points on Faces`` in *POISSON* mode with the
     composite mask as ``density_factor`` and ``distance_min``
     enforcing trunk separation.
  4. Per-instance non-uniform XYZ scale + Z random rotation. Base
     scale tapers from patch-centre to patch-edge using the captured
     noise value; X/Y/Z each receive an independent ``Random Value``.
  5. ``Instance on Points`` from the ``Tree_Assets`` collection with
     ``Pick Instance`` enabled for variety.

The modifier exposes ``Input_TopdownXYInflate`` so the top-down
preview render can temporarily inflate canopy XY (Z untouched) to
mimic remote-sensing tree-segmentation patches, then restore.

Instances are kept as native GN instances (NOT realized into
``bpy.data.objects``) to minimise memory & object overhead — Cycles
inherits pass_index / material from the originating template objects
in ``_tree_templates``, which keep their tags
(``is_tree_instance=True``, ``class=foliage``, ``class_id=2``,
``pass_index=2``). Downstream:
  - ``_dump_tree_instances_json`` reads them via
    ``depsgraph.object_instances``;
  - ``_render_preview_png`` / ``_render_topdown_ndsm`` work via
    template materials + object-index pass — unchanged.
  - ``_resolve_tree_building_conflicts`` is obsolete (proximity mask
    is enforced inside the GN graph).
"""
from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Exposed hyperparameters (CLI / yaml can override via build_*_modifier args).
# ---------------------------------------------------------------------------
TREE_COLLECTION_NAME = "Tree_Assets"     # collection of master tree meshes
SCATTER_SOURCE_COLLECTION_NAME = "GN_Scatter_Substrate"
GROUND_PLANE_NAME = "GN_GroundPlane"     # mesh that hosts the GN modifier
GN_MODIFIER_NAME = "GN_TreeScatter"      # idempotency anchor
GN_NODEGROUP_NAME = "GN_TreeScatterNG"

FOREST_NOISE_SCALE = 0.10                # macro-patch frequency (1/m units)
TREE_MIN_DISTANCE = 3.5                  # Poisson disk min spacing (m)
SAFE_DISTANCE_ROAD = 3.0                 # avoid road centreline (m)
SAFE_DISTANCE_BUILDING = 2.5             # avoid building footprint (m)
SAFE_DISTANCE_WATER = 2.0                # avoid water polygon (m)

DENSITY_MAX = 0.003                      # upper-bound trees / m² before mask
PATCH_CONTRAST_LO = 0.50                 # noise -> patch threshold low
PATCH_CONTRAST_HI = 0.68                 # noise -> patch threshold high

SCALE_BASE_LO = 0.5                      # tapered base scale at patch edge
SCALE_BASE_HI = 1.4                      # base scale at patch core
SCALE_XY_LO = 0.7                        # per-axis XY random multiplier
SCALE_XY_HI = 1.4
SCALE_Z_LO = 0.8                         # height random multiplier
SCALE_Z_HI = 1.6

# --- Refinement-pass tunables (2026-XX) -----------------------------------
# (1) Correlated XY scaling: one uniform base * (1 + small per-axis offset).
SCALE_BASE_UNIFORM_LO = 0.9              # main forest tree uniform base lo
SCALE_BASE_UNIFORM_HI = 1.3              # main forest tree uniform base hi
SCALE_XY_OFFSET = 0.15                   # ± offset added to X / Y only

# UI/CLI 0..1 stretch amounts are mapped to these physical multiplier ranges.
XY_STRETCH_MIN_AT_0 = 0.60
XY_STRETCH_MIN_AT_1 = 0.90
XY_STRETCH_MAX_AT_0 = 0.90
XY_STRETCH_MAX_AT_1 = 2.00
Z_STRETCH_MIN_AT_0 = 0.45
Z_STRETCH_MIN_AT_1 = 1.15
Z_STRETCH_MAX_AT_0 = 0.80
Z_STRETCH_MAX_AT_1 = 2.40

# (2) Slight instance tilt to break low-poly silhouettes.
TILT_MAX_RAD = 0.1396                    # ~8 deg, applied to instance X & Y

# (3) Bush understory stream: wider mask, much flatter Z.
BUSH_PATCH_THRESHOLD_LO = 0.36           # extends beyond tree threshold
BUSH_PATCH_THRESHOLD_HI = 0.60
BUSH_MIN_DISTANCE = 2.2
BUSH_DENSITY_MAX = 0.001
BUSH_SCALE_UNIFORM_LO = 0.55             # XY footprint slightly smaller
BUSH_SCALE_UNIFORM_HI = 1.0
BUSH_SCALE_Z_LO = 0.05                   # heavily flattened on Z (0.2–0.5 m
BUSH_SCALE_Z_HI = 0.10                   # on a 4–8m tree template)
BUSH_SAFE_DIST_SHRINK = 0.6              # bushes can grow closer to obstacles

# (4) Residential courtyard tree stream: low-density ring around buildings.
RESIDENTIAL_RING_INNER_M = 3.0           # >= main avoidance start
RESIDENTIAL_RING_OUTER_M = 8.0
RESIDENTIAL_MIN_DISTANCE = 6.0           # large Poisson spacing
RESIDENTIAL_DENSITY_MAX = 0.002
RESIDENTIAL_SCALE_UNIFORM_LO = 0.55      # smaller than forest trees
RESIDENTIAL_SCALE_UNIFORM_HI = 0.85

GROUND_SUBDIV_RES_M = 4.0                # plane face size (m); finer = better
                                         # spatial resolution for density,
                                         # but more verts.


# ---------------------------------------------------------------------------
# Helper: collection containers
# ---------------------------------------------------------------------------
def _ensure_collection(name):
    import bpy
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _build_tree_assets_collection(templates):
    """Move/link tree template objects into ``Tree_Assets`` collection.

    ``templates`` is the list returned by ``_load_tree_species_dir``
    in the main module:
        ``[(species_name, master_obj, base_h, base_z_min, is_bush), ...]``
    Returns the ``bpy.types.Collection`` handle.
    """
    import bpy
    from mathutils import Vector
    col = _ensure_collection(TREE_COLLECTION_NAME)
    seen = set()
    for tpl in templates:
        try:
            obj = tpl[1]
        except (IndexError, TypeError):
            continue
        try:
            base_z_min = float(tpl[3])
        except (IndexError, TypeError, ValueError):
            base_z_min = 0.0
        if obj is None or obj.name in seen:
            continue
        seen.add(obj.name)
        if obj.name not in col.objects:
            try:
                col.objects.link(obj)
            except RuntimeError:
                pass
        # Anchor each template so its lowest vertex sits at Z=0 before GN
        # instances apply non-uniform scaling. Otherwise increasing Z scale
        # expands around the template origin and pushes part of the canopy
        # below the ground plane.
        anchor_obj = obj
        while anchor_obj.parent is not None:
            anchor_obj = anchor_obj.parent
        if abs(base_z_min) > 1e-6:
            anchor_obj.location = Vector((anchor_obj.location.x,
                                          anchor_obj.location.y,
                                          anchor_obj.location.z - base_z_min))
        # Keep templates viewport/render enabled: Blender's GN collection
        # instances inherit source visibility, so hiding templates also hides
        # the virtual forest in the 3D viewport.
        obj.hide_viewport = False
        obj.hide_render = False
        obj.hide_select = True
        # Carry foliage tags so any object iteration in the preview
        # render path stamps the correct seg-palette material.
        obj["is_tree_instance"] = True
        obj["class"] = "foliage"
        obj["class_id"] = 2
        obj.pass_index = 2
    print(f"  [GN] Tree_Assets collection has {len(col.objects)} template(s)")
    return col


# ---------------------------------------------------------------------------
# Helper: ground plane that hosts the modifier
# ---------------------------------------------------------------------------
def _create_ground_plane(cfg):
    """Create a flat, subdivided plane covering the full tile.

    Tile scene convention (see ``main`` in ``3_blender_assemble.py``):
    SW corner at world (0, 0), tile spans [0, ortho_m] x [0, ortho_m]
    in the XY plane, Z=0 is ground level.
    """
    import bpy
    ortho_m = float(cfg["render"]["ortho_scale_m"])
    cx = cy = 0.5 * ortho_m

    # Clean any previous instance to keep idempotency.
    old = bpy.data.objects.get(GROUND_PLANE_NAME)
    if old is not None:
        mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    bpy.ops.mesh.primitive_plane_add(size=ortho_m, location=(cx, cy, 0.001))
    plane = bpy.context.active_object
    plane.name = GROUND_PLANE_NAME
    # subdivide so Distribute Points sees a face-density field with
    # adequate spatial resolution.
    n_cuts = max(1, int(ortho_m / float(GROUND_SUBDIV_RES_M)))
    # Cap to avoid >1M face explosion on very large tiles.
    n_cuts = min(n_cuts, 200)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=n_cuts)
    bpy.ops.object.mode_set(mode="OBJECT")
    # IMPORTANT: do NOT set ``hide_render=True`` here. The GN modifier's
    # evaluated output (all virtual tree instances) is bound to this host
    # object's depsgraph result, so hiding the host hides every instance.
    # Instead we mask the host's *own* mesh from the camera view by
    # using a transparent shader-less material and dropping it slightly
    # below scene Z=0; the GN-output instances remain visible.
    plane.hide_render = False
    plane.display_type = "WIRE"
    plane["is_gn_ground"] = True
    plane["is_gn_substrate"] = True  # preview can flag-skip emission tag
    print(f"  [GN] ground plane {ortho_m:.1f}m, {n_cuts+1}x{n_cuts+1} verts")
    return plane


# ---------------------------------------------------------------------------
# Helper: obstacle collections (Buildings / Roads / Water)
# ---------------------------------------------------------------------------
_OBSTACLE_GROUPS = {
    "GN_Obstacles_Buildings": ("building",),
    "GN_Obstacles_Roads": ("road",),
    "GN_Obstacles_Water": ("water",),
}


def _build_obstacle_collections():
    """Group already-imported KR2 GLB meshes by class into 3 collections
    that ``Geometry Proximity`` can target via ``Collection Info``.

    We *link* (not move) so the original scene graph is preserved.
    """
    import bpy
    cols = {}
    for cname, classes in _OBSTACLE_GROUPS.items():
        col = _ensure_collection(cname)
        # purge previous links to keep idempotency
        for ob in list(col.objects):
            try:
                col.objects.unlink(ob)
            except RuntimeError:
                pass
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            if obj.get("class") in classes:
                try:
                    col.objects.link(obj)
                except RuntimeError:
                    pass
        cols[cname] = col
        print(f"  [GN] obstacle group '{cname}' = {len(col.objects)} mesh(es)")
    return cols


def _build_scatter_source_collection():
    """Collect KR2 foliage meshes used as the GN scatter surface.

    The previous whole-tile ground-plane scatter creates a carpet of trees
    over every open pixel. The historical dataset style scatters trees from
    the vegetation substrate, then lets road/water/building masks cut holes.
    """
    import bpy
    col = _ensure_collection(SCATTER_SOURCE_COLLECTION_NAME)
    for ob in list(col.objects):
        try:
            col.objects.unlink(ob)
        except RuntimeError:
            pass
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.get("is_tree_instance") or obj.get("is_gn_ground"):
            continue
        if obj.get("class") != "foliage":
            continue
        try:
            col.objects.link(obj)
        except RuntimeError:
            pass
    print(f"  [GN] scatter substrate '{SCATTER_SOURCE_COLLECTION_NAME}' = "
          f"{len(col.objects)} foliage mesh(es)")
    return col


# ---------------------------------------------------------------------------
# Core: build the Geometry Nodes node tree
# ---------------------------------------------------------------------------
def _new_node(nt, ntype, name=None, location=None):
    node = nt.nodes.new(ntype)
    if name:
        node.name = name
        node.label = name
    if location:
        node.location = location
    return node


def _link(nt, a, b):
    nt.links.new(a, b)


def _socket_by_name(node, sockets, name):
    """Find a socket by its (possibly translated) display name.

    Blender's display labels for built-in node sockets are stable enough
    across 4.x/5.x for our purposes, but we accept *substring* matches
    so minor wording drift (e.g. ``"Distance Min"`` vs ``"Min Distance"``)
    doesn't break the wiring.
    """
    target = name.lower().replace(" ", "")
    for s in sockets:
        nm = s.name.lower().replace(" ", "")
        if nm == target or target in nm or nm in target:
            return s
    raise KeyError(f"socket '{name}' not found on {node.bl_idname} "
                   f"(have: {[s.name for s in sockets]})")


def _build_gn_node_tree(buildings_col, roads_col, water_col, tree_col,
                        scatter_source_col):
    """Construct the full Geometry Nodes graph; return the node group."""
    import bpy

    # Idempotency: kill stale node group first.
    if GN_NODEGROUP_NAME in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[GN_NODEGROUP_NAME])
    ng = bpy.data.node_groups.new(GN_NODEGROUP_NAME, "GeometryNodeTree")

    # ---- Interface (modifier-exposed inputs) -----------------------------
    iface = ng.interface
    iface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    iface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    def _add_float(name, default, mn=None, mx=None):
        s = iface.new_socket(name, in_out="INPUT", socket_type="NodeSocketFloat")
        s.default_value = float(default)
        if mn is not None:
            s.min_value = float(mn)
        if mx is not None:
            s.max_value = float(mx)
        return s

    _add_float("Safe Distance Building", SAFE_DISTANCE_BUILDING, 0.0, 50.0)
    _add_float("Safe Distance Road", SAFE_DISTANCE_ROAD, 0.0, 50.0)
    _add_float("Safe Distance Water", SAFE_DISTANCE_WATER, 0.0, 50.0)
    _add_float("Forest Noise Scale", FOREST_NOISE_SCALE, 0.001, 5.0)
    _add_float("Patch Threshold Lo", PATCH_CONTRAST_LO, 0.0, 1.0)
    _add_float("Patch Threshold Hi", PATCH_CONTRAST_HI, 0.0, 1.0)
    _add_float("Tree Min Distance", TREE_MIN_DISTANCE, 0.1, 20.0)
    _add_float("Density Max", DENSITY_MAX, 0.0, 5.0)
    _add_float("Tree Amount", 0.5, 0.0, 1.0)
    _add_float("XY Scale Min", SCALE_BASE_UNIFORM_LO, 0.1, 5.0)
    _add_float("XY Scale Max", SCALE_BASE_UNIFORM_HI, 0.1, 5.0)
    _add_float("Z Scale Min", SCALE_Z_LO, 0.1, 5.0)
    _add_float("Z Scale Max", SCALE_Z_HI, 0.1, 5.0)
    _add_float("Topdown XY Inflate", 1.0, 0.1, 10.0)
    _add_float("Random Seed", 0.0)

    # ---- Helper inner builders ------------------------------------------
    def new_math(op, loc, in0=None, in1=None, in2=None, clamp=False):
        n = _new_node(ng, "ShaderNodeMath", location=loc)
        n.operation = op
        n.use_clamp = clamp
        if in0 is not None:
            _link(ng, in0, n.inputs[0])
        if in1 is not None:
            _link(ng, in1, n.inputs[1])
        if in2 is not None and len(n.inputs) > 2:
            _link(ng, in2, n.inputs[2])
        return n

    # ---- Input / Output --------------------------------------------------
    n_in = _new_node(ng, "NodeGroupInput", "GroupInput", (-1800, 0))
    n_out = _new_node(ng, "NodeGroupOutput", "GroupOutput", (2400, 0))

    sock_geom_in = n_in.outputs["Geometry"]
    sock_safe_b = n_in.outputs["Safe Distance Building"]
    sock_safe_r = n_in.outputs["Safe Distance Road"]
    sock_safe_w = n_in.outputs["Safe Distance Water"]
    sock_noise_scale = n_in.outputs["Forest Noise Scale"]
    sock_thr_lo = n_in.outputs["Patch Threshold Lo"]
    sock_thr_hi = n_in.outputs["Patch Threshold Hi"]
    sock_min_dist = n_in.outputs["Tree Min Distance"]
    sock_dmax = n_in.outputs["Density Max"]
    sock_tree_amount = n_in.outputs["Tree Amount"]
    sock_xy_scale_min = n_in.outputs["XY Scale Min"]
    sock_xy_scale_max = n_in.outputs["XY Scale Max"]
    sock_z_scale_min = n_in.outputs["Z Scale Min"]
    sock_z_scale_max = n_in.outputs["Z Scale Max"]
    sock_xy_inflate = n_in.outputs["Topdown XY Inflate"]
    sock_seed = n_in.outputs["Random Seed"]

    amount_mul = new_math("MULTIPLY", (-1500, -780), sock_tree_amount)
    amount_mul.inputs[1].default_value = 1.9
    amount_scale = new_math("ADD", (-1320, -780), amount_mul.outputs[0])
    amount_scale.inputs[1].default_value = 0.1
    main_dmax = new_math("MULTIPLY", (-1120, -780), sock_dmax,
                         amount_scale.outputs[0])

    # ---- Step 1: Environmental mask (3x Geometry Proximity on
    #              joined collection geometries) ---------------------------
    def proximity_chain(col, safe_socket, y, label):
        col_info = _new_node(ng, "GeometryNodeCollectionInfo",
                             f"CollInfo_{label}", (-1600, y))
        col_info.transform_space = "RELATIVE"
        try:
            # New Blender API: collection on inputs[0]
            col_info.inputs["Collection"].default_value = col
        except KeyError:
            col_info.inputs[0].default_value = col
        # Separate Children = False -> single joined geometry stream
        try:
            col_info.inputs["Separate Children"].default_value = False
        except KeyError:
            pass
        join = _new_node(ng, "GeometryNodeJoinGeometry",
                         f"Join_{label}", (-1400, y))
        _link(ng, col_info.outputs["Instances"], join.inputs[0])
        # Realize so Proximity can sample its mesh surface.
        realize = _new_node(ng, "GeometryNodeRealizeInstances",
                            f"Realize_{label}", (-1250, y))
        _link(ng, join.outputs[0], realize.inputs[0])
        prox = _new_node(ng, "GeometryNodeProximity",
                         f"Proximity_{label}", (-1080, y))
        try:
            prox.target_element = "FACES"
        except (AttributeError, TypeError):
            pass
        _link(ng, realize.outputs[0], prox.inputs["Target"])
        # Compare distance > safe_distance => 1.0 else 0.0
        cmp = new_math("GREATER_THAN", (-880, y), prox.outputs["Distance"],
                       safe_socket)
        # Return BOTH the binary safe-mask and the raw distance socket
        # (needed for the residential ring mask in Stream C).
        return cmp.outputs[0], prox.outputs["Distance"]

    mask_b, dist_b = proximity_chain(buildings_col, sock_safe_b, 600,
                                     "Buildings")
    mask_r, dist_r = proximity_chain(roads_col, sock_safe_r, 300, "Roads")
    mask_w, dist_w = proximity_chain(water_col, sock_safe_w, 0, "Water")

    # AND-chain: mask_env = mask_b * mask_r * mask_w  (all binary)
    and_br = new_math("MULTIPLY", (-700, 450), mask_b, mask_r)
    mask_env = new_math("MULTIPLY", (-540, 380), and_br.outputs[0], mask_w)

    # Scatter on the KR2 foliage substrate, not on the whole tile plane.
    # The host plane still carries the modifier, but only GN instances are
    # output; its own geometry is not used as the distribution mesh unless no
    # foliage substrate exists.
    scatter_mesh = sock_geom_in
    if scatter_source_col is not None and len(scatter_source_col.objects) > 0:
        src_info = _new_node(ng, "GeometryNodeCollectionInfo",
                             "CollInfo_ScatterSubstrate", (-1600, 900))
        src_info.transform_space = "RELATIVE"
        try:
            src_info.inputs["Collection"].default_value = scatter_source_col
        except KeyError:
            src_info.inputs[0].default_value = scatter_source_col
        try:
            src_info.inputs["Separate Children"].default_value = False
        except KeyError:
            pass
        src_realize = _new_node(ng, "GeometryNodeRealizeInstances",
                                "Realize_ScatterSubstrate", (-1350, 900))
        _link(ng, src_info.outputs["Instances"], src_realize.inputs[0])
        scatter_mesh = src_realize.outputs[0]

    # ---- Step 2: Noise patch mask ---------------------------------------
    pos = _new_node(ng, "GeometryNodeInputPosition", "Position",
                    (-1600, -350))
    noise = _new_node(ng, "ShaderNodeTexNoise", "ForestNoise",
                      (-1380, -350))
    try:
        noise.noise_dimensions = "3D"
    except AttributeError:
        pass
    _link(ng, pos.outputs[0], noise.inputs["Vector"])
    _link(ng, sock_noise_scale, noise.inputs["Scale"])
    # Detail / Roughness defaults are fine for organic patches.

    # MapRange noise -> [0,1] with high contrast (clamp on).
    mr = _new_node(ng, "ShaderNodeMapRange", "PatchMapRange", (-1080, -350))
    mr.clamp = True
    _link(ng, noise.outputs["Fac"], mr.inputs["Value"])
    _link(ng, sock_thr_lo, mr.inputs["From Min"])
    _link(ng, sock_thr_hi, mr.inputs["From Max"])
    # To Min/Max default to 0..1.
    patch_mask = mr.outputs["Result"]

    # density_factor = mask_env * patch_mask
    density_factor = new_math("MULTIPLY", (-820, -250),
                              mask_env.outputs[0], patch_mask)

    # =====================================================================
    # Shared helpers — DRY scale / rotation / IoP construction reused by
    # the 3 scatter streams (forest trees / understory bushes /
    # residential ring trees).
    # =====================================================================
    def build_distribute(density_socket, min_dist_socket, dmax_socket,
                         seed_off, loc, label):
        d = _new_node(ng, "GeometryNodeDistributePointsOnFaces",
                      f"Distribute_{label}", loc)
        try:
            d.distribute_method = "POISSON"
        except (AttributeError, TypeError):
            pass
        _link(ng, scatter_mesh, d.inputs["Mesh"])
        _link(ng, min_dist_socket,
              _socket_by_name(d, d.inputs, "Distance Min"))
        _link(ng, dmax_socket,
              _socket_by_name(d, d.inputs, "Density Max"))
        _link(ng, density_socket,
              _socket_by_name(d, d.inputs, "Density Factor"))
        try:
            seed_n = new_math("ADD", (loc[0] - 220, loc[1] - 200), sock_seed)
            seed_n.inputs[1].default_value = float(seed_off)
            _link(ng, seed_n.outputs[0],
                  _socket_by_name(d, d.inputs, "Seed"))
        except KeyError:
            pass
        return d.outputs["Points"]

    def set_or_link(input_socket, value_or_socket):
        try:
            _link(ng, value_or_socket, input_socket)
        except Exception:
            input_socket.default_value = float(value_or_socket)

    def rand_float(name, lo, hi, loc, seed_offset):
        rv = _new_node(ng, "FunctionNodeRandomValue", name, loc)
        rv.data_type = "FLOAT"
        set_or_link(rv.inputs["Min"], lo)
        set_or_link(rv.inputs["Max"], hi)
        try:
            sum_node = new_math("ADD", (loc[0] - 200, loc[1] - 60),
                                sock_seed)
            sum_node.inputs[1].default_value = float(seed_offset)
            _link(ng, sum_node.outputs[0], rv.inputs["Seed"])
        except KeyError:
            pass
        return rv.outputs["Value"]

    def build_correlated_scale(*, base_lo, base_hi, xy_offset,
                               z_lo, z_hi, seed_off, loc, label,
                               use_xy_inflate=True):
        """OPT-1: correlated XY via shared uniform base + small offsets;
        independent Z. Avoids the "noodle-tree" artefact caused by fully
        independent per-axis random multipliers.

            Scale.X = (uniform_base + offset_x) * topdown_inflate
            Scale.Y = (uniform_base + offset_y) * topdown_inflate
            Scale.Z = independent random in [z_lo, z_hi]
        """
        base = rand_float(f"BaseUni_{label}", base_lo, base_hi,
                          (loc[0], loc[1]), seed_off + 0.0)
        ox = rand_float(f"OffX_{label}", -xy_offset, +xy_offset,
                        (loc[0], loc[1] - 120), seed_off + 1.0)
        oy = rand_float(f"OffY_{label}", -xy_offset, +xy_offset,
                        (loc[0], loc[1] - 240), seed_off + 2.0)
        rz = rand_float(f"RandZ_{label}", z_lo, z_hi,
                        (loc[0], loc[1] - 360), seed_off + 3.0)
        sx = new_math("ADD", (loc[0] + 220, loc[1]),       base, ox)
        sy = new_math("ADD", (loc[0] + 220, loc[1] - 120), base, oy)
        if use_xy_inflate:
            sx = new_math("MULTIPLY", (loc[0] + 400, loc[1]),
                          sx.outputs[0], sock_xy_inflate)
            sy = new_math("MULTIPLY", (loc[0] + 400, loc[1] - 120),
                          sy.outputs[0], sock_xy_inflate)
        comb = _new_node(ng, "ShaderNodeCombineXYZ",
                         f"ScaleCombine_{label}",
                         (loc[0] + 600, loc[1] - 120))
        _link(ng, sx.outputs[0], comb.inputs["X"])
        _link(ng, sy.outputs[0], comb.inputs["Y"])
        _link(ng, rz, comb.inputs["Z"])
        return comb.outputs[0]

    def build_tilt_rotation(seed_off, loc, label,
                            tilt_max=TILT_MAX_RAD):
        """OPT-2: random Z yaw + tiny X/Y tilt to break low-poly silhouettes."""
        yaw = rand_float(f"Yaw_{label}", 0.0, 6.283185307,
                         (loc[0], loc[1]), seed_off + 0.0)
        tilt_x = rand_float(f"TiltX_{label}", -tilt_max, +tilt_max,
                            (loc[0], loc[1] - 120), seed_off + 1.0)
        tilt_y = rand_float(f"TiltY_{label}", -tilt_max, +tilt_max,
                            (loc[0], loc[1] - 240), seed_off + 2.0)
        rc = _new_node(ng, "ShaderNodeCombineXYZ", f"RotCombine_{label}",
                       (loc[0] + 240, loc[1] - 120))
        _link(ng, tilt_x, rc.inputs["X"])
        _link(ng, tilt_y, rc.inputs["Y"])
        _link(ng, yaw, rc.inputs["Z"])
        return rc.outputs[0]

    def build_iop(points_socket, scale_socket, rot_socket,
                  seed_off, loc, label):
        col_assets = _new_node(ng, "GeometryNodeCollectionInfo",
                               f"TreeCollection_{label}", (loc[0], loc[1] + 200))
        col_assets.transform_space = "ORIGINAL"
        try:
            col_assets.inputs["Collection"].default_value = tree_col
        except KeyError:
            col_assets.inputs[0].default_value = tree_col
        try:
            col_assets.inputs["Separate Children"].default_value = True
        except KeyError:
            pass
        try:
            col_assets.inputs["Reset Children"].default_value = True
        except KeyError:
            pass
        pick_rv = _new_node(ng, "FunctionNodeRandomValue",
                            f"RandPick_{label}", (loc[0], loc[1] + 60))
        pick_rv.data_type = "INT"
        pick_rv.inputs[2].default_value = 0
        pick_rv.inputs[3].default_value = 1_000_000
        try:
            seed_p = new_math("ADD", (loc[0] - 180, loc[1] + 60), sock_seed)
            seed_p.inputs[1].default_value = float(seed_off)
            _link(ng, seed_p.outputs[0], pick_rv.inputs["Seed"])
        except KeyError:
            pass
        iop = _new_node(ng, "GeometryNodeInstanceOnPoints",
                        f"IoP_{label}", (loc[0] + 240, loc[1]))
        _link(ng, points_socket, iop.inputs["Points"])
        _link(ng, col_assets.outputs["Instances"], iop.inputs["Instance"])
        try:
            iop.inputs["Pick Instance"].default_value = True
        except KeyError:
            pass
        try:
            _link(ng, pick_rv.outputs[2], iop.inputs["Instance Index"])
        except (KeyError, IndexError):
            pass
        _link(ng, rot_socket, iop.inputs["Rotation"])
        _link(ng, scale_socket, iop.inputs["Scale"])
        return iop.outputs["Instances"]

    # =====================================================================
    # Stream A — main forest trees (existing pipeline, now with correlated
    #            XY scale + tilt rotation).
    # =====================================================================
    pts_a = build_distribute(density_factor.outputs[0], sock_min_dist,
                             main_dmax.outputs[0], seed_off=0.0,
                             loc=(-300, 0), label="A")
    scl_a = build_correlated_scale(
        base_lo=sock_xy_scale_min, base_hi=sock_xy_scale_max,
        xy_offset=SCALE_XY_OFFSET,
        z_lo=sock_z_scale_min, z_hi=sock_z_scale_max,
        seed_off=10.0, loc=(0, -50), label="A", use_xy_inflate=True)
    rot_a = build_tilt_rotation(seed_off=30.0, loc=(0, 320), label="A")
    inst_a = build_iop(pts_a, scl_a, rot_a, seed_off=50.0,
                       loc=(900, 0), label="A")

    # =====================================================================
    # Stream B — understory bushes (OPT-3).
    # Same noise field with a *wider* MapRange so bush patches extend
    # past tree patches, plus heavy Z flattening to read as low foliage.
    # Tighter Poisson spacing, higher density. Uses *shrunk* safe
    # distances so bushes can hug obstacle boundaries.
    # =====================================================================
    bush_mr = _new_node(ng, "ShaderNodeMapRange", "BushMapRange",
                        (-1080, -600))
    bush_mr.clamp = True
    _link(ng, noise.outputs["Fac"], bush_mr.inputs["Value"])
    bush_mr.inputs["From Min"].default_value = BUSH_PATCH_THRESHOLD_LO
    bush_mr.inputs["From Max"].default_value = BUSH_PATCH_THRESHOLD_HI
    bush_patch_mask = bush_mr.outputs["Result"]

    # Optional env-mask with shrunk safe distances (bush_safe = safe * shrink).
    def _shrunk_cmp(dist_sock, safe_sock, shrink, y, label):
        sh = new_math("MULTIPLY", (-880, y), safe_sock)
        sh.inputs[1].default_value = float(shrink)
        cmp = new_math("GREATER_THAN", (-720, y), dist_sock, sh.outputs[0])
        return cmp.outputs[0]

    bmask_b = _shrunk_cmp(dist_b, sock_safe_b,
                          BUSH_SAFE_DIST_SHRINK, -700, "Bldg")
    bmask_r = _shrunk_cmp(dist_r, sock_safe_r,
                          BUSH_SAFE_DIST_SHRINK, -800, "Road")
    bmask_w = _shrunk_cmp(dist_w, sock_safe_w,
                          BUSH_SAFE_DIST_SHRINK, -900, "Watr")
    b_and1 = new_math("MULTIPLY", (-560, -700), bmask_b, bmask_r)
    b_envmask = new_math("MULTIPLY", (-400, -780), b_and1.outputs[0], bmask_w)
    b_density = new_math("MULTIPLY", (-240, -700),
                         b_envmask.outputs[0], bush_patch_mask)

    # Constant sockets via Value nodes (so Distribute helper signature
    # stays uniform across streams).
    val_b_min = _new_node(ng, "ShaderNodeValue", "BushMinDist", (-300, -900))
    val_b_min.outputs[0].default_value = BUSH_MIN_DISTANCE
    val_b_dmax = _new_node(ng, "ShaderNodeValue", "BushDMax", (-300, -1000))
    val_b_dmax.outputs[0].default_value = BUSH_DENSITY_MAX
    bush_dmax = new_math("MULTIPLY", (-120, -1000), val_b_dmax.outputs[0],
                         amount_scale.outputs[0])

    pts_b = build_distribute(b_density.outputs[0], val_b_min.outputs[0],
                             bush_dmax.outputs[0], seed_off=200.0,
                             loc=(-50, -750), label="B")
    scl_b = build_correlated_scale(
        base_lo=BUSH_SCALE_UNIFORM_LO, base_hi=BUSH_SCALE_UNIFORM_HI,
        xy_offset=SCALE_XY_OFFSET,
        z_lo=BUSH_SCALE_Z_LO, z_hi=BUSH_SCALE_Z_HI,
        seed_off=210.0, loc=(280, -800), label="B", use_xy_inflate=True)
    rot_b = build_tilt_rotation(seed_off=230.0, loc=(280, -450), label="B")
    inst_b = build_iop(pts_b, scl_b, rot_b, seed_off=250.0,
                       loc=(900, -750), label="B")

    # =====================================================================
    # Stream C — residential ring trees (OPT-4).
    # Mask is the *anti*-shell of the building safe-mask: dist_b in
    # [RESIDENTIAL_RING_INNER_M, RESIDENTIAL_RING_OUTER_M], still
    # avoiding roads & water. Sparse Poisson; smaller uniform scale.
    # =====================================================================
    val_r_in = _new_node(ng, "ShaderNodeValue", "ResRingIn", (-1080, -1200))
    val_r_in.outputs[0].default_value = RESIDENTIAL_RING_INNER_M
    val_r_out = _new_node(ng, "ShaderNodeValue", "ResRingOut", (-1080, -1300))
    val_r_out.outputs[0].default_value = RESIDENTIAL_RING_OUTER_M
    ring_gt_in = new_math("GREATER_THAN", (-880, -1200), dist_b,
                          val_r_in.outputs[0])
    ring_lt_out = new_math("LESS_THAN", (-880, -1300), dist_b,
                           val_r_out.outputs[0])
    ring_band = new_math("MULTIPLY", (-720, -1240),
                         ring_gt_in.outputs[0], ring_lt_out.outputs[0])
    # Still avoid roads & water with normal safety.
    res_envmask = new_math("MULTIPLY", (-560, -1280),
                           ring_band.outputs[0], mask_r)
    res_envmask = new_math("MULTIPLY", (-400, -1320),
                           res_envmask.outputs[0], mask_w)
    val_r_min = _new_node(ng, "ShaderNodeValue", "ResMinDist", (-300, -1450))
    val_r_min.outputs[0].default_value = RESIDENTIAL_MIN_DISTANCE
    val_r_dmax = _new_node(ng, "ShaderNodeValue", "ResDMax", (-300, -1550))
    val_r_dmax.outputs[0].default_value = RESIDENTIAL_DENSITY_MAX
    res_dmax = new_math("MULTIPLY", (-120, -1550), val_r_dmax.outputs[0],
                        amount_scale.outputs[0])

    pts_c = build_distribute(res_envmask.outputs[0], val_r_min.outputs[0],
                             res_dmax.outputs[0], seed_off=400.0,
                             loc=(-50, -1300), label="C")
    scl_c = build_correlated_scale(
        base_lo=RESIDENTIAL_SCALE_UNIFORM_LO,
        base_hi=sock_xy_scale_max,
        xy_offset=SCALE_XY_OFFSET,
        z_lo=sock_z_scale_min, z_hi=sock_z_scale_max,
        seed_off=410.0, loc=(280, -1350), label="C", use_xy_inflate=True)
    rot_c = build_tilt_rotation(seed_off=430.0, loc=(280, -1050), label="C")
    inst_c = build_iop(pts_c, scl_c, rot_c, seed_off=450.0,
                       loc=(900, -1300), label="C")

    # =====================================================================
    # Output: join all three instance streams.
    # =====================================================================
    join_all = _new_node(ng, "GeometryNodeJoinGeometry",
                         "JoinStreams", (1500, 0))
    _link(ng, inst_a, join_all.inputs[0])
    _link(ng, inst_b, join_all.inputs[0])
    _link(ng, inst_c, join_all.inputs[0])
    _link(ng, join_all.outputs[0], n_out.inputs["Geometry"])
    return ng


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def scatter_geometry_nodes(cfg, *, tree_templates,
                           safe_b=None, safe_r=None, safe_w=None,
                           noise_scale=None, min_dist=None,
                           density_max=None, tree_amount=0.5,
                           xy_stretch=0.5, z_stretch=0.5,
                           xy_stretch_min_at_0=None,
                           xy_stretch_min_at_1=None,
                           xy_stretch_max_at_0=None,
                           xy_stretch_max_at_1=None,
                           z_stretch_min_at_0=None,
                           z_stretch_min_at_1=None,
                           z_stretch_max_at_0=None,
                           z_stretch_max_at_1=None,
                           seed=0):
    """Build & attach the GN tree-scatter modifier.

    Args:
      cfg: KR3 config dict.
      tree_templates: ``_load_tree_species_dir`` output list.
      safe_b / safe_r / safe_w / noise_scale / min_dist / density_max:
        per-call overrides for the modifier inputs (None = module default).
      seed: integer seed → passes through to the GN ``Random Seed`` socket.

    Returns:
      ``(ground_obj, modifier)`` so callers can read socket identifiers
      later (e.g. for the top-down XY inflate trick).
    """
    import bpy

    if not tree_templates:
        print("  [GN] no tree templates loaded; scatter skipped")
        return None, None

    tree_col = _build_tree_assets_collection(tree_templates)
    obstacle_cols = _build_obstacle_collections()
    scatter_source_col = _build_scatter_source_collection()
    ground = _create_ground_plane(cfg)

    ng = _build_gn_node_tree(
        obstacle_cols["GN_Obstacles_Buildings"],
        obstacle_cols["GN_Obstacles_Roads"],
        obstacle_cols["GN_Obstacles_Water"],
        tree_col,
        scatter_source_col,
    )

    # Idempotent modifier attach.
    if GN_MODIFIER_NAME in ground.modifiers:
        ground.modifiers.remove(ground.modifiers[GN_MODIFIER_NAME])
    mod = ground.modifiers.new(GN_MODIFIER_NAME, "NODES")
    mod.node_group = ng

    # Push exposed-input values via the per-modifier interface.
    def _set(name, value):
        if value is None:
            return
        for item in ng.interface.items_tree:
            if item.item_type == "SOCKET" and item.in_out == "INPUT" \
                    and item.name == name:
                try:
                    mod[item.identifier] = float(value)
                    return
                except (KeyError, TypeError):
                    pass
        print(f"  [GN][warn] could not set modifier input '{name}'")

    _set("Safe Distance Building", safe_b if safe_b is not None
         else SAFE_DISTANCE_BUILDING)
    _set("Safe Distance Road", safe_r if safe_r is not None
         else SAFE_DISTANCE_ROAD)
    _set("Safe Distance Water", safe_w if safe_w is not None
         else SAFE_DISTANCE_WATER)
    _set("Forest Noise Scale", noise_scale if noise_scale is not None
         else FOREST_NOISE_SCALE)
    _set("Tree Min Distance", min_dist if min_dist is not None
         else TREE_MIN_DISTANCE)
    _set("Density Max", density_max if density_max is not None
         else DENSITY_MAX)
    _set("Tree Amount", max(0.0, min(1.0, float(tree_amount))))
    xy_a = max(0.0, min(1.0, float(xy_stretch)))
    z_a = max(0.0, min(1.0, float(z_stretch)))
    xy_min_0 = (XY_STRETCH_MIN_AT_0 if xy_stretch_min_at_0 is None
                else float(xy_stretch_min_at_0))
    xy_min_1 = (XY_STRETCH_MIN_AT_1 if xy_stretch_min_at_1 is None
                else float(xy_stretch_min_at_1))
    xy_max_0 = (XY_STRETCH_MAX_AT_0 if xy_stretch_max_at_0 is None
                else float(xy_stretch_max_at_0))
    xy_max_1 = (XY_STRETCH_MAX_AT_1 if xy_stretch_max_at_1 is None
                else float(xy_stretch_max_at_1))
    z_min_0 = (Z_STRETCH_MIN_AT_0 if z_stretch_min_at_0 is None
               else float(z_stretch_min_at_0))
    z_min_1 = (Z_STRETCH_MIN_AT_1 if z_stretch_min_at_1 is None
               else float(z_stretch_min_at_1))
    z_max_0 = (Z_STRETCH_MAX_AT_0 if z_stretch_max_at_0 is None
               else float(z_stretch_max_at_0))
    z_max_1 = (Z_STRETCH_MAX_AT_1 if z_stretch_max_at_1 is None
               else float(z_stretch_max_at_1))
    _set("XY Scale Min", xy_min_0 + (xy_min_1 - xy_min_0) * xy_a)
    _set("XY Scale Max", xy_max_0 + (xy_max_1 - xy_max_0) * xy_a)
    _set("Z Scale Min", z_min_0 + (z_min_1 - z_min_0) * z_a)
    _set("Z Scale Max", z_max_0 + (z_max_1 - z_max_0) * z_a)
    _set("Random Seed", float(seed))
    _set("Topdown XY Inflate", 1.0)

    # Force depsgraph update so subsequent operations (preview render,
    # JSON dump) see the realized instance stream.
    bpy.context.view_layer.update()
    try:
        bpy.context.evaluated_depsgraph_get()
    except Exception:
        pass

    print(f"  [GN] attached modifier '{GN_MODIFIER_NAME}' on '{ground.name}'")
    return ground, mod


# ---------------------------------------------------------------------------
# Downstream helpers used by 3_blender_assemble.py
# ---------------------------------------------------------------------------
def set_topdown_xy_inflate(ground_obj, value):
    """Drive the modifier's ``Topdown XY Inflate`` socket at render time."""
    if ground_obj is None:
        return
    mod = ground_obj.modifiers.get(GN_MODIFIER_NAME)
    if mod is None or mod.node_group is None:
        return
    for item in mod.node_group.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == "INPUT" \
                and item.name == "Topdown XY Inflate":
            try:
                mod[item.identifier] = float(value)
            except (KeyError, TypeError):
                pass
            break
    # Tag depsgraph dirty so the modifier re-evaluates.
    try:
        ground_obj.update_tag()
        import bpy
        bpy.context.view_layer.update()
    except Exception:
        pass


def iter_gn_tree_instances(ground_obj):
    """Yield ``(world_matrix, base_h_m, radius_xy_m)`` for each GN tree.

    Reads the evaluated depsgraph: native GN instances are virtual but
    expose ``InstanceCollection`` entries with ``matrix_world`` and the
    underlying template ``object``.
    """
    import bpy
    from mathutils import Vector
    if ground_obj is None:
        return
    deps = bpy.context.evaluated_depsgraph_get()
    eval_ground = ground_obj.evaluated_get(deps)
    for inst in deps.object_instances:
        if not inst.is_instance:
            continue
        parent = inst.parent
        if parent is None or parent.original != ground_obj:
            continue
        src = inst.object
        if src is None or src.type != "MESH":
            continue
        mw = inst.matrix_world.copy()
        try:
            xs = [(mw @ Vector(c)).x for c in src.bound_box]
            ys = [(mw @ Vector(c)).y for c in src.bound_box]
            zs = [(mw @ Vector(c)).z for c in src.bound_box]
            h = float(max(zs) - min(zs))
            r_xy = 0.5 * float(max(max(xs) - min(xs), max(ys) - min(ys)))
        except Exception:
            h = 0.0
            r_xy = 0.0
        yield mw, h, r_xy


def dump_gn_tree_instances_json(cfg, ground_obj, out_path):
    """Write a ``tree_instances.json`` compatible with the legacy schema
    used by ``auto_pipeline.py``: ``{ortho_m, trees: [{x_centered, y_centered, h}]}``.
    """
    import json
    try:
        ortho_m = float(cfg["render"]["ortho_scale_m"])
    except Exception:
        ortho_m = 0.0
    half = 0.5 * ortho_m
    trees = []
    for mw, h, r_xy in iter_gn_tree_instances(ground_obj):
        x_local = float(mw.translation.x)
        y_local = float(mw.translation.y)
        trees.append({
            "x_centered": x_local - half,
            "y_centered": y_local - half,
            "h": h,
            "r_xy_m": r_xy,
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ortho_m": ortho_m,
        "trees": trees,
    }))
    print(f"  [GN] dumped {len(trees)} tree instances -> {out_path.name}")
    return len(trees)

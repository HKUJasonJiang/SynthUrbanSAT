"""KR3 — Blender scene assembly.

Loads `output/meshes/{city}.glb`, assigns Object Index = class_id per object,
sets up a top-down orthographic camera, and saves a `.blend` for KR4.

Run:
    blender --background --python scripts/3_blender_assemble.py -- \
        --config configs/default.yaml --city zurich
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ---- argv parse (Blender swallows args; everything after `--` is ours) ---
def _argv():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def _load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clear_scene():
    import bpy
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                  bpy.data.cameras, bpy.data.lights):
        for it in list(block):
            block.remove(it)


def _import_glb(glb_path: Path):
    import bpy
    import math
    from mathutils import Matrix
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    # trimesh exports Z-up but glTF spec is Y-up; the importer applies the
    # spec rotation, leaving our height axis along Blender's negative Y.
    # Undo it: rotate root objects -90 deg around X so north -> +Y, height -> +Z.
    rot = Matrix.Rotation(math.radians(-90.0), 4, "X")
    for o in bpy.context.selected_objects:
        if o.parent is None:
            o.matrix_world = rot @ o.matrix_world
    bpy.context.view_layer.update()


# Single source of truth lives in dataprep/osm_tags.py. Blender's bundled
# Python doesn't automatically see the project root, so we add it before
# importing.
def _load_class_ids(project_root: Path):
    import sys as _sys
    p = str(project_root)
    if p not in _sys.path:
        _sys.path.insert(0, p)
    from dataprep.osm_tags import CLASS_IDS as _CLS  # type: ignore
    return dict(_CLS)


CLASS_IDS: dict = {}  # populated in main() after we know project_root


def _fix_normals_outward():
    """Recompute face normals so they point outward / upward.

    The shapely polygons feeding ``flat_polygon_mesh`` don't have a
    guaranteed CCW exterior winding, so the triangulated planes for
    water / foliage / grass can end up with face normals pointing
    world-down after the glTF axis remap. Cycles' camera-ray hits the
    back face and the IndexOB pass records nothing for those pixels.
    Recompute consistent outward normals on every imported mesh.
    """
    import bpy
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        bpy.context.view_layer.objects.active = o
        for ob in bpy.data.objects:
            ob.select_set(False)
        o.select_set(True)
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError as e:
            print(f"  [warn] normals fix failed for {o.name}: {e}")
    print("[KR3] face normals recomputed outward")


def _assign_pass_indices():
    """Each imported object's name (or its parent collection) starts with the
    class name from KR2. Map name prefix -> class id and write to pass_index.
    """
    import bpy
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        cls = None
        n = obj.name.lower()
        for k in CLASS_IDS:
            if n.startswith(k):
                cls = k
                break
        if cls is None and obj.users_collection:
            cn = obj.users_collection[0].name.lower()
            for k in CLASS_IDS:
                if k in cn:
                    cls = k
                    break
        if cls is None:
            print(f"  [warn] no class match for {obj.name}, default ground")
            cls = "ground"
        obj.pass_index = CLASS_IDS[cls]
        # also stash on custom prop for downstream
        obj["class"] = cls
        obj["class_id"] = CLASS_IDS[cls]
        print(f"  {obj.name:32s} -> class={cls} pass_index={CLASS_IDS[cls]}")


def _setup_ortho_camera(cfg):
    import bpy
    cam_data = bpy.data.cameras.new("OrthoCam")
    cam = bpy.data.objects.new("OrthoCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = float(cfg["render"]["ortho_scale_m"])
    cam_data.clip_start = float(cfg["render"]["near_clip_m"])
    cam_data.clip_end = float(cfg["render"]["far_clip_m"])
    # place above scene center, look straight down (-Z)
    h = float(cfg["render"]["camera_height_m"])
    cam.location = (0.0, 0.0, h)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam
    return cam


def _setup_render(cfg):
    import bpy
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.samples = 1   # geometry-only passes; no need for many samples
    scn.cycles.use_denoising = False
    s = int(cfg["render"]["image_size"])
    scn.render.resolution_x = s
    scn.render.resolution_y = s
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "OPEN_EXR"
    scn.render.image_settings.color_depth = "32"
    # enable passes
    vl = scn.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_object_index = True


def _list_buildings():
    import bpy
    return [o for o in bpy.data.objects
            if o.type == "MESH" and o.get("class") == "building"]


def _dump_tree_instances_json(cfg, out_path):
    """Dump positions of all `is_tree_instance` objects to JSON.

    Must be called BEFORE :func:`_prepare_blend_for_user` shifts the
    scene to be centred on the origin — at this point the scene is
    still in tile-corner coords (SW corner at world (0,0), tile spans
    [0, ortho_m] in both x and y). We emit centre-relative coords
    (`x_centered = x - ortho_m/2`) so the consumer just adds
    `(cx_utm, cy_utm)` to recover absolute UTM.
    """
    import bpy
    import json
    from mathutils import Vector
    try:
        ortho_m = float(cfg["render"]["ortho_scale_m"])
    except Exception:
        ortho_m = 0.0
    half = 0.5 * ortho_m
    trees = []
    for obj in bpy.data.objects:
        if not obj.get("is_tree_instance"):
            continue
        # Skip template objects (kept hidden in `_tree_templates` collection).
        if obj.hide_render:
            continue
        x_local = float(obj.location.x)
        y_local = float(obj.location.y)
        try:
            xs = [(obj.matrix_world @ Vector(c)).x for c in obj.bound_box]
            ys = [(obj.matrix_world @ Vector(c)).y for c in obj.bound_box]
            zs = [(obj.matrix_world @ Vector(c)).z for c in obj.bound_box]
            h = float(max(zs) - min(zs))
            r_xy = 0.5 * float(max(max(xs) - min(xs), max(ys) - min(ys)))
        except Exception:
            h = 0.0
            r_xy = 0.0
        trees.append({
            "x_centered": x_local - half,
            "y_centered": y_local - half,
            "h": h,
            "r_xy_m": r_xy,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ortho_m": ortho_m,
        "trees": trees,
    }))
    print(f"  [KR3] dumped {len(trees)} tree instances -> {out_path.name}")


def _prepare_blend_for_user(cfg):
    """Make the saved .blend pleasant to open in Blender's GUI:

    * Re-centre the scene so the tile midpoint sits on the world origin
      (KR2 places the tile's SW corner at (0,0); shifting it to be
      symmetric around origin makes orbiting much more natural).
    * Set every 3D View's ``clip_end`` to 100 km so users do not have
      to manually raise the View > End every session.
    * Frame the scene in any 3D View so the geometry is centred when
      the file is reopened.
    """
    import bpy
    # ---- 1. Re-centre geometry around world origin --------------------
    # Tile origin is the SW corner; shift by -ortho/2 so centre sits at 0.
    try:
        ortho_m = float(cfg["render"]["ortho_scale_m"])
    except Exception:
        ortho_m = 0.0
    if ortho_m > 0.0:
        dx = -0.5 * ortho_m
        dy = -0.5 * ortho_m
        tree_assets = bpy.data.collections.get("Tree_Assets")
        tree_asset_names = set(tree_assets.objects.keys()) if tree_assets else set()
        for obj in bpy.data.objects:
            if obj.parent is not None:
                continue  # children move with their parent
            if obj.name in tree_asset_names:
                continue  # GN source templates must keep their source transform
            obj.location.x += dx
            obj.location.y += dy
        # Move the camera too so the topdown render still frames the tile.
        cam = bpy.context.scene.camera
        if cam is not None:
            cam.location.x += dx
            cam.location.y += dy
        bpy.context.view_layer.update()
        print(f"  [KR3] re-centred scene by ({dx:.1f}, {dy:.1f}) m "
              f"(tile centre now at origin)")

    # ---- 2. Bump 3D View clip_end to 100 km ---------------------------
    n_views = 0
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for sp in area.spaces:
                if sp.type == "VIEW_3D":
                    sp.clip_end = 100000.0
                    sp.clip_start = 0.1
                    n_views += 1
    print(f"  [KR3] set 3D-view clip_end=100000m on {n_views} viewport(s)")


def _export_scene_glb_for_web(out_glb):
    """Export the coloured KR3 scene as a separate browser-preview GLB."""
    import bpy

    out_glb = Path(out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    temp_objects = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for instance in depsgraph.object_instances:
        if not instance.is_instance:
            continue
        source_obj = instance.instance_object
        if source_obj is None or source_obj.type != "MESH":
            continue
        if not bool(source_obj.get("is_tree_instance")):
            continue
        dup = source_obj.copy()
        dup.data = source_obj.data.copy()
        dup.animation_data_clear()
        dup.matrix_world = instance.matrix_world.copy()
        dup.name = f"WebPreviewTree_{len(temp_objects):04d}"
        dup.hide_viewport = False
        dup.hide_render = False
        dup.hide_select = False
        bpy.context.collection.objects.link(dup)
        temp_objects.append(dup)
    if temp_objects:
        bpy.context.view_layer.update()
        n_tree_faces = sum(len(obj.data.polygons) for obj in temp_objects)
        print(f"[KR3] WebUI GLB: baked {len(temp_objects)} GN tree instances "
              f"({n_tree_faces} faces)")

    source_cols = {"Tree_Assets", "_tree_templates"}
    view_layer_objects = set(bpy.context.view_layer.objects)
    selected = []
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        if bool(obj.get("is_gn_ground")):
            continue
        if obj not in view_layer_objects:
            continue
        collection_names = {col.name for col in obj.users_collection}
        if collection_names & source_cols:
            continue
        try:
            obj.select_set(True)
        except RuntimeError:
            continue
        selected.append(obj)

    if not selected:
        raise RuntimeError("no mesh objects selected for WebUI GLB export")
    print(f"[KR3] WebUI GLB: selected {len(selected)} mesh objects")
    bpy.context.view_layer.objects.active = selected[0]
    bpy.context.view_layer.update()

    base_kwargs = dict(
        filepath=str(out_glb),
        export_format="GLB",
        use_selection=True,
        export_cameras=False,
        export_lights=False,
    )
    try:
        for kwargs in ({**base_kwargs, "export_apply": True}, base_kwargs):
            try:
                bpy.ops.export_scene.gltf(**kwargs)
                print(f"[KR3] WebUI preview GLB saved -> {out_glb}")
                return
            except TypeError:
                continue
        raise RuntimeError(f"failed to export WebUI preview GLB at {out_glb}")
    finally:
        for obj in temp_objects:
            mesh = obj.data
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
            try:
                if mesh is not None and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            except Exception:
                pass


def _render_topdown_ndsm(cfg, out_exr, out_png):
    """Render a top-down **nDSM** (height-above-ground) pass.

    nDSM = ``camera_height_m - depth_from_camera`` clamped to >= 0, in
    metres. Ground sits at z=0 in the scene (KR2 convention), so this
    pass produces a remote-sensing-style height map: ground -> 0,
    building/tree tops -> their physical heights.

    Outputs:
      * ``out_exr``: 32-bit single-channel OpenEXR with raw nDSM in
        metres (downstream consumers should read with OpenEXR /
        ``imageio.imread(..., flags='-FLAGS_NO_LIBRARY')``).
      * ``out_png``: 16-bit single-channel PNG, normalized [0, 1]
        across the per-tile nDSM range (for quick visualisation /
        overview stitching).

    Uses the same ortho camera as the topdown RGB pass, with real
    geometry (no tree x/y enlargement). Cycles samples=1 — Z pass is
    purely geometric so no shader noise.
    """
    import bpy
    out_exr = Path(out_exr)
    out_png = Path(out_png)
    out_exr.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    scn = bpy.context.scene
    cam = scn.camera
    cam_h = float(cfg["render"]["camera_height_m"])
    ortho_m = float(cfg["render"]["ortho_scale_m"])
    cx = cy = 0.5 * ortho_m
    if cam is not None:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho_m
        cam.location = (cx, cy, cam_h)
        cam.rotation_euler = (0.0, 0.0, 0.0)

    scn.render.engine = "CYCLES"
    _samples_backup = scn.cycles.samples
    _use_nodes_backup = scn.use_nodes
    scn.cycles.samples = 1
    scn.cycles.use_denoising = False
    vl = scn.view_layers[0]
    vl.use_pass_z = True

    # Keep KR2 substrate classes visible in the depth pass. The older output
    # style uses the small class z-offsets (road/water/grass/foliage/ground)
    # as broad grayscale context, with trees/buildings overlaid by real height.
    _hidden_non_objects = []

    # Compositor: RenderLayers.Depth -> (cam_h - Z) -> max(_, 0)
    # Blender 5.0 removed Scene.node_tree / Scene.use_nodes in favour of
    # Scene.compositing_node_group. Fall back gracefully on 4.x.
    if hasattr(scn, "compositing_node_group"):
        nt = bpy.data.node_groups.new(name="_ndsm_comp_", type="CompositorNodeTree")
        scn.compositing_node_group = nt
        scn.use_nodes = True
    else:
        scn.use_nodes = True
        nt = scn.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    # Blender 5.0 unified the compositor with shader nodes; the old
    # CompositorNodeMath was removed in favour of ShaderNodeMath.
    def _new_math():
        for _id in ("CompositorNodeMath", "ShaderNodeMath"):
            try:
                return nt.nodes.new(_id)
            except Exception:
                continue
        raise RuntimeError("no Math node available in this Blender build")
    sub = _new_math(); sub.operation = "SUBTRACT"
    sub.inputs[0].default_value = float(cam_h)
    nt.links.new(rl.outputs["Depth"], sub.inputs[1])
    clamp = _new_math(); clamp.operation = "MAXIMUM"
    clamp.inputs[1].default_value = 0.0
    nt.links.new(sub.outputs[0], clamp.inputs[0])

    # Smooth the nDSM via a Gaussian filter in the Blender compositor to make
    # the output look like a real satellite-derived nDSM (smooth canopy mounds
    # + flat building plateaus) instead of per-tree salt-and-pepper noise.
    # 12 px @ 0.5 m/px ≈ 6 m kernel — about twice the typical tree spacing —
    # so dense scatter blends into continuous canopy, while building edges
    # stay sharp enough to read.
    blur = nt.nodes.new("CompositorNodeBlur")
    _BLUR_PX = 12
    if "Size" in blur.inputs:
        blur.inputs["Size"].default_value = (float(_BLUR_PX), float(_BLUR_PX))
    else:
        blur.size_x = _BLUR_PX
        blur.size_y = _BLUR_PX
    if "Type" in blur.inputs:
        blur.inputs["Type"].default_value = "Gaussian"
    elif hasattr(blur, "filter_type"):
        blur.filter_type = "GAUSSIAN"
    nt.links.new(clamp.outputs[0], blur.inputs["Image"])

    # EXR sink (raw metres, float32)
    fo_exr = nt.nodes.new("CompositorNodeOutputFile")
    # Blender 5.0: `base_path` is now `directory`, `file_slots` is now
    # `file_output_items`. Single-slot output: keep the default first
    # input ("Image") and rename it to our prefix.
    _has_dir = hasattr(fo_exr, "directory")
    if _has_dir:
        fo_exr.directory = str(out_exr.parent)
    else:
        fo_exr.base_path = str(out_exr.parent)
    slot_exr_name = "_ndsm_exr_"
    if hasattr(fo_exr, "file_output_items"):
        # Rename existing default item (index 0) instead of clearing.
        if len(fo_exr.file_output_items) > 0:
            fo_exr.file_output_items[0].name = slot_exr_name
        else:
            fo_exr.file_output_items.new("FLOAT", slot_exr_name)
    else:
        fo_exr.file_slots.clear()
        fo_exr.file_slots.new(slot_exr_name)
    # Blender 5.0: OutputFile node defaults to MULTI_LAYER_IMAGE media
    # which forces file_format=OPEN_EXR_MULTILAYER. Switch to IMAGE so
    # we can write plain single-layer OPEN_EXR.
    if hasattr(fo_exr.format, "media_type"):
        try:
            fo_exr.format.media_type = "IMAGE"
        except Exception:
            pass
    fo_exr.format.file_format = "OPEN_EXR"
    fo_exr.format.color_depth = "32"
    fo_exr.format.color_mode = "BW"
    nt.links.new(blur.outputs["Image"], fo_exr.inputs[0])

    # PNG vis (16-bit normalized)
    fo_png = nt.nodes.new("CompositorNodeOutputFile")
    if _has_dir:
        fo_png.directory = str(out_png.parent)
    else:
        fo_png.base_path = str(out_png.parent)
    slot_png_name = "_ndsm_png_"
    if hasattr(fo_png, "file_output_items"):
        if len(fo_png.file_output_items) > 0:
            fo_png.file_output_items[0].name = slot_png_name
        else:
            fo_png.file_output_items.new("FLOAT", slot_png_name)
    else:
        fo_png.file_slots.clear()
        fo_png.file_slots.new(slot_png_name)
    if hasattr(fo_png.format, "media_type"):
        try:
            fo_png.format.media_type = "IMAGE"
        except Exception:
            pass
    fo_png.format.file_format = "PNG"
    fo_png.format.color_depth = "16"
    fo_png.format.color_mode = "BW"
    
    # Render LiDAR-realistic absolute scale DSM: map 0..NDSM_MAX_M physical
    # heights linearly to 0..1.0 (clamped), instead of dynamic normalisation.
    # 25 m ceiling makes typical 8–15 m tree canopies and 9–20 m buildings
    # land in the readable mid-to-upper grey range (similar to airborne
    # LiDAR nDSM rasters), instead of compressing them to near-black.
    _NDSM_MAX_M = 25.0
    div = _new_math()
    div.operation = "DIVIDE"
    nt.links.new(blur.outputs["Image"], div.inputs[0])
    div.inputs[1].default_value = _NDSM_MAX_M
    
    clamper = _new_math()
    clamper.operation = "MINIMUM"
    nt.links.new(div.outputs[0], clamper.inputs[0])
    clamper.inputs[1].default_value = 1.0
    
    nt.links.new(clamper.outputs[0], fo_png.inputs[0])

    # Trigger a render; main image_settings can stay PNG (we don't
    # use scn.render.filepath here, the File Output nodes write).
    scn.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=False)

    # File Output naming varies across Blender versions:
    #   4.x: "{slot}{frame:04d}.{ext}"
    #   5.0: "file_name{slot}.{ext}" (no frame, default template)
    # Glob for whatever the actual produced file is and move it to the
    # user-requested path.
    def _resolve_and_move(out_dir: Path, slot_name: str, ext: str,
                          target: Path) -> bool:
        candidates = sorted(out_dir.glob(f"*{slot_name}*.{ext}"))
        if not candidates:
            return False
        src = candidates[0]
        try:
            if target.exists():
                target.unlink()
            src.replace(target)
            return True
        except Exception as _e:
            print(f"[KR3] move {src} -> {target} failed: {_e}")
            return False

    try:
        if _resolve_and_move(out_exr.parent, slot_exr_name, "exr", out_exr):
            print(f"[KR3] nDSM EXR saved -> {out_exr}")
        else:
            print(f"[KR3] WARN nDSM EXR missing (slot={slot_exr_name})")
        if _resolve_and_move(out_png.parent, slot_png_name, "png", out_png):
            print(f"[KR3] nDSM PNG saved -> {out_png}")
        else:
            print(f"[KR3] WARN nDSM PNG missing (slot={slot_png_name})")
    except Exception as e:  # noqa: BLE001
        print(f"[KR3] nDSM rename failed: {e}")

    # Restore render visibility of non-object classes
    for obj in _hidden_non_objects:
        obj.hide_render = False

    # Clean up the compositor so it doesn't leak into later renders.
    if hasattr(scn, "compositing_node_group"):
        try:
            scn.compositing_node_group = None
            if nt.name in bpy.data.node_groups:
                bpy.data.node_groups.remove(nt)
        except Exception:
            pass
    else:
        scn.use_nodes = bool(_use_nodes_backup)
    scn.cycles.samples = int(_samples_backup)


def _render_topdown_shader_depth_png(cfg, out_png):
    """Render the official top-down depth PNG with a world-Z emission shader."""
    import bpy

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    scn = bpy.context.scene
    cam = scn.camera
    cam_h = float(cfg["render"]["camera_height_m"])
    ortho_m = float(cfg["render"]["ortho_scale_m"])
    size = int(cfg["render"].get("image_size", 1024))
    max_h = float(cfg["render"].get("depth_max_height_m", 30.0))
    cx = cy = 0.5 * ortho_m
    if cam is not None:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho_m
        cam.data.clip_start = 0.1
        cam.data.clip_end = cam_h + 100.0
        cam.location = (cx, cy, cam_h)
        cam.rotation_euler = (0.0, 0.0, 0.0)

    mat = bpy.data.materials.new("_DepthByWorldZ_Emission")
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    map_range = nt.nodes.new("ShaderNodeMapRange")
    emission = nt.nodes.new("ShaderNodeEmission")
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    map_range.inputs[1].default_value = 0.0
    map_range.inputs[2].default_value = max_h
    map_range.inputs[3].default_value = 0.0
    map_range.inputs[4].default_value = 1.0
    if hasattr(map_range, "clamp"):
        map_range.clamp = True
    emission.inputs[1].default_value = 1.0
    nt.links.new(geom.outputs["Position"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], map_range.inputs[0])
    nt.links.new(map_range.outputs[0], emission.inputs[0])
    nt.links.new(emission.outputs[0], out_node.inputs["Surface"])

    view_layer = bpy.context.view_layer
    old_override = getattr(view_layer, "material_override", None)
    old_engine = scn.render.engine
    old_samples = scn.cycles.samples if hasattr(scn, "cycles") else None
    old_denoising = scn.cycles.use_denoising if hasattr(scn, "cycles") else None
    old_filepath = scn.render.filepath
    old_format = scn.render.image_settings.file_format
    old_mode = scn.render.image_settings.color_mode
    old_depth = scn.render.image_settings.color_depth
    try:
        scn.render.engine = "CYCLES"
        scn.cycles.samples = 1
        scn.cycles.use_denoising = False
        scn.render.resolution_x = size
        scn.render.resolution_y = size
        scn.render.resolution_percentage = 100
        try:
            scn.view_settings.view_transform = "Standard"
            scn.view_settings.look = "None"
            scn.view_settings.exposure = 0.0
            scn.view_settings.gamma = 1.0
        except Exception:
            pass
        view_layer.material_override = mat
        scn.render.filepath = str(out_png)
        scn.render.image_settings.file_format = "PNG"
        scn.render.image_settings.color_mode = "BW"
        scn.render.image_settings.color_depth = "8"
        bpy.ops.render.render(write_still=True)
        print(f"[KR3] shader depth PNG saved -> {out_png}")
    finally:
        view_layer.material_override = old_override
        scn.render.engine = old_engine
        if old_samples is not None:
            scn.cycles.samples = old_samples
        if old_denoising is not None:
            scn.cycles.use_denoising = old_denoising
        scn.render.filepath = old_filepath
        scn.render.image_settings.file_format = old_format
        scn.render.image_settings.color_mode = old_mode
        scn.render.image_settings.color_depth = old_depth


def _render_preview_png(cfg, out_png, class_ids, iso_png=None,
                         hide_foliage_substrate=True,
                         topdown_tree_xy_scale=1.0,
                         depth_exr=None, depth_png=None):
    """Render top-down RGB PNG (and optionally an isometric/3D PNG).

    Mutates render settings and materials -- call AFTER saving the .blend.
    Used by osm_app to show 2D + 3D previews of the assembled scene without
    the user having to open Blender. Both renders use the same per-class
    emission shader so colors match KR1's seg overlay.

    When ``hide_foliage_substrate`` is True (the default) the foliage
    *polygon* mesh used as scatter substrate is hidden from render, so
    the green you see in the top-down view is exclusively from actual
    placed tree/bush instances. The substrate is restored before the iso
    render so the 3D view still shows the green ground plane underneath.
    """
    import bpy
    import math
    palette = {}
    for name, cid in class_ids.items():
        rgb = cfg.get("class_colors", {}).get(name, [180, 180, 180])
        palette[int(cid)] = (rgb[0] / 255.0, rgb[1] / 255.0,
                             rgb[2] / 255.0, 1.0)
    default_color = (0.7, 0.7, 0.7, 1.0)

    ortho_m = float(cfg["render"]["ortho_scale_m"])
    cx = cy = 0.5 * ortho_m
    cam = bpy.context.scene.camera
    cam_h = float(cfg["render"]["camera_height_m"])

    # Per-class emission so the preview matches the final seg palette.
    # Actual tree instances stay foliage green. Non-tree foliage/canopy
    # substrate is rendered as grass purple, matching auto_pipeline's
    # final 4_seg.png composition where growable substrate is grass and
    # only individual tree crowns are foliage.
    TREE_RGBA = palette.get(2, (0.0, 1.0, 0.0, 1.0))
    GRASS_RGBA = palette.get(4, (0.5, 0.0, 0.5, 1.0))
    TREE_STRENGTH = 1.0
    n_tree_instances = 0
    n_foliage_substrate = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        is_tree = bool(obj.get("is_tree_instance"))
        if is_tree:
            col = TREE_RGBA
            strength = TREE_STRENGTH
            n_tree_instances += 1
        elif obj.get("class") == "foliage":
            col = GRASS_RGBA
            strength = 1.0
            n_foliage_substrate += 1
        else:
            col = palette.get(int(obj.pass_index), default_color)
            strength = 1.0
        mat = bpy.data.materials.new(f"_prev_{obj.name}")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = col
        emit.inputs["Strength"].default_value = strength
        out_node = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
        if is_tree:
            # GN-virtual instances inherit the master mesh's material
            # slot list and per-poly material_index. To make EVERY face
            # of every instance render green we must REPLACE each slot
            # in place rather than clear+append (which would collapse to
            # a single slot and leave faces with stale material_index
            # pointing at empty slots, falling back to default gray).
            try:
                n_slots = max(1, len(obj.data.materials))
                # Resize materials list if empty
                if len(obj.data.materials) == 0:
                    obj.data.materials.append(mat)
                else:
                    for si in range(n_slots):
                        obj.data.materials[si] = mat
            except Exception:
                pass
        else:
            try:
                obj.data.materials.clear()
            except Exception:
                pass
            obj.data.materials.append(mat)
    print(f"[KR3] preview: tagged {n_tree_instances} tree instances "
          f"green and {n_foliage_substrate} foliage substrate object(s) "
          f"as grass-purple", flush=True)

    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.samples = 8
    scn.cycles.use_denoising = False
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.color_depth = "8"
    vl = scn.view_layers[0]
    if hasattr(vl, "use_pass_z"):
        vl.use_pass_z = False
    if hasattr(vl, "use_pass_object_index"):
        vl.use_pass_object_index = False

    # ---- (1) top-down ortho ------------------------------------------- #
    if cam is not None:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho_m
        cam.location = (cx, cy, cam_h)
        cam.rotation_euler = (0.0, 0.0, 0.0)

    # Optionally hide the foliage *substrate* polygons so the top-down
    # green is exclusively from placed tree / bush instances. Substrate
    # objects keep ``obj["class"] == "foliage"`` but their *name* starts
    # with "foliage" (the GLB class prefix from KR2), whereas instances
    # are spawned with names "tree_*" / "bush_*".
    _hidden_substrate = []
    if hide_foliage_substrate:
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            if obj.get("class") != "foliage":
                continue
            # Skip placed tree / bush instances (tagged in _spawn_instance).
            if obj.get("is_tree_instance"):
                continue
            n = obj.name.lower()
            if n.startswith(("tree_", "trees_", "bush_", "bushes_",
                              "cnp_", "street_", "child_")):
                continue
            if not obj.hide_render:
                obj.hide_render = True
                _hidden_substrate.append(obj)
        if _hidden_substrate:
            print(f"[KR3] top-view: hidden {len(_hidden_substrate)} foliage "
                  f"substrate object(s) so only tree crowns show as green",
                  flush=True)

    # Inflate tree-instance crowns in XY ONLY (Z unchanged) so the
    # top-down render reads as a remote-sensing-style canopy patch
    # without inflating tree height — heights stay driven by ETH canopy
    # / --tree-height-min/max.
    # NOTE: this scaling is now PERMANENT — applied once and never
    # restored — so the saved .blend 3D scene contains trees at the
    # exact same canopy size as the 3_seg.png mask. Restoring after the
    # render (former behaviour) caused the seg mask to show much fatter
    # canopies than what the 3D scene actually had.
    try:
        sxy = float(topdown_tree_xy_scale)
    except Exception:
        sxy = 1.0
    if sxy != 1.0 and sxy > 0.0:
        # GN scatter exposes a `Topdown XY Inflate` modifier input on the
        # ground-plane object; driving it inflates every virtual tree
        # instance's XY scale (Z kept) at zero memory cost. This is
        # PERMANENT for the saved .blend so the 3D scene matches the
        # top-down seg / depth maps 1:1.
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import _tree_scatter_gn as _gn  # type: ignore
        ground = bpy.data.objects.get("GN_GroundPlane")
        if ground is not None:
            _gn.set_topdown_xy_inflate(ground, sxy)
            print(f"[KR3] top-view: GN topdown XY inflate = {sxy:.2f}x "
                  f"(PERMANENT — seg & .blend will match 1:1)")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    scn.render.filepath = str(out_png)
    bpy.ops.render.render(write_still=True)
    print(f"[KR3] preview PNG saved -> {out_png}")

    # Restore visibility for the depth pass, iso render and any downstream measurement.
    for obj in _hidden_substrate:
        obj.hide_render = False

    # ---- (1b) top-down nDSM depth ------------------------------------- #
    if depth_exr or depth_png:
        try:
            _render_topdown_ndsm(
                cfg,
                out_exr=(depth_exr or (out_png.parent / "_ndsm_tmp.exr")),
                out_png=(depth_png or (out_png.parent / "_ndsm_tmp.png")),
            )
            if depth_png:
                _render_topdown_shader_depth_png(cfg, depth_png)
        except Exception as e:  # noqa: BLE001
            print(f"[KR3] nDSM render failed: {e}")

    # NOTE: tree XY scales are NOT restored after rendering — the
    # ``topdown_tree_xy_scale`` inflation is intentionally permanent so
    # the saved .blend 3D scene matches the topdown seg / depth maps
    # 1:1. (Pass --topdown-tree-xy-scale 1.0 to disable inflation.)

    # ---- (2) isometric / 3D ------------------------------------------- #
    if iso_png and cam is not None:
        from mathutils import Vector
        old_clip_start = cam.data.clip_start
        old_clip_end = cam.data.clip_end
        old_res_x = scn.render.resolution_x
        old_res_y = scn.render.resolution_y
        old_res_pct = scn.render.resolution_percentage
        cam.data.type = "ORTHO"
        elev = math.radians(60.0)   # match pointcloud preview angle
        az = math.radians(225.0)    # SW -> looking toward NE
        dist = ortho_m * 2.2
        center = Vector((cx, cy, 0.0))
        cam_pos = Vector((
            cx + dist * math.cos(elev) * math.cos(az),
            cy + dist * math.cos(elev) * math.sin(az),
            dist * math.sin(elev),
        ))
        cam.location = cam_pos
        direction = center - cam_pos
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        view_dir = direction.normalized()
        right_vec = view_dir.cross(Vector((0.0, 0.0, 1.0))).normalized()
        up_vec = right_vec.cross(view_dir).normalized()
        corners = [
            Vector((0.0, 0.0, 0.0)), Vector((ortho_m, 0.0, 0.0)),
            Vector((ortho_m, ortho_m, 0.0)), Vector((0.0, ortho_m, 0.0)),
            Vector((0.0, 0.0, 80.0)), Vector((ortho_m, 0.0, 80.0)),
            Vector((ortho_m, ortho_m, 80.0)), Vector((0.0, ortho_m, 80.0)),
        ]
        proj_x = [(p - center).dot(right_vec) for p in corners]
        proj_y = [(p - center).dot(up_vec) for p in corners]
        cam.data.ortho_scale = max(max(proj_x) - min(proj_x),
                                   max(proj_y) - min(proj_y)) * 1.08
        cam.data.clip_start = 0.05
        cam.data.clip_end = max(10000.0, dist + ortho_m * 4.0)
        scn.render.resolution_x = 1024
        scn.render.resolution_y = 1024
        scn.render.resolution_percentage = 100
        iso_png = Path(iso_png)
        iso_png.parent.mkdir(parents=True, exist_ok=True)
        scn.render.filepath = str(iso_png)
        bpy.ops.render.render(write_still=True)
        cam.data.clip_start = old_clip_start
        cam.data.clip_end = old_clip_end
        scn.render.resolution_x = old_res_x
        scn.render.resolution_y = old_res_y
        scn.render.resolution_percentage = old_res_pct
        print(f"[KR3] iso preview PNG saved -> {iso_png}")


def _semantic_color_map(cfg):
    colors = {}
    for class_name, rgb in (cfg.get("class_colors") or {}).items():
        try:
            colors[str(class_name)] = tuple(int(max(0, min(255, v))) for v in rgb[:3])
        except Exception:
            pass
    return colors


def _object_semantic_class(obj):
    if obj is None:
        return "ground"
    if bool(obj.get("is_tree_instance")):
        return "foliage"
    class_name = obj.get("class")
    if class_name in CLASS_IDS:
        return str(class_name)
    try:
        class_id = int(obj.get("class_id", obj.pass_index))
    except Exception:
        class_id = 0
    for name, value in CLASS_IDS.items():
        if int(value) == class_id:
            return str(name)
    return "ground"


def _mesh_local_triangles(source_obj, depsgraph, cache):
    cache_key = source_obj.name_full
    if cache_key in cache:
        return cache[cache_key]
    eval_obj = source_obj.evaluated_get(depsgraph)
    mesh = None
    try:
        mesh = eval_obj.to_mesh()
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            cache[cache_key] = None
            return None
        import numpy as np
        vertices = np.asarray([v.co[:] for v in mesh.vertices], dtype=np.float64)
        tri_indices = np.asarray(
            [[loop_tri.vertices[0], loop_tri.vertices[1], loop_tri.vertices[2]]
             for loop_tri in mesh.loop_triangles],
            dtype=np.int64,
        )
        local_tris = vertices[tri_indices]
        cache[cache_key] = local_tris
        return local_tris
    finally:
        if mesh is not None:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass


def _sample_surface_pointcloud(cfg, target_count=50000, voxel_size=0.5, seed=12345):
    import bpy
    import numpy as np

    target_count = max(1, int(target_count))
    voxel_size = max(0.05, float(voxel_size))
    rng = np.random.default_rng(int(seed))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    palette = _semantic_color_map(cfg)
    default_rgb = palette.get("ground", (0, 0, 0))

    tri_blocks = []
    area_blocks = []
    rgb_blocks = []
    class_blocks = []
    mesh_cache = {}
    n_sources = 0

    for instance in depsgraph.object_instances:
        source_obj = instance.instance_object if instance.is_instance else instance.object
        if source_obj is None or source_obj.type != "MESH":
            continue
        if source_obj.hide_render:
            continue
        if bool(source_obj.get("is_gn_ground")):
            continue
        if (not instance.is_instance) and bool(source_obj.get("is_tree_instance")):
            continue

        local_tris = _mesh_local_triangles(source_obj, depsgraph, mesh_cache)
        if local_tris is None or len(local_tris) == 0:
            continue

        matrix = instance.matrix_world
        matrix_np = np.asarray([[matrix[row][col] for col in range(4)] for row in range(4)], dtype=np.float64)
        world_tris = (local_tris.reshape(-1, 3) @ matrix_np[:3, :3].T + matrix_np[:3, 3]).reshape(-1, 3, 3)
        edge_1 = world_tris[:, 1, :] - world_tris[:, 0, :]
        edge_2 = world_tris[:, 2, :] - world_tris[:, 0, :]
        areas = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1)
        valid = np.isfinite(areas) & (areas > 1e-8)
        if not np.any(valid):
            continue

        class_name = _object_semantic_class(source_obj)
        rgb = palette.get(class_name, default_rgb)
        class_id = int(CLASS_IDS.get(class_name, 0))
        valid_count = int(valid.sum())
        tri_blocks.append(world_tris[valid])
        area_blocks.append(areas[valid])
        rgb_blocks.append(np.tile(np.asarray(rgb, dtype=np.uint8), (valid_count, 1)))
        class_blocks.append(np.full(valid_count, class_id, dtype=np.uint8))
        n_sources += 1

    if not area_blocks:
        raise RuntimeError("no renderable mesh surfaces found for pointcloud sampling")

    triangles = np.concatenate(tri_blocks, axis=0)
    areas = np.concatenate(area_blocks, axis=0)
    triangle_rgb = np.concatenate(rgb_blocks, axis=0)
    triangle_class_id = np.concatenate(class_blocks, axis=0)
    area_total = float(areas.sum())
    if not np.isfinite(area_total) or area_total <= 0.0:
        raise RuntimeError("pointcloud sampling found zero total surface area")

    oversample_count = max(target_count * 3, target_count + 5000)
    cdf = np.cumsum(areas)
    chosen = np.searchsorted(cdf, rng.random(oversample_count) * cdf[-1], side="right")
    chosen = np.clip(chosen, 0, len(areas) - 1)

    selected_tris = triangles[chosen]
    bary_u = rng.random(oversample_count)
    bary_v = rng.random(oversample_count)
    flip = (bary_u + bary_v) > 1.0
    bary_u[flip] = 1.0 - bary_u[flip]
    bary_v[flip] = 1.0 - bary_v[flip]
    points = (selected_tris[:, 0, :]
              + bary_u[:, None] * (selected_tris[:, 1, :] - selected_tris[:, 0, :])
              + bary_v[:, None] * (selected_tris[:, 2, :] - selected_tris[:, 0, :]))
    colors = triangle_rgb[chosen]
    class_ids = triangle_class_id[chosen]

    cells = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(cells, axis=0, return_index=True)
    unique_indices.sort()
    if len(unique_indices) >= target_count:
        keep = unique_indices[:target_count]
    else:
        remaining = target_count - len(unique_indices)
        fill = rng.choice(len(points), size=remaining, replace=(remaining > len(points)))
        keep = np.concatenate([unique_indices, fill])

    sampled_points = points[keep].astype(np.float32, copy=False)
    sampled_colors = colors[keep].astype(np.uint8, copy=False)
    sampled_class_ids = class_ids[keep].astype(np.uint8, copy=False)
    print(f"[KR3] pointcloud sampled {len(sampled_points)} points from "
          f"{len(areas)} triangles across {n_sources} surfaces "
          f"(voxel={voxel_size:.2f}m)")
    return sampled_points, sampled_colors, sampled_class_ids


def _write_pointcloud_ply(out_ply, points, colors, class_ids):
    import numpy as np
    out_ply = Path(out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    class_ids = np.asarray(class_ids, dtype=np.uint8)
    vertex_count = int(len(points))
    label_comments = "".join(
        f"comment class_label {int(class_id)} {name}\n"
        for name, class_id in sorted(CLASS_IDS.items(), key=lambda item: int(item[1]))
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated by SynthUrbanSAT OSM native Blender pointcloud exporter\n"
        + label_comments +
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar class_id\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("class_id", "u1"),
    ])
    data = np.empty(vertex_count, dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["red"] = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"] = colors[:, 2]
    data["class_id"] = class_ids
    with open(out_ply, "wb") as file_obj:
        file_obj.write(header)
        data.tofile(file_obj)
    print(f"[KR3] pointcloud PLY saved -> {out_ply} ({vertex_count} points)")


def _render_pointcloud_png(cfg, out_png, points, colors):
    import bpy
    import numpy as np
    from mathutils import Vector

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(points) == 0:
        raise RuntimeError("cannot render empty pointcloud")

    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        scene.camera = cam

    old_filepath = scene.render.filepath
    old_film_transparent = scene.render.film_transparent
    old_camera_location = cam.location.copy()
    old_camera_rotation = cam.rotation_euler.copy()
    old_camera_type = cam.data.type
    old_ortho_scale = cam.data.ortho_scale
    old_clip_start = cam.data.clip_start
    old_clip_end = cam.data.clip_end
    old_res_x = scene.render.resolution_x
    old_res_y = scene.render.resolution_y
    old_res_pct = scene.render.resolution_percentage
    old_hidden = {obj: obj.hide_render for obj in bpy.data.objects}

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    old_world_color = tuple(world.color)

    pc_obj = None
    try:
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        center = 0.5 * (min_xyz + max_xyz)
        span_xy = max(float(max_xyz[0] - min_xyz[0]), float(max_xyz[1] - min_xyz[1]), 1.0)
        span_z = max(float(max_xyz[2] - min_xyz[2]), 1.0)
        elev = math.radians(60.0)
        azimuth = math.radians(225.0)
        distance = max(span_xy * 2.2, span_z * 8.0, 500.0)
        cam_pos = Vector((
            float(center[0]) + distance * math.cos(elev) * math.cos(azimuth),
            float(center[1]) + distance * math.cos(elev) * math.sin(azimuth),
            float(center[2]) + distance * math.sin(elev),
        ))
        target = Vector((float(center[0]), float(center[1]), float(center[2])))
        direction = target - cam_pos
        cam.location = cam_pos
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.data.type = "ORTHO"
        cam.data.clip_start = 0.05
        cam.data.clip_end = max(10000.0, distance + span_xy * 4.0 + span_z * 8.0)

        view_dir = direction.normalized()
        right_vec = view_dir.cross(Vector((0.0, 0.0, 1.0))).normalized()
        up_vec = right_vec.cross(view_dir).normalized()
        centered = points - center[None, :]
        right_axis = np.asarray(right_vec[:], dtype=np.float32)
        up_axis = np.asarray(up_vec[:], dtype=np.float32)
        proj_x = centered @ right_axis
        proj_y = centered @ up_axis
        projected_width = max(float(proj_x.max() - proj_x.min()), 1.0)
        projected_height = max(float(proj_y.max() - proj_y.min()), 1.0)
        ortho_scale = max(projected_width, projected_height) * 1.08
        cam.data.ortho_scale = ortho_scale
        point_size = max(0.16, min(0.55, ortho_scale / 1800.0))
        right_np = right_axis * point_size
        up_np = up_axis * point_size

        offsets = np.stack([
            -right_np - up_np,
            right_np - up_np,
            right_np + up_np,
            -right_np + up_np,
        ], axis=0)
        quad_vertices = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        faces = [(base, base + 1, base + 2, base + 3)
                 for base in range(0, len(points) * 4, 4)]

        mesh = bpy.data.meshes.new("PointCloudPreviewMesh")
        mesh.from_pydata([tuple(v) for v in quad_vertices], [], faces)
        mesh.update()
        pc_obj = bpy.data.objects.new("PointCloudPreview", mesh)
        bpy.context.collection.objects.link(pc_obj)

        material_indices = {}
        for rgb_tuple in sorted({tuple(int(v) for v in rgb) for rgb in colors.tolist()}):
            mat = bpy.data.materials.new(f"_pc_{rgb_tuple[0]}_{rgb_tuple[1]}_{rgb_tuple[2]}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            for node in list(nodes):
                nodes.remove(node)
            emit = nodes.new("ShaderNodeEmission")
            emit.inputs["Color"].default_value = (
                rgb_tuple[0] / 255.0, rgb_tuple[1] / 255.0, rgb_tuple[2] / 255.0, 1.0)
            emit.inputs["Strength"].default_value = 1.2
            out_node = nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
            material_indices[rgb_tuple] = len(pc_obj.data.materials)
            pc_obj.data.materials.append(mat)

        for poly_index, poly in enumerate(pc_obj.data.polygons):
            rgb_tuple = tuple(int(v) for v in colors[poly_index])
            poly.material_index = material_indices.get(rgb_tuple, 0)

        for obj in bpy.data.objects:
            if obj != pc_obj:
                obj.hide_render = True
        pc_obj.hide_render = False
        scene.render.film_transparent = False
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024
        scene.render.resolution_percentage = 100
        world.color = (1.0, 1.0, 1.0)
        scene.render.filepath = str(out_png)
        bpy.ops.render.render(write_still=True)
        print(f"[KR3] pointcloud preview PNG saved -> {out_png}")
    finally:
        scene.render.filepath = old_filepath
        scene.render.film_transparent = old_film_transparent
        cam.location = old_camera_location
        cam.rotation_euler = old_camera_rotation
        cam.data.type = old_camera_type
        cam.data.ortho_scale = old_ortho_scale
        cam.data.clip_start = old_clip_start
        cam.data.clip_end = old_clip_end
        scene.render.resolution_x = old_res_x
        scene.render.resolution_y = old_res_y
        scene.render.resolution_percentage = old_res_pct
        world.color = old_world_color
        for obj, hide_render in old_hidden.items():
            try:
                obj.hide_render = hide_render
            except ReferenceError:
                pass
        if pc_obj is not None:
            mesh = pc_obj.data
            bpy.data.objects.remove(pc_obj, do_unlink=True)
            bpy.data.meshes.remove(mesh)


def _export_pointcloud_outputs(cfg, ply_path=None, png_path=None,
                               target_count=50000, voxel_size=0.5, seed=12345):
    points, colors, _class_ids = _sample_surface_pointcloud(
        cfg, target_count=target_count, voxel_size=voxel_size, seed=seed)
    if ply_path:
        _write_pointcloud_ply(ply_path, points, colors, _class_ids)
    if png_path:
        _render_pointcloud_png(cfg, png_path, points, colors)
    return len(points)


def _add_roof_clutter(rng, max_per_building=3, prob=0.5):
    """Tiny boxes on building roofs to break flat top.

    Baked into the .blend at KR3 so Cycles' depsgraph picks them up at
    render time. pass_index = building (3).
    """
    import bpy
    from mathutils import Vector
    n_added = 0
    for b in _list_buildings():
        if rng.random() > prob:
            continue
        bb_min = b.matrix_world @ Vector(b.bound_box[0])
        bb_max = b.matrix_world @ Vector(b.bound_box[6])
        for _ in range(rng.randint(1, max_per_building)):
            sx = rng.uniform(0.5, 2.5)
            sy = rng.uniform(0.5, 2.5)
            sz = rng.uniform(0.3, 1.5)
            cx = rng.uniform(bb_min.x + sx, bb_max.x - sx)
            cy = rng.uniform(bb_min.y + sy, bb_max.y - sy)
            cz = bb_max.z + sz / 2
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
            obj = bpy.context.active_object
            obj.scale = (sx, sy, sz)
            obj["class"] = "building"
            obj["class_id"] = 3
            obj.pass_index = 3
            n_added += 1
    print(f"  [KR3] roof clutter added: {n_added}")




def _load_tree_species_dir(asset_dir, species_names=None):
    """Load one master template per `<species>.blend` in ``asset_dir``.

    Each file is expected to contain at least one MESH object; the mesh
    with the largest world-space Z extent is taken as the master template
    for that species (other meshes are appended but hidden).

    Args:
        asset_dir: Directory holding `<name>.blend` files.
        species_names: If given, only these species (matched by file stem,
            case-insensitive) are loaded. ``None`` or empty -> load all.

    Returns:
        ``[(species_name, master_obj, base_h_m, base_z_min, is_bush), ...]``
        ``is_bush`` is auto-detected from the filename prefix ``bush_``.
    """
    import bpy
    from mathutils import Vector

    def _consolidate_species_mesh(species_name, source_objs, holder_col):
        mesh_objs = [obj for obj in source_objs if obj is not None and obj.type == "MESH"]
        if not mesh_objs:
            return None

        temp_dups = []
        try:
            for idx, src in enumerate(mesh_objs):
                dup = src.copy()
                dup.data = src.data.copy()
                dup.animation_data_clear()
                dup.matrix_world = src.matrix_world.copy()
                dup.parent = None
                dup.name = f"_tree_tmp_{species_name}_{idx:02d}"
                holder_col.objects.link(dup)
                dup.hide_viewport = True
                dup.hide_render = True
                temp_dups.append(dup)

            if not temp_dups:
                return None

            bpy.ops.object.select_all(action="DESELECT")
            for dup in temp_dups:
                dup.select_set(True)
            bpy.context.view_layer.objects.active = temp_dups[0]
            bpy.ops.object.join()
            merged = bpy.context.view_layer.objects.active
            merged.name = f"TreeTemplate_{species_name}"
            bpy.ops.object.select_all(action="DESELECT")
            return merged
        except Exception:
            for dup in temp_dups:
                try:
                    if dup.name in bpy.data.objects:
                        bpy.data.objects.remove(dup, do_unlink=True)
                except Exception:
                    pass
            raise

    out = []
    if not asset_dir:
        return out
    asset_dir = Path(asset_dir)
    if not asset_dir.is_dir():
        print(f"  [KR3] tree_assets dir not found: {asset_dir}")
        return out

    blends = sorted(asset_dir.glob("*.blend"))
    wanted = None
    if species_names:
        wanted = {s.strip().lower() for s in species_names if s.strip()}
    holder = bpy.data.collections.get("_tree_templates")
    if holder is None:
        holder = bpy.data.collections.new("_tree_templates")
        bpy.context.scene.collection.children.link(holder)

    for bf in blends:
        sp_name = bf.stem
        if wanted is not None and sp_name.lower() not in wanted:
            continue
        try:
            with bpy.data.libraries.load(str(bf), link=False) as (df, dt):
                dt.objects = [n for n in df.objects]
        except Exception as e:
            print(f"  [KR3] failed to load {bf.name}: {e}")
            continue

        mesh_objs = [obj for obj in dt.objects if obj is not None and obj.type == "MESH"]
        if not mesh_objs:
            print(f"  [KR3] {bf.name}: no usable mesh; skipping")
            continue

        try:
            master = _consolidate_species_mesh(sp_name, mesh_objs, holder)
        except Exception as e:
            print(f"  [KR3] failed to consolidate {bf.name}: {e}")
            continue
        if master is None:
            print(f"  [KR3] {bf.name}: consolidation produced no mesh; skipping")
            continue

        master.hide_viewport = True
        master.hide_render = True
        master["is_tree_instance"] = True
        master["class"] = "foliage"
        master["class_id"] = 2
        master.pass_index = 2
        zs = [(master.matrix_world @ Vector(c)).z for c in master.bound_box]
        base_h = float(max(zs) - min(zs))
        base_z = float(min(zs))
        if base_h < 0.05:
            print(f"  [KR3] {bf.name}: consolidated mesh too small; skipping")
            try:
                bpy.data.objects.remove(master, do_unlink=True)
            except Exception:
                pass
            continue
        is_bush = sp_name.lower().startswith("bush")
        out.append((sp_name, master, base_h, base_z, is_bush))
        print(f"  [KR3] species '{sp_name}' <- {bf.name} "
              f"(base_h={base_h:.2f}m, bush={is_bush})")
    return out


# ---------------------------------------------------------------------------
# NOTE: All Python-side scatter functions have been removed in favour of the
# Geometry Nodes pipeline in `scripts/_tree_scatter_gn.py`. See _code_modify.md.
# ---------------------------------------------------------------------------


def main():
    import bpy
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--out", default=None,
                    help="output .blend path (default: output/meshes/{city}.blend)")
    ap.add_argument("--no-foliage", action="store_true",
                    help="skip proxy-tree scattering")
    ap.add_argument("--no-clutter", action="store_true",
                    help="skip roof clutter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scatter-seed", type=int, default=None,
                    help="separate seed for tree scattering; falls back "
                         "to --seed when omitted")
    ap.add_argument("--tree-density", type=float, default=0.0008,
                    help="trees per square meter of foliage polygon area")
    ap.add_argument("--max-trees", type=int, default=3000)
    # ----- PCG: per-species library + multi-surface scatter ----- #
    ap.add_argument("--tree-species", default="",
                    help="comma-separated species names (file stems in "
                         "tree_assets.dir). Empty = use all *.blend in dir.")
    ap.add_argument("--tree-height-dist", default=None,
                    choices=["flat", "uniform", "lognormal", "bimodal",
                             "beta_u"],
                    help="per-tree height distribution (default from yaml)")
    ap.add_argument("--tree-height-seed", type=int, default=None)
    ap.add_argument("--tree-height-scale", type=float, default=None,
                    help="DEPRECATED: kept for backwards compat; "
                         "prefer --tree-height-min/--tree-height-max")
    ap.add_argument("--tree-height-min", type=float, default=None,
                    help="min tree height (m); overrides yaml height_range_m[0]")
    ap.add_argument("--tree-height-max", type=float, default=None,
                    help="max tree height (m); overrides yaml height_range_m[1]")
    ap.add_argument("--foliage-density", type=float, default=None,
                    help="trees / m^2 inside class==foliage (default yaml)")
    ap.add_argument("--open-ground-density", type=float, default=None,
                    help="trees / m^2 on class==ground+grass (default yaml)")
    ap.add_argument("--cluster-strength", type=float, default=None,
                    help="0=poisson, 1=fully clustered (default yaml)")
    ap.add_argument("--cluster-children", type=int, default=None)
    ap.add_argument("--cluster-radius", type=float, default=None,
                    help="gaussian sigma (m) around each cluster seed")
    # ----- Real-world cluster grouping (canopy_driven mode) ----------- #
    ap.add_argument("--cluster-size-min", type=int, default=10,
                    help="min trees per cluster in canopy_driven scatter")
    ap.add_argument("--cluster-size-max", type=int, default=20,
                    help="max trees per cluster in canopy_driven scatter")
    ap.add_argument("--cluster-disk-radius-min", type=float, default=4.0,
                    help="min cluster disk radius (m) in canopy_driven")
    ap.add_argument("--cluster-disk-radius-max", type=float, default=10.0,
                    help="max cluster disk radius (m) in canopy_driven")
    ap.add_argument("--cluster-disk-aspect", type=float, default=0.65,
                    help="ellipse minor/major axis ratio (1=circle)")
    ap.add_argument("--cluster-size-dist", default="uniform",
                    choices=["uniform", "bimodal", "beta_u"],
                    help="shape of cluster-size distribution within "
                         "[cluster-size-min, cluster-size-max]. "
                         "bimodal = mostly small clumps + a few big.")
    ap.add_argument("--cluster-size-low-frac", type=float, default=0.7,
                    help="for cluster-size-dist=bimodal: fraction of "
                         "clusters concentrated near the LOW end.")
    ap.add_argument("--tree-height-low-frac", type=float, default=0.65,
                    help="for tree-height-dist=bimodal: fraction of "
                         "trees concentrated near the LOW end.")
    ap.add_argument("--cluster-overlap-factor", type=float, default=0.45,
                    help="in-cluster spacing multiplier (canopy_prob). "
                         "<1 lets crowns of the same cluster merge "
                         "into a continuous blob (default 0.45). 1.0 "
                         "= legacy hard rejection.")
    ap.add_argument("--cluster-min-keep-ratio", type=float, default=0.6,
                    help="discard a whole cluster when fewer than "
                         "cluster_size_min*ratio trees fit (canopy_prob)."
                         " Prevents the unrealistic '2-3 lone tree' "
                         "fragments at land-class edges. Default 0.6.")
    ap.add_argument("--cluster-min-size-abs", type=int, default=0,
                    help="absolute minimum trees per cluster; clusters"
                         " placing fewer than this are rolled back"
                         " entirely (canopy_prob). 0 = disabled,"
                         " only ratio is used. Recommended 10-15 to"
                         " eliminate small isolated fragments.")
    ap.add_argument("--uniform-tree-scale", action="store_true",
                    help="force every tree to use the MAX height scale"
                         " (hmax * 1.5) instead of sampling per-tree;"
                         " makes every cluster a fat continuous canopy"
                         " blob (canopy_prob).")
    ap.add_argument("--use-pcg", action="store_true",
                    help="force PCG scatter (default: auto when any "
                         "tree_assets.* arg or tree_assets.dir is set)")
    # ----- Realism options ------------------------------------------- #
    ap.add_argument("--scatter-mode", default="canopy_prob",
                    choices=["cluster", "poisson_disk", "canopy_driven",
                             "canopy_prob", "canopy_prob_streets",
                             "linear_corridor", "noise_forest", "cp_nf_hybrid"],
                    help="placement strategy for the main foliage scatter. "
                         "canopy_prob = Ecological Continuous Canopy, "
                         "linear_corridor = Linear Green Corridor (road + water buffer), "
                         "noise_forest = Dense Forest Noise Clustering.")
    ap.add_argument("--tree-scale-xy-ratio", type=float, default=1.35,
                    help="Anisotropic scale multiplier on the horizontal (X/Y) axes "
                         "of tree instances in Blender to produce continuous crown coverage.")
    # ----- Phase 2 scatter realism toggles ---------------------------- #
    ap.add_argument("--allow-non-foliage", action="store_true",
                    help="(B2) allow trees to spawn outside the OSM "
                         "foliage class on grass/ground at reduced "
                         "weight (excludes building/road/water).")
    ap.add_argument("--enable-street-trees", action="store_true",
                    help="(B3) alias for --add-street-trees: place trees "
                         "along road buffers in addition to the main scatter.")
    ap.add_argument("--canopy-prob-scale", type=float, default=1.0,
                    help="(B1) multiplier on the per-cell canopy "
                         "probability in canopy_prob mode.")
    ap.add_argument("--procedural-augment-ratio", type=float, default=0.0,
                    help="(B1) extra fraction (0..1) of randomly placed "
                         "procedural trees on top of canopy-driven trees, "
                         "used to add stylization above real ETH data.")
    ap.add_argument("--canopy-npz", default=None,
                    help="path to the .npz produced by "
                         "dataprep/canopy_height.py for this tile")
    ap.add_argument("--canopy-as-heights", action="store_true",
                    help="override per-tree heights with the canopy grid "
                         "value when available (works in any mode)")
    ap.add_argument("--avoid-building-m", type=float, default=0.0,
                    help="reject scatter points within this many metres "
                         "of any building bounding box (0 = disabled)")
    ap.add_argument("--add-street-trees", action="store_true",
                    help="additionally place trees along road boundaries")
    ap.add_argument("--street-tree-spacing", type=float, default=10.0,
                    help="metres between adjacent street trees")
    ap.add_argument("--street-tree-offset", type=float, default=2.0,
                    help="metres offset perpendicular into the curb side")
    ap.add_argument("--street-tree-max", type=int, default=400,
                    help="cap on total street trees added in this pass")
    ap.add_argument("--show-foliage-substrate", action="store_true",
                    help="if set, the foliage scatter substrate (the green "
                         "polygon under the trees) is also rendered in the "
                         "top-down preview. Default: hidden, so the green "
                         "you see in the top-down PNG is exclusively from "
                         "placed tree / bush instances.")
    # ----- Iterative top-up to hit a target foliage ratio ----- #
    ap.add_argument("--topup-target-min", type=float, default=None,
                    help="if set, after PCG scatter keep adding trees on "
                         "ground+grass until the Blender top-down foliage "
                         "pixel ratio >= this value")
    ap.add_argument("--topup-target-max", type=float, default=0.55,
                    help="upper bound of the target ratio band; we stop "
                         "as soon as ratio enters [tgt_min, tgt_max]")
    ap.add_argument("--topup-max-iter", type=int, default=3)
    ap.add_argument("--topup-batch", type=int, default=300,
                    help="base trees added per iteration (auto-scaled by "
                         "the deficit)")
    ap.add_argument("--topup-grid", type=int, default=256,
                    help="raycast grid resolution for measuring foliage "
                         "ratio (NxN rays)")
    ap.add_argument("--preview-png", default=None,
                    help="if set, also render a top-down RGB preview PNG "
                         "to this path (post .blend save)")
    ap.add_argument("--preview-iso-png", default=None,
                    help="if set, render an axonometric 3D preview PNG "
                         "to this path (post .blend save)")
    ap.add_argument("--preview-glb", default=None,
                    help="if set, export a coloured tree-enriched GLB "
                         "for browser preview")
    ap.add_argument("--topdown-tree-xy-scale", type=float, default=3.5,
                    help="multiplier applied to tree-instance X/Y scale "
                         "during the top-down RGB pass only (Z unchanged). "
                         "Default 3.5 makes per-tile canopy read like a "
                         "remote-sensing tree-segmentation patch. Set to "
                         "1.0 to disable.")
    ap.add_argument("--gn-tree-amount", type=float, default=0.5,
                    help="0..1 GN density control; 0.5 = default density")
    ap.add_argument("--gn-safe-building", type=float, default=None,
                    help="GN safe distance from building geometry in metres")
    ap.add_argument("--gn-safe-road", type=float, default=None,
                    help="GN safe distance from road geometry in metres")
    ap.add_argument("--gn-safe-water", type=float, default=None,
                    help="GN safe distance from water geometry in metres")
    ap.add_argument("--gn-noise-scale", type=float, default=None,
                    help="GN forest patch noise scale")
    ap.add_argument("--gn-min-distance", type=float, default=None,
                    help="GN main-tree Poisson minimum distance in metres")
    ap.add_argument("--gn-xy-stretch", type=float, default=0.5,
                    help="0..1 GN instance XY scale amount; maps to a fixed min/max range")
    ap.add_argument("--gn-z-stretch", type=float, default=0.5,
                    help="0..1 GN instance Z scale amount; maps to a fixed min/max range")
    ap.add_argument("--gn-xy-stretch-min-at-0", type=float, default=None,
                    help="XY stretch minimum when --gn-xy-stretch=0")
    ap.add_argument("--gn-xy-stretch-min-at-1", type=float, default=None,
                    help="XY stretch minimum when --gn-xy-stretch=1")
    ap.add_argument("--gn-xy-stretch-max-at-0", type=float, default=None,
                    help="XY stretch maximum when --gn-xy-stretch=0")
    ap.add_argument("--gn-xy-stretch-max-at-1", type=float, default=None,
                    help="XY stretch maximum when --gn-xy-stretch=1")
    ap.add_argument("--gn-z-stretch-min-at-0", type=float, default=None,
                    help="Z stretch minimum when --gn-z-stretch=0")
    ap.add_argument("--gn-z-stretch-min-at-1", type=float, default=None,
                    help="Z stretch minimum when --gn-z-stretch=1")
    ap.add_argument("--gn-z-stretch-max-at-0", type=float, default=None,
                    help="Z stretch maximum when --gn-z-stretch=0")
    ap.add_argument("--gn-z-stretch-max-at-1", type=float, default=None,
                    help="Z stretch maximum when --gn-z-stretch=1")
    ap.add_argument("--depth-exr", default=None,
                    help="if set, render top-down nDSM (height above "
                         "ground, metres) as 32-bit OpenEXR to this path")
    ap.add_argument("--depth-png", default=None,
                    help="if set, render top-down nDSM as 16-bit "
                         "normalized PNG to this path (visual companion "
                         "of --depth-exr)")
    ap.add_argument("--pointcloud-ply", default=None,
                    help="if set, export native surface point cloud PLY to this path")
    ap.add_argument("--pointcloud-png", default=None,
                    help="if set, render an isometric semantic point cloud PNG to this path")
    ap.add_argument("--pointcloud-count", type=int, default=50000,
                    help="target point count for --pointcloud-ply/--pointcloud-png")
    ap.add_argument("--pointcloud-voxel-size", type=float, default=0.5,
                    help="voxel spacing in metres for Poisson-like pointcloud thinning")
    args = ap.parse_args(_argv())

    root = Path(args.config).resolve().parent.parent
    cfg = _load_config(args.config)

    # Populate CLASS_IDS from the canonical map.
    global CLASS_IDS
    CLASS_IDS = _load_class_ids(root)
    print(f"[KR3] CLASS_IDS = {CLASS_IDS}")

    # Per-tile output folder layout:
    #   <tile_root>/<city>/blender/<city>.glb     (input from KR2)
    #   <tile_root>/<city>/blender/<city>.blend   (our output)
    tile_root = (cfg["paths"].get("tile_root")
                 or cfg["paths"].get("blender_dir")
                 or cfg["paths"].get("meshes_dir") or "output")
    tile_dir = root / tile_root / args.city
    tile_dir.mkdir(parents=True, exist_ok=True)
    blender_subdir = tile_dir / "blender"
    blender_subdir.mkdir(parents=True, exist_ok=True)
    glb = blender_subdir / f"{args.city}.glb"
    if not glb.exists():
        # Fall back to legacy single-folder layouts for old GLBs.
        legacy_a = tile_dir / f"{args.city}.glb"
        legacy_b = root / "output" / "blender" / args.city / f"{args.city}.glb"
        legacy_c = (root
                    / cfg["paths"].get("meshes_dir", "output/meshes")
                    / f"{args.city}.glb")
        # Also try the previous layout where GLB lived under obj/.
        legacy_d = tile_dir / "obj" / f"{args.city}.glb"
        for cand in (legacy_a, legacy_b, legacy_c, legacy_d):
            if cand.exists():
                glb = cand
                break
        else:
            raise SystemExit(
                f"GLB not found at {glb} (or legacy locations). "
                "Run scripts/2_build_geometry.py first.")

    out_blend = Path(args.out) if args.out else (
        blender_subdir / f"{args.city}.blend")

    print(f"[KR3] assembling {glb} -> {out_blend}")
    _clear_scene()
    _import_glb(glb)
    _fix_normals_outward()
    _assign_pass_indices()
    _setup_ortho_camera(cfg)
    _setup_render(cfg)

    # Bake enrichment into the .blend so Cycles depsgraph picks it up.
    rng = random.Random(args.seed)
    gn_ground_obj = None  # set by _scatter_geometry_nodes for downstream
    if not args.no_foliage:
        scatter_seed = (args.scatter_seed if args.scatter_seed is not None
                        else args.seed)
        ta = cfg.get("tree_assets") or {}
        species_arg = (args.tree_species or "").strip()
        species_list = (
            [s.strip() for s in species_arg.split(",") if s.strip()]
            if species_arg else (ta.get("species") or [])
        )
        asset_dir = ta.get("dir", "assets/trees")
        asset_dir_abs = (root / asset_dir).resolve() if asset_dir else None
        pcg_templates = []
        # Resolved tree-asset params (still consumed by the GN node group
        # as modifier inputs; per-tree height distribution is no longer
        # used because GN handles height via random scale on Z).
        pcg_h_range = list(ta.get("height_range_m", [3.0, 14.0]))
        if args.tree_height_min is not None:
            pcg_h_range[0] = float(args.tree_height_min)
        if args.tree_height_max is not None:
            pcg_h_range[1] = float(args.tree_height_max)

        # ---------------------------------------------------------------- #
        # NEW: Geometry-Nodes-driven scatter (replaces all Python-loop     #
        # scatter modes — canopy_prob, linear_corridor, noise_forest, etc).#
        # All legacy CLI flags are kept for backwards-compat with          #
        # auto_pipeline.py / osm_app.py but only the relevant ones are     #
        # honoured: --tree-species, --tree-height-min/max, --seed,         #
        # --avoid-building-m, --topdown-tree-xy-scale.                     #
        # ---------------------------------------------------------------- #
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import _tree_scatter_gn as gn  # type: ignore

        if mode_legacy := str(args.scatter_mode or ""):
            print(f"[KR3] scatter_mode='{mode_legacy}' is now a no-op; "
                  f"using Geometry Nodes scatter for all modes.")

        pcg_templates = _load_tree_species_dir(
            str(asset_dir_abs) if asset_dir_abs else None,
            species_list,
        )
        if not pcg_templates:
            print("[KR3] no tree templates found; foliage scatter skipped.")
        else:
            # Map the legacy --avoid-building-m onto the GN
            # Safe Distance Building input when it exceeds the module
            # default; otherwise fall back to the module constant.
            safe_b = max(float(args.avoid_building_m or 0.0),
                         gn.SAFE_DISTANCE_BUILDING)
            if args.gn_safe_building is not None:
                safe_b = float(args.gn_safe_building)
            gn_ground_obj, _ = gn.scatter_geometry_nodes(
                cfg,
                tree_templates=pcg_templates,
                safe_b=safe_b,
                safe_r=args.gn_safe_road,
                safe_w=args.gn_safe_water,
                noise_scale=args.gn_noise_scale,
                min_dist=args.gn_min_distance,
                tree_amount=float(args.gn_tree_amount),
                xy_stretch=float(args.gn_xy_stretch),
                z_stretch=float(args.gn_z_stretch),
                xy_stretch_min_at_0=args.gn_xy_stretch_min_at_0,
                xy_stretch_min_at_1=args.gn_xy_stretch_min_at_1,
                xy_stretch_max_at_0=args.gn_xy_stretch_max_at_0,
                xy_stretch_max_at_1=args.gn_xy_stretch_max_at_1,
                z_stretch_min_at_0=args.gn_z_stretch_min_at_0,
                z_stretch_min_at_1=args.gn_z_stretch_min_at_1,
                z_stretch_max_at_0=args.gn_z_stretch_max_at_0,
                z_stretch_max_at_1=args.gn_z_stretch_max_at_1,
                seed=int(scatter_seed),
            )

    if not args.no_clutter:
        _add_roof_clutter(rng)

    # (tree/building conflict resolution is no longer needed — the GN
    # graph enforces Safe Distance via Geometry Proximity.)

    out_blend.parent.mkdir(parents=True, exist_ok=True)

    # Render previews FIRST while scene is still in tile-corner coords
    # (the preview camera assumes SW corner at origin); only then shift
    # the scene to be centred about origin and save the .blend so the
    # file users open in Blender's GUI is centred and friendly.
    if args.preview_png or args.preview_iso_png:
        try:
            _render_preview_png(cfg,
                                args.preview_png or (str(out_blend) + ".top.png"),
                                CLASS_IDS,
                                iso_png=args.preview_iso_png,
                                hide_foliage_substrate=(
                                    not args.show_foliage_substrate),
                                topdown_tree_xy_scale=float(
                                    args.topdown_tree_xy_scale),
                                depth_exr=args.depth_exr,
                                depth_png=args.depth_png)
        except Exception as e:
            print(f"[KR3] preview PNG failed: {e}")

    if args.pointcloud_ply or args.pointcloud_png:
        try:
            _export_pointcloud_outputs(
                cfg,
                ply_path=args.pointcloud_ply,
                png_path=args.pointcloud_png,
                target_count=int(args.pointcloud_count),
                voxel_size=float(args.pointcloud_voxel_size),
                seed=int(args.seed) + 1701,
            )
        except Exception as e:
            print(f"[KR3] pointcloud export failed: {e}")

    # Dump tree instance positions while scene is still in tile-corner
    # coords — this lets the pipeline composite trees in mercator with
    # zero pixel-level resampling error. With the GN scatter, instance
    # transforms live in the depsgraph; the new dumper walks them.
    try:
        if gn_ground_obj is not None:
            scripts_dir = str(Path(__file__).resolve().parent)
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            import _tree_scatter_gn as gn  # type: ignore
            gn.dump_gn_tree_instances_json(
                cfg, gn_ground_obj,
                blender_subdir / "tree_instances.json")
        else:
            _dump_tree_instances_json(
                cfg, blender_subdir / "tree_instances.json")
    except Exception as e:
        print(f"[KR3] tree dump failed: {e}")

    _prepare_blend_for_user(cfg)
    if args.preview_glb:
        try:
            _export_scene_glb_for_web(args.preview_glb)
        except Exception as e:
            print(f"[KR3] WebUI preview GLB export failed: {e}")
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"[KR3] saved {out_blend}")

if __name__ == "__main__":
    main()

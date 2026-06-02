# Tree species library

Drop one **`.blend`** file per tree/bush species into this folder. The
file's name (without `.blend`) becomes the species name shown in the
web UI checkbox group.

```
assets/trees/
  oak.blend
  pine.blend
  birch.blend
  bush_low.blend
  bush_tall.blend
```

## How a `.blend` is interpreted

* The file is opened with `bpy.data.libraries.load(..., link=False)` and
  every MESH object it contains is appended.
* The mesh with the **largest world-space Z extent** is picked as the
  species master template. Its base height (Δz) is used to rescale every
  instance to the target tree height drawn from the height distribution.
* All other meshes from that file are kept inside the hidden
  `_tree_templates` collection and are ignored by the renderer.

## Recommended workflow (extracting from Nature_Pack.blend)

1. Open `Nature_Pack.blend` in Blender.
2. Select one tree object, e.g. `Tree 03` from the `Plants` collection.
3. `File > Export > ...` is fine, but the simplest path is:
   - Press `Ctrl+C` (Copy) on that object,
   - File > New > General,
   - `Ctrl+V` (Paste) into the empty scene,
   - File > Save As > `assets/trees/<species_name>.blend`.
4. Repeat for each species you want available.

Tip: name bushes with a `bush_` prefix so the scatter algorithm can
auto-detect them. Otherwise everything is treated as a tree.

## Where to find the path

Inside this repository this folder is at:

```
SynthUrbanSAT/osm_pipeline/assets/trees/
```

It is also exposed in `configs/default.yaml` as `tree_assets.dir`.

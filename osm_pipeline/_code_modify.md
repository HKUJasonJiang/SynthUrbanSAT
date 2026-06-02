# Code Modification Log

## 2026-05-28

- Added automatic Gradio port fallback in `osm_app.py`.
- Default launch still tries `127.0.0.1:8765` first, then falls back to `8766`, `8767`, `8768`, `7860`, `7861`, and `7862`.
- `GRADIO_SERVER_PORT` is still respected as the first candidate when set.
- Verified `py_compile`, `build_ui()`, fallback selection with occupied `8765`, and actual launch on `http://127.0.0.1:8766`.
- Added best-effort cleanup for stale Python/Gradio processes occupying the default `127.0.0.1:8765` port before falling back to alternate ports.
- Cleanup can be disabled with `OSM_APP_KEEP_PORT_PROCESS=1`.
- Verified the cleanup released a stale `osm_app.py` PID and selected `8765` again.
- Added a separate KR3-derived WebUI preview GLB export: `blender/<tile>_scene.glb`.
- The original `blender/<tile>.glb` remains the clean KR2 input for tree reruns; the new `_scene.glb` contains semantic colors and baked GN tree instances for `gr.Model3D`.
- Verified direct Blender export baked 313 GN tree instances and produced a ~9.8 MB preview GLB for `ui_debug_city/tile_dev`.
- Aligned KR3 preview/Model3D colors with final `4_seg.png`: tree instances stay foliage green, while non-tree foliage/canopy substrate is rendered as grass purple.
- Verified `rerun_trees_only` regenerated `ui_debug_city/tile_dev/blender/tile_dev_scene.glb` successfully (~9.9 MB).
- Added batch runtime recording. `auto_pipeline.py` now writes `output/<city>/metadata/run_timing_latest.json` plus timestamped timing history with wall-clock duration, per-stage totals, and per-tile timings.
- Fixed WebUI Tab 2 full-batch execution to call the current `AutoPipeline.run()` signature and added UI-level timing files: `ui_run_timing_latest.json` and `ui_tree_rerun_timing_latest.json`.
- Avoided a second online `osmnx` building fetch in Stage F metadata writing by copying KR1's cached `<tile>_buildings.geojson` into `metadata/<tile>_osm_buildings.geojson`; `_write_tile_metadata()` now reuses that file when present.

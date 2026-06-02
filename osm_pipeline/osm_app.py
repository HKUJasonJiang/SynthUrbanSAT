"""Interactive Gradio app for the osm_pipeline automation.

Workflow (each step is one button + one row of preview images):
  1. 抓取卫星图与 OSM 底图
  2. 提取 OSM 6 类语义
  3. ETH canopy 树冠宽高 → 植被掩码
  4. Blender 装配 3D 场景并保存 .blend / .glb / PLY

Per-tile output layout (one folder per tile):
    output/<city>/<tile>/
        1_osm.png  2_rgb.png  3_blender_preview.png
        4_seg.png  5_depth.png/.exr
        6_polygon_outline.png  6_polygons.json
        7_pointcloud.png/.ply
        near-nadir-1..4/        # off-nadir RGB + depth preset folders
        blender/<tile>.blend    # KR3 Blender scene
        metadata/<tile>.json    # bbox / GSD / file index
        metadata/<tile>.meta.json               # KR2 sidecar (UTM, classes)
        metadata/<tile>_osm_buildings.geojson   # raw OSM IDs

Run:
    python osm_app.py
Then open http://127.0.0.1:8765
"""
from __future__ import annotations

import builtins
import html
import sys

_orig_print = builtins.print

def colored_print(*args, **kwargs):
    if args and isinstance(args[0], str):
        msg = args[0]
        if sys.platform == "win32":
            try:
                import os as _os
                _os.system("")
            except Exception:
                pass
        if msg.startswith("[auto]"):
            msg = "\033[36m[auto]\033[0m" + msg[6:]
        elif msg.startswith("[C0]"):
            msg = "\033[33m[C0]\033[0m" + msg[4:]
        elif msg.startswith("[app]"):
            msg = "\033[35m[app]\033[0m" + msg[5:]
        elif msg.startswith("[ui]"):
            msg = "\033[34m[ui]\033[0m" + msg[4:]
        
        if "successfully" in msg or "Success" in msg or "Successfully" in msg or "SUCCESS" in msg:
            msg = msg.replace("Successfully", "\033[32mSuccessfully\033[0m") \
                     .replace("successfully", "\033[32msuccessfully\033[0m") \
                     .replace("SUCCESS", "\033[32mSUCCESS\033[0m") \
                     .replace("success", "\033[32msuccess\033[0m") \
                     .replace("ok", "\033[32mok\033[0m")
        if "failed" in msg or "fail" in msg.lower() or "error" in msg.lower() or "crash" in msg.lower():
            msg = msg.replace("Failed", "\033[31mFailed\033[0m") \
                     .replace("failed", "\033[31mfailed\033[0m") \
                     .replace("FAILURE", "\033[31mFAILURE\033[0m") \
                     .replace("failure", "\033[31mfailure\033[0m") \
                     .replace("Error", "\033[31mError\033[0m") \
                     .replace("error", "\033[31merror\033[0m")
        args = (msg,) + args[1:]
    _orig_print(*args, **kwargs)

builtins.print = colored_print

import csv
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Sequence

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataprep.geocode import geocode
from dataprep.geometry_utils import (
    ensure_dir, make_transformer, reproject_geom, utm_crs_for_bbox,
)
from dataprep.osm_tags import CLASS_IDS, CLASS_PRIORITY, ROAD_HIGHWAY_KEEP
from dataprep.raster_utils import (
    class_geoms_to_local, colorize_seg, compose_seg_with_foliage_mask,
    depth_to_uint8, rasterize_depth_proxy, rasterize_seg,
)
from dataprep.satellite import satellite_image_for_bbox, osm_map_image_for_bbox

# ---- import KR1 fetchers (reuse them) ------------------------------------ #
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("kr1", ROOT / "scripts" / "1_fetch_osm.py")
kr1 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(kr1)  # type: ignore


CFG = yaml.safe_load(open(ROOT / "configs/default.yaml", "r", encoding="utf-8"))
PALETTE = CFG["class_colors"]
PRESETS = {c["name"]: c["bbox"] for c in CFG["cities"]}
DEFAULT_PRESET = next(iter(PRESETS), None)
PICKS_CSV = ROOT / "output" / "tile_picks.csv"


# ============================== core helpers ============================= #

def bbox_from_center(lat: float, lon: float, gsd: float, size: int):
    """Return (W,S,E,N) WGS84 bbox covering a (gsd*size) m square at (lat,lon)."""
    bbox_for_utm = (lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01)
    utm = utm_crs_for_bbox(bbox_for_utm)
    tr = make_transformer("EPSG:4326", utm)
    p = reproject_geom(Point(lon, lat), tr)
    cx, cy = p.x, p.y
    half = gsd * size / 2 * 1.05
    inv = make_transformer(utm, "EPSG:4326")
    sw = reproject_geom(Point(cx - half, cy - half), inv)
    ne = reproject_geom(Point(cx + half, cy + half), inv)
    return (sw.x, sw.y, ne.x, ne.y), utm, (cx, cy)


# overview map shown in the picker (wider than the tile so you have context)
MAP_VIEW_PX = 720
MAP_VIEW_M = 4000.0  # half-width in metres -> view is ~8 km across


def _overview_bbox(lat: float, lon: float, half_m: float = MAP_VIEW_M):
    """WGS84 (W,S,E,N) bbox of the overview map for clicking."""
    bbox_for_utm = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
    utm = utm_crs_for_bbox(bbox_for_utm)
    fwd = make_transformer("EPSG:4326", utm)
    inv = make_transformer(utm, "EPSG:4326")
    p = reproject_geom(Point(lon, lat), fwd)
    sw = reproject_geom(Point(p.x - half_m, p.y - half_m), inv)
    ne = reproject_geom(Point(p.x + half_m, p.y + half_m), inv)
    return (sw.x, sw.y, ne.x, ne.y)


def build_satellite_map(lat: float, lon: float, gsd: float, size: int):
    """Esri overview as a clickable PIL image; draws the tile bbox + center.

    Returns (PIL.Image, overview_bbox tuple) so click handlers can map
    pixel -> lat/lon using the bbox.
    """
    ov_bbox = _overview_bbox(lat, lon)
    try:
        img = satellite_image_for_bbox(ov_bbox, out_size=MAP_VIEW_PX).convert(
            "RGB")
    except Exception as exc:  # noqa: BLE001
        print(f"[app] overview tile fetch failed: {exc}", flush=True)
        img = Image.new("RGB", (MAP_VIEW_PX, MAP_VIEW_PX), (40, 40, 40))

    # draw tile bbox (in metres) on the overview
    half_m = gsd * size / 2
    bbox_for_utm = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
    utm = utm_crs_for_bbox(bbox_for_utm)
    fwd = make_transformer("EPSG:4326", utm)
    inv = make_transformer(utm, "EPSG:4326")
    c = reproject_geom(Point(lon, lat), fwd)
    sw = reproject_geom(Point(c.x - half_m, c.y - half_m), inv)
    ne = reproject_geom(Point(c.x + half_m, c.y + half_m), inv)

    w, s, e, n_ = ov_bbox
    W = MAP_VIEW_PX
    def to_px(lon_, lat_):
        x = (lon_ - w) / (e - w) * W
        y = (n_ - lat_) / (n_ - s) * W  # y flipped
        return x, y
    draw = ImageDraw.Draw(img)
    x1, y1 = to_px(sw.x, sw.y)
    x2, y2 = to_px(ne.x, ne.y)
    x0, y0 = to_px(lon, lat)
    draw.rectangle([x1, y2, x2, y1], outline=(255, 60, 60), width=3)
    draw.line([(x0 - 8, y0), (x0 + 8, y0)], fill=(255, 60, 60), width=2)
    draw.line([(x0, y0 - 8), (x0, y0 + 8)], fill=(255, 60, 60), width=2)
    return img, ov_bbox


# =============================== handlers =============================== #

def on_preset_change(preset_name: str, gsd, size):
    bb = PRESETS.get(preset_name)
    if bb is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    cx = (bb[0] + bb[2]) / 2; cy = (bb[1] + bb[3]) / 2
    img, ov = build_satellite_map(cy, cx, float(gsd), int(size))
    return gr.update(value=cy), gr.update(value=cx), img, list(ov)


def on_search(query: str, gsd, size):
    if not query or not query.strip():
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                "enter something to search.")
    res = geocode(query.strip(), limit=5)
    if not res:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                f"no result for '{query}'.")
    top = res[0]
    img, ov = build_satellite_map(top["lat"], top["lon"],
                                  float(gsd), int(size))
    msg = [f"found {len(res)} result(s); using top:",
           f"  {top['display_name']}",
           f"  lat={top['lat']:.5f}, lon={top['lon']:.5f}"]
    if len(res) > 1:
        msg.append("other matches:")
        for r in res[1:]:
            msg.append(f"  - {r['display_name']}")
    return (gr.update(value=top["lat"]), gr.update(value=top["lon"]),
            img, list(ov), "\n".join(msg))


def on_map_click(ov_bbox, gsd, size, evt: gr.SelectData):
    """Image click -> pixel coord -> lat/lon via overview bbox."""
    if evt is None or evt.index is None or ov_bbox is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    px, py = evt.index  # (x, y) in image pixels
    w, s, e, n_ = ov_bbox
    W = MAP_VIEW_PX
    lon = w + (px / W) * (e - w)
    lat = n_ - (py / W) * (n_ - s)
    img, ov = build_satellite_map(lat, lon, float(gsd), int(size))
    return gr.update(value=lat), gr.update(value=lon), img, list(ov)


def on_redraw_bbox(lat, lon, gsd, size):
    """User edited a slider -> redraw bbox without refetching tiles.
    (Cheap path: still calls Esri but cached locally for repeated centers.)
    """
    try:
        img, ov = build_satellite_map(float(lat), float(lon),
                                       float(gsd), int(size))
        return img, list(ov)
    except Exception:  # noqa: BLE001
        return gr.update(), gr.update()


def on_preview_satellite(lat, lon, gsd, size):
    import time
    t0 = time.time()
    lat = float(lat); lon = float(lon); gsd = float(gsd); size = int(size)
    bbox_wgs, utm, (cx_utm, cy_utm) = bbox_from_center(lat, lon, gsd, size)
    print(f"[app] sat-preview bbox={bbox_wgs}", flush=True)
    try:
        sat = satellite_image_for_bbox(bbox_wgs, out_size=size)
    except Exception as exc:  # noqa: BLE001
        print(f"[app] satellite fetch failed: {exc}", flush=True)
        sat = Image.new("RGB", (size, size), (40, 40, 40))
    # Also fetch the cartographic OSM map for the same bbox (cheap; same
    # XYZ-tile pipeline). Failures are non-fatal — show a placeholder.
    try:
        osm_map = osm_map_image_for_bbox(bbox_wgs, out_size=size)
    except Exception as exc:  # noqa: BLE001
        print(f"[app] osm map fetch failed: {exc}", flush=True)
        osm_map = Image.new("RGB", (size, size), (220, 220, 215))
    print(f"[app] sat+osm done @ {time.time()-t0:.1f}s", flush=True)

    state = {
        "lat": lat, "lon": lon, "gsd": gsd, "size": size,
        "bbox_wgs": list(bbox_wgs), "utm": utm.to_string(),
        "cx_utm": cx_utm, "cy_utm": cy_utm, "ratios": None,
        "sat_rgb": sat,  # PIL.Image cached for canopy/topview composition
        "osm_basemap": osm_map,  # PIL.Image cached for tile metadata
        "seg_osm": None,  # filled by Fetch OSM
    }
    info = (f"bbox_wgs = {tuple(round(x,5) for x in bbox_wgs)}\n"
            f"UTM CRS  = {utm.to_string()}\n"
            f"center_utm = ({cx_utm:.1f}, {cy_utm:.1f})\n"
            f"tile_m   = {gsd*size:.1f} x {gsd*size:.1f}\n"
            "satellite preview only - OSM not yet fetched.")
    return sat, osm_map, info, state


def fetch_classes_wgs(bbox):
    """Single-Overpass-POST OSM fetch (all 4 classes + roads in one call).

    Empirical benchmark on a 553x517 m Omaha tile (kumi mirror):
      - osmnx.features_from_bbox(building=True)  : 66.7 s
      - osmnx.graph_from_bbox(network=drive)     : ~60 s + drops service/footway
      - one combined raw Overpass POST (this fn) : 12-15 s
    See scripts/bench_osm_fetch.py + output/_bench_osm.json.

    The osmnx path is kept as a fallback if Overpass is fully down.
    """
    import time
    from shapely.ops import unary_union
    from dataprep.osm_overpass import fetch_all_classes_combined

    t0 = time.time()
    out: dict = {}
    fetch_source = "overpass"
    road_keep = set(CFG["osm"].get("road_keep", list(ROAD_HIGHWAY_KEEP)))

    def _has_non_ground_feature(classes: dict) -> bool:
        for cls in ("building", "water", "grass", "foliage", "road"):
            geom = classes.get(cls)
            if geom is not None and not getattr(geom, "is_empty", True):
                return True
        return False

    # ---------- fastest/stable path: local prepared Geofabrik PBF -------- #
    try:
        from dataprep.osm_local import (
            fetch_polygon_class_local,
            fetch_roads_local,
            pbf_for_bbox,
        )
        region_dir = pbf_for_bbox(bbox, ROOT / "cache" / "pbf")
        if region_dir is not None:
            fetch_source = "local-pbf"
            print(f"[app] using local PBF cache: {region_dir}", flush=True)
            for cls in ("building", "water", "grass"):
                out[cls] = fetch_polygon_class_local(region_dir, bbox, cls)
            out["foliage"] = None
            out["_buildings_with_id"] = None
            edges = fetch_roads_local(region_dir, bbox, road_keep)
            if edges is not None and len(edges) > 0:
                out["road"] = kr1._buffer_road_edges(
                    edges, bbox, CFG["osm"]["road_buffer_m"], road_keep)
            else:
                out["road"] = None
            print(f"[app]   local PBF OSM read in {time.time()-t0:.2f}s",
                  flush=True)
            if not _has_non_ground_feature(out):
                print("[app]   local PBF returned no usable OSM features; "
                      "falling back to Overpass", flush=True)
                out = {}
        else:
            print("[app] local PBF cache miss; falling back to Overpass",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[app] local PBF path unavailable ({type(e).__name__}: {e}); "
              "falling back to Overpass", flush=True)

    # ---------- fast path: single combined Overpass POST ----------- #
    try:
        if out:
            raise StopIteration("local PBF cache already supplied OSM classes")
        print(f"[app] fetching all 4 classes + roads via combined "
              f"Overpass POST ...", flush=True)
        bundle = fetch_all_classes_combined(bbox, road_keep=road_keep)
        meta = bundle.get("_meta", {})
        print(f"[app]   overpass: {meta.get('n_elements', 0)} elements "
              f"in {meta.get('elapsed_s', 0):.1f}s "
              f"(buildings={meta.get('n_buildings', 0)}, "
              f"roads={meta.get('n_roads', 0)})", flush=True)
        # If every mirror soft-failed and returned 0 elements, fall back
        # to osmnx instead of silently producing an all-ground seg.
        if meta.get("n_elements", 0) == 0:
            raise RuntimeError(
                "all overpass mirrors returned 0 elements (likely "
                "soft-timeout); falling back to osmnx")

        for cls in ("building", "water", "grass", "foliage"):
            out[cls] = bundle.get(cls)

        # Per-building OSM IDs (carried separately from the unioned
        # ``building`` MultiPolygon so the UE exporter can name each
        # building with its real OSM way id).
        out["_buildings_with_id"] = bundle.get("_buildings_with_id")

        # Buffer road LineStrings -> WGS84 polygon, mirrors KR1 path.
        edges = bundle.get("_road_edges")
        if edges is not None and len(edges) > 0:
            out["road"] = kr1._buffer_road_edges(
                edges, bbox, CFG["osm"]["road_buffer_m"], road_keep)
        else:
            out["road"] = None

    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        # ---------- slow fallback: per-class osmnx parallel ------- #
        fetch_source = "osmnx-fallback"
        print(f"[app] combined Overpass FAILED ({type(e).__name__}: "
              f"{e}); falling back to osmnx per-class parallel.",
              flush=True)
        from concurrent.futures import ThreadPoolExecutor
        from dataprep.osm_tags import TAG_QUERIES
        try:
            import osmnx as ox
            kr1._configure_osmnx(cache_dir=ROOT / "cache")
            ox.settings.overpass_url = kr1.OVERPASS_ENDPOINTS[0]
        except Exception:  # noqa: BLE001
            pass

        def _fetch_poly(cls, tags):
            try:
                return cls, kr1.fetch_polygon_class(bbox, tags,
                                                    class_name=cls)
            except Exception as ex:  # noqa: BLE001
                print(f"[app]   {cls} fetch FAILED: {ex}", flush=True)
                return cls, None

        def _fetch_roads():
            try:
                return "road", kr1.fetch_roads(
                    bbox, CFG["osm"]["road_buffer_m"], road_keep=road_keep)
            except Exception as ex:  # noqa: BLE001
                print(f"[app]   road fetch FAILED: {ex}", flush=True)
                return "road", None

        with ThreadPoolExecutor(max_workers=4) as ex:
            tasks = [(c, t) for c, t in TAG_QUERIES.items()]
            futs = [ex.submit(_fetch_poly, c, t) for c, t in tasks]
            futs.append(ex.submit(_fetch_roads))
            for f in futs:
                cls, g = f.result()
                out[cls] = g

    # Compute the 'ground' class as bbox - union(others).
    # Skip "ground" itself and any "_"-prefixed sidecar entries
    # (e.g. _buildings_with_id GeoDataFrame, _meta dict, _road_edges).
    bbox_poly = box(*bbox)
    others = [g for k, g in out.items()
              if k != "ground" and not k.startswith("_")
              and g is not None and not g.is_empty]
    union_others = unary_union(others) if others else None
    ground = bbox_poly.difference(union_others) if union_others else bbox_poly
    out["ground"] = ground if not ground.is_empty else None
    if not _has_non_ground_feature(out):
        raise RuntimeError(
            "OSM vector fetch returned only ground/no usable features. "
            "Overpass/osmnx likely failed or returned an empty cached "
            "response; refusing to generate a tree-only semantic tile.")
    print(f"[app] OSM total {time.time()-t0:.1f}s "
            f"({fetch_source})", flush=True)
    return out


def _overlay_inline(class_geoms_wgs, bbox) -> Image.Image:
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    min_lon, min_lat, max_lon, max_lat = bbox

    def _to_path(poly):
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            verts.extend(xy.tolist())
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(xy) - 2))
            codes.append(MplPath.CLOSEPOLY)
        return MplPath(verts, codes)

    order = sorted(
        (k for k in class_geoms_wgs.keys()
         if not (isinstance(k, str) and k.startswith("_"))),
        key=lambda k: CLASS_PRIORITY[k])
    for name in order:
        g = class_geoms_wgs.get(name)
        if g is None or g.is_empty:
            continue
        rgb = tuple(c / 255 for c in PALETTE[name])
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for p in polys:
            if p.geom_type != "Polygon" or p.is_empty:
                continue
            ax.add_patch(PathPatch(_to_path(p), facecolor=rgb,
                                   edgecolor="none", alpha=0.9))
    ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal"); ax.set_title("6-class overlay (WGS84)")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    buf.seek(0); return Image.open(buf).copy()


def ratio_bar_png(seg: np.ndarray) -> Image.Image:
    classes = ["ground", "foliage", "grass", "water", "building", "road"]
    total = seg.size
    ours = [(seg == CLASS_IDS[c]).sum() / total for c in classes]
    ref = [CFG.get("ref_class_ratios", {}).get(c, 0.0) for c in classes]
    x = np.arange(len(classes)); w = 0.4
    fig, ax = plt.subplots(figsize=(7, 3.0), dpi=120)
    ax.bar(x - w / 2, ours, w, label="this tile", color="#3b7dd8")
    ax.bar(x + w / 2, ref, w, label="ref (placeholder)", color="#d8893b")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylim(0, max(max(ours), max(ref)) * 1.25 + 0.05)
    ax.set_ylabel("pixel ratio"); ax.legend()
    for i, r in enumerate(ours):
        ax.text(i - w / 2, r + 0.005, f"{r:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    buf.seek(0); return Image.open(buf).copy()


def on_fetch_osm(lat, lon, gsd, size, prev_state):
    import time
    t0 = time.time()
    lat = float(lat); lon = float(lon); gsd = float(gsd); size = int(size)
    bbox_wgs, utm, (cx_utm, cy_utm) = bbox_from_center(lat, lon, gsd, size)
    print(f"[app] fetch-osm bbox={bbox_wgs}", flush=True)

    cls_wgs = fetch_classes_wgs(bbox_wgs)
    cls_utm, _ = class_geoms_to_local(cls_wgs, bbox_wgs)
    seg = rasterize_seg(cls_utm, cx_utm, cy_utm, gsd, size)
    seg_rgb = colorize_seg(seg, PALETTE)
    depth = rasterize_depth_proxy(
        cls_utm, cx_utm, cy_utm, gsd, size,
        building_h=CFG["osm"]["default_building_height_m"],
        camera_h=CFG["render"]["camera_height_m"],
    )
    depth_u8 = depth_to_uint8(depth)
    overlay = _overlay_inline(cls_wgs, bbox_wgs)
    bars = ratio_bar_png(seg)
    print(f"[app] fetch-osm done @ {time.time()-t0:.1f}s", flush=True)

    counts = {c: int((seg == CLASS_IDS[c]).sum()) for c in CLASS_IDS}
    total = seg.size
    lines = [
        f"bbox_wgs = {tuple(round(x,5) for x in bbox_wgs)}",
        f"UTM CRS  = {utm.to_string()}",
        f"center_utm = ({cx_utm:.1f}, {cy_utm:.1f})",
        f"tile_m   = {gsd*size:.1f} x {gsd*size:.1f}",
        "",
        "class pixel ratios (this tile):",
    ]
    for c in ["ground", "foliage", "grass", "water", "building", "road"]:
        lines.append(f"  {c:>9}: {counts[c]/total:6.4f}  ({counts[c]} px)")

    state = {
        "lat": lat, "lon": lon, "gsd": gsd, "size": size,
        "bbox_wgs": list(bbox_wgs), "utm": utm.to_string(),
        "cx_utm": cx_utm, "cy_utm": cy_utm,
        "ratios": {c: counts[c] / total for c in CLASS_IDS},
        "sat_rgb": (prev_state or {}).get("sat_rgb"),
        # Carry the OSM basemap forward so step 5 can persist it.
        "osm_basemap": (prev_state or {}).get("osm_basemap"),
        "seg_osm": seg,
        # Cache fetched OSM polygons so step 5 doesn't re-fetch.
        "cls_wgs": cls_wgs,
    }
    return (overlay, Image.fromarray(seg_rgb), Image.fromarray(depth_u8),
            bars, "\n".join(lines), state)


def on_save(state, city_name, blend_path: str | None = None):
    if not state:
        return "Run Preview or Fetch first."
    PICKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PICKS_CSV.exists()
    row = {
        "city": city_name or "custom",
        "lat": state["lat"], "lon": state["lon"],
        "gsd": state["gsd"], "size": state["size"],
        "cx_utm": state["cx_utm"], "cy_utm": state["cy_utm"],
        "utm": state["utm"],
        "bbox_wgs": json.dumps(state["bbox_wgs"]),
        "ratios": json.dumps(state.get("ratios")),
        "blend_path": blend_path or "",
    }
    with open(PICKS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    return (f"saved -> {PICKS_CSV.relative_to(ROOT)}  "
            f"(now {sum(1 for _ in open(PICKS_CSV))-1} tiles)")


# Path to the Blender executable used by KR3 build. Override with env var
# ``BLENDER_EXE`` if your install lives elsewhere.
import os as _os
BLENDER_EXE = _os.environ.get(
    "BLENDER_EXE", r"C:\Softwares\blender\blender.exe"
)


def _preview_height_distribution(dist: str, seed: int,
                                  hmin: float, hmax: float,
                                  n: int = 400) -> Image.Image:
    """Draw N samples from the chosen distribution, plot a histogram.

    Uses the same sampler as KR2 so the preview matches what gets baked
    into the .blend.
    """
    import importlib.util as _ilu
    import random
    spec = _ilu.spec_from_file_location(
        "kr2_h", ROOT / "scripts" / "2_build_geometry.py")
    kr2_h = _ilu.module_from_spec(spec)
    spec.loader.exec_module(kr2_h)  # type: ignore
    rng = random.Random(int(seed))
    hmin = float(hmin); hmax = float(max(hmax, hmin + 0.1))
    samples = np.array([
        max(0.5, kr2_h._sample_building_height(rng, dist, hmin, hmax))
        for _ in range(n)
    ])

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=110)
    ax.hist(samples, bins=30, color="#3a7bd5", edgecolor="white")
    ax.set_xlabel("building height (m)")
    ax.set_ylabel(f"count (n={n})")
    ax.set_title(
        f"{dist} | seed={int(seed)} | [{hmin:.1f}, {hmax:.1f}] m\n"
        f"min={samples.min():.1f}  max={samples.max():.1f}  "
        f"mean={samples.mean():.1f}  median={np.median(samples):.1f}"
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ----------------------- PCG tree helpers ----------------------- #

def _scan_tree_species_dir() -> list[str]:
    """Return sorted list of `<stem>` for every .blend file in tree_assets.dir."""
    ta = CFG.get("tree_assets") or {}
    rel = ta.get("dir", "assets/trees")
    p = (ROOT / rel).resolve()
    if not p.is_dir():
        return []
    return sorted(b.stem for b in p.glob("*.blend"))


def _resolve_existing_single_tile_name(city: str, requested_name: str,
                                       center: tuple[float, float] | None = None) -> str:
    """Resolve a UI tile alias back to an existing single-tile output folder."""
    city_dir = ROOT / "output" / city
    requested = (requested_name or "").strip().replace("/", "_").replace("\\", "_")
    if not city_dir.exists():
        return requested or "tile_0001"

    def _read_existing_tile_metadata(tile_dir: Path) -> dict:
        meta_dir = tile_dir / "metadata"
        if not meta_dir.exists():
            return {}
        for path in sorted(meta_dir.glob("*.json")):
            if path.name.endswith(".meta.json"):
                continue
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
        return {}

    def _has_blender_artifacts(tile_dir: Path) -> bool:
        blender_dir = tile_dir / "blender"
        return blender_dir.exists() and any(blender_dir.glob("*.glb"))

    if requested:
        requested_dir = city_dir / requested
        if requested_dir.exists() and _has_blender_artifacts(requested_dir):
            return requested

    candidates: list[tuple[str, float]] = []
    for tile_dir in sorted(city_dir.iterdir()):
        if not tile_dir.is_dir() or not _has_blender_artifacts(tile_dir):
            continue
        meta = _read_existing_tile_metadata(tile_dir)
        score = 0.0
        if center is not None and meta:
            center_wgs = meta.get("center_wgs84") or {}
            try:
                dlat = float(center_wgs.get("lat", 0.0)) - float(center[0])
                dlon = float(center_wgs.get("lon", 0.0)) - float(center[1])
                score = dlat * dlat + dlon * dlon
            except Exception:
                score = 0.0
        candidates.append((tile_dir.name, score))

    if not candidates:
        return requested or "tile_0001"
    if center is None:
        return candidates[0][0]
    return min(candidates, key=lambda item: item[1])[0]


def _sync_renamed_tile_artifacts(tile_dir: Path, old_name: str,
                                 new_name: str) -> None:
    """Keep internal file names consistent after a UI folder rename."""
    import shutil

    if old_name == new_name:
        return
    blender_dir = tile_dir / "blender"
    meta_dir = tile_dir / "metadata"
    pairs = [
        (blender_dir / f"{old_name}.blend", blender_dir / f"{new_name}.blend"),
        (blender_dir / f"{old_name}.glb", blender_dir / f"{new_name}.glb"),
        (blender_dir / f"{old_name}_scene.glb", blender_dir / f"{new_name}_scene.glb"),
        (meta_dir / f"{old_name}.json", meta_dir / f"{new_name}.json"),
        (meta_dir / f"{old_name}.meta.json", meta_dir / f"{new_name}.meta.json"),
        (meta_dir / f"{old_name}_osm_buildings.geojson",
         meta_dir / f"{new_name}_osm_buildings.geojson"),
    ]
    for src, dst in pairs:
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)


def _refresh_species_choices():
    species = _scan_tree_species_dir()
    return gr.update(choices=species, value=species)


def _preview_tree_height_distribution(dist: str, seed: int,
                                       hmin: float = 3.0, hmax: float = 14.0,
                                       n: int = 400) -> Image.Image:
    """Histogram of N tree-height samples; matches KR3 _sample_tree_height."""
    import math
    import random
    rng = random.Random(int(seed))
    samples = []
    hmin = float(hmin); hmax = float(max(hmax, hmin + 0.1))
    mid = 0.5 * (hmin + hmax)
    for _ in range(n):
        if dist == "flat":
            h = mid
        elif dist == "uniform":
            h = rng.uniform(hmin, hmax)
        elif dist == "bimodal":
            h = (rng.uniform(hmin, mid) if rng.random() < 0.5
                 else rng.uniform(mid, hmax))
        else:  # lognormal
            mu = math.log(max(mid, 0.5))
            sigma = max((math.log(max(hmax, mid + 0.5)) - mu) / 1.96, 0.15)
            h = math.exp(rng.gauss(mu, sigma))
        samples.append(max(0.5, min(h, hmax * 4.0)))
    samples = np.array(samples)
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=110)
    ax.hist(samples, bins=30, color="#2ea043", edgecolor="white")
    ax.set_xlabel("tree height (m)")
    ax.set_ylabel(f"count (n={n})")
    ax.set_title(
        f"{dist} | seed={int(seed)} | [{hmin:.1f}, {hmax:.1f}] m\n"
        f"min={samples.min():.1f}  max={samples.max():.1f}  "
        f"mean={samples.mean():.1f}  median={np.median(samples):.1f}"
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ----------------------- Canopy-height helpers ----------------------- #

def _compute_canopy_for_state(state, source: str, local_path: str | None,
                                city_name: str,
                                *, vectorize_threshold_m: float = 2.0,
                                target_ratio: float | None = None):
    """Build (or refresh) the canopy-height npz for the current tile.

    Uses the lat/lon/cx_utm/cy_utm/utm/gsd/size cached in ``state``
    (filled in by :func:`on_preview_satellite`).  Returns
    ``(npz_path_or_None, status_text, preview_image_or_None)``.

    Also writes ``{city}_foliage_canopy.geojson`` (unless the fetch
    failed) so KR2 will union the canopy mask into the foliage class.
    When ``target_ratio`` is given, the height threshold is auto-bumped
    so the mask covers at most that fraction of the tile.
    """
    if not state or "cx_utm" not in state:
        return None, "Run 'Preview satellite' first.", None
    try:
        from dataprep.canopy_height import (
            build_canopy_npz, render_canopy_preview,
            canopy_npz_to_foliage_geojson,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"canopy module import failed: {e}", None

    out_dir = ensure_dir(ROOT / "cache" / "canopy")
    cache_dir = ensure_dir(ROOT / "cache" / "canopy")
    name = (city_name or "tile").replace("/", "_").replace(" ", "_")
    npz = out_dir / f"{name}.npz"
    try:
        summary = build_canopy_npz(
            lat=float(state["lat"]), lon=float(state["lon"]),
            gsd=float(state["gsd"]), size=int(state["size"]),
            utm_crs=str(state["utm"]),
            cx_utm=float(state["cx_utm"]),
            cy_utm=float(state["cy_utm"]),
            cache_dir=cache_dir,
            out_npz=npz,
            source=str(source or "eth_10m"),
            local_tif=(local_path or None),
        )
    except Exception as e:  # noqa: BLE001
        return None, f"canopy build failed: {e}", None

    img = None
    try:
        img = render_canopy_preview(npz)
    except Exception as e:  # noqa: BLE001
        print(f"[app] canopy preview render failed: {e}", flush=True)

    # Vectorize canopy >= thr -> foliage geojson so KR2 unions it.
    n_polys = 0
    used_thr = float(vectorize_threshold_m)
    achieved = 0.0
    if summary["ok"] and float(vectorize_threshold_m) > 0:
        try:
            geojson_dir = ensure_dir(ROOT / CFG["paths"]["geojson_dir"])
            geojson_path = geojson_dir / f"{name}_foliage_canopy.geojson"
            n_polys, used_thr, achieved = canopy_npz_to_foliage_geojson(
                npz, tuple(state["bbox_wgs"]), geojson_path,
                height_threshold_m=float(vectorize_threshold_m),
                target_ratio=(float(target_ratio) if target_ratio
                              and float(target_ratio) > 0 else None),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[app] canopy vectorize failed: {e}", flush=True)

    if not summary["ok"]:
        msg = (f"canopy fetch FAILED  source='{source}'  "
                f"sources={summary['n_sources']}\n"
                f"  -> max={summary['max_m']:.1f}m  "
                f"frac>2m={summary['frac_treed']:.3f}\n"
                f"(check internet for ETH download, or supply a local TIFF)")
        return summary["npz_path"], msg, img
    target_str = (f"  target_ratio={target_ratio:.2f} -> "
                   f"thr={used_thr:.1f}m  ratio={achieved:.3f}\n"
                   if target_ratio and float(target_ratio) > 0
                   else "")
    msg = (f"canopy ok  source='{summary['source']}'  "
            f"sources={summary['n_sources']}  polys={n_polys}\n"
            f"  max={summary['max_m']:.1f}m  "
            f"mean(>1m)={summary['mean_canopy_m']:.1f}m  "
            f"frac>2m={summary['frac_treed']:.3f}\n"
            f"{target_str}"
            f"  -> {Path(summary['npz_path']).relative_to(ROOT)}")
    return summary["npz_path"], msg, img


def on_fetch_canopy(state, canopy_source, canopy_local_path, city_name,
                     target_foliage_ratio=None):
    """UI handler: fetch the canopy grid right now and show preview.

    The ``target_foliage_ratio`` (when > 0) is forwarded so the user can
    iterate on the canopy mask before running the full build pipeline.
    """
    npz_path, msg, img = _compute_canopy_for_state(
        state, canopy_source, canopy_local_path, city_name,
        target_ratio=(float(target_foliage_ratio)
                       if target_foliage_ratio else None))
    return img, msg


def _run_kr3(city: str, scatter_seed: int, tree_density: float,
             *, species: list | None = None,
             tree_h_dist: str = "lognormal",
             tree_h_seed: int = 11,
             tree_h_min: float = 3.0,
             tree_h_max: float = 14.0,
             canopy_npz: str | None = None,
             show_foliage_substrate: bool = False,
             scatter_mode: str = "canopy_prob",
             allow_non_foliage: bool = True,
             enable_street_trees: bool = False,
             procedural_augment_ratio: float = 0.0,
             canopy_prob_scale: float = 1.0,
             cluster_size_min: int = 10,
             cluster_size_max: int = 20,
             cluster_disk_radius_min: float = 4.0,
             cluster_disk_radius_max: float = 10.0,
             cluster_disk_aspect: float = 0.65,
             cluster_size_dist: str = "bimodal",
             cluster_size_low_frac: float = 0.7,
             tree_height_low_frac: float = 0.65,
             cluster_overlap_factor: float = 0.45,
             cluster_min_keep_ratio: float = 0.6,
             cluster_min_size_abs: int = 0,
             topdown_tree_xy_scale: float = 1.0,
             gn_tree_amount: float = 0.5,
             gn_safe_building: float = 2.5,
             gn_safe_road: float = 3.0,
             gn_safe_water: float = 2.0,
             gn_noise_scale: float = 0.10,
             gn_min_distance: float = 3.5,
             gn_xy_stretch: float = 0.75,
             gn_z_stretch: float = 0.5,
             gn_xy_stretch_min_at_0: float = 0.60,
             gn_xy_stretch_min_at_1: float = 0.90,
             gn_xy_stretch_max_at_0: float = 0.90,
             gn_xy_stretch_max_at_1: float = 4.00,
             gn_z_stretch_min_at_0: float = 0.45,
             gn_z_stretch_min_at_1: float = 1.15,
             gn_z_stretch_max_at_0: float = 0.80,
             gn_z_stretch_max_at_1: float = 2.40,
             uniform_tree_scale: bool = True,
             render_depth: bool = True):
    """Run KR3 (Blender headless) on an existing ``{city}.glb``.

    Internally always uses ``--scatter-mode canopy_driven`` so the result
    follows the real-world canopy-height grid; falls back to KR3's default
    cluster mode only if the canopy NPZ is missing.

    Returns ``(ok, log_tail, top_png, iso_png, depth_exr, depth_png)``
    where the paths live in a temp directory (no on-disk preview folder).
    ``depth_exr`` / ``depth_png`` are ``None`` when ``render_depth=False``
    or the depth render failed.
    """
    import subprocess
    import tempfile

    tile_root = (CFG["paths"].get("tile_root")
                 or CFG["paths"].get("blender_dir")
                 or CFG["paths"].get("meshes_dir") or "output")
    tile_dir = ROOT / tile_root / city
    glb = tile_dir / "blender" / f"{city}.glb"
    if not glb.exists():
        # Legacy fallbacks: pre-restructure had GLB at <tile>/<tile>.glb
        # or under output/blender/<tile>/, etc; older layout used obj/.
        legacy_a = tile_dir / f"{city}.glb"
        legacy_b = ROOT / "output" / "blender" / city / f"{city}.glb"
        legacy_c = (ROOT / CFG["paths"].get("meshes_dir", "output/meshes")
                    / f"{city}.glb")
        legacy_d = tile_dir / "obj" / f"{city}.glb"
        for cand in (legacy_a, legacy_b, legacy_c, legacy_d):
            if cand.exists():
                glb = cand
                break
        else:
            return False, f"GLB missing at {glb}", None, None, None, None
    if not Path(BLENDER_EXE).exists():
        return (False, f"Blender not found at {BLENDER_EXE}. "
                "Set BLENDER_EXE env var.", None, None, None, None)

    # Preview PNGs go to a temp dir; Gradio loads them and they may stay
    # until the OS cleans them up - they never pollute the project tree.
    tmp_dir = Path(tempfile.mkdtemp(prefix="procosm_preview_"))
    top_png = tmp_dir / f"{city}_topview.png"
    iso_png = tmp_dir / f"{city}_iso.png"
    depth_exr = tmp_dir / f"{city}_ndsm.exr" if render_depth else None
    depth_png = tmp_dir / f"{city}_ndsm.png" if render_depth else None

    cmd = [BLENDER_EXE, "--background",
           "--python", str(ROOT / "scripts" / "3_blender_assemble.py"),
           "--", "--config", str(ROOT / "configs" / "default.yaml"),
           "--city", city,
           "--scatter-seed", str(int(scatter_seed)),
           "--tree-density", str(float(tree_density)),
           "--tree-height-dist", str(tree_h_dist),
           "--tree-height-seed", str(int(tree_h_seed)),
           "--tree-height-min", str(float(tree_h_min)),
           "--tree-height-max", str(float(tree_h_max)),
           "--scatter-mode", str(scatter_mode),
           # Use the canopy heights for per-tree height too if NPZ present.
           "--canopy-as-heights",
           "--cluster-size-min", str(int(cluster_size_min)),
           "--cluster-size-max", str(int(cluster_size_max)),
           "--cluster-disk-radius-min", str(float(cluster_disk_radius_min)),
           "--cluster-disk-radius-max", str(float(cluster_disk_radius_max)),
           "--cluster-disk-aspect", str(float(cluster_disk_aspect)),
           "--cluster-size-dist", str(cluster_size_dist),
           "--cluster-size-low-frac", str(float(cluster_size_low_frac)),
           "--tree-height-low-frac", str(float(tree_height_low_frac)),
           "--cluster-overlap-factor", str(float(cluster_overlap_factor)),
           "--cluster-min-keep-ratio", str(float(cluster_min_keep_ratio)),
           "--cluster-min-size-abs", str(int(cluster_min_size_abs)),
           "--canopy-prob-scale", str(float(canopy_prob_scale)),
           "--procedural-augment-ratio", str(float(procedural_augment_ratio)),
           "--preview-png", str(top_png),
           "--preview-iso-png", str(iso_png),
           "--topdown-tree-xy-scale", str(float(topdown_tree_xy_scale)),
           "--gn-tree-amount", str(float(gn_tree_amount)),
           "--gn-safe-building", str(float(gn_safe_building)),
           "--gn-safe-road", str(float(gn_safe_road)),
           "--gn-safe-water", str(float(gn_safe_water)),
           "--gn-noise-scale", str(float(gn_noise_scale)),
           "--gn-min-distance", str(float(gn_min_distance)),
           "--gn-xy-stretch", str(float(gn_xy_stretch)),
           "--gn-z-stretch", str(float(gn_z_stretch)),
           "--gn-xy-stretch-min-at-0", str(float(gn_xy_stretch_min_at_0)),
           "--gn-xy-stretch-min-at-1", str(float(gn_xy_stretch_min_at_1)),
           "--gn-xy-stretch-max-at-0", str(float(gn_xy_stretch_max_at_0)),
           "--gn-xy-stretch-max-at-1", str(float(gn_xy_stretch_max_at_1)),
           "--gn-z-stretch-min-at-0", str(float(gn_z_stretch_min_at_0)),
           "--gn-z-stretch-min-at-1", str(float(gn_z_stretch_min_at_1)),
           "--gn-z-stretch-max-at-0", str(float(gn_z_stretch_max_at_0)),
           "--gn-z-stretch-max-at-1", str(float(gn_z_stretch_max_at_1))]
    if uniform_tree_scale:
        cmd += ["--uniform-tree-scale"]
    if depth_exr is not None:
        cmd += ["--depth-exr", str(depth_exr)]
    if depth_png is not None:
        cmd += ["--depth-png", str(depth_png)]
    if allow_non_foliage:
        cmd += ["--allow-non-foliage"]
    if enable_street_trees:
        cmd += ["--enable-street-trees"]
    if species:
        cmd += ["--tree-species", ",".join(species)]
    if canopy_npz:
        cmd += ["--canopy-npz", str(canopy_npz)]
    if show_foliage_substrate:
        cmd += ["--show-foliage-substrate"]

    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        return (False,
                f"rc={r.returncode}\nstdout: {r.stdout[-400:]}\n"
                f"stderr: {r.stderr[-400:]}",
                None, None, None, None)
    diag_lines = [
        ln for ln in r.stdout.splitlines()
        if ("[KR3]" in ln or "canopy_driven" in ln
            or "scatter mode" in ln or "PCG instanced" in ln
            or "PCG-canopy" in ln or "PCG poisson" in ln)
    ]
    log_msg = "ok"
    if diag_lines:
        log_msg = "ok\n" + "\n".join(diag_lines[-25:])
    return (True, log_msg,
            str(top_png) if top_png.exists() else None,
            str(iso_png) if iso_png.exists() else None,
            str(depth_exr) if (depth_exr and depth_exr.exists()) else None,
            str(depth_png) if (depth_png and depth_png.exists()) else None)


# --------------------- per-tile artifact persistence -------------------- #
def _persist_tile_artifacts(state, name: str, tile_dir: Path,
                            top_png: str | None,
                            depth_exr: str | None = None,
                            depth_png: str | None = None) -> None:
    """Save the 4 standard PNGs at the top of the tile folder.

    Files written (when sources available):
      - satellite_image.png        : Esri WorldImagery RGB
      - osm_basemap.png            : OSM cartographic basemap
      - seg_6class.png             : 6-class semantic palette image
      - topview_treeseg.png        : KR3 ortho top-view (canopy seg)
      - topview_depth.exr          : KR3 ortho nDSM (height-above-ground, m)
      - topview_depth.png          : 16-bit normalized vis of the EXR
    Missing sources are silently skipped (logged to stdout).
    """
    import shutil
    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)

    sat = state.get("sat_rgb")
    if sat is not None:
        try:
            sat.save(tile_dir / "satellite_image.png")
        except Exception as e:  # noqa: BLE001
            print(f"[app] save satellite_image.png failed: {e}")

    osm_b = state.get("osm_basemap")
    if osm_b is not None:
        try:
            osm_b.save(tile_dir / "osm_basemap.png")
        except Exception as e:  # noqa: BLE001
            print(f"[app] save osm_basemap.png failed: {e}")

    seg = state.get("seg_osm")
    if seg is not None:
        try:
            Image.fromarray(colorize_seg(seg, PALETTE)).save(
                tile_dir / "seg_6class.png")
        except Exception as e:  # noqa: BLE001
            print(f"[app] save seg_6class.png failed: {e}")

    if top_png and Path(top_png).exists():
        try:
            shutil.copyfile(top_png, tile_dir / "topview_treeseg.png")
        except Exception as e:  # noqa: BLE001
            print(f"[app] copy topview_treeseg.png failed: {e}")

    if depth_exr and Path(depth_exr).exists():
        try:
            shutil.copyfile(depth_exr, tile_dir / "topview_depth.exr")
        except Exception as e:  # noqa: BLE001
            print(f"[app] copy topview_depth.exr failed: {e}")

    if depth_png and Path(depth_png).exists():
        try:
            shutil.copyfile(depth_png, tile_dir / "topview_depth.png")
        except Exception as e:  # noqa: BLE001
            print(f"[app] copy topview_depth.png failed: {e}")


def _write_tile_metadata(state, name: str, tile_dir: Path,
                         bbox_wgs: tuple) -> str:
    """Write metadata/<name>.json + metadata/<name>_osm_buildings.geojson.

    The JSON bundle contains tile geo-info (bbox, GSD, image size, UTM
    CRS) and an array of synthetic building records (id, centroid in
    WGS84, height, footprint area) merged from KR2's sidecar so that
    downstream tools can match buildings to OSM features.

    The .geojson is fetched directly from osmnx for raw OSM IDs and
    geometry properties (no merging); skipped silently on failure.
    """
    meta_dir = ensure_dir(Path(tile_dir) / "metadata")
    msg_lines = []

    # ---- read KR2 sidecar (under metadata/<name>.meta.json) ---------- #
    geo_meta_path = meta_dir / f"{name}.meta.json"
    geom_meta = {}
    if geo_meta_path.exists():
        try:
            geom_meta = json.loads(geo_meta_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[app] read KR2 meta failed: {e}")

    # ---- convert per-building local centroids to WGS84 -------------- #
    buildings_wgs = []
    sw_utm = geom_meta.get("sw_utm")
    utm_crs = geom_meta.get("utm_crs")
    recs = geom_meta.get("buildings", [])
    if sw_utm and utm_crs and recs:
        try:
            inv = make_transformer(utm_crs, "EPSG:4326")
            sw_x, sw_y = float(sw_utm[0]), float(sw_utm[1])
            for r in recs:
                cx, cy = r["centroid_local_xy_m"]
                p = reproject_geom(Point(sw_x + cx, sw_y + cy), inv)
                buildings_wgs.append({
                    "id": r["building_id"],
                    "centroid_lon": float(p.x),
                    "centroid_lat": float(p.y),
                    "height_m": float(r["height_m"]),
                    "footprint_area_m2": float(r["footprint_area_m2"]),
                })
        except Exception as e:  # noqa: BLE001
            print(f"[app] reproject building centroids failed: {e}")

    # ---- assemble main metadata JSON ------------------------------- #
    meta = {
        "tile_name": name,
        "bbox_wgs84": [float(x) for x in bbox_wgs],
        "center_wgs84": {
            "lat": float(state.get("lat")),
            "lon": float(state.get("lon")),
        },
        "utm_crs": state.get("utm"),
        "sw_utm": sw_utm,
        "gsd_m_per_px": float(state.get("gsd")),
        "image_size_px": int(state.get("size")),
        "tile_extent_m": float(state.get("gsd")) * float(state.get("size")),
        "ratios": state.get("ratios"),
        "n_buildings": len(buildings_wgs),
        "buildings": buildings_wgs,
        "files": {
            "glb": f"blender/{name}.glb",
            "blend": f"blender/{name}.blend",
            "osm_basemap": "1_osm.png",
            "satellite": "2_rgb.png",
            "blender_preview": "3_blender_preview.png",
            "seg": "4_seg.png",
            "depth": "5_depth.png",
            "depth_exr": "5_depth.exr",
            "polygon_outline": "6_polygon_outline.png",
            "polygon_json": "6_polygons.json",
            "pointcloud_preview": "7_pointcloud.png",
            "pointcloud_ply": "7_pointcloud.ply",
            "geometry_meta": f"metadata/{name}.meta.json",
            "osm_buildings": f"metadata/{name}_osm_buildings.geojson",
        },
    }
    main_path = meta_dir / f"{name}.json"
    main_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    msg_lines.append(f"  metadata -> {main_path.relative_to(ROOT)}")

    # ---- raw OSM buildings geojson (preserves OSM IDs) ------------- #
    bldg_gj = meta_dir / f"{name}_osm_buildings.geojson"
    if bldg_gj.exists():
        msg_lines.append(f"  osm buildings reused -> {bldg_gj.relative_to(ROOT)}")
        return "\n".join(msg_lines)
    try:
        import osmnx as ox
        kr1._configure_osmnx(cache_dir=ROOT / "cache")
        # osmnx accepts (W,S,E,N) tuple as bbox.
        gdf = ox.features_from_bbox(bbox=tuple(bbox_wgs),
                                    tags={"building": True})
        if gdf is not None and len(gdf) > 0:
            # Drop columns with non-serializable types (lists/dicts often).
            gdf = gdf.reset_index()
            gdf.to_file(bldg_gj, driver="GeoJSON")
            msg_lines.append(f"  osm buildings ({len(gdf)}) -> "
                             f"{bldg_gj.relative_to(ROOT)}")
    except Exception as e:  # noqa: BLE001
        msg_lines.append(f"  osm buildings fetch skipped: {e}")

    return "\n".join(msg_lines)


def on_rescatter_trees(state, city_name, scatter_seed, tree_density,
                        tree_species, tree_h_dist, tree_h_seed,
                        tree_h_min, tree_h_max,
                        canopy_source, canopy_local_path,
                        scatter_mode="canopy_prob",
                        allow_non_foliage=True,
                        enable_street_trees=False,
                        canopy_prob_scale=1.0,
                        procedural_augment_ratio=0.0,
                        cluster_size_min=10,
                        cluster_size_max=20,
                        cluster_radius=10.0):
    """Re-run only KR3 (Blender) against the existing GLB."""
    import time
    name = (city_name or "tile").replace("/", "_").replace(" ", "_")
    t0 = time.time()
    canopy_npz_path, canopy_msg, _ = _compute_canopy_for_state(
        state, str(canopy_source or "eth_10m"),
        canopy_local_path, name)
    print(f"[app] {canopy_msg}", flush=True)
    ok, log, top_path, iso_path, _, _ = _run_kr3(
        name, int(scatter_seed), float(tree_density),
        species=list(tree_species) if tree_species else None,
        tree_h_dist=str(tree_h_dist), tree_h_seed=int(tree_h_seed),
        tree_h_min=float(tree_h_min), tree_h_max=float(tree_h_max),
        canopy_npz=canopy_npz_path,
        scatter_mode=str(scatter_mode),
        allow_non_foliage=bool(allow_non_foliage),
        enable_street_trees=bool(enable_street_trees),
        canopy_prob_scale=float(canopy_prob_scale),
        procedural_augment_ratio=float(procedural_augment_ratio),
        cluster_size_min=int(cluster_size_min),
        cluster_size_max=int(cluster_size_max),
        cluster_disk_radius_max=float(cluster_radius),
    )
    if not ok:
        return f"[Re-scatter FAILED] {log}", None, None
    msg = (f"trees re-scattered in {time.time()-t0:.0f}s "
           f"(seed={int(scatter_seed)}, density={tree_density}, "
           f"species={list(tree_species) if tree_species else 'auto'})")
    if canopy_msg:
        msg += "\n" + canopy_msg
    if log and log != "ok":
        msg += "\n" + log
    return msg, top_path, iso_path


# ================================== UI ================================== #


def _foliage_only_seg_image(seg_arr) -> Image.Image:
    """Colorize seg but keep only foliage class; everything else black."""
    if seg_arr is None:
        return Image.new("RGB", (256, 256), (10, 10, 10))
    seg_arr = np.asarray(seg_arr)
    out = np.zeros((seg_arr.shape[0], seg_arr.shape[1], 3), dtype=np.uint8)
    fid = CLASS_IDS["foliage"]
    rgb = PALETTE.get("foliage", [60, 200, 60])
    out[seg_arr == fid] = rgb
    return Image.fromarray(out)


# ================================ build_ui ============================== #

def build_ui():
    """新版 Gradio UI：双 Tab + 一键 Generate All。

    Tab 1 (单瓦片调试)：搜索/preset → 卫星图点击中心 → 1 个 tile 全套生成。
    Tab 2 (多瓦片批量)：搜索/preset → 卫星图点 NW + SE 两角 → 多 tile 批跑。

    所有可调参数集中在共享面板（精确镜像 ``AutoPipelineConfig``，
    每个 label 末尾标注对应 CLI flag），点击 🚀 Generate All 走与
    ``auto_pipeline.py`` 完全相同的 stage 函数，保证 UI 调出来的结果
    和 CLI 批量跑的字节一致。
    """
    # Local imports (auto_pipeline already imports osm_app at top — we
    # defer it so import order is safe).
    from auto_pipeline import (
        AutoPipeline, AutoPipelineConfig, STAGES, STAGE_NAMES,
        aggregate_city,
        rerun_trees_only, run_single_tile,
    )
    from dataprep.tile_grid import plan_tiles, grid_shape
    from dataprep.geometry_utils import (
        make_transformer, reproject_geom, utm_crs_for_bbox,
    )
    import threading
    import time as _time

    # ---------------- shared overview-map helpers (ported from
    # auto_osm_app.py) ---------------- #
    WIDE_MAP_PX = 800
    WIDE_HALF_M_DEFAULT = 6000.0  # ~12 km wide

    def _wide_bbox(lat: float, lon: float,
                   half_m: float = WIDE_HALF_M_DEFAULT):
        bbox_for_utm = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
        utm = utm_crs_for_bbox(bbox_for_utm)
        fwd = make_transformer("EPSG:4326", utm)
        inv = make_transformer(utm, "EPSG:4326")
        p = reproject_geom(Point(lon, lat), fwd)
        sw = reproject_geom(Point(p.x - half_m, p.y - half_m), inv)
        ne = reproject_geom(Point(p.x + half_m, p.y + half_m), inv)
        return (sw.x, sw.y, ne.x, ne.y)

    def _wide_overview(lat: float, lon: float, half_m: float,
                       corner1=None, corner2=None,
                       single_center=None, plans=None):
        ov = _wide_bbox(lat, lon, half_m=half_m)
        try:
            img = satellite_image_for_bbox(
                ov, out_size=WIDE_MAP_PX).convert("RGB")
        except Exception as e:
            print(f"[ui] overview fetch failed: {e}", flush=True)
            img = Image.new("RGB", (WIDE_MAP_PX, WIDE_MAP_PX),
                            (40, 40, 40))
        W, S, E, N = ov

        def to_px(lon_, lat_):
            x = (lon_ - W) / (E - W) * WIDE_MAP_PX
            y = (N - lat_) / (N - S) * WIDE_MAP_PX
            return x, y

        draw = ImageDraw.Draw(img)
        # NW/SE corner markers (yellow + cyan)
        for c, color in ((corner1, (255, 200, 0)),
                          (corner2, (0, 200, 255))):
            if c:
                cx, cy = to_px(c[1], c[0])
                draw.line([(cx - 8, cy), (cx + 8, cy)], fill=color, width=2)
                draw.line([(cx, cy - 8), (cx, cy + 8)], fill=color, width=2)
        if corner1 and corner2:
            lat1, lon1 = corner1; lat2, lon2 = corner2
            Wl = min(lon1, lon2); El = max(lon1, lon2)
            Sl = min(lat1, lat2); Nl = max(lat1, lat2)
            x1, y1 = to_px(Wl, Nl); x2, y2 = to_px(El, Sl)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)
        if single_center:
            slat, slon = single_center
            cx, cy = to_px(slon, slat)
            # red cross + tile-sized bbox preview (512 m at gsd=0.5)
            draw.line([(cx - 10, cy), (cx + 10, cy)],
                       fill=(255, 60, 60), width=3)
            draw.line([(cx, cy - 10), (cx, cy + 10)],
                       fill=(255, 60, 60), width=3)
            # Approx 512 m → degrees latitude (rough; for preview only)
            half_deg_lat = 256.0 / 111000.0
            half_deg_lon = half_deg_lat / max(0.1, np.cos(np.radians(slat)))
            x1, y1 = to_px(slon - half_deg_lon, slat + half_deg_lat)
            x2, y2 = to_px(slon + half_deg_lon, slat - half_deg_lat)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=2)
        if plans:
            for p in plans:
                tw, ts, te, tn = p.bbox_wgs
                x1, y1 = to_px(tw, tn); x2, y2 = to_px(te, ts)
                draw.rectangle([x1, y1, x2, y2],
                                outline=(255, 255, 255), width=1)
        return img, list(ov)

    # ---------------- shared progress tracker ---------------- #
    class _ProgressTracker:
        def __init__(self, n_tiles: int):
            self.n_tiles = n_tiles
            self.lock = threading.Lock()
            self.counts = {s: {"ok": 0, "failed": 0, "blocked": 0,
                                "skipped": 0, "running": 0}
                            for s in STAGES}
            self.lines: list[str] = []

        def cb(self, stage: str, name: str, status: str):
            with self.lock:
                if stage in self.counts and status in self.counts[stage]:
                    self.counts[stage][status] += 1
                self.lines.append(
                    f"[{_time.strftime('%H:%M:%S')}] "
                    f"{stage} {name:<14s} {status}")
                if len(self.lines) > 600:
                    self.lines = self.lines[-600:]

        def render(self) -> str:
            with self.lock:
                head = []
                for s in STAGES:
                    c = self.counts[s]
                    head.append(f"{s}({STAGE_NAMES[s][:14]}): "
                                 f"ok={c['ok']} fail={c['failed']} "
                                 f"blocked={c['blocked']} skip={c['skipped']}"
                                 f" / {self.n_tiles}")
                tail = "\n".join(self.lines[-25:])
                return "\n".join(head) + "\n\n" + tail

    def _progress_bar_html(percent: float, label: str) -> str:
        pct = max(0.0, min(100.0, float(percent)))
        safe_label = html.escape(label or "Idle")
        return f"""
<div style="position:fixed;inset:0;z-index:9999;background:rgba(15,23,42,0.48);backdrop-filter:blur(2px);display:flex;align-items:center;justify-content:center;pointer-events:auto;">
    <div style="width:min(560px,calc(100vw - 32px));border:1px solid #d0d7de;border-radius:16px;padding:18px 18px 14px;background:#fff;box-shadow:0 24px 80px rgba(0,0,0,0.25);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;font-size:16px;">
            <span><strong>{safe_label}</strong></span>
            <span>{pct:.0f}%</span>
        </div>
        <div style="width:100%;height:14px;background:#edf1f5;border-radius:999px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,#ff7a18,#ffb347);"></div>
        </div>
        <div style="margin-top:10px;font-size:13px;color:#475569;line-height:1.5;">
            生成进行中，页面已临时锁定。请等待当前任务完成后再继续拖动、点击或改参数。
        </div>
    </div>
</div>
"""

    def _tracker_progress_html(tracker: "_ProgressTracker",
                               stages: Sequence[str],
                               label: str) -> str:
        done_statuses = ("ok", "failed", "blocked", "skipped")
        with tracker.lock:
            total = max(1, len(stages) * tracker.n_tiles)
            done = 0
            for stage in stages:
                c = tracker.counts.get(stage, {})
                done += sum(int(c.get(st, 0)) for st in done_statuses)
        return _progress_bar_html(100.0 * done / total, label)

    class _LiveLogBuffer(io.TextIOBase):
        def __init__(self, max_lines: int = 300):
            self.max_lines = max_lines
            self.lock = threading.Lock()
            self.lines: list[str] = []
            self._partial = ""
            self._stdout = sys.__stdout__
            self._stderr = sys.__stderr__

        def writable(self):
            return True

        def write(self, data):
            if not data:
                return 0
            text = str(data)
            try:
                self._stdout.write(text)
                self._stdout.flush()
            except Exception:
                pass
            with self.lock:
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                text = self._partial + text
                parts = text.split("\n")
                self._partial = parts.pop() if parts else ""
                for part in parts:
                    self.lines.append(part)
                if len(self.lines) > self.max_lines:
                    self.lines = self.lines[-self.max_lines:]
            return len(data)

        def flush(self):
            try:
                self._stdout.flush()
            except Exception:
                pass

        def snapshot(self) -> str:
            with self.lock:
                lines = list(self.lines)
                if self._partial:
                    lines.append(self._partial)
            return "\n".join(lines[-self.max_lines:])

        def render(self, tracker_text: str) -> str:
            log_text = self.snapshot().strip()
            if not log_text:
                log_text = "[ui] 等待后台日志输出..."
            return f"{tracker_text}\n\n---- 实时日志 ----\n{log_text}"

    # ---------------- shared param panel builder ---------------- #
    # Each tab gets its own component instances (Gradio binds per-component).
    # Returns a dict of {field_name: Component}. Order is preserved.
    def _build_param_panel(tag: str) -> dict:
        d: dict = {}
        # 🏗️ Building height ---------------------------------------- #
        with gr.Accordion("🏗️ 建筑物高度 (height_*)", open=False):
            d["height_dist"] = gr.Dropdown(
                choices=["uniform", "lognormal", "bimodal"],
                value="lognormal", label="height_dist  (—)")
            with gr.Row():
                d["height_seed"] = gr.Number(
                    value=42, precision=0, label="height_seed  (—)")
                d["height_min"] = gr.Number(
                    value=3.0, label="height_min  (—) [m]")
                d["height_max"] = gr.Number(
                    value=30.0, label="height_max  (—) [m]")
        # 🌳 GN tree scatter --------------------------------------- #
        with gr.Accordion("🌳 GN 树木生成规则（当前真正生效）", open=True):
            gr.Markdown(
                "当前常用只看 3 组参数：\n"
                "1. `Green Area Size` 控制整体绿色覆盖面积；\n"
                "2. `Tree Density in Green Area` 控制绿色区域里的树木数量和密集程度；\n"
                "3. `XY / Z Stretch` 控制树冠横向与纵向拉伸；每个方向只需要设置一组 `Min / Max`，滑条 `0 -> 1` 会从无拉伸过渡到你设定的范围。"
            )
            d["target_foliage_ratio"] = gr.Slider(
                0.0, 1.0, value=0.25, step=0.05,
                label="Green Area Size  (--target-foliage-ratio)  控制整体绿色面积")
            d["gn_tree_amount"] = gr.Slider(
                0.0, 1.0, value=0.5, step=0.05,
                label="Tree Density in Green Area  (--gn-tree-amount)  控制绿色区域内树木数量/密度")
            d["tree_density"] = gr.Number(
                value=0.00015, visible=False, label="legacy tree_density")
            with gr.Row():
                d["gn_xy_stretch"] = gr.Slider(
                    0.0, 1.0, value=0.75, step=0.05,
                    label="Tree XY Stretch  (--gn-xy-stretch)  0=不拉伸, 1=使用下面 XY Min/Max")
                d["gn_z_stretch"] = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Tree Z Stretch  (--gn-z-stretch)  0=不拉伸, 1=使用下面 Z Min/Max")
            with gr.Row():
                d["gn_xy_stretch_min"] = gr.Number(
                    value=0.90, label="XY Stretch Min")
                d["gn_xy_stretch_max"] = gr.Number(
                    value=4.00, label="XY Stretch Max")
            with gr.Row():
                d["gn_z_stretch_min"] = gr.Number(
                    value=1.15, label="Z Stretch Min")
                d["gn_z_stretch_max"] = gr.Number(
                    value=2.40, label="Z Stretch Max")
            with gr.Accordion("高级树木参数（通常不用改）", open=False):
                d["scatter_seed"] = gr.Number(
                    value=11, precision=0, label="Random Seed  (--scatter-seed)")
                gr.Button("🎲 Random seed", size="sm").click(
                    lambda: int(np.random.randint(0, 1_000_000)),
                    None, d["scatter_seed"])
                with gr.Row():
                    d["gn_safe_building"] = gr.Slider(
                        0.0, 20.0, value=2.5, step=0.5,
                        label="Safe Distance Building  (--gn-safe-building) [m]")
                    d["gn_safe_road"] = gr.Slider(
                        0.0, 20.0, value=3.0, step=0.5,
                        label="Safe Distance Road  (--gn-safe-road) [m]")
                    d["gn_safe_water"] = gr.Slider(
                        0.0, 20.0, value=2.0, step=0.5,
                        label="Safe Distance Water  (--gn-safe-water) [m]")
                with gr.Row():
                    d["gn_noise_scale"] = gr.Slider(
                        0.01, 0.5, value=0.10, step=0.01,
                        label="Forest Noise Scale  (--gn-noise-scale)  越大斑块越碎")
                    d["gn_min_distance"] = gr.Slider(
                        0.5, 12.0, value=3.5, step=0.5,
                        label="Tree Min Distance  (--gn-min-distance) [m]")
                    d["topdown_tree_xy_scale"] = gr.Slider(
                        0.5, 6.0, value=1.0, step=0.1,
                        label="Topdown XY Inflate  (--topdown-tree-xy-scale)")
                species_init = _scan_tree_species_dir()
                d["tree_species"] = gr.CheckboxGroup(
                    choices=species_init, value=species_init,
                    label="Tree Species  assets/trees/*.blend")

        # Hidden legacy controls retained only to keep existing config wiring stable.
        d["scatter_mode"] = gr.Textbox(value="canopy_prob", visible=False)
        d["allow_non_foliage"] = gr.Checkbox(value=True, visible=False)
        d["enable_street_trees"] = gr.Checkbox(value=False, visible=False)
        d["canopy_prob_scale"] = gr.Number(value=1.0, visible=False)
        d["procedural_augment_ratio"] = gr.Number(value=0.0, visible=False)
        d["uniform_tree_scale"] = gr.Checkbox(value=False, visible=False)
        d["tree_h_dist"] = gr.Textbox(value="lognormal", visible=False)
        d["tree_h_seed"] = gr.Number(value=11, precision=0, visible=False)
        d["tree_h_min"] = gr.Number(value=6.0, visible=False)
        d["tree_h_max"] = gr.Number(value=20.0, visible=False)
        d["tree_height_low_frac"] = gr.Number(value=0.65, visible=False)
        d["cluster_size_min"] = gr.Number(value=10, precision=0, visible=False)
        d["cluster_size_max"] = gr.Number(value=20, precision=0, visible=False)
        d["cluster_min_size_abs"] = gr.Number(value=10, precision=0, visible=False)
        d["cluster_disk_radius_min"] = gr.Number(value=4.0, visible=False)
        d["cluster_disk_radius_max"] = gr.Number(value=10.0, visible=False)
        d["cluster_disk_aspect"] = gr.Number(value=0.65, visible=False)
        d["cluster_size_dist"] = gr.Textbox(value="uniform", visible=False)
        d["cluster_size_low_frac"] = gr.Number(value=0.7, visible=False)
        d["cluster_overlap_factor"] = gr.Number(value=0.45, visible=False)
        d["cluster_min_keep_ratio"] = gr.Number(value=0.6, visible=False)
        # 🎨 Render / global --------------------------------------- #
        with gr.Accordion("🎨 渲染 & 全局", open=False):
            d["use_blender_seg"] = gr.Checkbox(
                value=False,
                visible=False,
                label="use_blender_seg  (--use-blender-seg)")
            with gr.Row():
                d["canopy_source"] = gr.Dropdown(
                    choices=["eth_10m", "local"],
                    value="eth_10m",
                    label="canopy_source  (—)")
            with gr.Row():
                d["overlap"] = gr.Slider(
                    0.0, 0.5, value=0.0, step=0.05,
                    label="overlap  (--overlap)  仅多瓦片用")
                d["io_workers"] = gr.Number(
                    value=8, precision=0,
                    label="io_workers  (--io-workers)")
                d["osm_workers"] = gr.Number(
                    value=4, precision=0,
                    label="osm_workers  (--osm-workers)")
                d["canopy_workers"] = gr.Number(
                    value=4, precision=0,
                    label="canopy_workers  (--canopy-workers)")
        return d

    # ---------- helper: assemble AutoPipelineConfig from panel ---------- #
    def _cfg_from_panel(city: str, area_bbox: tuple, vals: dict
                          ) -> "AutoPipelineConfig":
        """Map raw panel values into an AutoPipelineConfig."""
        target_ratio = float(vals["target_foliage_ratio"])
        xy_min = float(vals["gn_xy_stretch_min"])
        xy_max = float(vals["gn_xy_stretch_max"])
        z_min = float(vals["gn_z_stretch_min"])
        z_max = float(vals["gn_z_stretch_max"])
        return AutoPipelineConfig(
            city=city,
            area_bbox_wgs=area_bbox,
            overlap=float(vals["overlap"]),
            height_dist=str(vals["height_dist"]),
            height_seed=int(vals["height_seed"]),
            height_min=float(vals["height_min"]),
            height_max=float(vals["height_max"]),
            scatter_seed=int(vals["scatter_seed"]),
            tree_density=float(vals["tree_density"]),
            tree_species=list(vals["tree_species"]) or None,
            tree_h_dist=str(vals["tree_h_dist"]),
            tree_h_seed=int(vals["tree_h_seed"]),
            tree_h_min=float(vals["tree_h_min"]),
            tree_h_max=float(vals["tree_h_max"]),
            cluster_size_min=int(vals["cluster_size_min"]),
            cluster_size_max=int(vals["cluster_size_max"]),
            cluster_disk_radius_min=float(vals["cluster_disk_radius_min"]),
            cluster_disk_radius_max=float(vals["cluster_disk_radius_max"]),
            cluster_disk_aspect=float(vals["cluster_disk_aspect"]),
            cluster_size_dist=str(vals["cluster_size_dist"]),
            cluster_size_low_frac=float(vals["cluster_size_low_frac"]),
            tree_height_low_frac=float(vals["tree_height_low_frac"]),
            cluster_overlap_factor=float(vals["cluster_overlap_factor"]),
            cluster_min_keep_ratio=float(vals["cluster_min_keep_ratio"]),
            cluster_min_size_abs=int(vals["cluster_min_size_abs"]),
            uniform_tree_scale=bool(vals["uniform_tree_scale"]),
            scatter_mode=str(vals["scatter_mode"]),
            allow_non_foliage=bool(vals["allow_non_foliage"]),
            enable_street_trees=bool(vals["enable_street_trees"]),
            procedural_augment_ratio=float(vals["procedural_augment_ratio"]),
            canopy_prob_scale=float(vals["canopy_prob_scale"]),
            topdown_tree_xy_scale=float(vals["topdown_tree_xy_scale"]),
            gn_tree_amount=float(vals["gn_tree_amount"]),
            gn_safe_building=float(vals["gn_safe_building"]),
            gn_safe_road=float(vals["gn_safe_road"]),
            gn_safe_water=float(vals["gn_safe_water"]),
            gn_noise_scale=float(vals["gn_noise_scale"]),
            gn_min_distance=float(vals["gn_min_distance"]),
            gn_xy_stretch=float(vals["gn_xy_stretch"]),
            gn_z_stretch=float(vals["gn_z_stretch"]),
            gn_xy_stretch_min_at_0=1.0,
            gn_xy_stretch_min_at_1=xy_min,
            gn_xy_stretch_max_at_0=1.0,
            gn_xy_stretch_max_at_1=xy_max,
            gn_z_stretch_min_at_0=1.0,
            gn_z_stretch_min_at_1=z_min,
            gn_z_stretch_max_at_0=1.0,
            gn_z_stretch_max_at_1=z_max,
            use_blender_seg=bool(vals["use_blender_seg"]),
            canopy_source=str(vals["canopy_source"]),
            target_foliage_ratio=(target_ratio if target_ratio > 0 else None),
            io_workers=int(vals["io_workers"]),
            osm_workers=int(vals["osm_workers"]),
            canopy_workers=int(vals["canopy_workers"]),
        )

    init_lat, init_lon = 41.260, -96.135

    with gr.Blocks(title="ProcedureOSM — 一键全套生成") as demo:
        gr.Markdown(
            "# ProcedureOSM — 一键 Generate All\n"
            "**Tab 1 单瓦片调试** = 选中心 → 一键出 4 PNG + .blend，"
            "用于参数试错。  \n"
            "**Tab 2 多瓦片批量** = 选 NW/SE 两角 → 等价于 "
            "`python auto_pipeline.py ...`，批量产出。  \n"
            "底部参数面板与 [`AutoPipelineConfig`](auto_pipeline.py#L775) "
            "一一对应（label 后括号是 CLI flag）。"
        )

        # ===================== Tab 1: single-tile debug ===================== #
        with gr.Tab("🔬 单瓦片调试"):
            t1_state_center = gr.State(value=(init_lat, init_lon))
            t1_ov_bbox = gr.State()

            with gr.Row():
                with gr.Column(scale=1):
                    t1_search = gr.Textbox(
                        label="🔍 搜索城市/地址",
                        placeholder="Omaha, Nebraska")
                    with gr.Row():
                        t1_search_btn = gr.Button("Search",
                                                    variant="primary")
                        t1_preset = gr.Dropdown(
                            list(PRESETS.keys()),
                            value=DEFAULT_PRESET, label="或选 preset")
                    with gr.Row():
                        t1_lat = gr.Number(value=init_lat,
                                            label="center lat")
                        t1_lon = gr.Number(value=init_lon,
                                            label="center lon")
                    t1_half_km = gr.Slider(
                        1.0, 20.0, value=6.0, step=0.5,
                        label="overview 半宽 (km)")
                    t1_city = gr.Textbox(
                        value="ui_debug_city",
                        label="city  (输出 output/<city>/)")
                    t1_tile_name = gr.Textbox(
                        value="tile_dev",
                        label="tile 名 (输出子目录)")
                    t1_msg = gr.Textbox(
                        label="状态", interactive=False, lines=2,
                        value="点击卫星图设置 tile 中心。")
                with gr.Column(scale=2):
                    t1_map = gr.Image(
                        label="overview Esri (click = set tile center)",
                        height=600, interactive=False)

            with gr.Row():
                with gr.Column(scale=2, min_width=520):
                    with gr.Accordion("⚙️ 全部参数面板（共享 AutoPipelineConfig）",
                                        open=True):
                        t1_panel = _build_param_panel("t1")
                with gr.Column(scale=1, min_width=360):
                    t1_model = gr.Model3D(label="GLB 3D 预览", height=640)

            with gr.Row():
                t1_run_btn = gr.Button(
                    "🚀 Generate All (B→F 全套)",
                    variant="primary", size="lg")
                t1_rerun_trees_btn = gr.Button(
                    "🌳 Regenerate Trees Only (F only: .blend + seg/depth)",
                    variant="secondary", size="lg")
            t1_progress = gr.HTML(value="")
            with gr.Row():
                t1_osm = gr.Image(label="1_osm.png", height=300)
                t1_rgb = gr.Image(label="2_rgb.png", height=300)
                t1_blender_preview = gr.Image(
                    label="3_blender_preview.png", height=300)
                t1_seg = gr.Image(label="4_seg.png", height=300)
            with gr.Row():
                t1_depth = gr.Image(label="5_depth.png", height=300)
                t1_poly = gr.Image(
                    label="6_polygon_outline.png", height=300)
                t1_pointcloud = gr.Image(
                    label="7_pointcloud.png", height=300)
                t1_near_nadir_depth = gr.Image(
                    label="near-nadir-1/2_depth.png", height=300)
            t1_blend_path = gr.Textbox(
                label=".blend / .glb 路径", interactive=False, lines=2)
            t1_log = gr.Textbox(
                label="运行日志", interactive=False, lines=12)
            t1_busy = gr.State(value=False)

            # ---- Tab 1 handlers ---- #
            def _t1_recenter(lat, lon, half_km, busy):
                if busy:
                    return (gr.update(), gr.update(), gr.update(),
                            "生成中，暂时锁定地图拖动/选择与参数修改。")
                img, ov = _wide_overview(
                    float(lat), float(lon),
                    half_m=float(half_km) * 1000.0,
                    single_center=(float(lat), float(lon)))
                return img, list(ov), (float(lat), float(lon)), gr.update()

            def _t1_preset(name, half_km, busy):
                if busy:
                    return (gr.update(), gr.update(), gr.update(),
                            gr.update(), gr.update(),
                            "生成中，暂时锁定地图拖动/选择与参数修改。")
                bb = PRESETS.get(name)
                if not bb:
                    return (gr.update(), gr.update(), gr.update(),
                            gr.update(), gr.update(), "preset not found")
                cy = (bb[1] + bb[3]) / 2.0
                cx = (bb[0] + bb[2]) / 2.0
                img, ov = _wide_overview(
                    cy, cx, half_m=float(half_km) * 1000.0,
                    single_center=(cy, cx))
                return (cy, cx, img, list(ov), (cy, cx),
                        f"preset '{name}' loaded.")

            def _t1_search(q, half_km, busy):
                if busy:
                    return (gr.update(), gr.update(), gr.update(),
                            gr.update(), gr.update(),
                            "生成中，暂时锁定地图拖动/选择与参数修改。")
                if not q or not q.strip():
                    return (gr.update(), gr.update(), gr.update(),
                            gr.update(), gr.update(),
                            "enter a query.")
                from dataprep.geocode import geocode as _g
                res = _g(q.strip(), limit=3)
                if not res:
                    return (gr.update(), gr.update(), gr.update(),
                            gr.update(), gr.update(),
                            f"no result for '{q}'.")
                top = res[0]
                img, ov = _wide_overview(
                    top["lat"], top["lon"],
                    half_m=float(half_km) * 1000.0,
                    single_center=(top["lat"], top["lon"]))
                return (top["lat"], top["lon"], img, list(ov),
                        (top["lat"], top["lon"]),
                        f"found: {top['display_name']}")

            def _t1_click(ov_bbox, lat, lon, half_km, busy, evt: gr.SelectData):
                if busy:
                    return (gr.update(), gr.update(), gr.update(),
                            gr.update(), gr.update(),
                            "生成中，暂时锁定地图拖动/选择与参数修改。")
                if evt is None or evt.index is None or ov_bbox is None:
                    return gr.update(), gr.update(), gr.update(), \
                            gr.update(), "click on the map first."
                px, py = evt.index
                W, S, E, N = ov_bbox
                new_lon = W + (px / WIDE_MAP_PX) * (E - W)
                new_lat = N - (py / WIDE_MAP_PX) * (N - S)
                img, ov = _wide_overview(
                    float(lat), float(lon),
                    half_m=float(half_km) * 1000.0,
                    single_center=(new_lat, new_lon))
                return (new_lat, new_lon, img, list(ov), (new_lat, new_lon),
                        f"tile center: lat={new_lat:.5f}, lon={new_lon:.5f}")

            t1_search_btn.click(
                _t1_search, [t1_search, t1_half_km, t1_busy],
                [t1_lat, t1_lon, t1_map, t1_ov_bbox, t1_state_center, t1_msg])
            t1_search.submit(
                _t1_search, [t1_search, t1_half_km, t1_busy],
                [t1_lat, t1_lon, t1_map, t1_ov_bbox, t1_state_center, t1_msg])
            t1_preset.change(
                _t1_preset, [t1_preset, t1_half_km, t1_busy],
                [t1_lat, t1_lon, t1_map, t1_ov_bbox, t1_state_center, t1_msg])
            t1_half_km.change(
                _t1_recenter, [t1_lat, t1_lon, t1_half_km, t1_busy],
                [t1_map, t1_ov_bbox, t1_state_center, t1_msg])
            t1_map.select(
                _t1_click,
                [t1_ov_bbox, t1_lat, t1_lon, t1_half_km, t1_busy],
                [t1_lat, t1_lon, t1_map, t1_ov_bbox, t1_state_center, t1_msg])
            demo.load(
                _t1_recenter, [t1_lat, t1_lon, t1_half_km, t1_busy],
                [t1_map, t1_ov_bbox, t1_state_center, t1_msg])

            # ---- Tab 1: Generate All ---- #
            t1_panel_keys = list(t1_panel.keys())
            t1_panel_components = [t1_panel[k] for k in t1_panel_keys]
            t1_lockables = [
                t1_search, t1_search_btn, t1_preset,
                t1_lat, t1_lon, t1_half_km,
                t1_city, t1_tile_name,
                t1_run_btn, t1_rerun_trees_btn,
            ] + t1_panel_components

            def _t1_control_updates(enabled: bool):
                return [gr.update(interactive=enabled) for _ in t1_lockables]

            def _t1_begin_locked_run(label: str):
                return (_t1_control_updates(False)
                        + [True, _progress_bar_html(1.0, label)])

            def _t1_finish_locked_run():
                return _t1_control_updates(True) + [False, ""]

            def _print_result_block(success: bool, folder_path: str, tile_name: str, error_msg: str = "") -> str:
                if success:
                    box_str = f"""
\033[32m###########################################################################
#                                                                         #
#     🎉 SUCCESS: STAGE F ASSEMBLED COMPLETED SUCCESSFULLY! 🎉             #
#                            (Auto CC0 App)                               #
#                                                                         #
#     📍 TILE:   {tile_name:<48s} #
#     📂 SAVED TO FOLDER:                                                 #
#        {folder_path:<56s}   #
#                                                                         #
###########################################################################\033[0m
"""
                    banner = f"""===========================================================================
🎉 SUCCESS: STAGE F ASSEMBLED COMPLETED SUCCESSFULLY!
📍 TILE: {tile_name}
📂 Saved Folder: {folder_path}
==========================================================================="""
                else:
                    box_str = f"""
\033[31m###########################################################################
#                                                                         #
#     ❌ FAILURE: STAGE F GENERATION FAILED! ❌                            #
#                            (Auto CC0 App)                               #
#                                                                         #
#     📍 TILE:   {tile_name:<48s} #
#     📂 FOLDER: {folder_path:<56s}   #
#     ⚠️ ERROR:  {error_msg[:54]:<54s}   #
#                                                                         #
###########################################################################\033[0m
"""
                    banner = f"""===========================================================================
❌ FAILURE: STAGE F GENERATION FAILED!
📍 TILE: {tile_name}
📂 Folder: {folder_path}
⚠️ Error: {error_msg}
==========================================================================="""
                _orig_print(box_str)
                return banner

            def _t1_generate_all(center, city, tile_name, *panel_vals):
                lat0, lon0 = center
                vals = dict(zip(t1_panel_keys, panel_vals))
                # Build a tight bbox so plan_tiles produces exactly 1 tile.
                # 256 m half-side @ gsd=0.5 = 1024 px tile.
                half_m = 256.0
                # Approx degrees
                d_lat = half_m / 111000.0
                d_lon = d_lat / max(0.1, np.cos(np.radians(lat0)))
                bbox = (lon0 - d_lon, lat0 - d_lat,
                          lon0 + d_lon, lat0 + d_lat)
                city_clean = (city or "ui_debug_city").strip().replace(
                    "/", "_").replace(" ", "_")
                cfg = _cfg_from_panel(city_clean, bbox, vals)
                tracker = _ProgressTracker(n_tiles=1)
                live_log = _LiveLogBuffer()
                holder = {}

                def _bg():
                    with contextlib.redirect_stdout(live_log), \
                         contextlib.redirect_stderr(live_log):
                        try:
                            holder["tr"] = run_single_tile(
                                cfg, progress_cb=tracker.cb)
                        except Exception as e:
                            import traceback
                            holder["err"] = (
                                f"{type(e).__name__}: {e}\n"
                                + traceback.format_exc()[-1500:])

                th = threading.Thread(target=_bg, daemon=True); th.start()
                # Stream progress.
                yield (None, None, None, None, None, None, None, None, None, "", tracker.render(),
                       _tracker_progress_html(
                           tracker, ("B", "C", "D", "E", "F"),
                           "正在 Generate All..."))
                while th.is_alive():
                    _time.sleep(1.0)
                    yield (None, None, None, None, None, None, None, None, None, "", tracker.render(),
                           _tracker_progress_html(
                               tracker, ("B", "C", "D", "E", "F"),
                               "正在 Generate All..."))
                th.join()
                log_text = live_log.render(tracker.render())
                if "err" in holder:
                    user_name = (tile_name or "").strip().replace("/", "_").replace("\\", "_")
                    err_folder = str(ROOT / "output" / city_clean)
                    banner = _print_result_block(False, err_folder, user_name or "tile_0001", str(holder['err']))
                    yield (None, None, None, None, None, None, None, None, None,
                            f"FAILED: {holder['err']}",
                            banner + "\n\n" + log_text,
                            _progress_bar_html(100.0, "Generate All 失败"))
                    return
                tr = holder["tr"]
                tile_dir = ROOT / "output" / city_clean / tr.plan.name
                # Optionally rename to user-supplied tile_name.
                user_name = (tile_name or "").strip().replace("/", "_").replace("\\", "_")
                if user_name and user_name != tr.plan.name:
                    new_dir = ROOT / "output" / city_clean / user_name
                    try:
                        import shutil as _shutil
                        if not tile_dir.exists():
                            print(f"[ui] tile rename skipped: source {tile_dir} not found")
                        else:
                            if new_dir.exists():
                                _shutil.rmtree(str(new_dir))
                            # Use copytree+rmtree instead of rename —
                            # Path.rename() can fail on Windows when the
                            # destination was just rmtree'd (lazy dir-entry
                            # release) or when anti-virus holds a handle.
                            _shutil.copytree(str(tile_dir), str(new_dir))
                            _shutil.rmtree(str(tile_dir))
                            tile_dir = new_dir
                            _sync_renamed_tile_artifacts(
                                tile_dir, tr.plan.name, user_name)
                    except Exception as e:
                        print(f"[ui] tile rename failed ({type(e).__name__}: {e}); "
                              f"using auto-name '{tr.plan.name}'")
                osm_p = tile_dir / "1_osm.png"
                rgb_p = tile_dir / "2_rgb.png"
                preview_p = tile_dir / "3_blender_preview.png"
                seg_p = tile_dir / "4_seg.png"
                dep_p = tile_dir / "5_depth.png"
                poly_p = tile_dir / "6_polygon_outline.png"
                poly_json_p = tile_dir / "6_polygons.json"
                pointcloud_png_p = tile_dir / "7_pointcloud.png"
                pointcloud_ply_p = tile_dir / "7_pointcloud.ply"
                near_nadir_depth_p = tile_dir / "near-nadir-1" / "2_depth.png"
                blend_p = tile_dir / "blender" / f"{tile_dir.name}.blend"
                glb_p = tile_dir / "blender" / f"{tile_dir.name}.glb"
                preview_glb_p = tile_dir / "blender" / f"{tile_dir.name}_scene.glb"
                paths = (f".blend = {blend_p}\n.glb   = {glb_p}\n"
                         f"preview glb = {preview_glb_p}\n"
                         f"polygon json = {poly_json_p}\n"
                         f"pointcloud ply = {pointcloud_ply_p}")
                
                banner = _print_result_block(True, str(tile_dir), tile_dir.name)
                
                yield (str(osm_p) if osm_p.exists() else None,
                        str(rgb_p) if rgb_p.exists() else None,
                    str(preview_p) if preview_p.exists() else None,
                        str(seg_p) if seg_p.exists() else None,
                        str(dep_p) if dep_p.exists() else None,
                    str(poly_p) if poly_p.exists() else None,
                    str(pointcloud_png_p) if pointcloud_png_p.exists() else None,
                    str(near_nadir_depth_p) if near_nadir_depth_p.exists() else None,
                    str(preview_glb_p if preview_glb_p.exists() else glb_p) if (preview_glb_p.exists() or glb_p.exists()) else None,
                    paths, banner + "\n\n" + log_text,
                    _progress_bar_html(100.0, "Generate All 完成"))
                        

            def _t1_rerun_trees(center, city, tile_name, *panel_vals):
                lat0, lon0 = center
                vals = dict(zip(t1_panel_keys, panel_vals))
                half_m = 256.0
                d_lat = half_m / 111000.0
                d_lon = d_lat / max(0.1, np.cos(np.radians(lat0)))
                bbox = (lon0 - d_lon, lat0 - d_lat,
                          lon0 + d_lon, lat0 + d_lat)
                city_clean = (city or "ui_debug_city").strip().replace(
                    "/", "_").replace(" ", "_")
                user_name = ((tile_name or "tile_0001").strip()
                             .replace("/", "_").replace("\\", "_"))
                internal_name = _resolve_existing_single_tile_name(
                    city_clean, user_name, center=(lat0, lon0))
                cfg = _cfg_from_panel(city_clean, bbox, vals)
                tracker = _ProgressTracker(n_tiles=1)
                live_log = _LiveLogBuffer()
                holder = {}
                loading_text = (f"正在重生成树木与 seg/depth...\n"
                                f"目标 tile: {user_name}\n"
                                f"实际复用: {internal_name}")

                def _bg():
                    with contextlib.redirect_stdout(live_log), \
                         contextlib.redirect_stderr(live_log):
                        try:
                            runs = rerun_trees_only(
                                cfg, tile_names=[internal_name],
                                progress_cb=tracker.cb)
                            holder["tr"] = runs[0]
                        except Exception as e:
                            import traceback
                            holder["err"] = (
                                f"{type(e).__name__}: {e}\n"
                                + traceback.format_exc()[-1500:])

                th = threading.Thread(target=_bg, daemon=True); th.start()
                yield (gr.update(), gr.update(),
                       gr.update(value=None),
                       gr.update(value=None), gr.update(value=None),
                       gr.update(),
                       gr.update(value=None),
                       gr.update(value=None),
                       gr.update(value=None),
                       loading_text,
                       loading_text + "\n\n" + tracker.render(),
                       _tracker_progress_html(
                           tracker, ("F",), "正在重生成树木与 seg/depth..."))
                while th.is_alive():
                    _time.sleep(1.0)
                    yield (gr.update(), gr.update(),
                                    gr.update(value=None),
                           gr.update(value=None), gr.update(value=None),
                           gr.update(),
                                    gr.update(value=None),
                                    gr.update(value=None),
                           gr.update(value=None),
                           loading_text,
                           loading_text + "\n\n" + live_log.render(tracker.render()),
                           _tracker_progress_html(
                               tracker, ("F",), "正在重生成树木与 seg/depth..."))
                th.join()
                log_text = live_log.render(tracker.render())
                tile_dir = ROOT / "output" / city_clean / internal_name
                if "err" in holder:
                    banner = _print_result_block(
                        False, str(tile_dir), user_name, str(holder["err"]))
                    yield (gr.update(), gr.update(),
                                    gr.update(value=None),
                           gr.update(value=None), gr.update(value=None),
                           gr.update(),
                                    gr.update(value=None),
                                    gr.update(value=None),
                           gr.update(value=None),
                           f"FAILED: {holder['err']}",
                           banner + "\n\n" + log_text,
                           _progress_bar_html(100.0, "重生成树木失败"))
                    return
                tr = holder.get("tr")
                if tr is not None and tr.status.get("F") != "ok":
                    err = tr.errors.get("F", "Stage F failed")
                    banner = _print_result_block(False, str(tile_dir), user_name, err)
                    yield (gr.update(), gr.update(),
                                    gr.update(value=None),
                           gr.update(value=None), gr.update(value=None),
                           gr.update(),
                                    gr.update(value=None),
                                    gr.update(value=None),
                           gr.update(value=None),
                           f"FAILED: {err}",
                           banner + "\n\n" + log_text,
                           _progress_bar_html(100.0, "重生成树木失败"))
                    return
                preview_p = tile_dir / "3_blender_preview.png"
                seg_p = tile_dir / "4_seg.png"
                dep_p = tile_dir / "5_depth.png"
                poly_p = tile_dir / "6_polygon_outline.png"
                poly_json_p = tile_dir / "6_polygons.json"
                pointcloud_png_p = tile_dir / "7_pointcloud.png"
                pointcloud_ply_p = tile_dir / "7_pointcloud.ply"
                near_nadir_depth_p = tile_dir / "near-nadir-1" / "2_depth.png"
                blender_dir = tile_dir / "blender"
                blend_p = next(iter(sorted(blender_dir.glob("*.blend"))),
                               blender_dir / f"{internal_name}.blend")
                glb_p = blender_dir / f"{internal_name}.glb"
                if not glb_p.exists():
                    glb_p = next(iter(sorted(blender_dir.glob("*.glb"))),
                                 glb_p)
                preview_glb_p = blender_dir / f"{internal_name}_scene.glb"
                paths = (f".blend = {blend_p}\n.glb   = {glb_p}\n"
                         f"preview glb = {preview_glb_p}\n"
                         f"polygon json = {poly_json_p}\n"
                         f"pointcloud ply = {pointcloud_ply_p}")
                if internal_name != user_name:
                    paths += (f"\n(requested tile alias: {user_name}; "
                              f"reused existing tile: {internal_name})")
                banner = _print_result_block(True, str(tile_dir), user_name)
                yield (gr.update(),
                        gr.update(),
                    str(preview_p) if preview_p.exists() else None,
                        str(seg_p) if seg_p.exists() else None,
                        str(dep_p) if dep_p.exists() else None,
                    str(poly_p) if poly_p.exists() else gr.update(),
                    str(pointcloud_png_p) if pointcloud_png_p.exists() else gr.update(),
                    str(near_nadir_depth_p) if near_nadir_depth_p.exists() else gr.update(),
                    str(preview_glb_p if preview_glb_p.exists() else glb_p) if (preview_glb_p.exists() or glb_p.exists()) else None,
                    paths, banner + "\n\n" + log_text,
                    _progress_bar_html(100.0, "重生成树木完成"))
                        
                
            t1_run_btn.click(
                lambda: _t1_begin_locked_run("正在 Generate All..."),
                None,
                t1_lockables + [t1_busy, t1_progress],
                queue=False
            ).then(
                _t1_generate_all,
                [t1_state_center, t1_city, t1_tile_name] + t1_panel_components,
                [t1_osm, t1_rgb, t1_blender_preview, t1_seg, t1_depth,
                 t1_poly, t1_pointcloud, t1_near_nadir_depth,
                 t1_model, t1_blend_path, t1_log, t1_progress]
            ).then(
                _t1_finish_locked_run,
                None,
                t1_lockables + [t1_busy, t1_progress],
                queue=False
            )
            t1_rerun_trees_btn.click(
                lambda: _t1_begin_locked_run("正在重生成树木与 seg/depth..."),
                None,
                t1_lockables + [t1_busy, t1_progress],
                queue=False
            ).then(
                _t1_rerun_trees,
                [t1_state_center, t1_city, t1_tile_name] + t1_panel_components,
                [t1_osm, t1_rgb, t1_blender_preview, t1_seg, t1_depth,
                 t1_poly, t1_pointcloud, t1_near_nadir_depth,
                 t1_model, t1_blend_path, t1_log, t1_progress]
            ).then(
                _t1_finish_locked_run,
                None,
                t1_lockables + [t1_busy, t1_progress],
                queue=False
            )

        # ===================== Tab 2: multi-tile batch ===================== #
        with gr.Tab("🗺️ 多瓦片批量"):
            t2_corner1 = gr.State()
            t2_corner2 = gr.State()
            t2_ov_bbox = gr.State()

            with gr.Row():
                with gr.Column(scale=1):
                    t2_search = gr.Textbox(
                        label="🔍 搜索城市/地址",
                        placeholder="Omaha, Nebraska")
                    with gr.Row():
                        t2_search_btn = gr.Button("Search",
                                                    variant="primary")
                        t2_preset = gr.Dropdown(
                            list(PRESETS.keys()),
                            value=DEFAULT_PRESET, label="或选 preset")
                    with gr.Row():
                        t2_lat = gr.Number(value=init_lat, label="lat")
                        t2_lon = gr.Number(value=init_lon, label="lon")
                    t2_half_km = gr.Slider(
                        1.0, 30.0, value=6.0, step=0.5,
                        label="overview 半宽 (km)")
                    t2_city = gr.Textbox(
                        value="ui_batch_city",
                        label="city  (输出 output/<city>/)")
                    t2_msg = gr.Textbox(
                        label="状态", interactive=False, lines=3,
                        value="先点 NW，再点 SE。第三次点击重置。")
                with gr.Column(scale=2):
                    t2_map = gr.Image(
                        label="overview Esri (click NW then SE corner)",
                        height=600, interactive=False)

            with gr.Accordion("⚙️ 全部参数面板（共享 AutoPipelineConfig）",
                                open=True):
                t2_panel = _build_param_panel("t2")

            with gr.Row():
                t2_estimate_btn = gr.Button("📐 Estimate grid")
                t2_run_btn = gr.Button(
                    "🚀 Generate All tiles (= auto_pipeline CLI)",
                    variant="primary", size="lg")
                t2_rerun_trees_btn = gr.Button(
                    "🌳 Regenerate Trees Only for existing tiles",
                    variant="secondary", size="lg")
            t2_estimate = gr.Textbox(label="estimate",
                                       interactive=False, lines=4)
            t2_log = gr.Textbox(label="批量进度",
                                  interactive=False, lines=18)
            with gr.Row():
                t2_city_sat = gr.Image(
                    label="city_overview_satellite.png", height=320)
                t2_city_seg = gr.Image(
                    label="city_overview_seg.png", height=320)

            # ---- Tab 2 handlers ---- #
            def _t2_recenter(lat, lon, half_km, c1, c2):
                img, ov = _wide_overview(
                    float(lat), float(lon),
                    half_m=float(half_km) * 1000.0,
                    corner1=c1, corner2=c2)
                return img, list(ov)

            def _t2_preset(name, half_km):
                bb = PRESETS.get(name)
                if not bb:
                    return (gr.update(), gr.update(), gr.update(),
                            gr.update(), None, None,
                            "preset not found")
                cy = (bb[1] + bb[3]) / 2.0
                cx = (bb[0] + bb[2]) / 2.0
                img, ov = _wide_overview(
                    cy, cx, half_m=float(half_km) * 1000.0)
                return (cy, cx, img, list(ov), None, None,
                        f"preset '{name}'; click NW then SE.")

            def _t2_search(q, half_km):
                if not q or not q.strip():
                    return (gr.update(), gr.update(), gr.update(),
                            gr.update(), None, None, "enter a query.")
                from dataprep.geocode import geocode as _g
                res = _g(q.strip(), limit=3)
                if not res:
                    return (gr.update(), gr.update(), gr.update(),
                            gr.update(), None, None, f"no result.")
                top = res[0]
                img, ov = _wide_overview(
                    top["lat"], top["lon"],
                    half_m=float(half_km) * 1000.0)
                return (top["lat"], top["lon"], img, list(ov),
                        None, None,
                        f"found: {top['display_name']}; click NW then SE.")

            def _t2_click(ov_bbox, c1, c2, lat, lon, half_km, overlap,
                            evt: gr.SelectData):
                if evt is None or evt.index is None or ov_bbox is None:
                    return gr.update(), c1, c2, ""
                px, py = evt.index
                W, S, E, N = ov_bbox
                lon_c = W + (px / WIDE_MAP_PX) * (E - W)
                lat_c = N - (py / WIDE_MAP_PX) * (N - S)
                plans = None
                if c1 is None:
                    c1 = (lat_c, lon_c); c2 = None
                    msg = (f"NW set: ({lat_c:.5f}, {lon_c:.5f})\n"
                            f"now click SE corner.")
                elif c2 is None:
                    c2 = (lat_c, lon_c)
                    Nl = max(c1[0], c2[0]); Sl = min(c1[0], c2[0])
                    Wl = min(c1[1], c2[1]); El = max(c1[1], c2[1])
                    c1 = (Nl, Wl); c2 = (Sl, El)
                    try:
                        plans = plan_tiles(
                            (Wl, Sl, El, Nl), gsd=0.5,
                            size_px=1024, overlap=float(overlap))
                    except Exception as e:
                        return gr.update(), c1, c2, f"plan error: {e}"
                    nr, nc = grid_shape(plans)
                    msg = (f"NW=({Nl:.5f}, {Wl:.5f})  SE=({Sl:.5f}, {El:.5f})\n"
                            f"grid {nr}×{nc} = {len(plans)} tiles "
                            f"(overlap={overlap})")
                else:
                    c1 = (lat_c, lon_c); c2 = None
                    msg = (f"reset; NW set: ({lat_c:.5f}, {lon_c:.5f})\n"
                            f"now click SE corner.")
                img, _ov = _wide_overview(
                    float(lat), float(lon),
                    half_m=float(half_km) * 1000.0,
                    corner1=c1, corner2=c2, plans=plans)
                return img, c1, c2, msg

            t2_search_btn.click(
                _t2_search, [t2_search, t2_half_km],
                [t2_lat, t2_lon, t2_map, t2_ov_bbox,
                 t2_corner1, t2_corner2, t2_msg])
            t2_search.submit(
                _t2_search, [t2_search, t2_half_km],
                [t2_lat, t2_lon, t2_map, t2_ov_bbox,
                 t2_corner1, t2_corner2, t2_msg])
            t2_preset.change(
                _t2_preset, [t2_preset, t2_half_km],
                [t2_lat, t2_lon, t2_map, t2_ov_bbox,
                 t2_corner1, t2_corner2, t2_msg])
            t2_half_km.change(
                _t2_recenter,
                [t2_lat, t2_lon, t2_half_km, t2_corner1, t2_corner2],
                [t2_map, t2_ov_bbox])
            t2_map.select(
                _t2_click,
                [t2_ov_bbox, t2_corner1, t2_corner2,
                 t2_lat, t2_lon, t2_half_km, t2_panel["overlap"]],
                [t2_map, t2_corner1, t2_corner2, t2_msg])
            demo.load(
                _t2_recenter,
                [t2_lat, t2_lon, t2_half_km, t2_corner1, t2_corner2],
                [t2_map, t2_ov_bbox])

            def _t2_estimate(c1, c2, overlap):
                if not c1 or not c2:
                    return "pick NW + SE on the map first."
                Nl, Wl = c1; Sl, El = c2
                plans = plan_tiles(
                    (Wl, Sl, El, Nl), gsd=0.5, size_px=1024,
                    overlap=float(overlap))
                nr, nc = grid_shape(plans)
                return (f"grid = {nr} × {nc} = {len(plans)} tiles\n"
                        f"each tile = 512 m × 512 m (1024 px @ gsd=0.5)\n"
                        f"~disk = {len(plans) * 18} MB\n"
                        f"~wall: B+C+D ≈ {len(plans)*2}s, "
                        f"E+F ≈ {len(plans)*40}s")

            t2_estimate_btn.click(
                _t2_estimate,
                [t2_corner1, t2_corner2, t2_panel["overlap"]],
                [t2_estimate])

            # ---- Tab 2: Generate All ---- #
            t2_panel_keys = list(t2_panel.keys())
            t2_panel_components = [t2_panel[k] for k in t2_panel_keys]

            def _t2_generate_all(c1, c2, city, *panel_vals):
                if not c1 or not c2:
                    yield "pick NW + SE corners first.", None, None
                    return
                ui_t0 = _time.time()
                ui_started_at = _time.strftime("%Y-%m-%d %H:%M:%S")
                vals = dict(zip(t2_panel_keys, panel_vals))
                Nl, Wl = c1; Sl, El = c2
                bbox = (Wl, Sl, El, Nl)
                city_clean = (city or "ui_batch_city").strip().replace(
                    "/", "_").replace(" ", "_")
                cfg = _cfg_from_panel(city_clean, bbox, vals)
                plans = plan_tiles(bbox, gsd=0.5, size_px=1024,
                                    overlap=cfg.overlap)
                tracker = _ProgressTracker(n_tiles=len(plans))
                pipe = AutoPipeline(cfg, progress_cb=tracker.cb)
                holder = {}

                def _bg():
                    try:
                        holder["s"] = pipe.run()
                    except Exception as e:
                        import traceback
                        holder["err"] = (
                            f"{type(e).__name__}: {e}\n"
                            + traceback.format_exc()[-1500:])

                th = threading.Thread(target=_bg, daemon=True); th.start()
                yield (f"running... {len(plans)} tiles\n" + tracker.render(),
                        None, None)
                while th.is_alive():
                    _time.sleep(1.5)
                    yield (f"running... {len(plans)} tiles\n"
                            + tracker.render(), None, None)
                th.join()
                if "err" in holder:
                    yield (f"FAILED: {holder['err']}\n" + tracker.render(),
                            None, None)
                    return
                summary = holder.get("s", {})
                ui_duration = _time.time() - ui_t0
                city_dir = ROOT / "output" / city_clean
                try:
                    meta_dir = city_dir / "metadata"
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    ui_timing = {
                        "city": city_clean,
                        "source": "osm_app_tab2_generate_all",
                        "started_at": ui_started_at,
                        "ended_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_sec": round(float(ui_duration), 3),
                        "duration_min": round(float(ui_duration) / 60.0, 3),
                        "n_tiles": len(plans),
                        "bbox_wgs": list(bbox),
                        "auto_pipeline_timing_path": summary.get("timing_path"),
                    }
                    (meta_dir / "ui_run_timing_latest.json").write_text(
                        json.dumps(ui_timing, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                except Exception as e:
                    print(f"[ui] failed to write UI timing: {e}", flush=True)
                sat_p = city_dir / "city_overview_satellite.png"
                seg_p = city_dir / "city_overview_seg.png"
                if not sat_p.exists():
                    sat_p = city_dir / f"{city_clean}_rgb.png"
                if not seg_p.exists():
                    seg_p = city_dir / f"{city_clean}_seg.png"
                yield (f"DONE in {ui_duration:.1f}s "
                        f"({ui_duration / 60.0:.2f} min)\n"
                        f"{json.dumps(summary, indent=2)}\n\n"
                        + tracker.render(),
                        str(sat_p) if sat_p.exists() else None,
                        str(seg_p) if seg_p.exists() else None)

            t2_run_btn.click(
                _t2_generate_all,
                [t2_corner1, t2_corner2, t2_city] + t2_panel_components,
                [t2_log, t2_city_sat, t2_city_seg])

            def _t2_rerun_trees(c1, c2, city, *panel_vals):
                if not c1 or not c2:
                    yield "pick NW + SE corners first.", None, None
                    return
                ui_t0 = _time.time()
                ui_started_at = _time.strftime("%Y-%m-%d %H:%M:%S")
                vals = dict(zip(t2_panel_keys, panel_vals))
                Nl, Wl = c1; Sl, El = c2
                bbox = (Wl, Sl, El, Nl)
                city_clean = (city or "ui_batch_city").strip().replace(
                    "/", "_").replace(" ", "_")
                cfg = _cfg_from_panel(city_clean, bbox, vals)
                plans = plan_tiles(bbox, gsd=0.5, size_px=1024,
                                    overlap=cfg.overlap)
                tracker = _ProgressTracker(n_tiles=len(plans))
                holder = {}

                def _bg():
                    try:
                        holder["runs"] = rerun_trees_only(
                            cfg, progress_cb=tracker.cb)
                        aggregate_city(city_clean, holder["runs"])
                    except Exception as e:
                        import traceback
                        holder["err"] = (
                            f"{type(e).__name__}: {e}\n"
                            + traceback.format_exc()[-1500:])

                th = threading.Thread(target=_bg, daemon=True); th.start()
                yield (f"rerunning trees only... {len(plans)} tiles\n"
                        + tracker.render(), None, None)
                while th.is_alive():
                    _time.sleep(1.5)
                    yield (f"rerunning trees only... {len(plans)} tiles\n"
                            + tracker.render(), None, None)
                th.join()
                if "err" in holder:
                    yield (f"FAILED: {holder['err']}\n" + tracker.render(),
                            None, None)
                    return
                city_dir = ROOT / "output" / city_clean
                ui_duration = _time.time() - ui_t0
                try:
                    meta_dir = city_dir / "metadata"
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    ui_timing = {
                        "city": city_clean,
                        "source": "osm_app_tab2_rerun_trees",
                        "started_at": ui_started_at,
                        "ended_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_sec": round(float(ui_duration), 3),
                        "duration_min": round(float(ui_duration) / 60.0, 3),
                        "n_tiles": len(plans),
                        "bbox_wgs": list(bbox),
                    }
                    (meta_dir / "ui_tree_rerun_timing_latest.json").write_text(
                        json.dumps(ui_timing, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                except Exception as e:
                    print(f"[ui] failed to write UI tree timing: {e}", flush=True)
                sat_p = city_dir / "city_overview_satellite.png"
                seg_p = city_dir / "city_overview_seg.png"
                if not sat_p.exists():
                    sat_p = city_dir / f"{city_clean}_rgb.png"
                if not seg_p.exists():
                    seg_p = city_dir / f"{city_clean}_seg.png"
                yield ("DONE: trees regenerated from existing OSM/GLB "
                        f"artifacts in {ui_duration:.1f}s "
                        f"({ui_duration / 60.0:.2f} min).\n"
                        + tracker.render(),
                        str(sat_p) if sat_p.exists() else None,
                        str(seg_p) if seg_p.exists() else None)

            t2_rerun_trees_btn.click(
                _t2_rerun_trees,
                [t2_corner1, t2_corner2, t2_city] + t2_panel_components,
                [t2_log, t2_city_sat, t2_city_seg])

    return demo


def _port_is_available(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def _listening_pids_on_port(port: int) -> list[int]:
    import subprocess

    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=8)
    except Exception as exc:
        print(f"[ui] cannot inspect port {port}: {exc}")
        return []
    if result.returncode != 0:
        return []

    pids: set[int] = set()
    marker = f":{int(port)}"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_addr, state, pid_text = parts[1], parts[3].upper(), parts[-1]
        if state == "LISTENING" and local_addr.endswith(marker):
            try:
                pids.add(int(pid_text))
            except ValueError:
                pass
    return sorted(pids)


def _process_command_line(pid: int) -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["wmic", "process", "where", f"ProcessId={int(pid)}",
             "get", "CommandLine", "/value"],
            capture_output=True, text=True, timeout=8)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("CommandLine="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        pass

    try:
        ps_cmd = (
            f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\")"
            ".CommandLine")
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=8)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _release_default_gradio_port(host: str, port: int) -> bool:
    import subprocess

    if port != 8765 or _port_is_available(host, port):
        return True
    if _os.environ.get("OSM_APP_KEEP_PORT_PROCESS") == "1":
        print(f"[ui] port {port} is busy; cleanup disabled by OSM_APP_KEEP_PORT_PROCESS=1")
        return False

    killed_any = False
    for pid in _listening_pids_on_port(port):
        if pid == _os.getpid():
            continue
        cmd = _process_command_line(pid)
        cmd_l = cmd.lower()
        looks_like_this_app = (
            "python" in cmd_l
            and ("osm_app.py" in cmd_l or "gradio" in cmd_l)
        )
        if not looks_like_this_app:
            print(f"[ui] port {port} is busy by PID {pid}; not killing unknown process")
            continue
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True, text=True, timeout=8)
            print(f"[ui] released port {port}: killed stale PID {pid}")
            killed_any = True
        except Exception as exc:
            print(f"[ui] failed to kill PID {pid} on port {port}: {exc}")
    return killed_any and _port_is_available(host, port)


def _pick_gradio_port(host: str = "127.0.0.1") -> int:
    env_port = _os.environ.get("GRADIO_SERVER_PORT")
    candidates: list[int] = []
    if env_port:
        try:
            candidates.append(int(env_port))
        except ValueError:
            print(f"[ui] ignoring invalid GRADIO_SERVER_PORT={env_port!r}")
    candidates.extend([8765, 8766, 8767, 8768, 7860, 7861, 7862])

    seen: set[int] = set()
    for port in candidates:
        if port in seen:
            continue
        seen.add(port)
        if port == 8765:
            _release_default_gradio_port(host, port)
        if _port_is_available(host, port):
            if port != 8765:
                print(f"[ui] port 8765 is busy; using http://{host}:{port}")
            return port
    raise OSError(
        "No free Gradio port found in candidates: "
        + ", ".join(str(p) for p in seen)
    )


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=4)
    server_name = "127.0.0.1"
    server_port = _pick_gradio_port(server_name)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        inbrowser=False,
        allowed_paths=[str(ROOT), str(ROOT.parent)]
    )


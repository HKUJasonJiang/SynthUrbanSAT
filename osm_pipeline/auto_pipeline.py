"""Auto pipeline: stage-batched batch generation of N x M city tiles.

Reuses the per-tile helpers in :mod:`osm_app` (no duplication of OSM
fetching, KR2/KR3 invocation, canopy build, metadata writer) by
importing them directly. The CFG dict's ``paths.tile_root`` and
``paths.geojson_dir`` are temporarily redirected to ``output/<city>/``
so each batch run is self-contained.

Stage order (per-tile granularity progress):
  A. plan_tiles                                                 (single)
  B. satellite + osm basemap                                    (IO pool 8)
  C. OSM 6-class fetch + rasterize                              (IO pool 4)
  D. canopy NPZ + foliage geojson                               (IO pool 4)
  E. KR2 -> GLB                                                 (process pool)
  F. KR3 -> .blend (Blender headless)                           (serial)
  H. aggregate metadata + mosaic + tile_index.geojson           (single)

State persistence:
  output/<city>/.pipeline_state.json   - full per-tile status snapshot
  output/<city>/_failures.log           - append-only human-readable log
  output/<city>/_failures.json          - structured failures for retry
Resume scans the state file: any tile whose stage status is "ok" is
skipped for that stage on the next run.
"""
from __future__ import annotations

import builtins
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

import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence
import warnings

# Suppress annoying GIS-related warnings during run
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
warnings.filterwarnings("ignore", category=UserWarning, module="shapely")
warnings.filterwarnings("ignore", message=".*GeoSeries.notna.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*'mode' parameter is deprecated.*")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Reuse osm_app's per-tile machinery wholesale.
import osm_app as oa  # noqa: E402
from dataprep.geometry_utils import ensure_dir  # noqa: E402
from dataprep.tile_grid import (  # noqa: E402
    TilePlan, plan_tiles, grid_shape, grid_id_array, area_bbox_union,
)


STAGES = ("B", "C", "D", "E", "F", "H")

STAGE_NAMES = {
    "B": "satellite + osm basemap",
    "C": "OSM 6-class fetch",
    "D": "canopy height",
    "E": "KR2 geometry build",
    "F": "KR3 Blender assemble",
    "H": "aggregate city",
}

# Module-level handle to the running pipeline so per-tile worker
# functions (which take only ``tr``) can access shared city-level
# resources (union OSM, global mercator grid, city seg array) without
# threading them through every signature.
_ACTIVE_PIPELINE: "AutoPipeline | None" = None


# --------------------------------------------------------------------- #
# Per-tile state (in-memory). Heavy fields are populated on demand and
# dropped between stages once persisted to disk.
# --------------------------------------------------------------------- #
@dataclass
class TileRun:
    plan: TilePlan
    state: dict = field(default_factory=dict)         # osm_app-shaped state
    status: dict = field(default_factory=dict)        # per-stage str
    errors: dict = field(default_factory=dict)        # per-stage str
    timings: dict = field(default_factory=dict)       # per-stage float seconds
    glb_path: str | None = None
    blend_path: str | None = None
    canopy_npz: str | None = None
    top_png: str | None = None
    iso_png: str | None = None

    def stage(self, s: str) -> str:
        return self.status.get(s, "pending")

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.as_dict(),
            "status": dict(self.status),
            "errors": dict(self.errors),
            "timings": dict(self.timings),
            "glb_path": self.glb_path,
            "blend_path": self.blend_path,
            "canopy_npz": self.canopy_npz,
        }


# --------------------------------------------------------------------- #
# Path redirection: route all per-tile artifacts to output/<city>/
# --------------------------------------------------------------------- #
@contextmanager
def _redirect_cfg_paths(city: str):
    """Mutate osm_app.CFG.paths in-place; restore on exit."""
    orig = dict(oa.CFG["paths"])
    city_root = (ROOT / "output" / city).as_posix()
    oa.CFG["paths"]["tile_root"] = f"output/{city}"
    oa.CFG["paths"]["geojson_dir"] = f"cache/geojson/{city}"
    oa.CFG["paths"]["fig_dir"] = f"cache/fig/{city}"
    ensure_dir(ROOT / "cache" / "geojson" / city)
    ensure_dir(ROOT / "cache" / "fig" / city)
    try:
        yield city_root
    finally:
        oa.CFG["paths"].clear()
        oa.CFG["paths"].update(orig)


# --------------------------------------------------------------------- #
# Resume helpers
# --------------------------------------------------------------------- #
def _detect_existing_outputs(city: str, plan: TilePlan) -> dict:
    """Inspect output/<city>/<tile>/ and infer per-stage completion.

    Returns a dict {stage: "ok"|"pending"} based on file presence.
    Does NOT mark a stage failed — only confirms ok.
    """
    base = ROOT / "output" / city / plan.name
    out: dict[str, str] = {}
    sat = base / "2_rgb.png" if (base / "2_rgb.png").exists() else base / "satellite_image.png"
    osm_b = base / "1_osm.png" if (base / "1_osm.png").exists() else base / "osm_basemap.png"
    seg = base / "4_seg.png" if (base / "4_seg.png").exists() else (base / "3_seg.png" if (base / "3_seg.png").exists() else base / "seg_6class.png")
    glb = base / "blender" / f"{plan.name}.glb"
    blend = base / "blender" / f"{plan.name}.blend"
    prompt = base / f"{plan.name}_prompt.json"
    canopy = ROOT / "cache" / "canopy" / f"{plan.name}.npz"
    canopy_geojson = (ROOT / "cache" / "geojson" / city
                       / f"{plan.name}_foliage_canopy.geojson")
    if sat.exists() and osm_b.exists():
        out["B"] = "ok"
    if seg.exists():
        out["C"] = "ok"
    # Stage D is only complete if BOTH the canopy NPZ and the
    # foliage_canopy.geojson are on disk. The geojson is what KR2/KR3
    # actually consume to scatter trees.
    if canopy.exists() and canopy_geojson.exists():
        out["D"] = "ok"
    # E (KR2) and F (KR3) consume the canopy geojson; if it's missing
    # we must re-run them even if the GLB / blend already exist.
    d_ok = out.get("D") == "ok"
    if glb.exists() and d_ok:
        out["E"] = "ok"
    if blend.exists() and d_ok:
        out["F"] = "ok"
    if prompt.exists():
        out["G"] = "ok"
    return out


def _hydrate_state_from_disk(city: str, plan: TilePlan,
                             tr: TileRun) -> None:
    """When resuming, re-load the cached PNGs/seg back into ``tr.state``
    so later stages still have what they need.

    We re-read PNGs (cheap) and recompute things that aren't trivially
    persisted (cls_wgs is dropped — KR2 will fetch from disk geojson).
    """
    base = ROOT / "output" / city / plan.name
    sat = base / "2_rgb.png" if (base / "2_rgb.png").exists() else base / "satellite_image.png"
    osm_b = base / "1_osm.png" if (base / "1_osm.png").exists() else base / "osm_basemap.png"
    seg = base / "4_seg.png" if (base / "4_seg.png").exists() else (base / "3_seg.png" if (base / "3_seg.png").exists() else base / "seg_6class.png")
    glb = base / "blender" / f"{plan.name}.glb"
    blend = base / "blender" / f"{plan.name}.blend"
    canopy = ROOT / "cache" / "canopy" / f"{plan.name}.npz"

    # Compute UTM/center fields needed by canopy + metadata.
    bbox_wgs, utm, (cx_utm, cy_utm) = oa.bbox_from_center(
        plan.lat, plan.lon, 0.5, 1024)
    tr.state.update({
        "lat": plan.lat, "lon": plan.lon,
        "gsd": 0.5, "size": 1024,
        "bbox_wgs": list(plan.bbox_wgs),
        "utm": utm.to_string(),
        "cx_utm": cx_utm, "cy_utm": cy_utm,
    })
    if sat.exists():
        try:
            tr.state["sat_rgb"] = Image.open(sat).convert("RGB").copy()
        except Exception:  # noqa: BLE001
            pass
    if osm_b.exists():
        try:
            tr.state["osm_basemap"] = Image.open(osm_b).convert("RGB").copy()
        except Exception:  # noqa: BLE001
            pass
    if glb.exists():
        tr.glb_path = str(glb)
    if blend.exists():
        tr.blend_path = str(blend)
    if canopy.exists():
        tr.canopy_npz = str(canopy)


def _find_existing_tile_metadata(tile_dir: Path) -> dict:
    meta_dir = tile_dir / "metadata"
    if not meta_dir.exists():
        return {}
    candidates = [p for p in sorted(meta_dir.glob("*.json"))
                  if not p.name.endswith(".meta.json")]
    for p in candidates:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return {}


def _copy_first_matching(src_dir: Path, pattern: str, dst: Path) -> None:
    if dst.exists() or not src_dir.exists():
        return
    matches = sorted(src_dir.glob(pattern))
    if not matches:
        return
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(matches[0], dst)


def _ensure_tree_rerun_aliases(city: str, tile_name: str) -> None:
    """Make renamed single-tile folders usable by KR3.

    The UI can rename ``tile_0001`` to a user-supplied folder name after
    generation. KR3 expects the GLB and metadata sidecars to share the folder
    name, so create lightweight copies when only the old internal names exist.
    """
    tile_dir = ROOT / "output" / city / tile_name
    blender_dir = tile_dir / "blender"
    meta_dir = tile_dir / "metadata"
    _copy_first_matching(blender_dir, "*.glb", blender_dir / f"{tile_name}.glb")
    _copy_first_matching(meta_dir, "*.json", meta_dir / f"{tile_name}.json")
    _copy_first_matching(meta_dir, "*_osm_buildings.geojson",
                         meta_dir / f"{tile_name}_osm_buildings.geojson")


def _load_class_geoms_from_geojson(city: str, tile_name: str) -> dict:
    """Load cached KR1 class GeoJSONs without touching Overpass/OSM."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from dataprep.osm_tags import CLASS_IDS

    geo_dir = ROOT / "cache" / "geojson" / city
    if not geo_dir.exists():
        return {}

    classes = list(CLASS_IDS.keys())
    prefixes = [tile_name]
    if not any((geo_dir / f"{tile_name}_{cls}.geojson").exists()
               for cls in classes):
        found = set()
        for p in geo_dir.glob("*.geojson"):
            for cls in classes:
                suffix = f"_{cls}.geojson"
                if p.name.endswith(suffix):
                    found.add(p.name[:-len(suffix)])
        if len(found) == 1:
            prefixes.append(next(iter(found)))

    out = {}
    for prefix in prefixes:
        for cls in classes:
            if cls in out:
                continue
            path = geo_dir / f"{prefix}_{cls}.geojson"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                geoms = [shape(f["geometry"]) for f in data.get("features", [])
                         if f.get("geometry")]
                if geoms:
                    g = unary_union(geoms)
                    out[cls] = g if not g.is_empty else None
            except Exception as e:  # noqa: BLE001
                print(f"[auto] cached geojson load failed for {path.name}: {e}")
        if out:
            break
    return out


def _tile_plan_from_existing_output(cfg: "AutoPipelineConfig",
                                    tile_name: str) -> TilePlan:
    tile_dir = ROOT / "output" / cfg.city / tile_name
    meta = _find_existing_tile_metadata(tile_dir)
    if meta:
        bbox = tuple(float(x) for x in meta.get("bbox_wgs84") or [])
        center = meta.get("center_wgs84") or {}
        if len(bbox) == 4:
            lat = float(center.get("lat", 0.5 * (bbox[1] + bbox[3])))
            lon = float(center.get("lon", 0.5 * (bbox[0] + bbox[2])))
            return TilePlan(tile_name, 0, 0, lat, lon, bbox)

    plans = plan_tiles(cfg.area_bbox_wgs, gsd=cfg.gsd,
                       size_px=cfg.size_px, overlap=cfg.overlap)
    if not plans:
        raise RuntimeError("planner produced 0 tiles")
    p = plans[0]
    return TilePlan(tile_name, p.row, p.col, p.lat, p.lon, p.bbox_wgs)


def _hydrate_tree_rerun_state(city: str, tr: TileRun) -> None:
    _hydrate_state_from_disk(city, tr.plan, tr)
    tile_dir = ROOT / "output" / city / tr.plan.name
    meta = _find_existing_tile_metadata(tile_dir)
    if meta:
        bbox = meta.get("bbox_wgs84")
        center = meta.get("center_wgs84") or {}
        if bbox and len(bbox) == 4:
            tr.state["bbox_wgs"] = [float(x) for x in bbox]
        if "lat" in center and "lon" in center:
            tr.state["lat"] = float(center["lat"])
            tr.state["lon"] = float(center["lon"])
    tr.state["cls_wgs"] = _load_class_geoms_from_geojson(city, tr.plan.name)
    if not tr.canopy_npz:
        canopy_dir = ROOT / "cache" / "canopy"
        direct = canopy_dir / f"{tr.plan.name}.npz"
        if direct.exists():
            tr.canopy_npz = str(direct)
        else:
            matches = sorted(canopy_dir.glob("*.npz"))
            if len(matches) == 1:
                tr.canopy_npz = str(matches[0])


# --------------------------------------------------------------------- #
# Per-stage workers
# --------------------------------------------------------------------- #
def _stage_b_one(tr: TileRun) -> TileRun:
    """Stage B: fetch satellite + osm basemap into tr.state.

    Uses ``tr.plan.bbox_wgs`` directly (already metric-exact from
    ``plan_tiles``) instead of going through ``oa.on_preview_satellite``
    which re-derives the bbox via ``bbox_from_center`` with a 1.05x
    padding factor — that padding makes adjacent tiles overlap by ~5%
    and shows up as visible seams in the city overview mosaic.
    """
    from dataprep.satellite import (
        satellite_image_for_bbox, osm_map_image_for_bbox)
    from dataprep.geometry_utils import (
        make_transformer, reproject_geom, utm_crs_for_bbox)
    from shapely.geometry import Point as _P
    t0 = time.time()
    try:
        bbox = tuple(float(x) for x in tr.plan.bbox_wgs)
        gsd, size = 0.5, 1024
        try:
            sat = satellite_image_for_bbox(bbox, out_size=size)
        except Exception as exc:  # noqa: BLE001
            print(f"[auto] sat fetch {tr.plan.name}: {exc}")
            sat = Image.new("RGB", (size, size), (40, 40, 40))
        try:
            osm_b = osm_map_image_for_bbox(bbox, out_size=size)
        except Exception as exc:  # noqa: BLE001
            print(f"[auto] osm map fetch {tr.plan.name}: {exc}")
            osm_b = Image.new("RGB", (size, size), (220, 220, 215))
        # UTM derived from the *tile* bbox so cx/cy match what
        # rasterize_seg / KR2 expect (centered on this tile).
        utm = utm_crs_for_bbox(bbox)
        fwd = make_transformer("EPSG:4326", utm)
        sw = reproject_geom(_P(bbox[0], bbox[1]), fwd)
        ne = reproject_geom(_P(bbox[2], bbox[3]), fwd)
        cx_utm = (sw.x + ne.x) / 2.0
        cy_utm = (sw.y + ne.y) / 2.0
        tr.state.update({
            "lat": tr.plan.lat, "lon": tr.plan.lon,
            "gsd": gsd, "size": size,
            "bbox_wgs": list(bbox), "utm": utm.to_string(),
            "cx_utm": cx_utm, "cy_utm": cy_utm, "ratios": None,
            "sat_rgb": sat, "osm_basemap": osm_b, "seg_osm": None,
        })
        tr.status["B"] = "ok"
    except Exception as e:  # noqa: BLE001
        tr.status["B"] = "failed"
        tr.errors["B"] = f"{type(e).__name__}: {e}"
    finally:
        tr.timings["B"] = time.time() - t0
    return tr


def _stage_c_one(tr: TileRun) -> TileRun:
    """Stage C: OSM 6-class extract + rasterize. Updates tr.state with
    seg_osm + cls_wgs + ratios (so KR2 doesn't re-fetch in stage E).

    When a city-level union OSM fetch (``_ACTIVE_PIPELINE.city_cls_wgs``)
    is available, this stage just clips the union geometries to the
    tile bbox instead of issuing its own Overpass POST. This guarantees
    that features crossing tile boundaries are topologically identical
    on both sides — a prerequisite for seamless seg PNG mosaicing.
    """
    t0 = time.time()
    try:
        # on_fetch_osm wraps fetch_classes_wgs + rasterize + ratio bars.
        # We don't need the UI artifacts so call the inner functions.
        from dataprep.raster_utils import (
            class_geoms_to_local, colorize_seg, rasterize_seg,
        )
        from dataprep.osm_tags import CLASS_IDS

        bbox = tuple(tr.state["bbox_wgs"])
        def _has_non_ground_feature(classes: dict) -> bool:
            for cls in ("building", "water", "grass", "foliage", "road"):
                geom = classes.get(cls)
                if geom is not None and not getattr(geom, "is_empty", True):
                    return True
            return False

        cls_wgs = {}
        pipe = _ACTIVE_PIPELINE
        if pipe is not None and hasattr(pipe, "cfg"):
            cached = _load_class_geoms_from_geojson(
                pipe.cfg.city, tr.plan.name)
            if _has_non_ground_feature(cached):
                cls_wgs = cached
                print(f"[auto] {tr.plan.name} loaded cached KR1 "
                      "class geojson", flush=True)

        # ---- City-level union fetch path (preferred) ---- #
        city_cls_wgs = (pipe.city_cls_wgs if pipe is not None else None)
        if not cls_wgs and city_cls_wgs:
            from dataprep.city_grid import clip_city_cls_to_tile
            cls_wgs = clip_city_cls_to_tile(city_cls_wgs, bbox)
            if not _has_non_ground_feature(cls_wgs):
                print(f"[auto] {tr.plan.name} city OSM clip has no usable "
                      "features; falling back to per-tile Overpass",
                      flush=True)
                cls_wgs = oa.fetch_classes_wgs(bbox)
        elif not cls_wgs:
            # Fallback: per-tile Overpass fetch (legacy behaviour)
            cls_wgs = oa.fetch_classes_wgs(bbox)
        if not _has_non_ground_feature(cls_wgs):
            raise RuntimeError("OSM fetch returned no building/water/grass/"
                               "foliage/road features; refusing to generate "
                               "ground-only semantic output")
        cls_utm, _ = class_geoms_to_local(cls_wgs, bbox)
        seg = rasterize_seg(cls_utm, tr.state["cx_utm"], tr.state["cy_utm"],
                            tr.state["gsd"], tr.state["size"])
        counts = {c: int((seg == CLASS_IDS[c]).sum()) for c in CLASS_IDS}
        total = seg.size
        tr.state["seg_osm"] = seg
        tr.state["cls_wgs"] = cls_wgs
        tr.state["ratios"] = {c: counts[c] / total for c in CLASS_IDS}
        tr.status["C"] = "ok"
    except Exception as e:  # noqa: BLE001
        tr.status["C"] = "failed"
        tr.errors["C"] = f"{type(e).__name__}: {e}"
    finally:
        tr.timings["C"] = time.time() - t0
    return tr


def _stage_d_one(tr: TileRun, canopy_source: str = "eth_10m",
                  target_foliage_ratio: float | None = None) -> TileRun:
    """Stage D: canopy NPZ + foliage paint into seg_osm + foliage geojson.

    Writes ``output/<city>/_geojson/<tile>_foliage_canopy.geojson`` so KR2
    will union the canopy mask into the foliage substrate (KR3 then
    scatters trees onto that substrate).  Also unions the canopy mask
    into ``cls_wgs['foliage']`` so the mercator-aligned per-tile seg
    PNG includes canopy.
    """
    t0 = time.time()
    try:
        npz_path, msg, _ = oa._compute_canopy_for_state(
            tr.state, canopy_source, None, tr.plan.name,
            target_ratio=target_foliage_ratio)
        tr.canopy_npz = str(npz_path) if npz_path else None
        thr = 2.0
        if "thr=" in (msg or ""):
            try:
                thr = float(msg.split("thr=")[1].split("m")[0])
            except Exception:  # noqa: BLE001
                pass
        # ---- Defensive: ensure canopy foliage geojson exists ---- #
        # _compute_canopy_for_state writes this, but make absolutely
        # sure (KR3 needs it for tree scatter).
        canopy_geojson = None
        if npz_path and Path(npz_path).exists():
            from dataprep.canopy_height import canopy_npz_to_foliage_geojson
            from dataprep.geometry_utils import ensure_dir as _ed
            geojson_dir = _ed(ROOT / oa.CFG["paths"]["geojson_dir"])
            canopy_geojson = geojson_dir / f"{tr.plan.name}_foliage_canopy.geojson"
            if not canopy_geojson.exists():
                try:
                    n_polys, used_thr, achieved = canopy_npz_to_foliage_geojson(
                        npz_path, tuple(tr.state["bbox_wgs"]), canopy_geojson,
                        height_threshold_m=thr,
                        target_ratio=(float(target_foliage_ratio)
                                       if target_foliage_ratio
                                       and float(target_foliage_ratio) > 0
                                       else None),
                    )
                    print(f"[auto] {tr.plan.name} canopy geojson "
                          f"polys={n_polys} thr={used_thr:.1f}m "
                          f"ratio={achieved:.3f}")
                except Exception as e:  # noqa: BLE001
                    print(f"[auto] {tr.plan.name} canopy vectorize failed: {e}")
                    canopy_geojson = None
        # ---- Paint canopy >=2m onto seg (mirrors osm_app.on_make_veg_mask) ---- #
        # Note: this only updates ratios; the on-disk seg_6class.png is
        # later re-rendered in mercator from cls_wgs (which we DON'T
        # union the canopy into, since the satellite already shows tree
        # cover and double-painting it makes the seg too green).
        seg = tr.state.get("seg_osm")
        if seg is not None and npz_path and Path(npz_path).exists():
            from dataprep.raster_utils import compose_seg_with_foliage_mask
            z = np.load(npz_path)
            key = "heights" if "heights" in z.files else (
                "canopy_m" if "canopy_m" in z.files else z.files[0])
            h = np.asarray(z[key], dtype=float)
            hh = np.array(Image.fromarray(h).resize(
                (seg.shape[1], seg.shape[0]),
                resample=Image.NEAREST))
            seg2 = compose_seg_with_foliage_mask(seg, hh >= thr)
            tr.state["seg_osm"] = seg2
            from dataprep.osm_tags import CLASS_IDS
            counts = {c: int((seg2 == CLASS_IDS[c]).sum()) for c in CLASS_IDS}
            total = seg2.size
            tr.state["ratios"] = {c: counts[c] / total for c in CLASS_IDS}
        # NOTE: we deliberately do NOT union the canopy polygons into
        # cls_wgs['foliage'] here. KR2 / KR3 read the canopy substrate
        # directly from the *_foliage_canopy.geojson file we just wrote,
        # so they still scatter trees on canopy. The per-tile
        # seg_6class.png keeps OSM-only foliage so the green mask
        # doesn't dominate the rendered seg.
        tr.status["D"] = "ok"
    except Exception as e:  # noqa: BLE001
        tr.status["D"] = "failed"
        tr.errors["D"] = f"{type(e).__name__}: {e}"
    finally:
        tr.timings["D"] = time.time() - t0
    return tr


def _stage_e_one(tr: TileRun, height_dist: str, height_seed: int,
                  height_min: float, height_max: float) -> TileRun:
    """Stage E: KR1 (write geojsons) + KR2 (build GLB)."""
    t0 = time.time()
    try:
        import importlib.util as ilu
        from dataprep.geometry_utils import ensure_dir as _ed

        bbox = tuple(tr.state["bbox_wgs"])
        geojson_dir = _ed(ROOT / oa.CFG["paths"]["geojson_dir"])
        fig_dir = _ed(ROOT / oa.CFG["paths"]["fig_dir"])

        # KR1: write per-class geojsons (uses cached cls_wgs from stage C).
        oa.kr1.process_city({"name": tr.plan.name, "bbox": list(bbox)},
                            oa.CFG, geojson_dir, fig_dir,
                            class_geoms_wgs=tr.state.get("cls_wgs"))

        # KR2: build GLB.
        spec = ilu.spec_from_file_location(
            "kr2", ROOT / "scripts" / "2_build_geometry.py")
        kr2 = ilu.module_from_spec(spec)
        spec.loader.exec_module(kr2)  # type: ignore
        kr2.process_city({"name": tr.plan.name, "bbox": list(bbox)},
                         oa.CFG,
                         height_dist=str(height_dist),
                         height_seed=int(height_seed),
                         height_min=float(height_min),
                         height_max=float(height_max))

        tile_root = oa.CFG["paths"]["tile_root"]
        glb = ROOT / tile_root / tr.plan.name / "blender" / f"{tr.plan.name}.glb"
        if not glb.exists():
            raise RuntimeError(f"KR2 produced no GLB at {glb}")
        tr.glb_path = str(glb)
        tr.status["E"] = "ok"
    except Exception as e:  # noqa: BLE001
        tr.status["E"] = "failed"
        tr.errors["E"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"
    finally:
        tr.timings["E"] = time.time() - t0
    return tr


def _city_config_path(city: str) -> Path:
    """Write a yaml override for this city with patched paths and return it.

    Needed because KR3 runs in a Blender subprocess that re-reads the
    YAML — our in-memory CFG mutation doesn't reach it. Idempotent:
    rewrites every call so any CFG path change is reflected.

    Important: KR3 derives the project root via
    ``Path(args.config).resolve().parent.parent``, so the file MUST live
    two levels under ROOT (we use ``configs/_city_<city>.yaml``) — not
    under ``output/<city>/`` which would make root point at ``output/``
    and break ``import dataprep`` plus ``tile_root`` joining.
    """
    import yaml as _yaml
    src = ROOT / "configs" / "default.yaml"
    cfg_text = src.read_text(encoding="utf-8")
    cfg = _yaml.safe_load(cfg_text)
    cfg.setdefault("paths", {})
    cfg["paths"]["tile_root"] = f"output/{city}"
    cfg["paths"]["geojson_dir"] = f"output/{city}/_geojson"
    cfg["paths"]["fig_dir"] = f"output/{city}/_fig"
    out = ensure_dir(ROOT / "configs") / f"_city_{city}.yaml"
    out.write_text(_yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    return out


def _call_kr3(city: str, tile_name: str, scatter_seed: int,
               tree_density: float,
               *, species: list | None = None,
               tree_h_dist: str = "lognormal",
               tree_h_seed: int = 11,
               tree_h_min: float = 6.0,
               tree_h_max: float = 20.0,
               canopy_npz: str | None = None,
               cluster_size_min: int = 10,
               cluster_size_max: int = 20,
               cluster_disk_radius_min: float = 4.0,
               cluster_disk_radius_max: float = 10.0,
               cluster_disk_aspect: float = 0.65,
               cluster_size_dist: str = "uniform",
               cluster_size_low_frac: float = 0.7,
               tree_height_low_frac: float = 0.65,
               cluster_overlap_factor: float = 0.45,
               cluster_min_keep_ratio: float = 0.6,
               cluster_min_size_abs: int = 0,
               scatter_mode: str = "canopy_prob",               allow_non_foliage: bool = True,
               enable_street_trees: bool = False,
               procedural_augment_ratio: float = 0.0,
               canopy_prob_scale: float = 1.0,
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
               uniform_tree_scale: bool = False,
               render_depth: bool = True):
    """Same shape as ``osm_app._run_kr3`` but uses the city-patched yaml.

    Returns (ok, log_tail, top_png, iso_png, depth_exr, depth_png,
    pointcloud_png, pointcloud_ply).
    """
    import subprocess
    import tempfile

    tile_root = oa.CFG["paths"]["tile_root"]
    tile_dir = ROOT / tile_root / tile_name
    glb = tile_dir / "blender" / f"{tile_name}.glb"
    if not glb.exists():
        return False, f"GLB missing at {glb}", None, None, None, None, None, None
    if not Path(oa.BLENDER_EXE).exists():
        return (False,
                f"Blender not found at {oa.BLENDER_EXE}. "
                f"Set BLENDER_EXE env var.", None, None, None, None, None, None)

    cfg_path = _city_config_path(city)
    tmp_dir = Path(tempfile.mkdtemp(prefix="auto_preview_"))
    top_png = tmp_dir / f"{tile_name}_topview.png"
    iso_png = tmp_dir / f"{tile_name}_iso.png"
    depth_exr = tmp_dir / f"{tile_name}_ndsm.exr" if render_depth else None
    depth_png = tmp_dir / f"{tile_name}_ndsm.png" if render_depth else None
    pointcloud_png = tmp_dir / f"{tile_name}_pointcloud.png"
    pointcloud_ply = tmp_dir / f"{tile_name}_pointcloud.ply"
    preview_glb = tile_dir / "blender" / f"{tile_name}_scene.glb"

    cmd = [oa.BLENDER_EXE, "--background",
           "--python", str(ROOT / "scripts" / "3_blender_assemble.py"),
           "--", "--config", str(cfg_path),
           "--city", tile_name,
           "--scatter-seed", str(int(scatter_seed)),
           "--tree-density", str(float(tree_density)),
           "--foliage-density", str(float(tree_density)),
           "--tree-height-dist", str(tree_h_dist),
           "--tree-height-seed", str(int(tree_h_seed)),
           "--tree-height-min", str(float(tree_h_min)),
           "--tree-height-max", str(float(tree_h_max)),
           "--scatter-mode", str(scatter_mode),
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
           "--gn-z-stretch-max-at-1", str(float(gn_z_stretch_max_at_1)),
           "--pointcloud-png", str(pointcloud_png),
           "--pointcloud-ply", str(pointcloud_ply),
           "--pointcloud-count", "50000",
           "--pointcloud-voxel-size", "0.5",
           "--preview-glb", str(preview_glb)]
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
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    # Dump full KR3 stdout/stderr to per-tile log for debugging.
    try:
        log_dir = tile_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "kr3_stdout.log").write_text(r.stdout or "", encoding="utf-8")
        (log_dir / "kr3_stderr.log").write_text(r.stderr or "", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if r.returncode != 0:
        return (False,
                f"rc={r.returncode}\nstdout: {r.stdout[-400:]}\n"
                f"stderr: {r.stderr[-400:]}",
                None, None, None, None, None, None)
    return (True, "ok",
            str(top_png) if top_png.exists() else None,
            str(iso_png) if iso_png.exists() else None,
            str(depth_exr) if (depth_exr and depth_exr.exists()) else None,
            str(depth_png) if (depth_png and depth_png.exists()) else None,
            str(pointcloud_png) if pointcloud_png.exists() else None,
            str(pointcloud_ply) if pointcloud_ply.exists() else None)


def _stage_f_one(tr: TileRun, scatter_seed: int, tree_density: float,
                  tree_species: list | None,
                  tree_h_dist: str, tree_h_seed: int,
                  tree_h_min: float, tree_h_max: float,
                  city: str = "",
                  cluster_size_min: int = 10,
                  cluster_size_max: int = 20,
                  cluster_disk_radius_min: float = 4.0,
                  cluster_disk_radius_max: float = 10.0,
                  cluster_disk_aspect: float = 0.65,
                  cluster_size_dist: str = "uniform",
                  cluster_size_low_frac: float = 0.7,
                  tree_height_low_frac: float = 0.65,
                  cluster_overlap_factor: float = 0.45,
                  cluster_min_keep_ratio: float = 0.6,
                  cluster_min_size_abs: int = 0,
                  scatter_mode: str = "canopy_prob",
                  allow_non_foliage: bool = True,
                  enable_street_trees: bool = False,
                  procedural_augment_ratio: float = 0.0,
                  canopy_prob_scale: float = 1.0,
                  uniform_tree_scale: bool = False,
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
                  gn_z_stretch_max_at_1: float = 2.40) -> TileRun:
    """Stage F: KR3 Blender assemble (serial)."""
    t0 = time.time()
    try:
        ok, log, top_png, iso_png, depth_exr, depth_png, pointcloud_png, pointcloud_ply = _call_kr3(
            city, tr.plan.name, int(scatter_seed), float(tree_density),
            species=tree_species,
            tree_h_dist=str(tree_h_dist), tree_h_seed=int(tree_h_seed),
            tree_h_min=float(tree_h_min), tree_h_max=float(tree_h_max),
            canopy_npz=tr.canopy_npz,
            cluster_size_min=int(cluster_size_min),
            cluster_size_max=int(cluster_size_max),
            cluster_disk_radius_min=float(cluster_disk_radius_min),
            cluster_disk_radius_max=float(cluster_disk_radius_max),
            cluster_disk_aspect=float(cluster_disk_aspect),
            cluster_size_dist=str(cluster_size_dist),
            cluster_size_low_frac=float(cluster_size_low_frac),
            tree_height_low_frac=float(tree_height_low_frac),
            cluster_overlap_factor=float(cluster_overlap_factor),
            cluster_min_keep_ratio=float(cluster_min_keep_ratio),
            cluster_min_size_abs=int(cluster_min_size_abs),
            scatter_mode=str(scatter_mode),
            allow_non_foliage=bool(allow_non_foliage),
            enable_street_trees=bool(enable_street_trees),
            procedural_augment_ratio=float(procedural_augment_ratio),
            canopy_prob_scale=float(canopy_prob_scale),
            uniform_tree_scale=bool(uniform_tree_scale),
            topdown_tree_xy_scale=float(topdown_tree_xy_scale),
            gn_tree_amount=float(gn_tree_amount),
            gn_safe_building=float(gn_safe_building),
            gn_safe_road=float(gn_safe_road),
            gn_safe_water=float(gn_safe_water),
            gn_noise_scale=float(gn_noise_scale),
            gn_min_distance=float(gn_min_distance),
            gn_xy_stretch=float(gn_xy_stretch),
            gn_z_stretch=float(gn_z_stretch),
            gn_xy_stretch_min_at_0=float(gn_xy_stretch_min_at_0),
            gn_xy_stretch_min_at_1=float(gn_xy_stretch_min_at_1),
            gn_xy_stretch_max_at_0=float(gn_xy_stretch_max_at_0),
            gn_xy_stretch_max_at_1=float(gn_xy_stretch_max_at_1),
            gn_z_stretch_min_at_0=float(gn_z_stretch_min_at_0),
            gn_z_stretch_min_at_1=float(gn_z_stretch_min_at_1),
            gn_z_stretch_max_at_0=float(gn_z_stretch_max_at_0),
            gn_z_stretch_max_at_1=float(gn_z_stretch_max_at_1),
        )
        if not ok:
            raise RuntimeError(f"KR3 failed: {log}")
        tile_root = oa.CFG["paths"]["tile_root"]
        blend = ROOT / tile_root / tr.plan.name / "blender" / f"{tr.plan.name}.blend"
        tr.blend_path = str(blend) if blend.exists() else None
        tr.top_png = top_png
        tr.iso_png = iso_png
        # Persist 4 PNGs + tile metadata json now (uses tr.state).
        tile_dir = ROOT / tile_root / tr.plan.name
        oa._persist_tile_artifacts(
            tr.state, tr.plan.name, tile_dir, top_png,
            depth_exr=depth_exr, depth_png=depth_png)
        # Overwrite seg_6class.png with a mercator-aligned render first.
        try:
            _render_tile_seg_mercator(tr, tile_dir / "seg_6class.png")
        except Exception as e:  # noqa: BLE001
            print(f"[auto] {tr.plan.name} mercator seg render failed: {e}")
        # Prefer categorical tree-instance composition for final seg output.
        # The direct Blender preview is RGB-rendered/anti-aliased and is not a
        # semantic label map.
        use_blender_seg = False
        pipe = _ACTIVE_PIPELINE
        if pipe is not None and hasattr(pipe, "cfg"):
            use_blender_seg = getattr(pipe.cfg, "use_blender_seg", False)

        if use_blender_seg:
            print(f"[auto] {tr.plan.name} keeping direct Blender-rendered topview for 1-to-1 depth alignment.")
        else:
            tree_mask_png = tile_dir / "topview_tree_mask.png"
            try:
                if (tile_dir / "topview_treeseg.png").exists():
                    _write_blender_tree_mask_mercator(
                        tr, tile_dir / "topview_treeseg.png", tree_mask_png)
            except Exception as e:  # noqa: BLE001
                print(f"[auto] {tr.plan.name} Blender tree-mask extraction failed: {e}")
            # Compose topview_treeseg.png directly in Web Mercator grid to keep the
            # render flat (平铺) and bypass lossy / crooked UTM warping rotation/slanted edges.
            try:
                _compose_topview_treeseg_mercator(
                    tr, tile_dir / "topview_treeseg.png")
            except Exception as e:  # noqa: BLE001
                print(f"[auto] {tr.plan.name} topview compose failed: {e}")
        
        # Overwrite seg_6class.png with the composed topview_treeseg.png to ensure
        # pixel-alignment and tree-crown coherence matching physical reality. Only fallback
        # to OSM-only layout if Blender render is unavailable.
        try:
            warped_treeseg = tile_dir / "topview_treeseg.png"
            if warped_treeseg.exists():
                import shutil
                # Create a non-tree segmentation copy (seg_6class_notree.png) before overwriting!
                shutil.copyfile(tile_dir / "seg_6class.png", tile_dir / "seg_6class_notree.png")
                # Now overwrite seg_6class.png (this becomes the "with tree" seg)
                shutil.copyfile(warped_treeseg, tile_dir / "seg_6class.png")
                print(f"[auto] Saved seg_6class_notree.png & overwrote seg_6class.png with composed topview_treeseg.png for {tr.plan.name}")
        except Exception as e:
            print(f"[auto] Failed to overwrite seg_6class.png / save seg_6class_notree.png: {e}")
        # Warp final visual files back to Web Mercator projection so they are perfectly flat (平铺)
        # and aligned with 1_osm.png and 2_rgb.png, correcting any crooked rotation (歪了) and corner gaps.
        print(f"[auto] Warping visual outputs to Web Mercator coordinates for {tr.plan.name}")

        # ---- Copy mapped files to the final clean layout ---- #
        try:
            import shutil
            shutil.copyfile(tile_dir / "osm_basemap.png", tile_dir / "1_osm.png")
            shutil.copyfile(tile_dir / "satellite_image.png", tile_dir / "2_rgb.png")
            if iso_png and Path(iso_png).exists():
                shutil.copyfile(iso_png, tile_dir / "3_blender_preview.png")
            
            # 4_seg.png is the composed topview_treeseg.png (always with trees)
            if (tile_dir / "topview_treeseg.png").exists():
                shutil.copyfile(tile_dir / "topview_treeseg.png", tile_dir / "4_seg.png")
            elif (tile_dir / "seg_6class.png").exists():
                shutil.copyfile(tile_dir / "seg_6class.png", tile_dir / "4_seg.png")
                
            shutil.copyfile(tile_dir / "topview_depth.png", tile_dir / "5_depth.png")
            shutil.copyfile(tile_dir / "topview_depth.exr", tile_dir / "5_depth.exr")
            if pointcloud_png and Path(pointcloud_png).exists():
                shutil.copyfile(pointcloud_png, tile_dir / "7_pointcloud.png")
            if pointcloud_ply and Path(pointcloud_ply).exists():
                shutil.copyfile(pointcloud_ply, tile_dir / "7_pointcloud.ply")

            # Apply coordinate warping to output images to remove any UTM rotation angle ("歪了")
            mapped_files = ["5_depth.png", "5_depth.exr"]
            if use_blender_seg:
                mapped_files.insert(0, "4_seg.png")
            for mapped_file in mapped_files:
                p = tile_dir / mapped_file
                if p.exists():
                    print(f"[auto] Warping {mapped_file} back to Web Mercator coordinate system to straighten content.")
                    try:
                        _warp_topview_to_mercator(tr, p)
                    except Exception as we:
                        print(f"[auto] Warning: warping failed for {mapped_file}: {we}")

            try:
                _write_polygon_outline_outputs(tr, tile_dir)
            except Exception as pe:
                print(f"[auto] Warning: polygon outline export failed for {tr.plan.name}: {pe}")

            try:
                from scripts.render_sat_depth_tests import write_offnadir_preset_folders
                sat_dirs = write_offnadir_preset_folders(
                    tile_dir / "5_depth.png",
                    tile_dir / "4_seg.png",
                    tile_dir,
                    max_height_m=30.0,
                    gsd_m=float(tr.state.get("gsd", 0.5)),
                )
                print(f"[auto] Saved {len(sat_dirs)} off-nadir camera/sun preset folders for {tr.plan.name}")
            except Exception as se:
                print(f"[auto] Warning: off-nadir sat preset export failed for {tr.plan.name}: {se}")

            print(f"[auto] Successfully saved and straightened clean standard layout files [1_osm.png, 2_rgb.png, 3_blender_preview.png, 4_seg.png, 5_depth.png, 5_depth.exr, 6_polygon_outline.png, 6_polygons.json, 7_pointcloud.png, 7_pointcloud.ply] for {tr.plan.name}")
        except Exception as e:
            print(f"[auto] Failed to copy standard clean layout for {tr.plan.name}: {e}")

        # Reuse KR1's already-written raw building GeoJSON for metadata. This
        # avoids a second online osmnx request per tile during batch runs.
        try:
            import shutil
            src_buildings = (ROOT / oa.CFG["paths"]["geojson_dir"]
                             / f"{tr.plan.name}_buildings.geojson")
            dst_buildings = (tile_dir / "metadata"
                             / f"{tr.plan.name}_osm_buildings.geojson")
            if src_buildings.exists() and not dst_buildings.exists():
                dst_buildings.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_buildings, dst_buildings)
        except Exception as e:
            print(f"[auto] Warning: cached OSM building metadata copy failed for {tr.plan.name}: {e}")

        # ---- Cleanup redundant intermediate files ---- #
        try:
            redundant_files = [
                "osm_basemap.png",
                "satellite_image.png",
                "seg_6class.png",
                "seg_6class_notree.png",
                "topview_depth.exr",
                "topview_depth.png",
                "topview_tree_mask.png",
                "topview_treeseg.png",
                "3_seg.png",
                "4_depth.png",
                "kr3_stdout.log",
                "kr3_stderr.log"
            ]
            for f_name in redundant_files:
                path_to_del = tile_dir / f_name
                if path_to_del.exists():
                    try:
                        path_to_del.unlink()
                    except Exception:
                        pass
        except Exception as e:
            print(f"[auto] Failed to clean up redundant files inside {tr.plan.name}: {e}")

        oa._write_tile_metadata(tr.state, tr.plan.name, tile_dir,
                                tuple(tr.state["bbox_wgs"]))
        tr.status["F"] = "ok"
    except Exception as e:  # noqa: BLE001
        tr.status["F"] = "failed"
        tr.errors["F"] = f"{type(e).__name__}: {e}"
    finally:
        tr.timings["F"] = time.time() - t0
    return tr

# --------------------------------------------------------------------- #
# Top-level orchestrator
# --------------------------------------------------------------------- #
@dataclass
class AutoPipelineConfig:
    city: str
    area_bbox_wgs: tuple
    overlap: float = 0.0
    # Building height (mirrors osm_app.on_save_and_build defaults)
    height_dist: str = "lognormal"
    height_seed: int = 42
    height_min: float = 3.0
    height_max: float = 30.0
    # Tree scatter
    scatter_seed: int = 11
    tree_density: float = 0.00015
    tree_species: list | None = None
    tree_h_dist: str = "lognormal"
    tree_h_seed: int = 11
    tree_h_min: float = 6.0
    tree_h_max: float = 20.0
    # Real-world cluster grouping for canopy_driven scatter mode.
    # Each cluster places `U(cluster_size_min, cluster_size_max)` trees in
    # an ellipse of radius `U(cluster_disk_radius_min, ..._max)` metres
    # with minor/major aspect = cluster_disk_aspect (1=circle, <1=ellipse).
    cluster_size_min: int = 10
    cluster_size_max: int = 20
    cluster_disk_radius_min: float = 4.0
    cluster_disk_radius_max: float = 10.0
    cluster_disk_aspect: float = 0.65
    # Distribution shape for cluster size and per-tree height. Both
    # support: 'uniform' (legacy random in [min,max]), 'bimodal'
    # (two narrow peaks near min/max with low_frac controlling LOW
    # peak weight), 'beta_u' (symmetric U-shape via Beta(0.5,0.5)).
    cluster_size_dist: str = "uniform"
    cluster_size_low_frac: float = 0.7
    tree_height_low_frac: float = 0.65
    # In-cluster overlap & fragment cleanup (canopy_prob).
    # ``overlap_factor`` < 1 lets trees inside the SAME cluster pack
    # tightly so crowns merge into a continuous blob; cross-cluster
    # spacing keeps full r_min. ``min_keep_ratio`` discards a whole
    # cluster if fewer than cluster_size_min * ratio trees fit (this
    # eliminates lone 2-3 tree fragments at class-foliage edges).
    cluster_overlap_factor: float = 0.45
    cluster_min_keep_ratio: float = 0.6
    # Default 10: delete any cluster with fewer than 10 trees outright.
    # Eliminates the visible "isolated 3-5 tree fragments" noise that
    # bypasses the soft ratio threshold. Set to 0 to disable.
    cluster_min_size_abs: int = 10
    uniform_tree_scale: bool = False
    # ---- Tree scatter algorithm toggles (Phase 2) ---- #
    # Scatter mode: ``canopy_prob`` (B1, default) uses ETH canopy as a
    # probability density field; ``cluster`` is the legacy clustered
    # Poisson disk; ``canopy_prob_streets`` is canopy_prob + B3 street
    # trees along road centrelines.
    scatter_mode: str = "canopy_prob"
    # B2 (soft constraint): allow trees to spawn outside OSM ``foliage``
    # at a reduced density (on grass / ground), excluding building /
    # road / water surfaces.
    allow_non_foliage: bool = True
    # B3: also place equally-spaced trees along OSM road buffers.
    enable_street_trees: bool = False
    # Procedural augmentation on top of real ETH canopy (0..1). 0.0 =
    # 100% real-data driven; 0.5 = +50% extra random trees on eligible
    # surfaces. Only used by ``canopy_prob`` modes.
    procedural_augment_ratio: float = 0.0
    # Multiplier applied to per-cell canopy probability in B1.
    canopy_prob_scale: float = 1.0
    # Crown horizontal scaling factor for 2D/3D trees
    topdown_tree_xy_scale: float = 1.0
    # Geometry Nodes tree scatter controls. 0.5 is the tuned default;
    # internally mapped to density scale 1.0 via 0.1 + 1.9 * amount.
    gn_tree_amount: float = 0.5
    gn_safe_building: float = 2.5
    gn_safe_road: float = 3.0
    gn_safe_water: float = 2.0
    gn_noise_scale: float = 0.10
    gn_min_distance: float = 3.5
    gn_xy_stretch: float = 0.75
    gn_z_stretch: float = 0.5
    gn_xy_stretch_min_at_0: float = 1.00
    gn_xy_stretch_min_at_1: float = 0.90
    gn_xy_stretch_max_at_0: float = 1.00
    gn_xy_stretch_max_at_1: float = 4.00
    gn_z_stretch_min_at_0: float = 1.00
    gn_z_stretch_min_at_1: float = 1.15
    gn_z_stretch_max_at_0: float = 1.00
    gn_z_stretch_max_at_1: float = 2.40
    # Use direct Blender-rendered topview for segmentation to guarantee 1-to-1 depth alignment
    use_blender_seg: bool = False
    # Canopy
    canopy_source: str = "eth_10m"
    target_foliage_ratio: float | None = 0.25
    # Stage toggles
    # Concurrency
    io_workers: int = 8
    osm_workers: int = 4
    canopy_workers: int = 4
    # Filenames are fixed; tile gsd/size are pinned at 0.5 / 1024 px
    gsd: float = 0.5
    size_px: int = 1024


class AutoPipeline:
    def __init__(self, cfg: AutoPipelineConfig,
                 progress_cb: Callable[[str, str, str], None] | None = None):
        """progress_cb(stage_letter, tile_name, status_str) -> None.

        Called once per tile at the end of every stage attempt.
        """
        self.cfg = cfg
        self.city_dir = ROOT / "output" / cfg.city
        ensure_dir(self.city_dir / "metadata")
        self.state_path = self.city_dir / "metadata" / ".pipeline_state.json"
        self.failures_log = self.city_dir / "metadata" / "_failures.log"
        self.failures_json = self.city_dir / "metadata" / "_failures.json"
        self.timing_latest_path = self.city_dir / "metadata" / "run_timing_latest.json"
        self.runs: list[TileRun] = []
        self.progress_cb = progress_cb or (lambda *a: None)
        # ---- City-level shared resources (populated by stage C0) ----
        # When ``city_cls_wgs`` is non-empty, stage C clips per-tile
        # geometries from this single fetch instead of issuing a tile-
        # local Overpass POST → seamless cross-tile topology.
        self.city_cls_wgs: dict = {}
        self.city_grid_info = None  # CityGridInfo | None
        self.city_seg = None        # np.ndarray | None  (uint8 H×W)

    # ---------- planning + state IO ---------- #
    def plan(self, force_clean: bool = False) -> list[TileRun]:
        plans = plan_tiles(self.cfg.area_bbox_wgs,
                           gsd=self.cfg.gsd,
                           size_px=self.cfg.size_px,
                           overlap=self.cfg.overlap)
        self.runs = [TileRun(plan=p) for p in plans]
        ensure_dir(self.city_dir)
        # Resume from disk (state + existing artifacts).
        existing = {} if force_clean else self._load_state()
        for tr in self.runs:
            saved = existing.get(tr.plan.name, {})
            tr.status.update(saved.get("status", {}))
            tr.errors.update(saved.get("errors", {}))
            tr.timings.update(saved.get("timings", {}))
            tr.glb_path = saved.get("glb_path")
            tr.blend_path = saved.get("blend_path")
            tr.canopy_npz = saved.get("canopy_npz")
            if not force_clean:
                # Trust on-disk artifacts over saved state.
                disk = _detect_existing_outputs(self.cfg.city, tr.plan)
                for s, v in disk.items():
                    tr.status[s] = v
        self._save_state()
        return self.runs

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            return d.get("tiles", {})
        except Exception:  # noqa: BLE001
            return {}

    def _save_state(self) -> None:
        ensure_dir(self.city_dir)
        n_rows, n_cols = grid_shape([tr.plan for tr in self.runs])
        bbox_union = (area_bbox_union([tr.plan for tr in self.runs])
                      if self.runs else None)
        d = {
            "city": self.cfg.city,
            "area_bbox_wgs": list(self.cfg.area_bbox_wgs),
            "bbox_union_wgs": list(bbox_union) if bbox_union else None,
            "gsd": self.cfg.gsd, "size_px": self.cfg.size_px,
            "overlap": self.cfg.overlap,
            "n_rows": n_rows, "n_cols": n_cols,
            "tile_index_grid": grid_id_array([tr.plan for tr in self.runs]),
            "tiles": {tr.plan.name: tr.to_dict() for tr in self.runs},
        }
        self.state_path.write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    def _log_failure(self, tr: TileRun, stage: str) -> None:
        ensure_dir(self.city_dir)
        msg = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
               f"{tr.plan.name} stage={stage} ({STAGE_NAMES[stage]}): "
               f"{tr.errors.get(stage, '?')}\n")
        with open(self.failures_log, "a", encoding="utf-8") as f:
            f.write(msg)

    def _save_failures_json(self) -> None:
        failed = []
        for tr in self.runs:
            for s in STAGES:
                if tr.status.get(s) == "failed":
                    failed.append({
                        "tile": tr.plan.name, "stage": s,
                        "stage_name": STAGE_NAMES[s],
                        "error": tr.errors.get(s, ""),
                    })
        self.failures_json.write_text(
            json.dumps({"failures": failed}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def _save_run_timing(self, started_at: str, ended_at: str,
                         duration_sec: float, summary: dict,
                         error: str | None = None) -> None:
        ensure_dir(self.city_dir / "metadata")
        stage_seconds = {
            s: round(sum(float(tr.timings.get(s, 0.0))
                         for tr in self.runs), 3)
            for s in STAGES
        }
        payload = {
            "city": self.cfg.city,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_sec": round(float(duration_sec), 3),
            "duration_min": round(float(duration_sec) / 60.0, 3),
            "n_tiles": len(self.runs),
            "status": "failed" if error else "ok",
            "error": error,
            "summary": summary,
            "stage_seconds_total": stage_seconds,
            "stage_seconds_mean_per_tile": {
                s: round(stage_seconds[s] / max(1, len(self.runs)), 3)
                for s in STAGES
            },
            "tiles": {
                tr.plan.name: {
                    "status": dict(tr.status),
                    "timings": {k: round(float(v), 3)
                                for k, v in tr.timings.items()},
                }
                for tr in self.runs
            },
            "config": asdict(self.cfg),
        }
        stamp = time.strftime("%Y%m%d_%H%M%S")
        history_path = self.city_dir / "metadata" / f"run_timing_{stamp}.json"
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.timing_latest_path.write_text(text, encoding="utf-8")
        history_path.write_text(text, encoding="utf-8")
        summary["duration_sec"] = payload["duration_sec"]
        summary["duration_min"] = payload["duration_min"]
        summary["timing_path"] = str(self.timing_latest_path)

    # ---------- run helpers ---------- #
    def _need(self, tr: TileRun, stage: str) -> bool:
        # Skip stages already marked "ok"; everything else (pending, failed)
        # gets retried.
        return tr.status.get(stage) != "ok"

    def _emit(self, stage: str, tr: TileRun) -> None:
        self.progress_cb(stage, tr.plan.name, tr.status.get(stage, "?"))

    def _run_stage_c0(self) -> None:
        """Pre-stage C: city-level union OSM fetch + global mercator grid.

        Result: ``self.city_cls_wgs`` (dict of class → WGS84 geom) and
        ``self.city_grid_info`` (mercator pixel grid covering all tiles)
        are populated. Stage C then clips per-tile geom from
        ``city_cls_wgs`` instead of fetching independently per tile.

        On failure, both stay empty and stage C falls back to per-tile
        Overpass fetch (legacy behaviour).
        """
        if not self.runs:
            return
        from dataprep.city_grid import (
            compute_city_grid_info, rasterize_city_seg,
        )
        from dataprep.osm_tags import CLASS_IDS, CLASS_PRIORITY
        plans = [tr.plan for tr in self.runs]
        ubox = area_bbox_union(plans)
        if ubox is None:
            return
        n_rows, n_cols = grid_shape(plans)
        target_long_px = max(n_cols, n_rows) * self.cfg.size_px

        # 1) Build the global mercator grid (cheap, deterministic).
        try:
            self.city_grid_info = compute_city_grid_info(
                tuple(ubox), target_long_px=target_long_px)
            print(f"[C0] city grid: out=({self.city_grid_info.out_w} x "
                  f"{self.city_grid_info.out_h}) ubox={ubox}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[C0] city grid build failed: {e}; will use per-tile",
                   flush=True)
            self.city_grid_info = None

        def _has_non_ground_feature(classes: dict) -> bool:
            for cls in ("building", "water", "grass", "foliage", "road"):
                geom = classes.get(cls)
                if geom is not None and not getattr(geom, "is_empty", True):
                    return True
            return False

        cached_all = True
        for tr in self.runs:
            cached = _load_class_geoms_from_geojson(self.cfg.city,
                                                    tr.plan.name)
            if not _has_non_ground_feature(cached):
                cached_all = False
                break
        if cached_all:
            print("[C0] all tiles have cached KR1 class geojson; "
                  "skipping union Overpass fetch", flush=True)
            return

        # 2) Single Overpass union fetch.  Falls back to per-tile if
        #    the city is too big or Overpass times out. Cached on disk
        #    keyed by bbox so consecutive identical-bbox runs (e.g. the
        #    veg-strategy comparison) don't repeatedly hit Overpass and
        #    risk transient partial responses.
        import hashlib as _hashlib
        import pickle as _pickle
        cache_key = _hashlib.md5(
            f"{ubox[0]:.6f},{ubox[1]:.6f},{ubox[2]:.6f},{ubox[3]:.6f}"
            .encode()).hexdigest()[:10]
        cache_dir = ROOT / "output" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_pkl = cache_dir / f"c0_osm_{cache_key}.pkl"

        def _is_complete(cls_wgs: dict) -> bool:
            road_g = cls_wgs.get("road")
            return (road_g is not None and not road_g.is_empty)

        loaded = False
        if cache_pkl.exists():
            try:
                self.city_cls_wgs = _pickle.loads(cache_pkl.read_bytes())
                if _is_complete(self.city_cls_wgs):
                    n_keys = sum(1 for k, v in self.city_cls_wgs.items()
                                  if not k.startswith("_") and v is not None
                                  and not v.is_empty)
                    print(f"[C0] union OSM cache hit ({cache_pkl.name}): "
                          f"{n_keys} non-empty classes", flush=True)
                    loaded = True
                else:
                    print(f"[C0] cached OSM is incomplete (no roads); "
                          f"refetching", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[C0] cache read failed: {e}; refetching",
                       flush=True)

        if not loaded:
            self.city_cls_wgs = {}
            for attempt in range(3):
                try:
                    print(f"[C0] union OSM fetch over {ubox} "
                          f"(attempt {attempt + 1}/3) ...", flush=True)
                    self.city_cls_wgs = oa.fetch_classes_wgs(
                        tuple(ubox)) or {}
                    n_keys = sum(1 for k, v in self.city_cls_wgs.items()
                                  if not k.startswith("_") and v is not None
                                  and not v.is_empty)
                    print(f"[C0] union OSM ok: {n_keys} non-empty classes",
                           flush=True)
                    if _is_complete(self.city_cls_wgs):
                        break
                    print(f"[C0] partial OSM response (no roads); "
                          f"retrying", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[C0] union OSM fetch attempt {attempt + 1} "
                          f"failed: {e}", flush=True)
                    self.city_cls_wgs = {}
            # Cache only complete responses.
            if self.city_cls_wgs and _is_complete(self.city_cls_wgs):
                try:
                    cache_pkl.write_bytes(_pickle.dumps(self.city_cls_wgs))
                    print(f"[C0] cached union OSM -> {cache_pkl.name}",
                           flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[C0] cache write failed: {e}", flush=True)
            elif not self.city_cls_wgs:
                print(f"[C0] union OSM fetch failed all retries; "
                       f"per-tile fallback", flush=True)

        # 3) Rasterize the city seg once on the global grid (used by
        #    aggregate_city for city_overview_seg_aligned.png AND by
        #    per-tile mercator seg slicing).
        if self.city_cls_wgs and self.city_grid_info is not None:
            try:
                self.city_seg = rasterize_city_seg(
                    self.city_cls_wgs, self.city_grid_info,
                    CLASS_IDS, CLASS_PRIORITY)
                print(f"[C0] city_seg shape={self.city_seg.shape}",
                       flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[C0] city_seg rasterize failed: {e}", flush=True)
                self.city_seg = None

    # ---------- main run ---------- #
    def run(self) -> dict:
        """Execute all stages. Returns final summary dict."""
        run_t0 = time.time()
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        summary: dict | None = None
        run_error: str | None = None
        if not self.runs:
            self.plan()
        if not self.runs:
            summary = {"status": "empty", "n_tiles": 0}
            ended_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_run_timing(started_at, ended_at,
                                  time.time() - run_t0, summary)
            return summary

        try:
            with _redirect_cfg_paths(self.cfg.city):
                # Hydrate state for tiles that will need later stages.
                for tr in self.runs:
                    _hydrate_state_from_disk(self.cfg.city, tr.plan, tr)

                # Publish self as the active pipeline so stage workers can
                # access shared city-level resources without arg threading.
                global _ACTIVE_PIPELINE
                _ACTIVE_PIPELINE = self
                # ---- Stage B: imagery (parallel IO) ---- #
                self._run_pool("B", _stage_b_one,
                               workers=self.cfg.io_workers)
                self._save_state()

                # ---- Stage C0: city-level union OSM + global grid ---- #
                self._run_stage_c0()

                # ---- Stage C: OSM (parallel IO, smaller pool) ---- #
                self._run_pool("C", _stage_c_one,
                               workers=self.cfg.osm_workers,
                               require=("B",))
                self._save_state()

                # ---- Stage D: canopy ---- #
                self._run_pool(
                    "D",
                    lambda tr: _stage_d_one(tr,
                                            canopy_source=self.cfg.canopy_source,
                                            target_foliage_ratio=self.cfg.target_foliage_ratio),
                    workers=self.cfg.canopy_workers,
                    require=("B", "C"))
                self._save_state()

                # ---- Stage E: KR2 (serial; KR2 mutates global state) ---- #
                for tr in self.runs:
                    if not self._need(tr, "E"):
                        self._emit("E", tr); continue
                    if any(tr.status.get(s) != "ok" for s in ("B", "C", "D")):
                        tr.status["E"] = "blocked"
                        tr.errors["E"] = "upstream stage failed"
                        self._log_failure(tr, "E"); self._emit("E", tr); continue
                    _stage_e_one(tr, self.cfg.height_dist, self.cfg.height_seed,
                                 self.cfg.height_min, self.cfg.height_max)
                    if tr.status["E"] == "failed":
                        self._log_failure(tr, "E")
                    self._emit("E", tr)
                self._save_state()

                # ---- Stage F: KR3 Blender (serial) ---- #
                for tr in self.runs:
                    if not self._need(tr, "F"):
                        self._emit("F", tr); continue
                    if tr.status.get("E") != "ok":
                        tr.status["F"] = "blocked"
                        tr.errors["F"] = "stage E not ok"
                        self._log_failure(tr, "F"); self._emit("F", tr); continue
                    _stage_f_one(tr, self.cfg.scatter_seed, self.cfg.tree_density,
                                 self.cfg.tree_species, self.cfg.tree_h_dist,
                                 self.cfg.tree_h_seed, self.cfg.tree_h_min,
                                 self.cfg.tree_h_max, city=self.cfg.city,
                                 cluster_size_min=self.cfg.cluster_size_min,
                                 cluster_size_max=self.cfg.cluster_size_max,
                                 cluster_disk_radius_min=self.cfg.cluster_disk_radius_min,
                                 cluster_disk_radius_max=self.cfg.cluster_disk_radius_max,
                                 cluster_disk_aspect=self.cfg.cluster_disk_aspect,
                                 cluster_size_dist=self.cfg.cluster_size_dist,
                                 cluster_size_low_frac=self.cfg.cluster_size_low_frac,
                                 tree_height_low_frac=self.cfg.tree_height_low_frac,
                                 cluster_overlap_factor=self.cfg.cluster_overlap_factor,
                                 cluster_min_keep_ratio=self.cfg.cluster_min_keep_ratio,
                                 cluster_min_size_abs=self.cfg.cluster_min_size_abs,
                                 scatter_mode=self.cfg.scatter_mode,
                                 allow_non_foliage=self.cfg.allow_non_foliage,
                                 enable_street_trees=self.cfg.enable_street_trees,
                                 procedural_augment_ratio=self.cfg.procedural_augment_ratio,
                                 canopy_prob_scale=self.cfg.canopy_prob_scale,
                                 uniform_tree_scale=self.cfg.uniform_tree_scale,
                                 topdown_tree_xy_scale=self.cfg.topdown_tree_xy_scale,
                                 gn_tree_amount=self.cfg.gn_tree_amount,
                                 gn_safe_building=self.cfg.gn_safe_building,
                                 gn_safe_road=self.cfg.gn_safe_road,
                                 gn_safe_water=self.cfg.gn_safe_water,
                                 gn_noise_scale=self.cfg.gn_noise_scale,
                                 gn_min_distance=self.cfg.gn_min_distance,
                                 gn_xy_stretch=self.cfg.gn_xy_stretch,
                                 gn_z_stretch=self.cfg.gn_z_stretch,
                                 gn_xy_stretch_min_at_0=self.cfg.gn_xy_stretch_min_at_0,
                                 gn_xy_stretch_min_at_1=self.cfg.gn_xy_stretch_min_at_1,
                                 gn_xy_stretch_max_at_0=self.cfg.gn_xy_stretch_max_at_0,
                                 gn_xy_stretch_max_at_1=self.cfg.gn_xy_stretch_max_at_1,
                                 gn_z_stretch_min_at_0=self.cfg.gn_z_stretch_min_at_0,
                                 gn_z_stretch_min_at_1=self.cfg.gn_z_stretch_min_at_1,
                                 gn_z_stretch_max_at_0=self.cfg.gn_z_stretch_max_at_0,
                                 gn_z_stretch_max_at_1=self.cfg.gn_z_stretch_max_at_1)
                    if tr.status["F"] == "failed":
                        self._log_failure(tr, "F")
                    self._emit("F", tr)
                self._save_state()

                # ---- Stage H: aggregate ---- #
                try:
                    aggregate_city(self.cfg.city, self.runs)
                except Exception as e:  # noqa: BLE001
                    with open(self.failures_log, "a", encoding="utf-8") as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                f"AGGREGATE failed: {e}\n")
                self._save_state()
                self._save_failures_json()

            summary = self.summary()
            return summary
        except Exception as e:
            run_error = f"{type(e).__name__}: {e}"
            raise
        finally:
            _ACTIVE_PIPELINE = None
            if self.runs:
                ended_at = time.strftime("%Y-%m-%d %H:%M:%S")
                if summary is None:
                    summary = self.summary()
                if run_error:
                    summary["status"] = "failed"
                    summary["error"] = run_error
                self._save_run_timing(started_at, ended_at,
                                      time.time() - run_t0, summary,
                                      error=run_error)

    def _run_pool(self, stage: str, fn,
                   workers: int = 4,
                   require: Sequence[str] = ()) -> None:
        """Run ``fn(tr)`` in a thread pool over self.runs that need stage."""
        targets: list[TileRun] = []
        for tr in self.runs:
            if not self._need(tr, stage):
                self._emit(stage, tr); continue
            if any(tr.status.get(s) != "ok" for s in require):
                tr.status[stage] = "blocked"
                tr.errors[stage] = (f"upstream {require} not ok "
                                     f"(got {[tr.status.get(s) for s in require]})")
                self._log_failure(tr, stage)
                self._emit(stage, tr); continue
            targets.append(tr)
        if not targets:
            return
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {ex.submit(fn, tr): tr for tr in targets}
            for fut in as_completed(futs):
                tr = futs[fut]
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    tr.status[stage] = "failed"
                    tr.errors[stage] = f"{type(e).__name__}: {e}"
                if tr.status.get(stage) == "failed":
                    self._log_failure(tr, stage)
                self._emit(stage, tr)

    # ---------- summary ---------- #
    def summary(self) -> dict:
        ok = lambda s: sum(1 for tr in self.runs
                            if tr.status.get(s) == "ok")
        failed = lambda s: sum(1 for tr in self.runs
                                if tr.status.get(s) in ("failed", "blocked"))
        return {
            "city": self.cfg.city,
            "n_tiles": len(self.runs),
            "per_stage_ok": {s: ok(s) for s in STAGES},
            "per_stage_failed": {s: failed(s) for s in STAGES},
            "city_dir": str(self.city_dir),
        }


# --------------------------------------------------------------------- #
# Aggregation (Stage H)
# --------------------------------------------------------------------- #
def aggregate_city(city: str, runs: list[TileRun]) -> dict:
    """Combine all tile metadata + assemble overview images directly from standard tiles.

    Writes:
      output/<city>/metadata/<city>.json
      output/<city>/metadata/tile_index.geojson
      output/<city>/<city>_osm.png
      output/<city>/<city>_rgb.png
      output/<city>/<city>_seg.png
      output/<city>/<city>_depth.png
    """
    city_dir = ensure_dir(ROOT / "output" / city)
    meta_dir = ensure_dir(city_dir / "metadata")
    n_rows, n_cols = grid_shape([tr.plan for tr in runs])
    grid = grid_id_array([tr.plan for tr in runs])
    union_bbox = list(area_bbox_union([tr.plan for tr in runs]))

    # Collect per-tile metadata files.
    tile_summaries = []
    all_buildings: dict[str, dict] = {}
    for tr in runs:
        tile_dir = city_dir / tr.plan.name
        meta_path = tile_dir / "metadata" / f"{tr.plan.name}.json"
        rec = {
            "tile_name": tr.plan.name,
            "row": tr.plan.row, "col": tr.plan.col,
            "lat": tr.plan.lat, "lon": tr.plan.lon,
            "bbox_wgs84": list(tr.plan.bbox_wgs),
            "status": dict(tr.status),
            "ok": all(tr.status.get(s) == "ok" for s in ("E", "F")),
        }
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                rec["utm_crs"] = m.get("utm_crs")
                rec["sw_utm"] = m.get("sw_utm")
                rec["n_buildings"] = m.get("n_buildings", 0)
                rec["files"] = m.get("files", {})
                for b in m.get("buildings", []) or []:
                    gid = f"{tr.plan.name}__{b['id']}"
                    all_buildings[gid] = {
                        "global_id": gid,
                        "tile_name": tr.plan.name,
                        "local_id": b["id"],
                        "centroid_lon": b["centroid_lon"],
                        "centroid_lat": b["centroid_lat"],
                        "height_m": b["height_m"],
                        "footprint_area_m2": b["footprint_area_m2"],
                    }
            except Exception as e:  # noqa: BLE001
                rec["meta_read_error"] = str(e)
        tile_summaries.append(rec)

    city_meta = {
        "city": city,
        "area_bbox_wgs84": union_bbox,
        "n_rows": n_rows, "n_cols": n_cols,
        "n_tiles": len(runs),
        "tile_index_grid": grid,
        "tiles": tile_summaries,
        "buildings_global": list(all_buildings.values()),
        "n_buildings": len(all_buildings),
    }
    
    # Save metadatas inside metadata directory - clean space!
    (meta_dir / f"{city}.json").write_text(
        json.dumps(city_meta, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # tile_index.geojson
    feats = []
    for tr in runs:
        W, S, E, N = tr.plan.bbox_wgs
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [W, S], [E, S], [E, N], [W, N], [W, S]]]},
            "properties": {
                "tile_name": tr.plan.name,
                "row": tr.plan.row, "col": tr.plan.col,
                "ok": all(tr.status.get(s) == "ok" for s in ("E", "F")),
            },
        })
    (meta_dir / "tile_index.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    # Standard grid-based mosaic (useful for UTM-aligned adjacent tiles with zero overlap, e.g. unwarped 5_depth.png)
    def _mosaic_grid(filename: str, out_name: str) -> None:
        cell = None
        for tr in runs:
            p = city_dir / tr.plan.name / filename
            if p.exists():
                try:
                    cell = Image.open(p).size
                    break
                except Exception:  # noqa: BLE001
                    pass
        if cell is None:
            return
        cw, ch = cell
        canvas = Image.new("RGB", (cw * n_cols, ch * n_rows), (40, 40, 40))
        for tr in runs:
            p = city_dir / tr.plan.name / filename
            if not p.exists():
                continue
            try:
                im = Image.open(p).convert("RGB")
                if im.size != (cw, ch):
                    interp = Image.NEAREST if "seg" in filename else Image.BILINEAR
                    im = im.resize((cw, ch), interp)
                canvas.paste(im, (tr.plan.col * cw, tr.plan.row * ch))
            except Exception as e:  # noqa: BLE001
                print(f"[agg] mosaic grid paste {tr.plan.name}: {e}")
        # Cap final size to ~4096 on the long side to keep PNGs sane.
        max_side = max(canvas.size)
        if max_side > 4096:
            scale = 4096.0 / max_side
            interp = Image.NEAREST if "seg" in filename else Image.BILINEAR
            canvas = canvas.resize(
                (int(canvas.size[0] * scale), int(canvas.size[1] * scale)),
                interp)
        canvas.save(city_dir / out_name)

    # Stitch the standard aligned tiles straight up - fast, 100% offline, perfectly matching!
    _mosaic_grid("1_osm.png", f"{city}_osm.png")
    _mosaic_grid("2_rgb.png", f"{city}_rgb.png")
    _mosaic_grid("4_seg.png", f"{city}_seg.png")
    _mosaic_grid("5_depth.png", f"{city}_depth.png")

    # Dump strategy sidecar (safely inside metadata)
    try:
        pipe = _ACTIVE_PIPELINE
        if pipe is not None:
            _dump_strategy_sidecar(city_dir, pipe.cfg)
    except Exception as e:
        print(f"[agg] failed to dump strategy sidecar: {e}")

    return city_meta


def _render_tile_seg_mercator(tr: TileRun, out_png: Path) -> None:
    """Rasterize a single tile's cls_wgs in EPSG:3857 onto a 1024×1024
    grid that exactly matches ``satellite_image.png`` for that tile.

    KR1 / KR2 / KR3 keep using UTM internally (they need true ground
    metres for heights, scatter, etc.). Only the user-facing PNG is
    re-rendered in mercator so the seg pixel-aligns with the Esri
    satellite tile.

    When the active pipeline has a city-level mercator seg array
    (``_ACTIVE_PIPELINE.city_seg``), this function instead slices the
    tile sub-rect from that single global rasterization. That guarantees
    seamless cross-tile boundaries (a road crossing two tiles is rendered
    once on a unified grid → identical pixels in both tiles).
    """
    from dataprep.raster_utils import colorize_seg
    PALETTE = oa.CFG["class_colors"]

    # ---- Fast path: slice from city_seg if available ---- #
    pipe = _ACTIVE_PIPELINE
    if (pipe is not None and pipe.city_seg is not None
            and pipe.city_grid_info is not None):
        try:
            from dataprep.city_grid import tile_pixel_subrect
            bbox_wgs = tuple(tr.state["bbox_wgs"])
            px0, py0, px1, py1 = tile_pixel_subrect(
                bbox_wgs, pipe.city_grid_info)
            sub = pipe.city_seg[py0:py1, px0:px1]
            size = int(tr.state.get("size") or 1024)
            sub_im = Image.fromarray(sub).resize(
                (size, size), Image.NEAREST)
            seg = np.array(sub_im, dtype=np.uint8)
            Image.fromarray(colorize_seg(seg, PALETTE)).save(out_png)
            return
        except Exception as e:  # noqa: BLE001
            print(f"[auto] city_seg slice failed for {tr.plan.name}: {e}; "
                   f"falling back to per-tile rasterize", flush=True)

    # ---- Fallback: per-tile rasterize (legacy path) ---- #
    cls_wgs = tr.state.get("cls_wgs")
    if not cls_wgs:
        return
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import from_bounds
    from dataprep.geometry_utils import make_transformer, reproject_geom
    from dataprep.osm_tags import CLASS_IDS
    from dataprep.osm_tags import CLASS_IDS, CLASS_PRIORITY

    fwd = make_transformer("EPSG:4326", "EPSG:3857")
    bbox_wgs = tuple(tr.state["bbox_wgs"])
    W, S, E, N = bbox_wgs

    # Mercator bounds of the tile's WGS84 bbox.
    import math
    R = 6378137.0
    def _merc_x(lon: float) -> float:
        return R * math.radians(lon)
    def _merc_y(lat: float) -> float:
        return R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    west_m = _merc_x(W); east_m = _merc_x(E)
    south_m = _merc_y(S); north_m = _merc_y(N)

    # satellite_image_for_bbox squashes the cropped mercator image to
    # (size, size) regardless of mercator aspect, so we render the seg
    # at the same (size, size) for pixel-for-pixel alignment.
    size = int(tr.state.get("size") or 1024)
    transform = from_bounds(west_m, south_m, east_m, north_m, size, size)

    ground_id = CLASS_IDS["ground"]
    seg = np.full((size, size), ground_id, dtype=np.uint8)
    order = sorted(CLASS_IDS.keys(), key=lambda k: CLASS_PRIORITY[k])
    for name in order:
        g_wgs = cls_wgs.get(name)
        if g_wgs is None or g_wgs.is_empty:
            continue
        g_merc = reproject_geom(g_wgs, fwd)
        if g_merc is None or g_merc.is_empty:
            continue
        cls_id = CLASS_IDS[name]
        out = _rasterize([(g_merc, cls_id)], out_shape=(size, size),
                          transform=transform, fill=ground_id,
                          default_value=cls_id, dtype="uint8",
                          all_touched=False)
        if cls_id == 0:
            hit = _rasterize([(g_merc, 1)], out_shape=(size, size),
                             transform=transform, fill=0,
                             default_value=1, dtype="uint8",
                             all_touched=False) > 0
        else:
            hit = out == cls_id
        seg[hit] = cls_id
    Image.fromarray(colorize_seg(seg, PALETTE)).save(out_png)


def _write_polygon_outline_outputs(tr: TileRun, tile_dir: Path) -> None:
    """Write pixel-aligned polygon outline PNG + vertex JSON.

    Outputs are aligned to the same Web-Mercator-squashed 1024x1024 image
    grid used by 1_osm/2_rgb/4_seg. Unlike seg masks, this is boundary-only:
    black background with class-colored outlines plus explicit vertex rings.
    """
    import math
    from PIL import ImageDraw
    from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
    from dataprep.geometry_utils import make_transformer, reproject_geom
    from dataprep.osm_tags import CLASS_IDS

    size = int(tr.state.get("size") or 1024)
    cls_wgs = tr.state.get("cls_wgs") or {}
    tile_dir = Path(tile_dir)
    out_png = tile_dir / "6_polygon_outline.png"
    out_json = tile_dir / "6_polygons.json"

    colors = {
        "building": (255, 0, 0),
        "road": (0, 0, 255),
        "water": (0, 225, 255),
    }
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    bbox_wgs = tuple(float(x) for x in tr.state["bbox_wgs"])
    W_lon, S_lat, E_lon, N_lat = bbox_wgs
    R = 6378137.0
    W_m = R * math.radians(W_lon)
    E_m = R * math.radians(E_lon)
    N_m = R * math.log(math.tan(math.pi / 4 + math.radians(N_lat) / 2))
    S_m = R * math.log(math.tan(math.pi / 4 + math.radians(S_lat) / 2))
    fwd = make_transformer("EPSG:4326", "EPSG:3857")

    def _to_px(x_m: float, y_m: float) -> list[float]:
        x = (float(x_m) - W_m) / max(1e-9, E_m - W_m) * size
        y = (N_m - float(y_m)) / max(1e-9, N_m - S_m) * size
        return [round(x, 3), round(y, 3)]

    def _line_to_px(coords) -> list[list[float]]:
        return [_to_px(x, y) for x, y in coords]

    def _iter_polys(geom):
        if geom is None or geom.is_empty:
            return
        if isinstance(geom, Polygon):
            yield geom
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                if poly is not None and not poly.is_empty:
                    yield poly
        elif hasattr(geom, "geoms"):
            for part in geom.geoms:
                yield from _iter_polys(part)

    def _iter_lines(geom):
        if geom is None or geom.is_empty:
            return
        if isinstance(geom, LineString):
            yield geom
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                if line is not None and not line.is_empty:
                    yield line
        elif hasattr(geom, "geoms"):
            for part in geom.geoms:
                yield from _iter_lines(part)

    features = []
    for class_name in ("building", "road", "water"):
        geom_wgs = cls_wgs.get(class_name)
        if geom_wgs is None or geom_wgs.is_empty:
            continue
        geom = reproject_geom(geom_wgs, fwd)
        color = colors[class_name]
        feature_index = 0
        for poly in _iter_polys(geom):
            exterior = _line_to_px(poly.exterior.coords)
            interiors = [_line_to_px(ring.coords) for ring in poly.interiors]
            if len(exterior) >= 2:
                draw.line([tuple(p) for p in exterior], fill=color, width=3, joint="curve")
            for ring in interiors:
                if len(ring) >= 2:
                    draw.line([tuple(p) for p in ring], fill=color, width=2, joint="curve")
            features.append({
                "id": f"{class_name}_{feature_index:05d}",
                "class": class_name,
                "class_id": int(CLASS_IDS[class_name]),
                "geometry_type": "polygon",
                "exterior_px": exterior,
                "interiors_px": interiors,
                "properties": {},
            })
            feature_index += 1
        for line in _iter_lines(geom):
            points = _line_to_px(line.coords)
            if len(points) >= 2:
                draw.line([tuple(p) for p in points], fill=color, width=3)
            features.append({
                "id": f"{class_name}_{feature_index:05d}",
                "class": class_name,
                "class_id": int(CLASS_IDS[class_name]),
                "geometry_type": "line",
                "points_px": points,
                "properties": {},
            })
            feature_index += 1

    payload = {
        "tile_name": tr.plan.name,
        "size_px": size,
        "coordinate_space": "pixel",
        "pixel_origin": "top_left",
        "x_range": [0, size],
        "y_range": [0, size],
        "aligned_to": ["1_osm.png", "2_rgb.png", "4_seg.png", "5_depth.png"],
        "classes": ["building", "road", "water"],
        "features": features,
    }
    img.save(out_png)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[auto] {tr.plan.name} wrote polygon outline outputs "
          f"({len(features)} features) -> 6_polygon_outline.png, 6_polygons.json")


def _warp_topview_to_mercator(tr: TileRun, png_path: Path) -> None:
    """Warp KR3's topview PNG, EXR or depth image (UTM) onto the Web Mercator grid in-place.

    KR3 renders 1024×1024 in UTM (cx±256m, cy±256m, gsd=0.5m).  Esri
    satellite arrives in Web Mercator and is squashed to 1024×1024 over
    the WGS bbox. The two grids differ by UTM grid convergence and the
    per-latitude N/S mercator stretch. We resample the UTM image onto
    the mercator grid so all per-tile PNGs are pixel-aligned.
    """
    if not png_path.exists():
        return
    import math
    from scipy.ndimage import map_coordinates
    from dataprep.geometry_utils import make_transformer

    is_exr = png_path.suffix.lower() == ".exr"
    is_depth_png = (not is_exr) and "depth" in png_path.name.lower()
    if is_exr:
        try:
            import OpenEXR
            import Imath
            import array
            f_in = OpenEXR.InputFile(str(png_path))
            dw = f_in.header()['dataWindow']
            W = dw.max.x - dw.min.x + 1
            H = dw.max.y - dw.min.y + 1
            # Read single-channel float 'V' depth channel
            channel_data = f_in.channel('V', Imath.PixelType(Imath.PixelType.FLOAT))
            src = np.frombuffer(channel_data, dtype=np.float32).reshape(H, W)
        except Exception as e:
            print(f"[auto] OpenEXR native read failed: {e}")
            return
    else:
        src = np.asarray(Image.open(png_path))
        
    H, W = src.shape[:2]
    if H != W:
        return
    size = H

    bbox_wgs = tuple(float(x) for x in tr.state["bbox_wgs"])
    utm_crs = str(tr.state["utm"])
    cx_u = float(tr.state["cx_utm"])
    cy_u = float(tr.state["cy_utm"])
    gsd = float(tr.state.get("gsd") or 0.5)
    half_m = 0.5 * gsd * size
    sw_x_utm = cx_u - half_m
    sw_y_utm = cy_u - half_m

    R = 6378137.0
    W_lon, S_lat, E_lon, N_lat = bbox_wgs
    W_m = R * math.radians(W_lon)
    E_m = R * math.radians(E_lon)
    N_m = R * math.log(math.tan(math.pi / 4 + math.radians(N_lat) / 2))
    S_m = R * math.log(math.tan(math.pi / 4 + math.radians(S_lat) / 2))

    xs_m = np.linspace(W_m, E_m, size)
    ys_m = np.linspace(N_m, S_m, size)
    XX, YY = np.meshgrid(xs_m, ys_m)
    lons = np.degrees(XX / R)
    lats = np.degrees(2.0 * np.arctan(np.exp(YY / R)) - math.pi / 2.0)

    fwd = make_transformer("EPSG:4326", utm_crs)
    xs_u, ys_u = fwd.transform(lons.ravel(), lats.ravel())
    xs_u = np.asarray(xs_u).reshape(size, size)
    ys_u = np.asarray(ys_u).reshape(size, size)

    px = (xs_u - sw_x_utm) / gsd
    py = size - 1 - (ys_u - sw_y_utm) / gsd

    coords = np.stack([py, px], axis=0)
    out = np.zeros_like(src)
    if is_exr:
        out = map_coordinates(
            src, coords, order=1, mode="constant", cval=0.0
        ).astype(np.float32)
        try:
            # Write out single-channel float 'V' to EXR
            header = OpenEXR.Header(size, size)
            header['channels'] = {'V': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
            f_out = OpenEXR.OutputFile(str(png_path), header)
            f_out.writePixels({'V': out.tobytes()})
            f_out.close()
        except Exception as e:
            print(f"[auto] OpenEXR native write failed: {e}")
    else:
        if len(src.shape) == 2:
            order = 1 if is_depth_png else 0
            sampled = map_coordinates(
                src.astype(np.float32), coords, order=order,
                mode="constant", cval=0.0
            )
            if is_depth_png and src.dtype == np.uint16:
                out = np.clip(sampled / 257.0, 0, 255).astype(np.uint8)
            else:
                out = np.clip(sampled, 0, 255).astype(np.uint8)
        else:
            for c in range(src.shape[2]):
                out[..., c] = map_coordinates(
                    src[..., c], coords, order=0, mode="constant", cval=0
                ).astype(np.uint8)
        Image.fromarray(out).save(png_path)


def _write_blender_tree_mask_mercator(tr: TileRun, preview_png: Path,
                                      out_mask_png: Path) -> None:
    """Extract tree crowns from KR3's topdown render and warp to mercator.

    KR3 renders tree instances in bright green with foliage substrate hidden.
    Using that rendered footprint as the final semantic tree mask keeps the
    seg outline tied to the same Blender geometry as the depth image.
    """
    if not preview_png.exists():
        return
    arr = np.asarray(Image.open(preview_png).convert("RGB"))
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    mask = (g >= 120) & (g >= r + 45) & (g >= b + 45)
    out_mask_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_mask_png)
    _warp_topview_to_mercator(tr, out_mask_png)


def _compose_topview_treeseg_mercator(tr: TileRun, out_png: Path) -> None:
    """Compose ``topview_treeseg.png`` directly in Web Mercator.

    Bypasses the lossy UTM->mercator bitmap warp by:
      1. Rasterizing the tile's OSM vectors (cls_wgs) in mercator at
         the same (size, size) grid as ``satellite_image.png``.
      2. Reading ``blender/tree_instances.json`` (positions emitted by
         KR3 in tile-centred UTM-local metres) and drawing one filled
         circle per tree at its exact mercator pixel location.

    Result: every tree is rendered at its *true* WGS84 lat/lon and
    therefore pixel-aligns with the Esri satellite tile to within
    rasterio's anti-aliasing precision.
    """
    import math
    import json
    from PIL import ImageDraw
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import Point as _P
    from dataprep.geometry_utils import make_transformer, reproject_geom
    from dataprep.osm_tags import CLASS_IDS, CLASS_PRIORITY
    from dataprep.raster_utils import colorize_seg
    PALETTE = oa.CFG["class_colors"]

    cls_wgs = tr.state.get("cls_wgs") or {}
    bbox_wgs = tuple(float(x) for x in tr.state["bbox_wgs"])
    W_lon, S_lat, E_lon, N_lat = bbox_wgs

    R = 6378137.0
    W_m = R * math.radians(W_lon)
    E_m = R * math.radians(E_lon)
    N_m = R * math.log(math.tan(math.pi / 4 + math.radians(N_lat) / 2))
    S_m = R * math.log(math.tan(math.pi / 4 + math.radians(S_lat) / 2))

    size = int(tr.state.get("size") or 1024)
    transform = from_bounds(W_m, S_m, E_m, N_m, size, size)
    fwd_wgs2merc = make_transformer("EPSG:4326", "EPSG:3857")

    # ---- 1. Mercator seg base layer ---- #
    ground_id = CLASS_IDS["ground"]
    seg = np.full((size, size), ground_id, dtype=np.uint8)
    hard_surface_mask = np.zeros((size, size), dtype=bool)
    building_mask = np.zeros((size, size), dtype=bool)
    order = sorted(CLASS_IDS.keys(), key=lambda k: CLASS_PRIORITY[k])
    for name in order:
        g_wgs = cls_wgs.get(name)
        if g_wgs is None or g_wgs.is_empty:
            continue
        g_merc = reproject_geom(g_wgs, fwd_wgs2merc)
        if g_merc is None or g_merc.is_empty:
            continue
        cls_id = CLASS_IDS[name]
        rast = _rasterize([(g_merc, cls_id)], out_shape=(size, size),
                           transform=transform, fill=ground_id,
                           default_value=cls_id, dtype="uint8",
                           all_touched=False)
        hit = (rast == cls_id)
        if cls_id == 0:
            hit = _rasterize([(g_merc, 1)], out_shape=(size, size),
                             transform=transform, fill=0,
                             default_value=1, dtype="uint8",
                             all_touched=False) > 0
        seg[hit] = cls_id
        if name in {"building", "road", "water"}:
            hard_surface_mask |= hit
        if name == "building":
            building_mask |= hit

    # Foliage/canopy polygons are the substrate used to decide where trees can
    # grow. Keep that area visually grouped with grass; only actual tree
    # instance crowns drawn below remain green foliage in the final label map.
    foliage_id = CLASS_IDS["foliage"]
    grass_id = CLASS_IDS["grass"]
    seg[(seg == foliage_id) & (~hard_surface_mask)] = grass_id
    base_seg = seg.copy()

    # ---- 1b. Paint dense canopy polygons as grass substrate ---- #
    # The canopy/green-area mask is where trees may grow, but it is not an
    # individual tree crown label. Paint it as grass so grass/substrate and
    # actual green tree crowns remain separable in the semantic map.
    n_canopy_polys = 0
    canopy_geojson = (ROOT / oa.CFG["paths"]["geojson_dir"]
                      / f"{tr.plan.name}_foliage_canopy.geojson")
    if canopy_geojson.exists():
        try:
            import geopandas as _gpd
            cdf = _gpd.read_file(canopy_geojson)
            shapes = []
            for geom in cdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                geom_merc = reproject_geom(geom, fwd_wgs2merc)
                if geom_merc is not None and not geom_merc.is_empty:
                    shapes.append((geom_merc, grass_id))
            if shapes:
                rast = _rasterize(shapes, out_shape=(size, size),
                                  transform=transform, fill=0,
                                  default_value=grass_id, dtype="uint8",
                                  all_touched=False)
                hit = (rast == grass_id) & (~hard_surface_mask)
                seg[hit] = grass_id
                base_seg[hit] = grass_id
                n_canopy_polys = len(shapes)
        except Exception as e:  # noqa: BLE001
            print(f"[auto] {tr.plan.name} canopy substrate grass paint failed: {e}")

    # ---- 2. Tree-instance & Building height-level occlusion ---- #
    import geopandas as gpd
    tile_root = oa.CFG["paths"]["tile_root"]

    # Initialize building height map
    bld_height_map = np.zeros((size, size), dtype=np.float32)

    # A. Map osm_id to height_m from metadata JSON
    tile_meta_path = ROOT / tile_root / tr.plan.name / "metadata" / f"{tr.plan.name}.json"
    height_by_id = {}
    if tile_meta_path.exists():
        try:
            m = json.loads(tile_meta_path.read_text(encoding="utf-8"))
            for b in m.get("buildings", []) or []:
                bid = b.get("id")
                if bid:
                    if bid.startswith("osm_"):
                        osm_id_str = bid[4:]
                    else:
                        osm_id_str = bid
                    try:
                        height_by_id[int(osm_id_str)] = float(b["height_m"])
                    except ValueError:
                        height_by_id[osm_id_str] = float(b["height_m"])
        except Exception as e:  # noqa: BLE001
            print(f"[auto] height mapping parse exception: {e}")

    # B. Project individual building footprints and rasterize with their designated heights
    geo_path = ROOT / tile_root / tr.plan.name / "metadata" / f"{tr.plan.name}_osm_buildings.geojson"
    if geo_path.exists():
        try:
            gdf = gpd.read_file(geo_path)
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                geom_merc = reproject_geom(geom, fwd_wgs2merc)
                if geom_merc is None or geom_merc.is_empty:
                    continue
                fid = row.get("id")
                h_val = height_by_id.get(fid) or height_by_id.get(str(fid)) or 15.0
                rast = _rasterize([(geom_merc, h_val)], out_shape=(size, size),
                                   transform=transform, fill=0,
                                   default_value=h_val, dtype="float32",
                                   all_touched=False)
                bld_height_map = np.maximum(bld_height_map, rast)
        except Exception as e:  # noqa: BLE001
            print(f"[auto] building height mapping failed: {e}")

    # C. Fallback building height layer if geojson not present but building footprints exist in cls_wgs
    if (bld_height_map == 0).all() and "building" in cls_wgs:
        g_wgs = cls_wgs["building"]
        if g_wgs is not None and not g_wgs.is_empty:
            g_merc = reproject_geom(g_wgs, fwd_wgs2merc)
            if g_merc is not None and not g_merc.is_empty:
                rast = _rasterize([(g_merc, 15.0)], out_shape=(size, size),
                                   transform=transform, fill=0,
                                   default_value=15.0, dtype="float32",
                                   all_touched=False)
                bld_height_map = np.maximum(bld_height_map, rast)

    # D. Determine custom scaling factor
    pipe = _ACTIVE_PIPELINE
    xy_scale = pipe.cfg.topdown_tree_xy_scale if pipe is not None else 1.0

    tree_mask_path = ROOT / tile_root / tr.plan.name / "topview_tree_mask.png"
    tree_json = (ROOT / tile_root / tr.plan.name / "blender"
                 / "tree_instances.json")
    if tree_mask_path.exists():
        try:
            tree_mask = np.asarray(Image.open(tree_mask_path).convert("L")) > 0
        except Exception as e:  # noqa: BLE001
            print(f"[auto] {tr.plan.name} topview_tree_mask.png parse: {e}")
            tree_mask = np.zeros((size, size), dtype=bool)
        if tree_mask.shape != (size, size):
            tree_mask = np.zeros((size, size), dtype=bool)
        # This mask comes from the same top-down Blender render family as the
        # depth map, so it already represents the visible top surface. Trees
        # should be allowed to cover road/water/grass; building pixels remain
        # building only where the Blender render did not show tree foliage.
        is_foliage = tree_mask
        seg[is_foliage] = foliage_id
        n = int(is_foliage.sum())
        print(f"[auto] {tr.plan.name} composed topview_treeseg "
              f"from Blender tree mask ({n} foliage px) + "
              f"{n_canopy_polys} canopy polys (mercator-native base)")
    elif tree_json.exists():
        try:
            data = json.loads(tree_json.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[auto] {tr.plan.name} tree_instances.json parse: {e}")
            data = {"trees": []}
        trees = data.get("trees") or []

        # Create height and mask image to capture the physical footprint & tallest vertical profile of drawn tree crowns
        tree_height_img = Image.new("F", (size, size), 0.0)
        tree_mask_img = Image.new("L", (size, size), 0)
        tdraw = ImageDraw.Draw(tree_height_img)
        m_draw = ImageDraw.Draw(tree_mask_img)

        # UTM -> WGS so we can project each tree to mercator.
        utm_crs = str(tr.state["utm"])
        cx_u = float(tr.state["cx_utm"])
        cy_u = float(tr.state["cy_utm"])
        utm2wgs = make_transformer(utm_crs, "EPSG:4326")
        px_w = (E_m - W_m) / size  # mercator metres per pixel (x)
        px_h = (N_m - S_m) / size  # mercator metres per pixel (y)
        n = 0
        for t in trees:
            try:
                xc = float(t["x_centered"]); yc = float(t["y_centered"])
                h = float(t.get("h") or 0.0)
                r_xy_m = float(t.get("r_xy_m") or 0.0)
            except Exception:
                continue
            ux = cx_u + xc; uy = cy_u + yc
            try:
                pll = reproject_geom(_P(ux, uy), utm2wgs)
                lon, lat = float(pll.x), float(pll.y)
            except Exception:
                continue
            mx = R * math.radians(lon)
            my = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
            col = (mx - W_m) / px_w
            row = (N_m - my) / px_h
            # Keep 3D mesh stretch separate from 2D semantic labels. GN can
            # export a very large stretched mesh bbox; using it directly here
            # makes the whole canopy substrate green. For seg labels, draw a
            # conservative crown footprint and leave the canopy substrate as
            # grass so tree-vs-grass remains separable.
            crown_m = max(2.5, 0.35 * h) if h > 0.0 else 4.0
            r_m_ground = min(r_xy_m, crown_m) if r_xy_m > 0.0 else crown_m
            cos_lat = max(0.05, math.cos(math.radians(lat)))
            r_px = max(5.0, (r_m_ground / cos_lat / px_w) * xy_scale)
            tdraw.ellipse([col - r_px, row - r_px, col + r_px, row + r_px], fill=h)
            m_draw.ellipse([col - r_px, row - r_px, col + r_px, row + r_px], fill=1)
            n += 1

        # Use fast vectorized height comparisons
        tree_height_arr = np.array(tree_height_img)
        tree_mask = np.array(tree_mask_img) > 0
        
        building_id = CLASS_IDS["building"]

        # Mask rules:
        # 1. Trees are above road/water/grass/ground in top view. They only
        # compete with building footprints, where the taller surface wins.
        is_foliage = tree_mask & ((~building_mask) | (tree_height_arr >= bld_height_map))
        seg[is_foliage] = foliage_id

        # 2. Keep the pixel as building if building is taller than tree height
        is_occluded_by_building = tree_mask & building_mask & (tree_height_arr < bld_height_map) & (bld_height_map > 0)
        seg[is_occluded_by_building] = building_id

        print(f"[auto] {tr.plan.name} composed topview_treeseg "
              f"with {n} trees + {n_canopy_polys} canopy polys "
              f"(mercator-native) using height occlusion")
    else:
        print(f"[auto] {tr.plan.name} tree_instances.json missing; "
              "topview_treeseg will be tree-less")

    # ---- 3. Colorize final categorical map ---- #
    rgb = colorize_seg(seg, PALETTE)
    img = Image.fromarray(rgb).convert("RGB")
    img.save(out_png)


def _compose_lidar_depth_mercator(tr: TileRun, out_png: Path) -> None:
    """Compose a crisp 1024x1024 LiDAR-like DSM depth PNG in Mercator.

    This intentionally does not use Blender/Cycles color/depth rendering. It
    rasterizes footprints and GN tree instance crowns directly onto the final
    tile grid, using nearest/categorical writes so buildings and trees are
    sharp rather than shaded gradients.
    """
    import math
    import json
    from PIL import ImageDraw
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import Point as _P
    from dataprep.geometry_utils import make_transformer, reproject_geom
    from dataprep.osm_tags import CLASS_IDS

    cls_wgs = tr.state.get("cls_wgs") or {}
    bbox_wgs = tuple(float(x) for x in tr.state["bbox_wgs"])
    W_lon, S_lat, E_lon, N_lat = bbox_wgs
    size = int(tr.state.get("size") or 1024)

    R = 6378137.0
    W_m = R * math.radians(W_lon)
    E_m = R * math.radians(E_lon)
    N_m = R * math.log(math.tan(math.pi / 4 + math.radians(N_lat) / 2))
    S_m = R * math.log(math.tan(math.pi / 4 + math.radians(S_lat) / 2))
    transform = from_bounds(W_m, S_m, E_m, N_m, size, size)
    fwd_wgs2merc = make_transformer("EPSG:4326", "EPSG:3857")

    seg_base = np.zeros((size, size), dtype=np.uint8)
    height_m = np.zeros((size, size), dtype=np.float32)
    hard_surface_mask = np.zeros((size, size), dtype=bool)

    # Raster masks for hard surfaces; trees are not allowed to overwrite them.
    for name in ("road", "water", "building"):
        g_wgs = cls_wgs.get(name)
        if g_wgs is None or g_wgs.is_empty:
            continue
        g_merc = reproject_geom(g_wgs, fwd_wgs2merc)
        if g_merc is None or g_merc.is_empty:
            continue
        cls_id = CLASS_IDS[name]
        rast = _rasterize([(g_merc, cls_id)], out_shape=(size, size),
                          transform=transform, fill=0,
                          default_value=cls_id, dtype="uint8",
                          all_touched=False)
        hit = (rast > 0)
        if cls_id == 0:
            hit = _rasterize([(g_merc, 1)], out_shape=(size, size),
                             transform=transform, fill=0,
                             default_value=1, dtype="uint8",
                             all_touched=False) > 0
        seg_base[hit] = cls_id
        hard_surface_mask |= hit

    tile_root = oa.CFG["paths"]["tile_root"]

    # Building heights from per-building metadata when available.
    tile_meta_path = ROOT / tile_root / tr.plan.name / "metadata" / f"{tr.plan.name}.json"
    height_by_id = {}
    if tile_meta_path.exists():
        try:
            meta = json.loads(tile_meta_path.read_text(encoding="utf-8"))
            for b in meta.get("buildings", []) or []:
                bid = b.get("id")
                if not bid:
                    continue
                key = bid[4:] if str(bid).startswith("osm_") else bid
                try:
                    height_by_id[int(key)] = float(b["height_m"])
                except Exception:
                    height_by_id[str(key)] = float(b["height_m"])
        except Exception as e:  # noqa: BLE001
            print(f"[auto] LiDAR depth height metadata parse failed: {e}")

    geo_path = ROOT / tile_root / tr.plan.name / "metadata" / f"{tr.plan.name}_osm_buildings.geojson"
    wrote_building_heights = False
    if geo_path.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(geo_path)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                geom_merc = reproject_geom(geom, fwd_wgs2merc)
                if geom_merc is None or geom_merc.is_empty:
                    continue
                fid = row.get("id")
                h_val = float(height_by_id.get(fid) or
                              height_by_id.get(str(fid)) or 15.0)
                rast = _rasterize([(geom_merc, h_val)],
                                  out_shape=(size, size),
                                  transform=transform, fill=0,
                                  default_value=h_val, dtype="float32",
                                  all_touched=False)
                height_m = np.maximum(height_m, rast)
                wrote_building_heights = True
        except Exception as e:  # noqa: BLE001
            print(f"[auto] LiDAR depth building height raster failed: {e}")

    if not wrote_building_heights:
        building_mask = seg_base == CLASS_IDS["building"]
        height_m[building_mask] = np.maximum(height_m[building_mask], 15.0)

    tree_json = ROOT / tile_root / tr.plan.name / "blender" / "tree_instances.json"
    if tree_json.exists():
        try:
            trees = (json.loads(tree_json.read_text()).get("trees") or [])
        except Exception as e:  # noqa: BLE001
            print(f"[auto] LiDAR depth tree_instances parse failed: {e}")
            trees = []

        utm_crs = str(tr.state["utm"])
        cx_u = float(tr.state["cx_utm"])
        cy_u = float(tr.state["cy_utm"])
        utm2wgs = make_transformer(utm_crs, "EPSG:4326")
        px_w = (E_m - W_m) / size
        px_h = (N_m - S_m) / size
        pipe = _ACTIVE_PIPELINE
        xy_scale = pipe.cfg.topdown_tree_xy_scale if pipe is not None else 1.0
        yy, xx = np.ogrid[:size, :size]
        n_tree_pixels = 0
        n_trees = 0

        for t in sorted(trees, key=lambda x: float(x.get("h") or 0.0)):
            try:
                xc = float(t["x_centered"])
                yc = float(t["y_centered"])
                h = float(t.get("h") or 8.0)
            except Exception:
                continue
            try:
                pll = reproject_geom(_P(cx_u + xc, cy_u + yc), utm2wgs)
                lon, lat = float(pll.x), float(pll.y)
            except Exception:
                continue
            mx = R * math.radians(lon)
            my = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
            col = (mx - W_m) / px_w
            row = (N_m - my) / px_h
            r_m = max(5.0, 0.75 * h)
            cos_lat = max(0.05, math.cos(math.radians(lat)))
            r_px = max(6.0, (r_m / cos_lat / px_w) * xy_scale)
            # A LiDAR DSM crown is treated as a crisp canopy-return blob,
            # not a shaded 3D silhouette, so fill with one height value.
            mask = ((xx - col) ** 2 + (yy - row) ** 2) <= (r_px ** 2)
            mask &= ~hard_surface_mask
            if mask.any():
                height_m[mask] = np.maximum(height_m[mask], h)
                n_tree_pixels += int(mask.sum())
                n_trees += 1
        print(f"[auto] {tr.plan.name} LiDAR-like depth wrote "
              f"{n_trees} tree crowns ({n_tree_pixels} px)")

    # Normalize to a stable DSM visualization range. Values above 30m clip;
    # output remains a crisp 8-bit PNG companion while 5_depth.exr keeps float.
    depth = np.clip(height_m / 30.0 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(depth, mode="L").save(out_png)


def _compose_depth_proxy_from_seg(seg_png: Path, depth_png: Path) -> None:
    """Restore the historical grayscale 4_depth.png visual style.

    The Blender nDSM/EXR remains available as `5_depth.exr`; this PNG is the
    compact training/preview proxy used by the earlier pipeline: each semantic
    class maps to a stable grayscale level, so low-height context such as water,
    grass and foliage substrate remains visible instead of collapsing to black.
    """
    if not seg_png.exists():
        return
    palette = oa.CFG.get("class_colors", {})
    levels = {
        "ground": 0,
        "road": 6,
        "water": 65,
        "grass": 131,
        "foliage": 197,
        "building": 217,
    }
    seg = np.asarray(Image.open(seg_png).convert("RGB"))
    class_names = [name for name in levels if palette.get(name) is not None]
    if not class_names:
        return
    colors = np.asarray([palette[name] for name in class_names], dtype=np.float32)
    level_arr = np.asarray([levels[name] for name in class_names], dtype=np.uint8)
    flat = seg.reshape(-1, 3).astype(np.float32)
    d2 = ((flat[:, None, :] - colors[None, :, :]) ** 2).sum(axis=2)
    out = level_arr[np.argmin(d2, axis=1)].reshape(seg.shape[:2])
    Image.fromarray(out, mode="L").save(depth_png)


# --------------------------------------------------------------------- #
# Strategy caption (burned into city overviews) + sidecar JSON          #
# --------------------------------------------------------------------- #
def _strategy_caption_lines() -> list[str] | None:
    """Build a 2-line caption from the active pipeline's cfg.

    Returns ``None`` when no pipeline is active (e.g. the renderer is
    being called outside a pipeline run); callers then skip stamping.
    """
    pipe = _ACTIVE_PIPELINE
    if pipe is None:
        return None
    cfg = pipe.cfg
    tag = getattr(cfg, "_strategy_tag", None) or "default"
    line1 = (
        f"[{tag}] mode={cfg.scatter_mode} "
        f"density={cfg.tree_density:g} "
        f"prob_scale={cfg.canopy_prob_scale:g} "
        f"augment={cfg.procedural_augment_ratio:g}")
    line2 = (
        f"cluster=[{cfg.cluster_size_min},{cfg.cluster_size_max}] "
        f"r={cfg.cluster_disk_radius_max:g}m "
        f"cdist={cfg.cluster_size_dist}"
        f"({cfg.cluster_size_low_frac:g}) "
        f"hdist={cfg.tree_h_dist}"
        f"[{cfg.tree_h_min:g},{cfg.tree_h_max:g}]"
        f"({cfg.tree_height_low_frac:g}) "
        f"non_foliage={int(bool(cfg.allow_non_foliage))} "
        f"streets={int(bool(cfg.enable_street_trees))}")
    return [line1, line2]


def _stamp_strategy_caption(png_path: Path) -> None:
    """Burn a 2-line strategy caption into the top-left of an existing
    PNG. Uses PIL's default bitmap font so no font file is needed.
    Silently no-ops on any error — captioning is purely cosmetic."""
    try:
        lines = _strategy_caption_lines()
        if not lines:
            return
        from PIL import ImageDraw, ImageFont
        im = Image.open(png_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        # Measure max width.
        widths, heights = [], []
        for ln in lines:
            try:
                bbox = draw.textbbox((0, 0), ln, font=font)
                widths.append(bbox[2] - bbox[0])
                heights.append(bbox[3] - bbox[1])
            except Exception:
                widths.append(8 * len(ln)); heights.append(14)
        pad = 8
        box_w = max(widths) + 2 * pad
        line_h = max(heights) + 4
        box_h = line_h * len(lines) + 2 * pad
        # Translucent dark background strip.
        from PIL import Image as _Im
        overlay = _Im.new("RGBA", (box_w, box_h), (0, 0, 0, 170))
        im_rgba = im.convert("RGBA")
        im_rgba.paste(overlay, (0, 0), overlay)
        draw = ImageDraw.Draw(im_rgba)
        for i, ln in enumerate(lines):
            draw.text((pad, pad + i * line_h), ln,
                      fill=(255, 255, 255, 255), font=font)
        im_rgba.convert("RGB").save(png_path)
    except Exception as e:  # noqa: BLE001
        print(f"[agg] caption stamp failed for {png_path.name}: {e}")


def _dump_strategy_sidecar(city_dir: Path, cfg: "AutoPipelineConfig",
                              n_trees: int | None = None) -> None:
    """Write ``strategy.json`` next to the city overviews so each run
    is self-documenting (folder is enough to know what params produced
    it)."""
    try:
        tag = getattr(cfg, "_strategy_tag", None) or "default"
        payload = {
            "tag": tag,
            "city": cfg.city,
            "bbox_wgs": list(cfg.area_bbox_wgs),
            "scatter": {
                "mode": cfg.scatter_mode,
                "tree_density": cfg.tree_density,
                "canopy_prob_scale": cfg.canopy_prob_scale,
                "procedural_augment_ratio": cfg.procedural_augment_ratio,
                "allow_non_foliage": bool(cfg.allow_non_foliage),
                "enable_street_trees": bool(cfg.enable_street_trees),
                "cluster_size_min": cfg.cluster_size_min,
                "cluster_size_max": cfg.cluster_size_max,
                "cluster_disk_radius_min": cfg.cluster_disk_radius_min,
                "cluster_disk_radius_max": cfg.cluster_disk_radius_max,
                "cluster_size_dist": cfg.cluster_size_dist,
                "cluster_size_low_frac": cfg.cluster_size_low_frac,
                "tree_h_dist": cfg.tree_h_dist,
                "tree_h_min": cfg.tree_h_min,
                "tree_h_max": cfg.tree_h_max,
                "tree_height_low_frac": cfg.tree_height_low_frac,
                "scatter_seed": cfg.scatter_seed,
            },
        }
        if n_trees is not None:
            payload["n_trees_painted"] = int(n_trees)
        ensure_dir(city_dir / "metadata")
        (city_dir / "metadata" / "strategy.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[agg] strategy.json dump failed: {e}")


def _render_city_seg_mercator(city_dir: Path, runs: list[TileRun],
                                grid: dict) -> "np.ndarray | None":
    """Rasterize union OSM vectors in EPSG:3857 to match satellite grid.

    Writes ``city_overview_seg.png`` (the *only* city seg overview;
    the legacy per-tile mosaic version was retired because per-tile
    mercator squashes drifted at the seams). Each pixel matches the
    same lat/lon as the corresponding pixel in
    ``city_overview_satellite.png``.

    Returns the uint8 seg array (HxW, class ids) so callers can re-use
    it as a base canvas for downstream overlays (e.g. tree dots in
    :func:`_render_city_treeseg_mercator`). Returns ``None`` if both
    the C0 cache and the per-tile aggregation produce nothing usable.
    """
    from dataprep.raster_utils import colorize_seg
    PALETTE = oa.CFG["class_colors"]

    pipe = _ACTIVE_PIPELINE
    out_w, out_h = int(grid["out_w"]), int(grid["out_h"])
    if (pipe is not None and pipe.city_seg is not None
            and pipe.city_grid_info is not None):
        # Both grids share the same union bbox in mercator; if pixel
        # sizes differ (different target_long_px), nearest-resample.
        seg = pipe.city_seg
        if seg.shape != (out_h, out_w):
            seg = np.array(
                Image.fromarray(seg).resize((out_w, out_h), Image.NEAREST),
                dtype=np.uint8)
        Image.fromarray(colorize_seg(seg, PALETTE)).save(
            city_dir / "city_overview_seg.png")
        _stamp_strategy_caption(city_dir / "city_overview_seg.png")
        return seg

    # ---- Fallback: legacy per-tile aggregation ---- #
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import from_bounds
    from shapely.ops import unary_union
    from dataprep.geometry_utils import make_transformer, reproject_geom
    from dataprep.osm_tags import CLASS_IDS, CLASS_PRIORITY

    # Aggregate per-tile cls_wgs (already cached on tr.state from stage C).
    merged: dict[str, list] = {k: [] for k in CLASS_IDS}
    for tr in runs:
        cls = tr.state.get("cls_wgs")
        if not cls:
            continue
        for k, geom in cls.items():
            if k.startswith("_") or geom is None or geom.is_empty:
                continue
            if k in merged:
                merged[k].append(geom)
    union_geoms_wgs = {k: (unary_union(v) if v else None)
                        for k, v in merged.items()}

    # Reproject to EPSG:3857 (Web Mercator, units = metres).
    fwd = make_transformer("EPSG:4326", "EPSG:3857")
    union_geoms_merc = {
        k: (reproject_geom(g, fwd) if g is not None else None)
        for k, g in union_geoms_wgs.items()
    }

    # Mercator bounds of the satellite overview crop.
    import math
    R = 6378137.0
    def _merc_x(lon: float) -> float:
        return R * math.radians(lon)
    def _merc_y(lat: float) -> float:
        return R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    W, S, E, N = grid["ubox"]
    west_m = _merc_x(W); east_m = _merc_x(E)
    south_m = _merc_y(S); north_m = _merc_y(N)
    out_w, out_h = int(grid["out_w"]), int(grid["out_h"])
    transform = from_bounds(west_m, south_m, east_m, north_m, out_w, out_h)

    seg = np.zeros((out_h, out_w), dtype=np.uint8)  # 0 == ground
    order = sorted(CLASS_IDS.keys(), key=lambda k: CLASS_PRIORITY[k])
    for name in order:
        g = union_geoms_merc.get(name)
        if g is None or g.is_empty:
            continue
        cls_id = CLASS_IDS[name]
        out = _rasterize([(g, cls_id)], out_shape=(out_h, out_w),
                          transform=transform, fill=0,
                          default_value=cls_id, dtype="uint8",
                          all_touched=False)
        seg[out > 0] = cls_id

    Image.fromarray(colorize_seg(seg, PALETTE)).save(
        city_dir / "city_overview_seg.png")
    _stamp_strategy_caption(city_dir / "city_overview_seg.png")
    return seg


# --------------------------------------------------------------------- #
# City-level treeseg overlay (Phase B): paint every tile's              #
# tree_instances.json on top of the *aligned* city seg canvas. This     #
# bypasses the per-tile mosaic entirely so the result is guaranteed     #
# seamless and pixel-aligned to ``city_overview_satellite.png``.        #
# --------------------------------------------------------------------- #

def _render_city_treeseg_mercator(city_dir: Path, runs: list[TileRun],
                                    grid: dict,
                                    city_seg: np.ndarray) -> int:
    """Compose ``city_overview_treeseg.png`` by stitching per-tile
    Blender-rendered ``topview_treeseg.png`` files onto the city
    mercator grid.

    Each tile's PNG is already warped to mercator and matches the
    tile's lat/lon bbox exactly, so we just compute the destination
    pixel rectangle from the bbox and paste with a non-black mask so
    neighbouring tiles do not bleed each other's black backgrounds.
    A tile-count proxy is returned (sum of non-black pixel fractions
    only loosely correlates with tree count, so we simply report
    number of tiles successfully pasted).
    """
    import math
    out_w, out_h = int(grid["out_w"]), int(grid["out_h"])
    R = 6378137.0
    W, S, E, N = grid["ubox"]
    west_m = R * math.radians(W)
    east_m = R * math.radians(E)
    north_m = R * math.log(math.tan(math.pi / 4 + math.radians(N) / 2))
    south_m = R * math.log(math.tan(math.pi / 4 + math.radians(S) / 2))
    px_w = (east_m - west_m) / out_w
    px_h = (north_m - south_m) / out_h

    # Seamless base canvas: initialize with the colorized city_seg overview
    from dataprep.raster_utils import colorize_seg
    PALETTE = oa.CFG["class_colors"]
    rgb = colorize_seg(city_seg, PALETTE)
    canvas = Image.fromarray(rgb).convert("RGB")

    tile_root = oa.CFG["paths"]["tile_root"]
    n_pasted = 0
    for tr in runs:
        tile_png = ROOT / tile_root / tr.plan.name / "topview_treeseg.png"
        if not tile_png.exists():
            continue
        try:
            tw, ts, te, tn = (float(x) for x in tr.plan.bbox_wgs)
            mx_W = R * math.radians(tw)
            mx_E = R * math.radians(te)
            my_N = R * math.log(math.tan(math.pi / 4 + math.radians(tn) / 2))
            my_S = R * math.log(math.tan(math.pi / 4 + math.radians(ts) / 2))
            col_lo = int(round((mx_W - west_m) / px_w))
            col_hi = int(round((mx_E - west_m) / px_w))
            row_lo = int(round((north_m - my_N) / px_h))
            row_hi = int(round((north_m - my_S) / px_h))
            w_px = max(1, col_hi - col_lo)
            h_px = max(1, row_hi - row_lo)
            src = Image.open(tile_png).convert("RGB").resize(
                (w_px, h_px), Image.NEAREST)
            # Since topview_treeseg.png is now composed directly in Mercator
            # with seamless alignment, we can simply paste it onto the canvas.
            canvas.paste(src, (col_lo, row_lo))
            n_pasted += 1
        except Exception as e:  # noqa: BLE001
            print(f"[agg] {tr.plan.name} treeseg paste failed: {e}")

    canvas.save(city_dir / "city_overview_treeseg.png")
    _stamp_strategy_caption(city_dir / "city_overview_treeseg.png")
    print(f"[agg] city_overview_treeseg.png: stitched {n_pasted} "
          "per-tile Blender renders")
    return int(n_pasted)


def _render_city_depth_mercator(city_dir: Path, runs: list[TileRun],
                                 grid: dict) -> int:
    """Compose ``city_overview_depth.png`` by stitching per-tile
    Blender-rendered ``topview_depth.png`` files (nDSM, height above
    ground in metres, 16-bit normalized for visualisation).

    The raw per-tile ``topview_depth.exr`` files (float meters) are
    kept in their tile folders untouched; this overview is a stitched
    visualisation only.
    """
    import math
    out_w, out_h = int(grid["out_w"]), int(grid["out_h"])
    R = 6378137.0
    W, S, E, N = grid["ubox"]
    west_m = R * math.radians(W)
    east_m = R * math.radians(E)
    north_m = R * math.log(math.tan(math.pi / 4 + math.radians(N) / 2))
    south_m = R * math.log(math.tan(math.pi / 4 + math.radians(S) / 2))
    px_w = (east_m - west_m) / out_w
    px_h = (north_m - south_m) / out_h

    canvas = Image.new("L", (out_w, out_h), 0)
    tile_root = oa.CFG["paths"]["tile_root"]
    n_pasted = 0
    for tr in runs:
        tile_png = ROOT / tile_root / tr.plan.name / "topview_depth.png"
        if not tile_png.exists():
            continue
        try:
            tw, ts, te, tn = (float(x) for x in tr.plan.bbox_wgs)
            mx_W = R * math.radians(tw)
            mx_E = R * math.radians(te)
            my_N = R * math.log(math.tan(math.pi / 4 + math.radians(tn) / 2))
            my_S = R * math.log(math.tan(math.pi / 4 + math.radians(ts) / 2))
            col_lo = int(round((mx_W - west_m) / px_w))
            col_hi = int(round((mx_E - west_m) / px_w))
            row_lo = int(round((north_m - my_N) / px_h))
            row_hi = int(round((north_m - my_S) / px_h))
            w_px = max(1, col_hi - col_lo)
            h_px = max(1, row_hi - row_lo)
            src = Image.open(tile_png).convert("L").resize(
                (w_px, h_px), Image.NEAREST)
            src_arr = np.asarray(src)
            mask_arr = (src_arr > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask_arr, mode="L")
            canvas.paste(src, (col_lo, row_lo), mask)
            n_pasted += 1
        except Exception as e:  # noqa: BLE001
            print(f"[agg] {tr.plan.name} depth paste failed: {e}")

    canvas.save(city_dir / "city_overview_depth.png")
    print(f"[agg] city_overview_depth.png: stitched {n_pasted} "
          "per-tile nDSM renders")
    return int(n_pasted)


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--clean", action="store_true",
                    help="Wipe the output/<city> directory before running to force regeneration of all stages.")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                     metavar=("W", "S", "E", "N"),
                     help="WGS84 bbox of the area")
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--io-workers", type=int, default=8)
    ap.add_argument("--osm-workers", type=int, default=4)
    ap.add_argument("--canopy-workers", type=int, default=4)
    # ----- Tree clustering (canopy_driven scatter) -------------------- #
    ap.add_argument("--cluster-size-min", type=int, default=10,
                    help="min trees per cluster")
    ap.add_argument("--cluster-size-max", type=int, default=20,
                    help="max trees per cluster")
    ap.add_argument("--cluster-disk-radius-min", type=float, default=4.0,
                    help="min cluster disk radius (m)")
    ap.add_argument("--cluster-disk-radius-max", type=float, default=10.0,
                    help="max cluster disk radius (m)")
    ap.add_argument("--cluster-disk-aspect", type=float, default=0.65,
                    help="ellipse aspect ratio (1=circle)")
    ap.add_argument("--cluster-size-dist", default=None,
                    choices=["uniform", "bimodal", "beta_u"],
                    help="cluster-size shape (default uniform). bimodal"
                         " => mostly small clumps + a few big.")
    ap.add_argument("--cluster-size-low-frac", type=float, default=None,
                    help="bimodal cluster-size: fraction of clusters near the LOW end (0..1)")
    ap.add_argument("--tree-height-dist", default=None,
                    choices=["flat", "uniform", "lognormal", "bimodal",
                             "beta_u"],
                    help="per-tree height distribution")
    ap.add_argument("--tree-height-min", type=float, default=None,
                    help="min tree height (m)")
    ap.add_argument("--tree-height-max", type=float, default=None,
                    help="max tree height (m)")
    ap.add_argument("--tree-height-low-frac", type=float, default=None,
                    help="bimodal tree-height: fraction near LOW end (0..1)")
    ap.add_argument("--cluster-overlap-factor", type=float, default=None,
                    help="in-cluster spacing multiplier (<1 = crowns merge,"
                         " 1.0 = legacy hard reject). Default 0.45.")
    ap.add_argument("--cluster-min-keep-ratio", type=float, default=None,
                    help="drop fragment clusters smaller than "
                         "cluster_size_min*ratio. Default 0.6.")
    ap.add_argument("--cluster-min-size-abs", type=int, default=None,
                    help="absolute minimum trees per cluster; smaller"
                         " clusters get rolled back. 0 = disabled."
                         " Try 10-15 to kill small isolated fragments.")
    ap.add_argument("--target-foliage-ratio", type=float, default=None,
                    help="Target fraction of the tile covered by green canopy (0.001 to 1.0). "
                         "Bumps up the height threshold so only tallest portion of forest is kept (e.g. 0.1).")
    # ----- Vegetation realism knobs (Phase-2 scatter) ----------------- #
    ap.add_argument("--tree-density", type=float, default=None,
                    help="overall tree density (#/m^2)")
    ap.add_argument("--scatter-mode", default=None,
                    choices=["canopy_prob", "canopy_prob_streets",
                             "cluster", "poisson_disk", "canopy_driven",
                             "linear_corridor", "noise_forest", "cp_nf_hybrid"],
                    help="scatter mode")
    ap.add_argument("--canopy-prob-scale", type=float, default=None,
                    help="multiplier on per-cell ETH canopy probability")
    ap.add_argument("--procedural-augment-ratio", type=float, default=None,
                    help="extra procedural trees on top of ETH (0..1)")
    ap.add_argument("--allow-non-foliage", dest="allow_non_foliage",
                    action="store_true", default=None,
                    help="allow trees on non-foliage land (grass etc.)")
    ap.add_argument("--no-allow-non-foliage", dest="allow_non_foliage",
                    action="store_false",
                    help="restrict trees to OSM foliage only")
    ap.add_argument("--enable-street-trees", dest="enable_street_trees",
                    action="store_true", default=None,
                    help="also place trees along road centrelines")
    ap.add_argument("--uniform-tree-scale", dest="uniform_tree_scale",
                    action="store_true", default=None,
                    help="scale trees uniformly based on height")
    ap.add_argument("--no-uniform-tree-scale", dest="uniform_tree_scale",
                    action="store_false",
                    help="do not scale trees uniformly based on height")
    ap.add_argument("--topdown-tree-xy-scale", type=float, default=None,
                    help="Crown horizontal scaling factor for 2D/3D trees")
    ap.add_argument("--gn-tree-amount", type=float, default=None,
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
    ap.add_argument("--gn-xy-stretch", type=float, default=None,
                    help="0..1 GN instance XY stretch control")
    ap.add_argument("--gn-z-stretch", type=float, default=None,
                    help="0..1 GN instance Z stretch control")
    ap.add_argument("--gn-xy-stretch-min-at-0", type=float, default=None,
                    help="XY stretch min when GN XY stretch slider = 0")
    ap.add_argument("--gn-xy-stretch-min-at-1", type=float, default=None,
                    help="XY stretch min when GN XY stretch slider = 1")
    ap.add_argument("--gn-xy-stretch-max-at-0", type=float, default=None,
                    help="XY stretch max when GN XY stretch slider = 0")
    ap.add_argument("--gn-xy-stretch-max-at-1", type=float, default=None,
                    help="XY stretch max when GN XY stretch slider = 1")
    ap.add_argument("--gn-z-stretch-min-at-0", type=float, default=None,
                    help="Z stretch min when GN Z stretch slider = 0")
    ap.add_argument("--gn-z-stretch-min-at-1", type=float, default=None,
                    help="Z stretch min when GN Z stretch slider = 1")
    ap.add_argument("--gn-z-stretch-max-at-0", type=float, default=None,
                    help="Z stretch max when GN Z stretch slider = 0")
    ap.add_argument("--gn-z-stretch-max-at-1", type=float, default=None,
                    help="Z stretch max when GN Z stretch slider = 1")
    ap.add_argument("--use-blender-seg", dest="use_blender_seg",
                    action="store_true", default=None,
                    help="Use direct Blender-rendered topview for segmentation to guarantee 1-to-1 depth alignment")
    ap.add_argument("--no-blender-seg", dest="use_blender_seg",
                    action="store_false",
                    help="Do not use direct Blender-rendered topview; use PIL-composed Mercator layout")
    ap.add_argument("--strategy-tag", default=None,
                    help="short label baked into overview PNGs / sidecar")
    args = ap.parse_args()

    cfg_kwargs = dict(
        city=args.city,
        area_bbox_wgs=tuple(args.bbox),
        overlap=float(args.overlap),
        io_workers=args.io_workers,
        osm_workers=args.osm_workers,
        canopy_workers=args.canopy_workers,
        cluster_size_min=int(args.cluster_size_min),
        cluster_size_max=int(args.cluster_size_max),
        cluster_disk_radius_min=float(args.cluster_disk_radius_min),
        cluster_disk_radius_max=float(args.cluster_disk_radius_max),
        cluster_disk_aspect=float(args.cluster_disk_aspect),
    )
    if args.tree_density is not None:
        cfg_kwargs["tree_density"] = float(args.tree_density)
    if args.scatter_mode is not None:
        cfg_kwargs["scatter_mode"] = str(args.scatter_mode)
    if args.canopy_prob_scale is not None:
        cfg_kwargs["canopy_prob_scale"] = float(args.canopy_prob_scale)
    if args.procedural_augment_ratio is not None:
        cfg_kwargs["procedural_augment_ratio"] = float(
            args.procedural_augment_ratio)
    if args.allow_non_foliage is not None:
        cfg_kwargs["allow_non_foliage"] = bool(args.allow_non_foliage)
    if args.enable_street_trees is not None:
        cfg_kwargs["enable_street_trees"] = bool(args.enable_street_trees)
    if args.cluster_size_dist is not None:
        cfg_kwargs["cluster_size_dist"] = str(args.cluster_size_dist)
    if args.cluster_size_low_frac is not None:
        cfg_kwargs["cluster_size_low_frac"] = float(args.cluster_size_low_frac)
    if args.tree_height_dist is not None:
        cfg_kwargs["tree_h_dist"] = str(args.tree_height_dist)
    if args.tree_height_min is not None:
        cfg_kwargs["tree_h_min"] = float(args.tree_height_min)
    if args.tree_height_max is not None:
        cfg_kwargs["tree_h_max"] = float(args.tree_height_max)
    if args.tree_height_low_frac is not None:
        cfg_kwargs["tree_height_low_frac"] = float(args.tree_height_low_frac)
    if args.cluster_overlap_factor is not None:
        cfg_kwargs["cluster_overlap_factor"] = float(args.cluster_overlap_factor)
    if args.cluster_min_keep_ratio is not None:
        cfg_kwargs["cluster_min_keep_ratio"] = float(args.cluster_min_keep_ratio)
    if args.cluster_min_size_abs is not None:
        cfg_kwargs["cluster_min_size_abs"] = int(args.cluster_min_size_abs)
    if args.target_foliage_ratio is not None:
        cfg_kwargs["target_foliage_ratio"] = float(args.target_foliage_ratio)
    if args.uniform_tree_scale is not None:
        cfg_kwargs["uniform_tree_scale"] = bool(args.uniform_tree_scale)
    if args.topdown_tree_xy_scale is not None:
        cfg_kwargs["topdown_tree_xy_scale"] = float(args.topdown_tree_xy_scale)
    if args.gn_tree_amount is not None:
        cfg_kwargs["gn_tree_amount"] = float(args.gn_tree_amount)
    if args.gn_safe_building is not None:
        cfg_kwargs["gn_safe_building"] = float(args.gn_safe_building)
    if args.gn_safe_road is not None:
        cfg_kwargs["gn_safe_road"] = float(args.gn_safe_road)
    if args.gn_safe_water is not None:
        cfg_kwargs["gn_safe_water"] = float(args.gn_safe_water)
    if args.gn_noise_scale is not None:
        cfg_kwargs["gn_noise_scale"] = float(args.gn_noise_scale)
    if args.gn_min_distance is not None:
        cfg_kwargs["gn_min_distance"] = float(args.gn_min_distance)
    if args.gn_xy_stretch is not None:
        cfg_kwargs["gn_xy_stretch"] = float(args.gn_xy_stretch)
    if args.gn_z_stretch is not None:
        cfg_kwargs["gn_z_stretch"] = float(args.gn_z_stretch)
    if args.gn_xy_stretch_min_at_0 is not None:
        cfg_kwargs["gn_xy_stretch_min_at_0"] = float(args.gn_xy_stretch_min_at_0)
    if args.gn_xy_stretch_min_at_1 is not None:
        cfg_kwargs["gn_xy_stretch_min_at_1"] = float(args.gn_xy_stretch_min_at_1)
    if args.gn_xy_stretch_max_at_0 is not None:
        cfg_kwargs["gn_xy_stretch_max_at_0"] = float(args.gn_xy_stretch_max_at_0)
    if args.gn_xy_stretch_max_at_1 is not None:
        cfg_kwargs["gn_xy_stretch_max_at_1"] = float(args.gn_xy_stretch_max_at_1)
    if args.gn_z_stretch_min_at_0 is not None:
        cfg_kwargs["gn_z_stretch_min_at_0"] = float(args.gn_z_stretch_min_at_0)
    if args.gn_z_stretch_min_at_1 is not None:
        cfg_kwargs["gn_z_stretch_min_at_1"] = float(args.gn_z_stretch_min_at_1)
    if args.gn_z_stretch_max_at_0 is not None:
        cfg_kwargs["gn_z_stretch_max_at_0"] = float(args.gn_z_stretch_max_at_0)
    if args.gn_z_stretch_max_at_1 is not None:
        cfg_kwargs["gn_z_stretch_max_at_1"] = float(args.gn_z_stretch_max_at_1)
    if args.use_blender_seg is not None:
        cfg_kwargs["use_blender_seg"] = bool(args.use_blender_seg)

    if args.clean:
        import shutil
        city_dir = ROOT / "output" / args.city
        if city_dir.exists():
            print(f"[auto] --clean specified. Wiping output directory of {args.city} to force clean rebuild: {city_dir}")
            shutil.rmtree(city_dir, ignore_errors=True)

    cfg = AutoPipelineConfig(**cfg_kwargs)
    # Stash optional strategy tag for the aggregator (not a config field).
    cfg._strategy_tag = args.strategy_tag  # type: ignore[attr-defined]

    def _print_progress(stage, name, status):
        print(f"  [{stage}] {name:<14s} {status}", flush=True)

    pipe = AutoPipeline(cfg, progress_cb=_print_progress)
    runs = pipe.plan()
    n_rows, n_cols = grid_shape([tr.plan for tr in runs])
    print(f"[auto] city={args.city}  grid={n_rows}x{n_cols}={len(runs)} tiles",
          flush=True)
    summary = pipe.run()
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------- #
# Public helper: run pipeline on a single tile (used by osm_app UI)     #
# --------------------------------------------------------------------- #
def run_single_tile(cfg: "AutoPipelineConfig",
                    progress_cb: Callable[[str, str, str], None] | None = None
                    ) -> "TileRun":
    """Run all stages (B→H) on a tile-sized ``cfg`` and return its
    single ``TileRun``.

    The caller is responsible for setting ``cfg.area_bbox_wgs`` to a
    bbox that fits in exactly one tile (≈ ``size_px * gsd`` metres on
    a side ≈ 512 m at default 1024 px * 0.5 m/px). If the bbox produces
    multiple tiles, the first one is returned.

    This thin wrapper exists so the Gradio debug UI can re-use the
    exact same code path as the batch CLI — guaranteeing the single-
    tile preview is byte-equivalent to what ``auto_pipeline`` produces
    for that same tile in a batch run.
    """
    import shutil
    # For single tile UI runs, we always want to force regenerate to prevent
    # stale folder renames or caching bugs when users repeatedly hit Generate All.
    city_dir = ROOT / "output" / cfg.city
    if city_dir.exists():
        shutil.rmtree(city_dir, ignore_errors=True)

    pipe = AutoPipeline(cfg, progress_cb=progress_cb)
    pipe.plan(force_clean=True)
    if not pipe.runs:
        raise RuntimeError(
            f"run_single_tile: planner produced 0 tiles for bbox "
            f"{cfg.area_bbox_wgs}")
    pipe.run()
    tr = pipe.runs[0]
    failed = {s: tr.errors.get(s, tr.status.get(s, "?"))
              for s in ("B", "C", "D", "E", "F")
              if tr.status.get(s) != "ok"}
    if failed:
        raise RuntimeError(f"single-tile pipeline failed: {failed}")
    return tr


def _run_stage_f_from_config(tr: TileRun, cfg: "AutoPipelineConfig") -> TileRun:
    return _stage_f_one(
        tr, cfg.scatter_seed, cfg.tree_density,
        cfg.tree_species, cfg.tree_h_dist, cfg.tree_h_seed,
        cfg.tree_h_min, cfg.tree_h_max, city=cfg.city,
        cluster_size_min=cfg.cluster_size_min,
        cluster_size_max=cfg.cluster_size_max,
        cluster_disk_radius_min=cfg.cluster_disk_radius_min,
        cluster_disk_radius_max=cfg.cluster_disk_radius_max,
        cluster_disk_aspect=cfg.cluster_disk_aspect,
        cluster_size_dist=cfg.cluster_size_dist,
        cluster_size_low_frac=cfg.cluster_size_low_frac,
        tree_height_low_frac=cfg.tree_height_low_frac,
        cluster_overlap_factor=cfg.cluster_overlap_factor,
        cluster_min_keep_ratio=cfg.cluster_min_keep_ratio,
        cluster_min_size_abs=cfg.cluster_min_size_abs,
        scatter_mode=cfg.scatter_mode,
        allow_non_foliage=cfg.allow_non_foliage,
        enable_street_trees=cfg.enable_street_trees,
        procedural_augment_ratio=cfg.procedural_augment_ratio,
        canopy_prob_scale=cfg.canopy_prob_scale,
        uniform_tree_scale=cfg.uniform_tree_scale,
        topdown_tree_xy_scale=cfg.topdown_tree_xy_scale,
        gn_tree_amount=cfg.gn_tree_amount,
        gn_safe_building=cfg.gn_safe_building,
        gn_safe_road=cfg.gn_safe_road,
        gn_safe_water=cfg.gn_safe_water,
        gn_noise_scale=cfg.gn_noise_scale,
        gn_min_distance=cfg.gn_min_distance,
        gn_xy_stretch=cfg.gn_xy_stretch,
        gn_z_stretch=cfg.gn_z_stretch,
        gn_xy_stretch_min_at_0=cfg.gn_xy_stretch_min_at_0,
        gn_xy_stretch_min_at_1=cfg.gn_xy_stretch_min_at_1,
        gn_xy_stretch_max_at_0=cfg.gn_xy_stretch_max_at_0,
        gn_xy_stretch_max_at_1=cfg.gn_xy_stretch_max_at_1,
        gn_z_stretch_min_at_0=cfg.gn_z_stretch_min_at_0,
        gn_z_stretch_min_at_1=cfg.gn_z_stretch_min_at_1,
        gn_z_stretch_max_at_0=cfg.gn_z_stretch_max_at_0,
        gn_z_stretch_max_at_1=cfg.gn_z_stretch_max_at_1)


def rerun_trees_only(cfg: "AutoPipelineConfig",
                     tile_names: Sequence[str] | None = None,
                     progress_cb: Callable[[str, str, str], None] | None = None
                     ) -> list["TileRun"]:
    """Reuse existing OSM/GLB artifacts and rerun only KR3 tree outputs.

    This is the fast UI path for parameter iteration: no satellite fetch,
    no OSM fetch, no canopy rebuild, and no KR2 geometry rebuild. It refreshes
    the Blender scene plus final ``4_seg.png``, ``5_depth.png`` and
    ``5_depth.exr`` from the existing tile folder.
    """
    cb = progress_cb or (lambda *a: None)
    pipe = AutoPipeline(cfg, progress_cb=cb)
    if tile_names:
        pipe.runs = [TileRun(_tile_plan_from_existing_output(cfg, name))
                     for name in tile_names]
    else:
        pipe.plan(force_clean=False)
    if not pipe.runs:
        raise RuntimeError("rerun_trees_only: no tiles to rerun")

    with _redirect_cfg_paths(cfg.city):
        global _ACTIVE_PIPELINE
        _ACTIVE_PIPELINE = pipe
        try:
            for tr in pipe.runs:
                _ensure_tree_rerun_aliases(cfg.city, tr.plan.name)
                _hydrate_tree_rerun_state(cfg.city, tr)
                glb = ROOT / "output" / cfg.city / tr.plan.name / "blender" / f"{tr.plan.name}.glb"
                if not glb.exists():
                    tr.status["F"] = "failed"
                    tr.errors["F"] = f"existing GLB not found: {glb}"
                    cb("F", tr.plan.name, "failed")
                    continue
                tr.status.update({"B": "ok", "C": "ok", "D": "ok", "E": "ok"})
                tr.status.pop("F", None)
                _run_stage_f_from_config(tr, cfg)
                cb("F", tr.plan.name, tr.status.get("F", "?"))
        finally:
            _ACTIVE_PIPELINE = None
    pipe._save_state()
    return pipe.runs



if __name__ == "__main__":
    main()

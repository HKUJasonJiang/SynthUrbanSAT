"""Rasterize 6-class polygons to a square uint8 PNG at given GSD.

Pure-Python (rasterio.features). No Blender. Used by:
  * the gradio app for live seg / depth-proxy preview
  * sanity-check during KR1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from .geometry_utils import make_transformer, reproject_geom, utm_crs_for_bbox
from .osm_tags import CLASS_IDS, CLASS_PRIORITY


def tile_transform(cx_utm: float, cy_utm: float, gsd: float, size: int):
    """Return rasterio Affine for an ortho tile centered at (cx,cy) in UTM.

    Image rows go N→S (top row = north). `cy_utm` is the tile center.
    """
    half = gsd * size / 2
    west = cx_utm - half
    north = cy_utm + half
    return from_origin(west, north, gsd, gsd)


def class_geoms_to_local(class_geoms_wgs: dict, bbox_wgs):
    """WGS84 → local UTM. Returns (dict[name -> Polygon/MultiPolygon in UTM], crs).

    Underscore-prefixed keys (e.g. ``_buildings_with_id`` GeoDataFrame
    sidecars) are passed through untouched — they aren't single
    shapely geometries.
    """
    utm = utm_crs_for_bbox(bbox_wgs)
    tr = make_transformer("EPSG:4326", utm)
    out = {}
    for k, g in class_geoms_wgs.items():
        if isinstance(k, str) and k.startswith("_"):
            out[k] = g
            continue
        if g is None or g.is_empty:
            out[k] = None
            continue
        out[k] = reproject_geom(g, tr)
    return out, utm


def rasterize_seg(class_geoms_utm: dict, cx_utm: float, cy_utm: float,
                  gsd: float, size: int) -> np.ndarray:
    """Return uint8 HxW array, values in {0..5} (CLASS_IDS).

    Painters algorithm by CLASS_PRIORITY (low first, high overwrites).
    """
    transform = tile_transform(cx_utm, cy_utm, gsd, size)
    ground_id = CLASS_IDS["ground"]
    seg = np.full((size, size), ground_id, dtype=np.uint8)
    order = sorted(CLASS_IDS.keys(), key=lambda k: CLASS_PRIORITY[k])
    for name in order:
        g = class_geoms_utm.get(name)
        if g is None or g.is_empty:
            continue
        cls_id = CLASS_IDS[name]
        shapes = [(g, cls_id)]
        out = rasterize(
            shapes, out_shape=(size, size), transform=transform,
            fill=ground_id, default_value=cls_id,
            dtype="uint8", all_touched=False,
        )
        if cls_id == 0:
            hit = rasterize(
                [(g, 1)], out_shape=(size, size), transform=transform,
                fill=0, default_value=1, dtype="uint8", all_touched=False,
            ) > 0
        else:
            hit = out == cls_id
        seg[hit] = cls_id
    return seg


def colorize_seg(seg: np.ndarray, palette: dict) -> np.ndarray:
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for name, idx in CLASS_IDS.items():
        m = seg == idx
        if m.any():
            out[m] = palette[name]
    return out


def compose_seg_with_foliage_mask(
    seg_osm: np.ndarray, foliage_mask: np.ndarray,
) -> np.ndarray:
    """Stamp a SAM3-derived foliage binary mask onto an OSM-rasterized seg.

    Priority is the LOWEST: foliage only overwrites pixels that are still
    `ground` after OSM rasterization. Any OSM-confirmed class
    (road/water/building/grass) is kept verbatim — this prevents SAM3 from
    repainting green-tile rooftops or grass parks as foliage.

    Args:
        seg_osm: (H,W) uint8 with class IDs from osm_tags.CLASS_IDS.
        foliage_mask: (H,W) bool or 0/255 uint8. Must match seg shape.
    Returns:
        new (H,W) uint8 seg with foliage class painted onto eligible pixels.
    """
    if seg_osm.shape != foliage_mask.shape:
        raise ValueError(
            f"shape mismatch: seg {seg_osm.shape} vs mask {foliage_mask.shape}"
        )
    out = seg_osm.copy()
    G = CLASS_IDS["ground"]
    F = CLASS_IDS["foliage"]
    fol_b = foliage_mask.astype(bool) if foliage_mask.dtype != bool \
        else foliage_mask
    take = (out == G) & fol_b
    out[take] = F
    return out


def rasterize_depth_proxy(class_geoms_utm: dict, cx_utm: float, cy_utm: float,
                          gsd: float, size: int,
                          building_h: float = 7.0,
                          camera_h: float = 500.0) -> np.ndarray:
    """Cheap depth preview: depth = camera_h - height(z) at each pixel.

    Buildings extruded to `building_h`, all else z=0. NOT the final depth —
    Blender produces that. Useful only for app-side sanity check.
    """
    transform = tile_transform(cx_utm, cy_utm, gsd, size)
    z = np.zeros((size, size), dtype=np.float32)
    g = class_geoms_utm.get("building")
    if g is not None and not g.is_empty:
        out = rasterize(
            [(g, building_h)], out_shape=(size, size), transform=transform,
            fill=0.0, dtype="float32",
        )
        z = np.maximum(z, out)
    depth = camera_h - z  # meters from camera plane
    return depth.astype(np.float32)


def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-3:
        return np.zeros_like(depth, dtype=np.uint8)
    n = (depth - lo) / (hi - lo)
    return (n * 255).astype(np.uint8)

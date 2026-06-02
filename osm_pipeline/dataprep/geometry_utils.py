"""Geometry helpers shared by KR1/KR2/KR5.

Pure-Python (shapely + pyproj + trimesh). No bpy imports here so this module
is importable from both the standard interpreter and Blender's bundled Python.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, mapping, shape,
)
from shapely.ops import transform as shp_transform, unary_union


# ----------------------------- CRS helpers ------------------------------- #

def utm_crs_for_bbox(bbox: tuple[float, float, float, float]) -> CRS:
    """Pick the UTM zone whose central meridian is closest to bbox center.

    bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    cx = 0.5 * (min_lon + max_lon)
    cy = 0.5 * (min_lat + max_lat)
    zone = int(np.floor((cx + 180) / 6) + 1)
    south = cy < 0
    epsg = (32700 if south else 32600) + zone
    return CRS.from_epsg(epsg)


def make_transformer(src: CRS, dst: CRS) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def reproject_geom(geom, transformer: Transformer):
    return shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


# --------------------------- Geometry cleaning --------------------------- #

def to_multipolygon(geom) -> MultiPolygon | None:
    """Coerce arbitrary geometry to MultiPolygon; drop non-areal parts."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom]) if geom.is_valid else None
    if isinstance(geom, MultiPolygon):
        return geom
    # try buffer(0) for line/point fallbacks
    try:
        g = geom.buffer(0)
        if isinstance(g, Polygon):
            return MultiPolygon([g])
        if isinstance(g, MultiPolygon):
            return g
    except Exception:
        pass
    return None


def buffer_lines(lines: Iterable, half_width_m: float):
    """Buffer LineString/MultiLineString to flat-cap polygon strips."""
    bufs = []
    for ln in lines:
        if ln is None or ln.is_empty:
            continue
        bufs.append(ln.buffer(half_width_m, cap_style=2, join_style=2))
    if not bufs:
        return None
    return unary_union(bufs)


def clip_to_bbox(geom, bbox_poly: Polygon):
    if geom is None:
        return None
    g = geom.intersection(bbox_poly)
    return g if not g.is_empty else None


# ---------------------------- Mesh building ------------------------------ #

def triangulate_polygon(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Return (verts Nx2, faces Mx3) for a 2D polygon using mapbox_earcut.

    mapbox_earcut expects `ring_indices` as the END index of each ring; the
    last value MUST equal the total vertex count.
    """
    import mapbox_earcut as earcut

    if poly.is_empty:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    ext = np.asarray(poly.exterior.coords)[:-1]  # drop closing vertex
    rings = [ext]
    ring_ends = [len(ext)]
    cursor = len(ext)
    for hole in poly.interiors:
        h = np.asarray(hole.coords)[:-1]
        rings.append(h)
        cursor += len(h)
        ring_ends.append(cursor)
    verts = np.concatenate(rings, axis=0)
    tri = earcut.triangulate_float64(verts, np.array(ring_ends, dtype=np.uint32))
    faces = np.asarray(tri).reshape(-1, 3)
    return verts, faces


def extrude_polygon(poly: Polygon, base_z: float, top_z: float):
    """Extrude a 2D polygon between base_z..top_z. Returns (verts Nx3, faces Mx3)."""
    import trimesh

    height = max(top_z - base_z, 1e-3)
    mesh = trimesh.creation.extrude_polygon(poly, height=height)
    # extrude_polygon places base at z=0; lift to base_z
    mesh.apply_translation((0.0, 0.0, base_z))
    return mesh


def flat_polygon_mesh(poly: Polygon, z: float = 0.0):
    """Triangulate a flat polygon at given z. Returns trimesh.Trimesh."""
    import trimesh
    v2, f = triangulate_polygon(poly)
    if len(v2) == 0 or len(f) == 0:
        return None
    v = np.column_stack([v2, np.full(len(v2), z)])
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def merge_meshes(meshes: list):
    import trimesh
    meshes = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


# ----------------------------- IO helpers -------------------------------- #

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

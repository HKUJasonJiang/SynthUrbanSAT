"""Tile-grid planner for the auto pipeline.

Given a WGS84 bbox (the user-selected area on the overview map), a fixed
GSD and tile size in pixels, return a row-major list of sub-tiles that
tile the area. Each tile is `gsd * size_px` metres on a side. Optional
overlap (0..0.5) shrinks the stride so adjacent tiles share `overlap *
tile_m` metres on each shared edge.

Tile ids are zero-padded ``tile_NNNN`` (1-based), row-major from the
**north-west** corner: row 0 is the topmost (largest latitude), col 0 is
the westmost (smallest longitude).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

from shapely.geometry import Point

from .geometry_utils import make_transformer, reproject_geom, utm_crs_for_bbox


@dataclass
class TilePlan:
    name: str           # "tile_0001"
    row: int            # 0-based, NW origin (top-down)
    col: int            # 0-based, NW origin (left-right = west-east)
    lat: float          # tile center latitude (WGS84)
    lon: float          # tile center longitude (WGS84)
    bbox_wgs: tuple     # (W, S, E, N) of THIS tile

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bbox_wgs"] = list(self.bbox_wgs)
        return d


def plan_tiles(area_bbox_wgs: tuple,
               gsd: float = 0.5,
               size_px: int = 1024,
               overlap: float = 0.0) -> List[TilePlan]:
    """Tile an arbitrary WGS84 bbox into fixed-size sub-tiles.

    Args:
        area_bbox_wgs: (W, S, E, N) in degrees.
        gsd: metres per pixel for each tile.
        size_px: side length of each tile in pixels.
        overlap: fraction in [0, 0.5). 0 = no overlap, 0.1 = 10% overlap.

    Returns:
        Row-major list of TilePlan, NW origin. The grid covers the
        full input bbox (last row/col may extend slightly beyond if the
        input isn't an exact multiple of the tile size).
    """
    if not (0.0 <= overlap < 0.5):
        raise ValueError(f"overlap must be in [0, 0.5), got {overlap}")
    W, S, E, N = (float(x) for x in area_bbox_wgs)
    if not (E > W and N > S):
        raise ValueError(f"bad bbox (need E>W and N>S): {area_bbox_wgs}")

    tile_m = float(gsd) * int(size_px)
    stride_m = tile_m * (1.0 - float(overlap))

    # Local UTM for honest metric tiling.
    utm = utm_crs_for_bbox((W, S, E, N))
    fwd = make_transformer("EPSG:4326", utm)
    inv = make_transformer(utm, "EPSG:4326")

    sw = reproject_geom(Point(W, S), fwd)
    ne = reproject_geom(Point(E, N), fwd)
    minx, miny = sw.x, sw.y
    maxx, maxy = ne.x, ne.y
    width_m = maxx - minx
    height_m = maxy - miny

    # Number of cells needed to cover the area; ceil so we always cover
    # the whole bbox (last column may extend a bit east/last row south).
    import math
    n_cols = max(1, math.ceil((width_m - tile_m) / stride_m) + 1) \
        if width_m > tile_m else 1
    n_rows = max(1, math.ceil((height_m - tile_m) / stride_m) + 1) \
        if height_m > tile_m else 1

    plans: list[TilePlan] = []
    idx = 1
    half = tile_m / 2.0
    for row in range(n_rows):
        # row 0 is NORTH (top): y_center starts near maxy and decreases.
        cy_utm = maxy - half - row * stride_m
        for col in range(n_cols):
            cx_utm = minx + half + col * stride_m
            sw_t = reproject_geom(Point(cx_utm - half, cy_utm - half), inv)
            ne_t = reproject_geom(Point(cx_utm + half, cy_utm + half), inv)
            center = reproject_geom(Point(cx_utm, cy_utm), inv)
            plans.append(TilePlan(
                name=f"tile_{idx:04d}",
                row=row,
                col=col,
                lat=float(center.y),
                lon=float(center.x),
                bbox_wgs=(float(sw_t.x), float(sw_t.y),
                          float(ne_t.x), float(ne_t.y)),
            ))
            idx += 1

    return plans


def grid_shape(plans: List[TilePlan]) -> tuple[int, int]:
    """Return (n_rows, n_cols) of a row-major TilePlan list."""
    if not plans:
        return (0, 0)
    n_rows = max(p.row for p in plans) + 1
    n_cols = max(p.col for p in plans) + 1
    return (n_rows, n_cols)


def grid_id_array(plans: List[TilePlan]) -> List[List[str]]:
    """Return a [row][col] -> tile_name 2D list."""
    n_rows, n_cols = grid_shape(plans)
    arr: list[list[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for p in plans:
        arr[p.row][p.col] = p.name
    return arr


def area_bbox_union(plans: List[TilePlan]) -> tuple:
    """Return the WGS84 (W,S,E,N) union of all tile bboxes."""
    Ws = [p.bbox_wgs[0] for p in plans]
    Ss = [p.bbox_wgs[1] for p in plans]
    Es = [p.bbox_wgs[2] for p in plans]
    Ns = [p.bbox_wgs[3] for p in plans]
    return (min(Ws), min(Ss), max(Es), max(Ns))

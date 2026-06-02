"""City-level mercator grid for seamless multi-tile rasterization.

When the auto-pipeline processes multiple tiles, each tile's seg PNG
must align with adjacent tiles at their shared boundary. The historical
per-tile rasterization (each tile renders its own ``cls_wgs`` into a
1024×1024 mercator grid using ``from_bounds(tile_merc_bounds, 1024,
1024)``) drifts by sub-pixel amounts at boundaries and, more crucially,
fetches OSM independently per tile, which can produce slightly
different topology for features crossing the boundary.

This module solves both problems:

  1. ``compute_city_grid_info`` builds a single mercator pixel grid
     covering the whole union bbox at the same zoom that the satellite
     overview uses. All tiles will sub-slice this one grid.
  2. ``rasterize_city_seg`` rasterizes the unified ``city_cls_wgs`` once
     onto that grid, producing a uint8 seg array.
  3. ``tile_pixel_subrect`` computes the integer pixel rect for a given
     tile's bbox within the global grid, so per-tile seg PNGs can be
     extracted as exact slices (no per-tile re-rasterization).

KR1 / KR2 / KR3 keep using UTM internally for honest metric heights and
scatter — only the user-facing PNGs (and city_overview_seg_aligned.png)
use the mercator city grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

R_EARTH_M = 6378137.0


def _merc_x(lon: float) -> float:
    return R_EARTH_M * math.radians(lon)


def _merc_y(lat: float) -> float:
    return R_EARTH_M * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


@dataclass
class CityGridInfo:
    """Global Web-Mercator pixel grid for a city bbox."""
    ubox: tuple                  # (W, S, E, N) WGS84 union bbox
    out_w: int
    out_h: int
    west_m: float
    east_m: float
    south_m: float
    north_m: float

    def as_dict(self) -> dict:
        return {"ubox": list(self.ubox),
                "out_w": int(self.out_w), "out_h": int(self.out_h),
                "west_m": self.west_m, "east_m": self.east_m,
                "south_m": self.south_m, "north_m": self.north_m}

    @classmethod
    def from_dict(cls, d: dict) -> "CityGridInfo":
        return cls(ubox=tuple(d["ubox"]),
                   out_w=int(d["out_w"]), out_h=int(d["out_h"]),
                   west_m=float(d["west_m"]), east_m=float(d["east_m"]),
                   south_m=float(d["south_m"]), north_m=float(d["north_m"]))

    @property
    def transform(self):
        from rasterio.transform import from_bounds
        return from_bounds(self.west_m, self.south_m,
                           self.east_m, self.north_m,
                           self.out_w, self.out_h)


def compute_city_grid_info(union_bbox_wgs: tuple,
                            target_long_px: int) -> CityGridInfo:
    """Build a global mercator grid for the union bbox.

    ``target_long_px`` controls resolution: the longer side of the
    output grid will be at least this many pixels, preserving the
    mercator aspect ratio of the bbox so each tile sub-slice has roughly
    1024 px on the long side.
    """
    W, S, E, N = (float(v) for v in union_bbox_wgs)
    if not (E > W and N > S):
        raise ValueError(f"bad union bbox: {union_bbox_wgs}")
    west_m = _merc_x(W); east_m = _merc_x(E)
    south_m = _merc_y(S); north_m = _merc_y(N)
    width_m = east_m - west_m
    height_m = north_m - south_m
    long_m = max(width_m, height_m)
    # pixel size in mercator metres
    px_m = long_m / float(target_long_px)
    out_w = max(1, int(round(width_m / px_m)))
    out_h = max(1, int(round(height_m / px_m)))
    return CityGridInfo(ubox=(W, S, E, N),
                        out_w=out_w, out_h=out_h,
                        west_m=west_m, east_m=east_m,
                        south_m=south_m, north_m=north_m)


def rasterize_city_seg(city_cls_wgs: dict,
                        grid: CityGridInfo,
                        class_ids: dict,
                        class_priority: dict) -> np.ndarray:
    """Rasterize a city's union OSM class geoms into a single uint8 seg.

    Pixels with no class fall through to 0 (== ``ground``).
    """
    from rasterio.features import rasterize as _rasterize
    from shapely.ops import unary_union
    from .geometry_utils import make_transformer, reproject_geom

    fwd = make_transformer("EPSG:4326", "EPSG:3857")
    seg = np.zeros((grid.out_h, grid.out_w), dtype=np.uint8)
    transform = grid.transform

    order = sorted(class_ids.keys(), key=lambda k: class_priority[k])
    for name in order:
        g_wgs = city_cls_wgs.get(name)
        if g_wgs is None or g_wgs.is_empty:
            continue
        g_merc = reproject_geom(g_wgs, fwd)
        if g_merc is None or g_merc.is_empty:
            continue
        cls_id = class_ids[name]
        out = _rasterize([(g_merc, cls_id)],
                         out_shape=(grid.out_h, grid.out_w),
                         transform=transform, fill=0,
                         default_value=cls_id, dtype="uint8",
                         all_touched=False)
        seg[out > 0] = cls_id
    return seg


def tile_pixel_subrect(tile_bbox_wgs: tuple,
                        grid: CityGridInfo) -> tuple:
    """Return integer pixel rect (px0, py0, px1, py1) of tile in city grid.

    Coordinates are inclusive-of-px0/py0, exclusive-of-px1/py1
    (numpy slicing convention). py0 corresponds to the NORTH edge of
    the tile (top row of array).
    """
    W, S, E, N = (float(v) for v in tile_bbox_wgs)
    tw = _merc_x(W); te = _merc_x(E)
    ts = _merc_y(S); tn = _merc_y(N)
    px_m_x = (grid.east_m - grid.west_m) / grid.out_w
    px_m_y = (grid.north_m - grid.south_m) / grid.out_h
    px0 = int(round((tw - grid.west_m) / px_m_x))
    px1 = int(round((te - grid.west_m) / px_m_x))
    # y axis: row 0 = north → larger mercator y. Convert: row = (north_m - y) / px_m_y
    py0 = int(round((grid.north_m - tn) / px_m_y))
    py1 = int(round((grid.north_m - ts) / px_m_y))
    # Clamp.
    px0 = max(0, min(grid.out_w, px0))
    px1 = max(0, min(grid.out_w, px1))
    py0 = max(0, min(grid.out_h, py0))
    py1 = max(0, min(grid.out_h, py1))
    if px1 <= px0 or py1 <= py0:
        raise ValueError(
            f"tile bbox {tile_bbox_wgs} maps to empty pixel rect "
            f"(px0={px0},px1={px1},py0={py0},py1={py1}) on city grid "
            f"{grid.ubox}")
    return (px0, py0, px1, py1)


def clip_city_cls_to_tile(city_cls_wgs: dict,
                           tile_bbox_wgs: tuple) -> dict:
    """Return tile-local cls_wgs by clipping each class geom to tile bbox.

    Clipping is done in WGS84 (sufficient for small tiles ~512m). The
    output dict has the same keys as the input. ``_meta`` and other
    underscore-prefixed entries are passed through untouched.
    """
    from shapely.geometry import box as _box
    W, S, E, N = (float(v) for v in tile_bbox_wgs)
    tile_box = _box(W, S, E, N)
    out: dict = {}
    for k, g in city_cls_wgs.items():
        if k.startswith("_") or g is None:
            out[k] = g
            continue
        try:
            clipped = g.intersection(tile_box)
        except Exception:  # noqa: BLE001
            clipped = None
        out[k] = clipped if (clipped is not None
                              and not clipped.is_empty) else None
    # Special: ``_buildings_with_id`` may be either a GeoDataFrame (one
    # row per OSM building, the format produced by
    # ``fetch_all_classes_combined``) OR a list of ``(osm_id, polygon)``
    # tuples (legacy fallback path). Handle both.
    bld = city_cls_wgs.get("_buildings_with_id")
    if bld is not None:
        try:
            import geopandas as _gpd
        except Exception:  # noqa: BLE001
            _gpd = None
        if _gpd is not None and isinstance(bld, _gpd.GeoDataFrame):
            if len(bld) > 0:
                # GeoDataFrame.intersection -> GeoSeries with original
                # column metadata preserved on the parent frame.
                gdf = bld.copy()
                gdf["geometry"] = bld.geometry.intersection(tile_box)
                gdf = gdf[~gdf.geometry.is_empty
                          & gdf.geometry.notna()]
                out["_buildings_with_id"] = gdf.reset_index(drop=True)
            else:
                out["_buildings_with_id"] = bld
        elif isinstance(bld, (list, tuple)):
            clipped_list = []
            for entry in bld:
                try:
                    if (isinstance(entry, (tuple, list))
                            and len(entry) == 2):
                        osm_id, poly = entry
                    else:
                        continue
                    if poly is None or poly.is_empty:
                        continue
                    inter = poly.intersection(tile_box)
                    if inter is None or inter.is_empty:
                        continue
                    clipped_list.append((osm_id, inter))
                except Exception:  # noqa: BLE001
                    continue
            out["_buildings_with_id"] = clipped_list
    return out

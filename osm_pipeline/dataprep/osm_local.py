"""Read OSM features from per-class GeoParquets prepared by `prep_pbf.py`.

Pipeline:
  1. `python scripts/0_download_pbf.py nebraska`     # 30s, 97 MB
  2. `python scripts/prep_pbf.py nebraska`           # 80s -> 5 parquets
  3. now any bbox in Nebraska -> sub-second feature read.

`pbf_for_bbox(bbox, cache_dir)` returns the region directory if all 5
class parquets exist; else None (caller falls back to Overpass).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union


PBF_REGISTRY: list[tuple[str, tuple[float, float, float, float]]] = [
    ("nebraska",          (-104.06, 39.99, -95.30, 43.01)),
    ("massachusetts",     (-73.51,  41.18, -69.86, 42.89)),
    ("hong-kong",         (113.83,  22.15, 114.45, 22.57)),
    ("malaysia-singapore-brunei", (99.64, 0.85, 119.27, 7.36)),
    ("switzerland",       (5.96,    45.82, 10.49, 47.81)),
]

CLASSES = ("grass", "water", "building", "road")


def pbf_for_bbox(bbox, cache_dir: Path) -> Path | None:
    """Return path to region dir whose parquets cover bbox; else None."""
    w, s, e, n = bbox
    for stem, (W, S, E, N) in PBF_REGISTRY:
        if w >= W and s >= S and e <= E and n <= N:
            d = cache_dir / stem
            if all((d / f"{c}.parquet").exists() for c in CLASSES):
                return d
    return None


@lru_cache(maxsize=32)
def _load(region_dir_str: str, cls: str) -> gpd.GeoDataFrame:
    """Cache parquet loads in memory (one GDF per (region, class))."""
    return gpd.read_parquet(Path(region_dir_str) / f"{cls}.parquet")


def fetch_polygon_class_local(region_dir: Path, bbox, cls: str):
    if cls not in {"grass", "water", "building"}:
        return None
    gdf = _load(str(region_dir), cls)
    bb = box(*bbox)
    idx = gdf.sindex.query(bb, predicate="intersects")
    if len(idx) == 0:
        return None
    sub = gdf.iloc[idx]
    polys = []
    for g in sub.geometry:
        if g is None or g.is_empty:
            continue
        try:
            g2 = g.intersection(bb)
        except Exception:  # noqa: BLE001
            continue
        if not g2.is_empty:
            polys.append(g2)
    if not polys:
        return None
    return unary_union(polys)


def fetch_roads_local(region_dir: Path, bbox, road_keep: Iterable[str]):
    """Return highways GeoDataFrame intersecting bbox (LineString rows)."""
    gdf = _load(str(region_dir), "road")
    keep = set(road_keep)
    bb = box(*bbox)
    idx = gdf.sindex.query(bb, predicate="intersects")
    if len(idx) == 0:
        return None
    sub = gdf.iloc[idx]
    sub = sub[sub["highway"].isin(keep)]
    if len(sub) == 0:
        return None
    sub = sub.copy()
    sub["geometry"] = sub.geometry.intersection(bb)
    return sub

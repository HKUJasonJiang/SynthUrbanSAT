"""Fast Overpass-only OSM fetcher (bypasses osmnx).

Why this exists
---------------
Empirical benchmark on a 553x517 m Omaha tile (see scripts/bench_osm_fetch.py
results in output/_bench_osm.json):

    osmnx.features_from_bbox(building=True)   :  66.7 s   186 buildings
    raw Overpass POST (one combined query)    :  12.4 s   ALL 4 classes
    osmnx graph_from_bbox(network_type=drive) : ~60 s + drops service+footway

So we use a single combined Overpass-QL POST that returns buildings,
roads, water, grass, foliage tags in one round-trip; we then bucketize
into the 6 project classes on the Python side.

Public API
----------
    fetch_all_classes_combined(bbox, road_keep=None, endpoint=None)
        -> dict[class_name, shapely (Multi)Polygon | None]
            for the polygon classes ("building", "water", "grass",
            "foliage")
        + a special "_road_edges" key holding the WGS84 road edges
          GeoDataFrame (LineString rows + ``highway`` column) so callers
          can buffer them in metres exactly like the osmnx path.

Endpoint policy
---------------
We try ``OVERPASS_ENDPOINTS`` in order. ``overpass-api.de`` is often
unreachable from outside Europe; ``kumi.systems`` is the de-facto fast
alternative for North-American tiles. Failures fall over to the next
mirror; if all fail we raise.
"""
from __future__ import annotations

import os
from typing import Iterable

import geopandas as gpd
import requests
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

from .osm_tags import ROAD_HIGHWAY_KEEP


OVERPASS_ENDPOINTS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)


def _env_endpoints() -> list[str] | None:
    raw = os.environ.get("PROCEDUREOSM_OVERPASS_ENDPOINTS", "").strip()
    if not raw:
        return None
    eps = [x.strip() for x in raw.replace(";", ",").split(",")]
    return [x for x in eps if x]


def _env_timeout(default: int) -> int:
    raw = os.environ.get("PROCEDUREOSM_OVERPASS_TIMEOUT", "").strip()
    if not raw:
        return int(default)
    try:
        return max(3, int(float(raw)))
    except Exception:
        return int(default)


# Tag → class buckets. Order matters because some ways have multiple
# tags (e.g. a "way[building][landuse=cemetery]" -> we want building).
def _bucket(tags: dict) -> str | None:
    if "building" in tags:
        return "building"
    if "highway" in tags:
        return "_highway"
    nat = tags.get("natural")
    way_w = tags.get("waterway")
    lu = tags.get("landuse")
    leis = tags.get("leisure")
    if nat in {"water", "bay", "strait", "coastline"}:
        return "water"
    if way_w in {"river", "stream", "canal", "riverbank"}:
        return "water"
    if lu in {"reservoir", "basin"}:
        return "water"
    if lu in {"grass", "meadow", "recreation_ground", "village_green",
              "cemetery", "allotments"}:
        return "grass"
    if leis in {"park", "garden", "pitch", "playground", "golf_course"}:
        return "grass"
    if nat in {"grassland", "heath"}:
        return "grass"
    if nat in {"wood", "scrub", "tree", "tree_row"}:
        return "foliage"
    if lu in {"forest", "orchard"}:
        return "foliage"
    if leis == "nature_reserve":
        return "foliage"
    return None


def _build_query(bbox) -> str:
    """One Overpass-QL string that pulls all 4 polygon classes + roads.

    Notes:
      - We use ``out tags geom;`` to include way coordinates inline (no
        recurse needed).
      - Relations are queried but rendered without inner geom; for our
        small-tile use case (<1 km^2) virtually no buildings are
        relations, so this is fine. If a relation is critical, the
        caller can fall back to osmnx.
      - We deliberately do NOT include ``area``/``coastline``/admin
        boundary queries -- those slow Overpass to a crawl.
    """
    w, s, e, n = bbox
    bb = f"{s},{w},{n},{e}"
    return f"""[out:json][timeout:60];
(
  way["building"]({bb});
  way["highway"]({bb});
  way["natural"~"^(water|bay|coastline|grassland|heath|wood|scrub|tree_row)$"]({bb});
  way["waterway"~"^(river|stream|canal|riverbank)$"]({bb});
  way["landuse"~"^(grass|meadow|recreation_ground|village_green|cemetery|allotments|reservoir|basin|forest|orchard)$"]({bb});
  way["leisure"~"^(park|garden|pitch|playground|golf_course|nature_reserve)$"]({bb});
  relation["building"]({bb});
  relation["natural"~"^(water|bay|wood)$"]({bb});
  relation["landuse"~"^(reservoir|basin|forest)$"]({bb});
);
out tags geom;
"""


def _post(query: str, endpoint: str, timeout: int = 60) -> dict:
    r = requests.post(endpoint, data={"data": query}, timeout=timeout,
                      headers={"User-Agent": "ProcedureOSM/1.0 "
                                              "(github.com/ProcedureOSM)"})
    r.raise_for_status()
    return r.json()


def _post_with_failover(query: str, endpoints, timeout: int = 60) -> dict:
    """Try mirrors in order. Detects Overpass *soft* failures (HTTP 200
    with ``remark`` containing "timed out" / "Query timed out", or with
    ``elements=[]`` after a long wall-time) and treats them as a hard
    failure so we fail over to the next mirror.
    """
    import time
    last = None
    for ep in endpoints:
        try:
            t0 = time.time()
            js = _post(query, ep, timeout=timeout)
            dt = time.time() - t0
            osm3s = js.get("osm3s") or {}
            ts = str(osm3s.get("timestamp_osm_base") or "")
            if ts and "T" not in ts:
                msg = f"soft-fail invalid timestamp_osm_base={ts!r}"
                last = RuntimeError(msg)
                print(f"[overpass] {ep} {msg} after {dt:.1f}s; "
                      f"trying next mirror")
                continue
            remark = (js.get("remark") or "").lower()
            n_el = len(js.get("elements") or [])
            if "timed out" in remark or "runtime error" in remark:
                msg = f"soft-fail remark={js.get('remark')!r}"
                last = RuntimeError(msg)
                print(f"[overpass] {ep} {msg} after {dt:.1f}s; "
                      f"trying next mirror")
                continue
            # 200 OK + 0 elements + slow response = treat as soft fail.
            # Real empty tiles return in <2s; >5s with 0 elements means
            # the mirror gave up silently.
            if n_el == 0 and dt > 5.0:
                msg = f"soft-fail 0 elements after {dt:.1f}s"
                last = RuntimeError(msg)
                print(f"[overpass] {ep} {msg}; trying next mirror")
                continue
            return js
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[overpass] {ep} failed: {type(e).__name__}: "
                  f"{str(e)[:120]}; trying next mirror")
    raise RuntimeError(
        f"all overpass endpoints failed; last error: {last}")


def _way_to_polygon(geom_pts: list[dict]) -> Polygon | None:
    """Overpass ``geometry`` is a list of {lat, lon} dicts. Closed
    ways become Polygons; we silently drop ways with <4 points."""
    if not geom_pts or len(geom_pts) < 4:
        return None
    coords = [(p["lon"], p["lat"]) for p in geom_pts]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        p = Polygon(coords)
        return p if p.is_valid and not p.is_empty else p.buffer(0)
    except Exception:  # noqa: BLE001
        return None


def _way_to_linestring(geom_pts: list[dict]) -> LineString | None:
    if not geom_pts or len(geom_pts) < 2:
        return None
    try:
        return LineString([(p["lon"], p["lat"]) for p in geom_pts])
    except Exception:  # noqa: BLE001
        return None


def fetch_all_classes_combined(
    bbox,
    road_keep: Iterable[str] | None = None,
    endpoints: Iterable[str] | None = None,
    timeout: int = 60,
) -> dict:
    """Single Overpass POST, all 4 classes + roads in one shot.

    Returns dict with keys:
        building, water, grass, foliage : shapely (Multi)Polygon | None
        _road_edges : GeoDataFrame[geometry=LineString, highway, lanes,
                                   maxspeed, oneway, name] in EPSG:4326
        _meta : {"endpoint": ..., "elapsed_s": ..., "n_elements": ...}
    """
    import time

    keep = set(road_keep) if road_keep is not None else set(ROAD_HIGHWAY_KEEP)
    eps = list(endpoints) if endpoints else (_env_endpoints()
                                             or list(OVERPASS_ENDPOINTS))
    timeout = _env_timeout(timeout)
    query = _build_query(bbox)

    t0 = time.time()
    js = _post_with_failover(query, eps, timeout=timeout)
    elapsed = time.time() - t0

    # Bucket elements -> per-class polygon lists + road records.
    polys = {"building": [], "water": [], "grass": [], "foliage": []}
    road_rows = []
    # Per-building records preserving the OSM way/relation id so KR2 +
    # the UE exporter can name each building uniquely. We DO NOT union
    # these so each entry stays one OSM feature.
    buildings_with_id: list[dict] = []

    for el in js.get("elements", []):
        tags = el.get("tags", {}) or {}
        kind = _bucket(tags)
        if kind is None:
            continue
        if kind == "_highway":
            hw = tags.get("highway")
            if hw not in keep:
                continue
            ls = _way_to_linestring(el.get("geometry", []))
            if ls is None:
                continue
            road_rows.append({
                "geometry": ls,
                "highway": hw,
                "name": tags.get("name"),
                "lanes": tags.get("lanes"),
                "maxspeed": tags.get("maxspeed"),
                "oneway": tags.get("oneway"),
            })
            continue
        # Polygonal classes: ways become Polygons.
        if el.get("type") == "way":
            p = _way_to_polygon(el.get("geometry", []))
            if p is not None and not p.is_empty:
                polys[kind].append(p)
                if kind == "building":
                    buildings_with_id.append({
                        "geometry": p,
                        "osm_id": int(el.get("id", 0)),
                        "osm_type": "way",
                        "name": tags.get("name"),
                        "height_tag": tags.get("height"),
                        "levels_tag": tags.get("building:levels"),
                    })
        # Relations: skipped (would need inner-geom recurse). Acceptable
        # tradeoff for sub-km tiles.

    # Union per class.
    out: dict = {}
    for cls in ("building", "water", "grass", "foliage"):
        if not polys[cls]:
            out[cls] = None
            continue
        try:
            u = unary_union(polys[cls])
        except Exception:  # noqa: BLE001
            # try buffer(0) salvage
            u = unary_union([p.buffer(0) for p in polys[cls]])
        if u.is_empty:
            out[cls] = None
        elif isinstance(u, (Polygon, MultiPolygon)):
            out[cls] = u
        else:
            # GeometryCollection -> keep only polygons
            from shapely.geometry import GeometryCollection
            if isinstance(u, GeometryCollection):
                pp = [g for g in u.geoms
                      if isinstance(g, (Polygon, MultiPolygon))]
                out[cls] = unary_union(pp) if pp else None
            else:
                out[cls] = None

    # Road edges as a GeoDataFrame in WGS84.
    if road_rows:
        edges = gpd.GeoDataFrame(road_rows, crs="EPSG:4326")
    else:
        edges = gpd.GeoDataFrame(
            {"geometry": [], "highway": [], "name": [],
             "lanes": [], "maxspeed": [], "oneway": []},
            crs="EPSG:4326")
    out["_road_edges"] = edges
    # Per-building GeoDataFrame (one row per OSM building way) so KR2 +
    # the UE export can carry osm_id + tags through.
    if buildings_with_id:
        out["_buildings_with_id"] = gpd.GeoDataFrame(
            buildings_with_id, crs="EPSG:4326")
    else:
        out["_buildings_with_id"] = gpd.GeoDataFrame(
            {"geometry": [], "osm_id": [], "osm_type": [],
             "name": [], "height_tag": [], "levels_tag": []},
            crs="EPSG:4326")
    out["_meta"] = {
        "elapsed_s": round(elapsed, 2),
        "n_elements": len(js.get("elements", [])),
        "n_buildings": sum(1 for el in js.get("elements", [])
                            if (el.get("tags") or {}).get("building")),
        "n_roads": len(road_rows),
    }
    return out

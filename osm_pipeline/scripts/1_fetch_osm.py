"""KR1 — Fetch OSM and produce 6-class GeoJSON + KR1 figures.

Outputs (per city):
    output/geojson/{city}_{class}.geojson   x 6  (in WGS84 EPSG:4326)
    output/geojson/{city}_meta.json         (bbox, UTM CRS, area stats)
    fig/kr1_{city}_osm_classes.png          6-class polygon overlay
    fig/kr1_{city}_class_pixel_ratio.png    bar chart of class area ratios

Run:
    python scripts/1_fetch_osm.py --config configs/default.yaml
    python scripts/1_fetch_osm.py --config configs/default.yaml --city zurich
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from shapely.geometry import box, mapping
from shapely.ops import unary_union

# allow running from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataprep.osm_tags import TAG_QUERIES, ROAD_HIGHWAY_KEEP, CLASS_PRIORITY  # noqa: E402
from dataprep.osm_local import (  # noqa: E402
    pbf_for_bbox, fetch_polygon_class_local, fetch_roads_local,
)
from dataprep.geometry_utils import (  # noqa: E402
    utm_crs_for_bbox, make_transformer, reproject_geom,
    to_multipolygon, buffer_lines, clip_to_bbox, ensure_dir,
)


# Overpass mirrors, tried in order. The official endpoint is heavily
# rate-limited / often saturated from outside Europe; the mirrors below are
# community-maintained and usually faster. See
# https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances
OVERPASS_ENDPOINTS = [
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
_OSMNX_CONFIGURED = False


def _configure_osmnx(cache_dir: Path | None = None,
                     timeout: int = 60):
    """Set up osmnx once: medium timeout, on-disk cache, fast mirror."""
    global _OSMNX_CONFIGURED
    if _OSMNX_CONFIGURED:
        return
    import osmnx as ox
    s = ox.settings
    s.use_cache = True
    s.cache_only_mode = False
    if cache_dir is not None:
        s.cache_folder = str(cache_dir)
    s.requests_timeout = timeout       # was osmnx default 180 s — too long
    s.overpass_url = OVERPASS_ENDPOINTS[0]
    s.log_console = False
    _OSMNX_CONFIGURED = True


def _with_endpoint_failover(call):
    """Run `call()` against each Overpass endpoint until one succeeds.

    `InsufficientResponseError` from osmnx means "query OK, zero features" --
    that is a valid empty answer for sparse bboxes (e.g. a suburb with no
    water). Do NOT fail over on that; re-raise so the caller turns it into
    an empty geometry.
    """
    import osmnx as ox
    try:
        from osmnx._errors import InsufficientResponseError  # osmnx 2.x
    except Exception:  # noqa: BLE001
        InsufficientResponseError = None  # type: ignore
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = url
        try:
            return call()
        except Exception as e:
            if (InsufficientResponseError is not None
                    and isinstance(e, InsufficientResponseError)):
                raise
            last_err = e
            warnings.warn(f"  overpass {url} failed: "
                          f"{type(e).__name__}: {str(e)[:120]}")
            continue
    raise last_err if last_err else RuntimeError("all overpass endpoints failed")

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_polygon_class(bbox, tags, class_name: str | None = None):
    """Return a unioned shapely (Multi)Polygon in WGS84 for a tag dict.

    Tries local Geofabrik .osm.pbf first (if `class_name` is given and an
    extract covering `bbox` exists in cache/pbf/), else falls back to
    Overpass with mirror failover.
    """
    # ---- fast path: local PBF ---- #
    if class_name is not None:
        pbf = pbf_for_bbox(bbox, ROOT / "cache" / "pbf")
        if pbf is not None:
            try:
                return fetch_polygon_class_local(pbf, bbox, class_name)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"  local PBF read failed for {class_name}: "
                              f"{type(e).__name__}: {e}; falling back to "
                              "Overpass")
    # ---- slow path: Overpass ---- #
    import osmnx as ox
    _configure_osmnx(cache_dir=ROOT / "cache")

    min_lon, min_lat, max_lon, max_lat = bbox

    def _do():
        try:
            return ox.features_from_bbox(
                bbox=(min_lon, min_lat, max_lon, max_lat), tags=tags,
            )
        except TypeError:
            return ox.features_from_bbox(
                max_lat, min_lat, max_lon, min_lon, tags=tags,
            )

    try:
        gdf = _with_endpoint_failover(_do)
    except Exception as e:
        # InsufficientResponseError = "query worked, no features" — silent empty.
        try:
            from osmnx._errors import InsufficientResponseError
            if isinstance(e, InsufficientResponseError):
                return None
        except Exception:  # noqa: BLE001
            pass
        warnings.warn(f"  features fetch failed for tags={tags}: {e}")
        return None

    if gdf is None or len(gdf) == 0:
        return None
    polys = []
    for g in gdf.geometry:
        mp = to_multipolygon(g)
        if mp is not None:
            polys.append(mp)
    if not polys:
        return None
    return unary_union(polys)


def fetch_roads(bbox, road_buffer_m: dict, road_keep: set[str] | None = None):
    """Fetch road graph and buffer per highway type. Returns polygon in WGS84.

    We buffer in local UTM (meters) then project back to WGS84.
    `road_keep`: set of highway tag values to keep. Falls back to module default.
    Tries local PBF first.
    """
    if road_keep is None:
        road_keep = ROAD_HIGHWAY_KEEP

    # ---- fast path: local PBF ---- #
    pbf = pbf_for_bbox(bbox, ROOT / "cache" / "pbf")
    if pbf is not None:
        try:
            edges = fetch_roads_local(pbf, bbox, road_keep)
            if edges is None or len(edges) == 0:
                return None
            return _buffer_road_edges(edges, bbox, road_buffer_m, road_keep)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"  local PBF road read failed: "
                          f"{type(e).__name__}: {e}; falling back to Overpass")

    # ---- slow path: Overpass ---- #
    import osmnx as ox
    _configure_osmnx(cache_dir=ROOT / "cache")

    min_lon, min_lat, max_lon, max_lat = bbox

    def _do():
        try:
            return ox.graph_from_bbox(
                bbox=(min_lon, min_lat, max_lon, max_lat),
                network_type="drive", simplify=True, retain_all=True,
            )
        except TypeError:
            return ox.graph_from_bbox(
                max_lat, min_lat, max_lon, min_lon,
                network_type="drive", simplify=True, retain_all=True,
            )

    try:
        G = _with_endpoint_failover(_do)
    except Exception as e:
        warnings.warn(f"  road graph fetch failed: {e}")
        return None

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    if edges is None or len(edges) == 0:
        return None
    return _buffer_road_edges(edges, bbox, road_buffer_m, road_keep)


def _buffer_road_edges(edges, bbox, road_buffer_m: dict, road_keep):
    """Common road-edges -> buffered polygon (WGS84). Used by both PBF and
    Overpass paths.
    """
    utm = utm_crs_for_bbox(bbox)
    edges = edges.to_crs(utm)

    bufs_by_type: dict[float, list] = {}
    for _, row in edges.iterrows():
        hwy = row.get("highway")
        if isinstance(hwy, list):
            hwy = hwy[0] if hwy else None
        if hwy not in road_keep:
            continue
        half = road_buffer_m.get(hwy, road_buffer_m.get("default", 4.0))
        bufs_by_type.setdefault(half, []).append(row.geometry)

    polys_utm = []
    for half, lines in bufs_by_type.items():
        b = buffer_lines(lines, half)
        if b is not None:
            polys_utm.append(b)
    if not polys_utm:
        return None
    merged_utm = unary_union(polys_utm)

    # back to WGS84
    tr = make_transformer(utm, "EPSG:4326")
    return reproject_geom(merged_utm, tr)


def write_geojson(geom, out_path: Path, class_name: str, class_id: int):
    feature = {
        "type": "Feature",
        "properties": {"class": class_name, "class_id": class_id},
        "geometry": mapping(geom),
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)


def compute_areas_m2(class_geoms: dict, bbox) -> dict:
    """Compute area of each class polygon in UTM meters^2."""
    utm = utm_crs_for_bbox(bbox)
    tr = make_transformer("EPSG:4326", utm)
    out = {}
    for name, g in class_geoms.items():
        if g is None or g.is_empty:
            out[name] = 0.0
        else:
            out[name] = float(reproject_geom(g, tr).area)
    return out


def plot_overlay(class_geoms, bbox, colors, out_png: Path, title: str):
    """Draw 6-class overlay using matplotlib PathPatch (holes rendered correctly)."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 8))
    min_lon, min_lat, max_lon, max_lat = bbox

    def _poly_to_path(poly):
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            verts.extend(xy.tolist())
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(xy) - 2))
            codes.append(MplPath.CLOSEPOLY)
        return MplPath(verts, codes)

    # draw classes in priority order (low first, high on top)
    order = sorted(class_geoms.keys(), key=lambda k: CLASS_PRIORITY.get(k, 0))
    for name in order:
        g = class_geoms.get(name)
        if g is None or g.is_empty:
            continue
        rgb = tuple(c / 255.0 for c in colors[name])
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for p in polys:
            if p.geom_type != "Polygon" or p.is_empty:
                continue
            ax.add_patch(PathPatch(_poly_to_path(p),
                                   facecolor=rgb, edgecolor="none", alpha=0.9))
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=tuple(c / 255.0 for c in colors[n]), label=n)
               for n in ["ground", "foliage", "grass", "water", "building", "road"]]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ratio_bar(areas: dict, bbox_area: float, out_png: Path,
                   ref_ratios: dict | None = None, title: str = ""):
    classes = ["ground", "foliage", "grass", "water", "building", "road"]
    ratios = [areas.get(c, 0.0) / max(bbox_area, 1e-9) for c in classes]
    x = np.arange(len(classes))
    width = 0.4 if ref_ratios else 0.6
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - (width / 2 if ref_ratios else 0), ratios, width=width,
           label="ours", color="#3b7dd8")
    if ref_ratios:
        ref = [ref_ratios.get(c, 0.0) for c in classes]
        ax.bar(x + width / 2, ref, width=width, label="train_set",
               color="#d8893b")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("area ratio")
    ax.set_ylim(0, max(max(ratios), max(ref) if ref_ratios else 0) * 1.2 + 0.05)
    ax.set_title(title)
    if ref_ratios:
        ax.legend()
    for i, r in enumerate(ratios):
        ax.text(i - (width / 2 if ref_ratios else 0), r + 0.005, f"{r:.2f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def process_city(city_cfg: dict, cfg: dict, out_geojson: Path, out_fig: Path,
                  class_geoms_wgs: dict | None = None):
    """Run KR1 for a single city.

    If ``class_geoms_wgs`` is provided, the slow per-class Overpass fetch
    is skipped and we go straight to clip + write + plot. The dict must
    contain WGS84 (Multi)Polygons for keys
    ``{building, road, water, grass, foliage, ground}`` (None allowed).
    """
    name = city_cfg["name"]
    bbox = tuple(city_cfg["bbox"])  # (W,S,E,N)
    print(f"[KR1] {name}: bbox={bbox}")

    if class_geoms_wgs is not None:
        print("  using pre-fetched class geoms (skipping Overpass)")
        # Pull out and stash special keys ("_xxx") so they don't pollute
        # the class-geom dict that gets clipped + plotted.
        buildings_with_id = class_geoms_wgs.get("_buildings_with_id")
        class_geoms_wgs = {k: v for k, v in class_geoms_wgs.items()
                           if not str(k).startswith("_")}
        # Recompute ground from the supplied polygons to be safe.
        bbox_poly = box(*bbox)
        others = [g for k, g in class_geoms_wgs.items()
                  if k != "ground" and g is not None and not g.is_empty]
        if others:
            union_others = unary_union(others)
            gr = bbox_poly.difference(union_others)
            class_geoms_wgs["ground"] = gr if not gr.is_empty else None
    else:
        buildings_with_id = None
        # 1) fetch the 5 explicit classes
        class_geoms_wgs = {}
        for cls_name, tags in TAG_QUERIES.items():
            print(f"  fetching {cls_name} ...")
            g = fetch_polygon_class(bbox, tags, class_name=cls_name)
            class_geoms_wgs[cls_name] = g

        # 2) roads
        print("  fetching roads ...")
        road_keep = set(cfg["osm"].get("road_keep", list(ROAD_HIGHWAY_KEEP)))
        class_geoms_wgs["road"] = fetch_roads(
            bbox, cfg["osm"]["road_buffer_m"], road_keep=road_keep,
        )

        # 3) ground = bbox - union(others)
        bbox_poly = box(*bbox)
        others = [g for k, g in class_geoms_wgs.items()
                  if g is not None and not g.is_empty]
        union_others = unary_union(others) if others else None
        ground = bbox_poly.difference(union_others) if union_others else bbox_poly
        if ground.is_empty:
            ground = None
        class_geoms_wgs["ground"] = ground

    bbox_poly = box(*bbox)

    # 4) clip every class to bbox
    for k in list(class_geoms_wgs.keys()):
        class_geoms_wgs[k] = clip_to_bbox(class_geoms_wgs[k], bbox_poly)

    # 5) write geojson
    from dataprep.osm_tags import CLASS_IDS
    for k, g in class_geoms_wgs.items():
        if g is None or g.is_empty:
            print(f"  [warn] empty class: {k}")
            continue
        out = out_geojson / f"{name}_{k}.geojson"
        write_geojson(g, out, k, CLASS_IDS[k])
        print(f"  wrote {out}")

    # 5b) per-building FeatureCollection (one feature per OSM way) so
    #     KR2 + UE export can preserve osm_id and per-building tags.
    if buildings_with_id is not None and len(buildings_with_id) > 0:
        bld_path = out_geojson / f"{name}_buildings.geojson"
        bld_clipped = buildings_with_id.copy()
        bld_clipped["geometry"] = bld_clipped.geometry.intersection(bbox_poly)
        bld_clipped = bld_clipped[~bld_clipped.geometry.is_empty]
        bld_clipped.to_file(bld_path, driver="GeoJSON")
        print(f"  wrote {bld_path}  ({len(bld_clipped)} buildings)")

    # 6) stats + figs
    areas = compute_areas_m2(class_geoms_wgs, bbox)
    utm = utm_crs_for_bbox(bbox)
    tr = make_transformer("EPSG:4326", utm)
    bbox_area = reproject_geom(bbox_poly, tr).area
    print(f"  bbox_area_m2={bbox_area:.0f}")
    for k, a in areas.items():
        print(f"    {k:>9}: {a:>12.0f} m^2  ({100*a/bbox_area:5.2f}%)")

    meta = {
        "city": name,
        "bbox_wgs84": list(bbox),
        "utm_crs": utm.to_string(),
        "bbox_area_m2": bbox_area,
        "areas_m2": areas,
        "ratios": {k: a / bbox_area for k, a in areas.items()},
    }
    with open(out_geojson / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # figures
    colors = {k: cfg["class_colors"][k] for k in CLASS_IDS}
    plot_overlay(
        class_geoms_wgs, bbox, colors,
        out_fig / f"kr1_{name}_osm_classes.png",
        title=f"KR1 — OSM 6-class overlay ({name})",
    )
    ref_ratios = cfg.get("ref_class_ratios")  # placeholder until real stats land
    plot_ratio_bar(
        areas, bbox_area,
        out_fig / f"kr1_{name}_class_pixel_ratio.png",
        ref_ratios=ref_ratios,
        title=f"KR1 — class area ratios ({name}) vs ref (placeholder)",
    )
    print(f"  figures -> {out_fig}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--city", default=None, help="run only this city by name")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    out_geojson = ensure_dir(ROOT / cfg["paths"]["geojson_dir"])
    out_fig = ensure_dir(ROOT / cfg["paths"]["fig_dir"])

    cities = cfg["cities"]
    if args.city:
        cities = [c for c in cities if c["name"] == args.city]
        if not cities:
            raise SystemExit(f"city '{args.city}' not in config")

    for city in cities:
        process_city(city, cfg, out_geojson, out_fig)

    print("[KR1] done.")


if __name__ == "__main__":
    main()

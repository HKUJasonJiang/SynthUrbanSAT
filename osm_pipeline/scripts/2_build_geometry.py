"""KR2 — Build 3D geometry from KR1 GeoJSON.

For each city we produce a single GLB scene with up to 6 nodes
(one per class), each tagged with `extras = {"class_id": <int>, "class": <name>}`
so the Blender step can read the class tag back.

Building heights are estimated from OSM tags when present; otherwise from the
config defaults. Roads / grass / foliage / water / ground are flat strips.
Foliage 'zones' are kept flat here — actual tree instancing is done in
Blender (KR4) for performance.

Run:
    python scripts/2_build_geometry.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import shape, MultiPolygon, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataprep.osm_tags import CLASS_IDS  # noqa: E402
from dataprep.geometry_utils import (  # noqa: E402
    utm_crs_for_bbox, make_transformer, reproject_geom,
    flat_polygon_mesh, extrude_polygon, merge_meshes, ensure_dir,
)


def load_config(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_geom(geojson_path: Path):
    if not geojson_path.exists():
        return None
    with open(geojson_path, "r", encoding="utf-8") as f:
        fc = json.load(f)
    geoms = []
    for feat in fc["features"]:
        try:
            g = shape(feat["geometry"])
        except Exception:
            continue
        if g.is_empty:
            continue
        # explode MultiPolygon -> individual Polygons
        if g.geom_type == "MultiPolygon":
            geoms.extend(list(g.geoms))
        elif g.geom_type == "Polygon":
            geoms.append(g)
        elif hasattr(g, "geoms"):
            for sub in g.geoms:
                if sub.geom_type == "Polygon":
                    geoms.append(sub)
    if not geoms:
        return None
    if len(geoms) == 1:
        return geoms[0]
    return MultiPolygon(geoms)


def project_to_local(geom, bbox):
    """WGS84 -> local UTM (origin shifted to bbox SW corner) in meters.

    Returns (geom_local, sw_xy_utm). `sw_xy_utm` is the UTM coords of the
    bbox SW corner (the local origin) so callers can map absolute UTM points
    back into mesh coords later.
    """
    utm = utm_crs_for_bbox(bbox)
    tr = make_transformer("EPSG:4326", utm)
    g_utm = reproject_geom(geom, tr)
    from shapely.geometry import Point
    sw = reproject_geom(Point(bbox[0], bbox[1]), tr)
    dx, dy = -sw.x, -sw.y

    from shapely.affinity import translate
    return translate(g_utm, xoff=dx, yoff=dy), (sw.x, sw.y)


def iter_polys(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            yield p
    elif hasattr(geom, "geoms"):
        # GeometryCollection or other heterogeneous container -- recurse
        # and keep only polygonal pieces.
        for sub in geom.geoms:
            yield from iter_polys(sub)


def building_height(props: dict, default_h: float, m_per_lvl: float) -> float:
    h = props.get("height") or props.get("building:height")
    if h is not None:
        try:
            return float(str(h).split()[0])
        except Exception:
            pass
    lvl = props.get("building:levels") or props.get("levels")
    if lvl is not None:
        try:
            return float(lvl) * m_per_lvl
        except Exception:
            pass
    return default_h


def _sample_building_height(rng, dist: str, hmin: float, hmax: float) -> float:
    """Return a single building height (m) drawn from ``dist`` and clipped
    into ``[hmin, hmax]``. The four supported distributions:

      - ``"flat"``    : always (hmin + hmax) / 2
      - ``"uniform"`` : U(hmin, hmax)
      - ``"lognormal"`` (default): long-tail centred on (hmin+hmax)/2,
         with sigma chosen so ~95% of samples fall inside the range
      - ``"bimodal"`` : 70% N(low,1) + 30% N(high,4) clipped to range,
         where ``low`` / ``high`` are picked at the lower and upper third
         of the range respectively (suburban + landmark mix).
    """
    import math
    hmin = float(hmin)
    hmax = float(max(hmax, hmin + 0.1))
    mid = 0.5 * (hmin + hmax)
    if dist == "flat":
        return mid
    if dist == "uniform":
        return rng.uniform(hmin, hmax)
    if dist == "lognormal":
        mu = math.log(max(mid, 0.5))
        sigma = max((math.log(max(hmax, mid + 0.5)) - mu) / 1.96, 0.15)
        h = math.exp(rng.gauss(mu, sigma))
        return min(max(h, hmin), hmax)
    if dist == "bimodal":
        low = hmin + (hmax - hmin) / 3.0
        high = hmax - (hmax - hmin) / 3.0
        if rng.random() < 0.7:
            h = rng.gauss(low, max(1.0, (hmax - hmin) * 0.08))
        else:
            h = rng.gauss(high, max(1.0, (hmax - hmin) * 0.15))
        return min(max(h, hmin), hmax)
    return mid


def build_class_mesh(class_name: str, geom_local, cfg: dict,
                     height_dist: str = "lognormal", height_seed: int = 42,
                     height_min: float | None = None,
                     height_max: float | None = None,
                     building_features: list | None = None):
    """Return a trimesh.Trimesh with class_id metadata, or None.

    ``building_features``: optional list of dicts in *local UTM* coords with
    keys ``geometry`` (shapely Polygon), ``osm_id`` (int), ``height_tag``,
    ``levels_tag``. When provided (and ``class_name == 'building'``), each
    record gets its own building_record entry tagged with the real OSM id.
    Heights from OSM tags are honoured when present; otherwise they're
    sampled from the ``height_dist`` distribution.
    """
    if geom_local is None or geom_local.is_empty:
        return None

    z_offset = {
        "ground":   0.0,
        "water":    0.05,
        "grass":    0.10,
        "foliage":  0.15,
        "road":     0.20,   # slightly above ground to avoid z-fighting
    }

    meshes = []
    building_records: list[dict] = []
    if class_name == "building":
        # KR1 stores buildings as a unioned MultiPolygon; iter_polys splits
        # it back into individual buildings, so we can sample a height per
        # polygon. Touching buildings get merged in OSM and share one h --
        # acceptable for our use.
        import random
        rng = random.Random(int(height_seed))
        rng_range = cfg["osm"].get("building_height_range_m", [3.0, 30.0])
        hmin = float(height_min if height_min is not None else rng_range[0])
        hmax = float(height_max if height_max is not None else rng_range[1])
        meters_per_level = float(cfg["osm"].get("meters_per_level", 3.0))

        # Use per-building features if available (preserves osm_id +
        # honours height/levels tags), otherwise fall back to splitting
        # the unioned MultiPolygon.
        if building_features:
            iterator = enumerate(building_features)
            use_features = True
        else:
            iterator = enumerate(iter_polys(geom_local))
            use_features = False

        for idx, item in iterator:
            try:
                if use_features:
                    p = item["geometry"]
                    osm_id = int(item.get("osm_id", 0))
                    h_tag = item.get("height_tag")
                    l_tag = item.get("levels_tag")
                    h = None

                    def _finite_float(value):
                        try:
                            v = float(str(value).split()[0].replace("m", ""))
                        except Exception:  # noqa: BLE001
                            return None
                        return v if math.isfinite(v) else None

                    # Prefer explicit OSM height tag (handles "12", "12 m",
                    # "12.5"). Fall back to building:levels * meters_per_level.
                    h_val = _finite_float(h_tag)
                    if h_val is not None:
                        h = h_val
                    if h is None:
                        lvl_val = _finite_float(l_tag)
                        if lvl_val is not None:
                            h = lvl_val * meters_per_level
                    if h is None or not math.isfinite(float(h)):
                        h = _sample_building_height(rng, height_dist,
                                                     hmin, hmax)
                    h = max(0.5, float(h))
                    bid = (f"osm_{osm_id}" if osm_id
                            else f"b_{idx:05d}")
                else:
                    p = item
                    h = _sample_building_height(rng, height_dist, hmin, hmax)
                    h = max(0.5, float(h))
                    osm_id = 0
                    bid = f"b_{idx:05d}"
                m = extrude_polygon(p, base_z=0.0, top_z=h)
                meshes.append(m)
                c = p.centroid
                building_records.append({
                    "building_id": bid,
                    "osm_id": int(osm_id),
                    "centroid_local_xy_m": [float(c.x), float(c.y)],
                    "height_m": float(round(h, 3)),
                    "footprint_area_m2": float(round(p.area, 2)),
                })
            except Exception as e:
                warnings.warn(f"  building extrude failed: {e}")
    else:
        z = z_offset.get(class_name, 0.0)
        for p in iter_polys(geom_local):
            m = flat_polygon_mesh(p, z=z)
            if m is not None:
                meshes.append(m)

    merged = merge_meshes(meshes)
    if merged is None:
        return None
    merged.metadata["class"] = class_name
    merged.metadata["class_id"] = CLASS_IDS[class_name]
    if building_records:
        merged.metadata["building_records"] = building_records
    return merged


def export_glb(class_meshes: dict, out_path: Path):
    """Export a multi-node GLB; each class is its own geometry node."""
    import trimesh

    scene = trimesh.Scene()
    for name, m in class_meshes.items():
        if m is None:
            continue
        # attach extras via geometry name + a sidecar dict
        m.metadata["name"] = name
        scene.add_geometry(m, node_name=name, geom_name=name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path.as_posix())
    print(f"  exported {out_path}  ({len(class_meshes)} classes)")


def process_city(city_cfg: dict, cfg: dict,
                 height_dist: str = "lognormal", height_seed: int = 42,
                 height_min: float | None = None,
                 height_max: float | None = None):
    name = city_cfg["name"]
    bbox = tuple(city_cfg["bbox"])
    geojson_dir = ROOT / cfg["paths"]["geojson_dir"]
    # Per-tile output folder: <tile_root>/<name>/obj/<name>.glb
    # Falls back to legacy blender_dir/meshes_dir layout if tile_root
    # isn't set. Sidecar meta also goes under metadata/.
    tile_root_cfg = (cfg["paths"].get("tile_root")
                     or cfg["paths"].get("blender_dir")
                     or cfg["paths"].get("meshes_dir") or "output")
    tile_dir = ensure_dir(ROOT / tile_root_cfg / name)
    # GLB is a Blender-flavoured asset (KR3 imports it back into a
    # .blend), so it lives next to the .blend file under blender/.
    glb_dir = ensure_dir(tile_dir / "blender")
    meta_dir = ensure_dir(tile_dir / "metadata")

    print(f"[KR2] {name}: building meshes")
    utm_crs = utm_crs_for_bbox(bbox)
    class_meshes = {}
    sw_xy = (0.0, 0.0)
    sidecar = {
        "city": name,
        "bbox_wgs84": list(bbox),
        "utm_crs": utm_crs.to_string(),
        "classes": {},
    }

    # ---- Pre-load the masks we will subtract from foliage ------------- #
    # The canopy / SAM3 mask doesn't know about OSM streets / buildings,
    # so without subtraction the green substrate floods over roads,
    # roofs and lawns. We carve those out so foliage only lives in
    # genuinely tree-eligible space.
    from shapely.ops import unary_union as _uunion
    _excl_classes = ("building", "road", "water", "grass")
    _excl_geoms = []
    for _ec in _excl_classes:
        _ep = geojson_dir / f"{name}_{_ec}.geojson"
        if _ep.exists():
            _eg = load_class_geom(_ep)
            if _eg is not None and not _eg.is_empty:
                _excl_geoms.append(_eg)
    foliage_exclude = _uunion(_excl_geoms) if _excl_geoms else None

    for cls in CLASS_IDS:
        gp = geojson_dir / f"{name}_{cls}.geojson"
        geom = load_class_geom(gp) if gp.exists() else None

        # If this is foliage, also union in the SAM3-derived geojson
        # (written by osm_app.on_save when a SAM3 mask is present).
        if cls == "foliage":
            sam3_gp = geojson_dir / f"{name}_foliage_sam3.geojson"
            sam3_geom = load_class_geom(sam3_gp) if sam3_gp.exists() else None
            if sam3_geom is not None:
                if geom is None:
                    geom = sam3_geom
                else:
                    geom = _uunion([geom, sam3_geom])
                print(f"  foliage: merged SAM3 geojson "
                      f"({sam3_gp.name})")
            # Also union in canopy-derived foliage (from ETH/Meta canopy
            # height grid, written by osm_app._compute_canopy_for_state).
            canopy_gp = geojson_dir / f"{name}_foliage_canopy.geojson"
            canopy_geom = (load_class_geom(canopy_gp)
                            if canopy_gp.exists() else None)
            if canopy_geom is not None:
                if geom is None:
                    geom = canopy_geom
                else:
                    geom = _uunion([geom, canopy_geom])
                print(f"  foliage: merged canopy geojson "
                      f"({canopy_gp.name})")
            # Carve out buildings / roads / water / grass so trees never
            # spawn on top of them.
            if geom is not None and foliage_exclude is not None:
                _before_a = float(geom.area)
                geom = geom.difference(foliage_exclude)
                _after_a = float(geom.area) if geom is not None else 0.0
                if geom is not None and geom.is_empty:
                    geom = None
                print(f"  foliage: subtracted {_excl_classes}  "
                      f"area {_before_a:.0f} -> {_after_a:.0f} m^2")

        if geom is None:
            print(f"  skip {cls} (no geojson)")
            continue
        geom_local, sw_xy = project_to_local(geom, bbox)

        # For buildings: load the per-building FeatureCollection if KR1
        # wrote it, project each polygon to local coords, and pass them
        # in so we keep osm_id + honour OSM height/levels tags.
        building_features = None
        if cls == "building":
            bld_gp = geojson_dir / f"{name}_buildings.geojson"
            if bld_gp.exists():
                try:
                    import geopandas as _gpd
                    gdf = _gpd.read_file(bld_gp)
                    gdf_utm = gdf.to_crs(utm_crs)
                    sw_x, sw_y = sw_xy
                    from shapely.affinity import translate as _xlate
                    building_features = []
                    for _, row in gdf_utm.iterrows():
                        g_local = _xlate(row.geometry,
                                          xoff=-sw_x, yoff=-sw_y)
                        if g_local.is_empty:
                            continue
                        building_features.append({
                            "geometry": g_local,
                            "osm_id": int(row.get("osm_id") or 0),
                            "height_tag": row.get("height_tag"),
                            "levels_tag": row.get("levels_tag"),
                        })
                    if building_features:
                        from shapely.ops import unary_union as _uunion
                        feat_bounds = _uunion([
                            b["geometry"] for b in building_features
                        ]).bounds
                        geom_bounds = geom_local.bounds
                        tol_m = 2.0
                        if (feat_bounds[0] < geom_bounds[0] - tol_m or
                                feat_bounds[1] < geom_bounds[1] - tol_m or
                                feat_bounds[2] > geom_bounds[2] + tol_m or
                                feat_bounds[3] > geom_bounds[3] + tol_m):
                            print("  buildings: per-building features are "
                                  "outside clipped tile bounds; using "
                                  "unioned clipped building geometry")
                            building_features = None
                    print(f"  buildings: loaded {len(building_features)}"
                          f" per-building features (with osm_id)")
                except Exception as _be:
                    print(f"  buildings: per-building load failed: "
                          f"{_be}; falling back to unioned polygons")

        m = build_class_mesh(cls, geom_local, cfg,
                             height_dist=height_dist,
                             height_seed=height_seed,
                             height_min=height_min,
                             height_max=height_max,
                             building_features=building_features)
        if m is None:
            print(f"  empty mesh: {cls}")
            continue
        class_meshes[cls] = m
        sidecar["classes"][cls] = {
            "class_id": CLASS_IDS[cls],
            "n_vertices": int(len(m.vertices)),
            "n_faces": int(len(m.faces)),
            "z_min": float(m.bounds[0, 2]),
            "z_max": float(m.bounds[1, 2]),
        }
        if cls == "building" and "building_records" in m.metadata:
            recs = m.metadata["building_records"]
            sidecar["buildings"] = recs
            sidecar["classes"][cls]["n_buildings"] = len(recs)
            heights = [r["height_m"] for r in recs]
            if heights:
                sidecar["classes"][cls]["height_stats_m"] = {
                    "min": float(min(heights)),
                    "max": float(max(heights)),
                    "mean": float(sum(heights) / len(heights)),
                }
        print(f"  {cls}: V={len(m.vertices)}  F={len(m.faces)}")

    out_glb = glb_dir / f"{name}.glb"
    export_glb(class_meshes, out_glb)
    sidecar["sw_utm"] = list(sw_xy)
    sidecar["glb_path"] = str(out_glb.relative_to(ROOT)).replace("\\", "/")
    with open(meta_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--city", default=None)
    ap.add_argument("--height-dist", default="lognormal",
                    choices=["flat", "uniform", "lognormal", "bimodal"],
                    help="building height distribution (default: lognormal)")
    ap.add_argument("--height-seed", type=int, default=42)
    ap.add_argument("--height-min", type=float, default=None,
                    help="min building height (m); default from yaml")
    ap.add_argument("--height-max", type=float, default=None,
                    help="max building height (m); default from yaml")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    cities = cfg["cities"]
    if args.city:
        cities = [c for c in cities if c["name"] == args.city]
    for city in cities:
        process_city(city, cfg,
                     height_dist=args.height_dist,
                     height_seed=args.height_seed,
                     height_min=args.height_min,
                     height_max=args.height_max)
    print("[KR2] done.")


if __name__ == "__main__":
    main()

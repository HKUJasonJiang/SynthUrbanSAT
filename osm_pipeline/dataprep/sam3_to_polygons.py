"""Convert SAM3 binary masks (image-space) into WGS84 polygons.

SAM3 returns a (H,W) bool array on the satellite tile. To feed it into the
KR2 geometry builder we need it as a GeoJSON of WGS84 polygons in the same
file format as the OSM-derived `<city>_<class>.geojson` files. Pixels are
mapped to lon/lat by *linear interpolation* between the four bbox corners
in WGS84 — accurate enough at the scale of one tile (1 km square).

Public API:
    mask_to_wgs84_polygons(mask, bbox_wgs, simplify_m=0.5) -> list[Polygon]
    write_foliage_geojson(polys, out_path) -> None
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.validation import make_valid


def _pixels_to_wgs(pts_xy: np.ndarray, bbox_wgs: Tuple[float, float, float, float],
                   img_w: int, img_h: int) -> np.ndarray:
    """Map (N,2) pixel coords (x=col, y=row, origin top-left) -> (N,2) lon,lat.

    Bbox is (W, S, E, N). Pixel (0,0) -> NW corner; (W-1,H-1) -> SE corner.
    """
    w, s, e, n = bbox_wgs
    x = pts_xy[:, 0].astype(np.float64) / max(img_w - 1, 1)
    y = pts_xy[:, 1].astype(np.float64) / max(img_h - 1, 1)
    lon = w + x * (e - w)
    lat = n - y * (n - s)  # row 0 is north
    return np.stack([lon, lat], axis=1)


def mask_to_wgs84_polygons(
    mask: np.ndarray,
    bbox_wgs: Tuple[float, float, float, float],
    simplify_px: float = 1.5,
    min_area_px: int = 30,
) -> List[Polygon]:
    """Vectorize a binary (H,W) mask to WGS84 shapely polygons.

    Uses cv2.findContours with RETR_CCOMP so holes are preserved.
    Polygons smaller than `min_area_px` (in pixels) are dropped — these are
    typically isolated false positives.
    """
    import cv2
    if mask.dtype != np.uint8:
        m8 = (mask.astype(bool).astype(np.uint8)) * 255
    else:
        m8 = (mask > 0).astype(np.uint8) * 255
    H, W = m8.shape

    contours, hierarchy = cv2.findContours(
        m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if not contours:
        return []
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # (N, 4): [next, prev, child, parent]

    # Collect outer contours and their child holes.
    polys: List[Polygon] = []
    for i, cnt in enumerate(contours):
        if hierarchy is not None and hierarchy[i][3] != -1:
            continue  # this is a hole, handled by its parent
        if cv2.contourArea(cnt) < min_area_px:
            continue
        # Simplify in pixel space (cheap) before reprojecting.
        approx = cv2.approxPolyDP(cnt, epsilon=simplify_px, closed=True)
        outer = approx.reshape(-1, 2)
        if len(outer) < 3:
            continue
        outer_wgs = _pixels_to_wgs(outer, bbox_wgs, W, H)

        holes_wgs = []
        if hierarchy is not None:
            child = hierarchy[i][2]
            while child != -1:
                ch = contours[child]
                if cv2.contourArea(ch) >= min_area_px:
                    ch_approx = cv2.approxPolyDP(
                        ch, epsilon=simplify_px, closed=True
                    ).reshape(-1, 2)
                    if len(ch_approx) >= 3:
                        holes_wgs.append(
                            _pixels_to_wgs(ch_approx, bbox_wgs, W, H)
                        )
                child = hierarchy[child][0]

        try:
            poly = Polygon(outer_wgs, holes=holes_wgs)
            poly = make_valid(poly)
            if not poly.is_empty:
                polys.append(poly)
        except Exception:  # noqa: BLE001
            continue
    return polys


def write_foliage_geojson(polys: List[Polygon], out_path: Path) -> int:
    """Write polygons as a FeatureCollection to disk (WGS84). Returns count."""
    feats = []
    for poly in polys:
        # explode any MultiPolygon that may have come out of make_valid
        geoms = list(getattr(poly, "geoms", [poly]))
        for g in geoms:
            if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            feats.append({
                "type": "Feature",
                "properties": {"source": "sam3"},
                "geometry": mapping(g),
            })
    fc = {"type": "FeatureCollection", "features": feats}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)
    return len(feats)


if __name__ == "__main__":
    # tiny smoke test: synthetic mask -> ~1 polygon
    m = np.zeros((256, 256), dtype=bool)
    m[40:120, 30:200] = True
    bbox = (-96.10, 41.30, -96.09, 41.31)  # tiny WGS box
    polys = mask_to_wgs84_polygons(m, bbox)
    print(f"got {len(polys)} polys; first: {polys[0] if polys else None}")

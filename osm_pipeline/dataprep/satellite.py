"""Fetch a satellite-image mosaic for a WGS84 bbox from Esri World Imagery.

No API key required. Downloads XYZ tiles from
https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer
and stitches/crops them to the bbox at a target output size.

Public function:
    satellite_image_for_bbox(bbox, out_size=1024) -> PIL.Image (RGB)

bbox = (W, S, E, N) in WGS84 degrees.
"""
from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TILE = 256  # XYZ tiles are 256x256
USER_AGENT = "ProcedureOSM/0.1 (+research)"


def _lonlat_to_tile(lon, lat, z):
    """WGS84 → XYZ tile (x, y) at zoom z (Web Mercator scheme)."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _pick_zoom(bbox, out_size):
    """Choose the smallest zoom whose tile-pixel width covers `out_size`
    pixels across the bbox horizontally. Caps at 19 (Esri max).
    """
    w, s, e, n = bbox
    for z in range(0, 20):
        x0, _ = _lonlat_to_tile(w, (s + n) / 2, z)
        x1, _ = _lonlat_to_tile(e, (s + n) / 2, z)
        px = (x1 - x0) * TILE
        if px >= out_size:
            return z
    return 19


def _download_tile(z, x, y, timeout=15):
    url = ESRI_URL.format(z=z, x=x, y=y)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def satellite_image_for_bbox(bbox, out_size: int = 1024,
                             max_workers: int = 8) -> Image.Image:
    """Return a PIL RGB image of size (out_size, out_size) covering bbox."""
    w, s, e, n = bbox
    z = _pick_zoom(bbox, out_size)

    x0, y0 = _lonlat_to_tile(w, n, z)   # NW corner (top-left)
    x1, y1 = _lonlat_to_tile(e, s, z)   # SE corner (bottom-right)

    tx0 = int(math.floor(x0)); tx1 = int(math.floor(x1))
    ty0 = int(math.floor(y0)); ty1 = int(math.floor(y1))

    cols = list(range(tx0, tx1 + 1))
    rows = list(range(ty0, ty1 + 1))
    canvas = Image.new("RGB", (len(cols) * TILE, len(rows) * TILE), "black")

    jobs = [(z, x, y) for y in rows for x in cols]

    def fetch(job):
        z_, x_, y_ = job
        try:
            return job, _download_tile(z_, x_, y_)
        except Exception as exc:  # noqa: BLE001
            return job, exc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for job, result in pool.map(fetch, jobs):
            z_, x_, y_ = job
            if isinstance(result, Exception):
                continue  # leave that tile black
            px = (x_ - tx0) * TILE
            py = (y_ - ty0) * TILE
            canvas.paste(result, (px, py))

    # crop to the precise sub-pixel bbox
    crop_left = (x0 - tx0) * TILE
    crop_top = (y0 - ty0) * TILE
    crop_right = (x1 - tx0) * TILE
    crop_bottom = (y1 - ty0) * TILE
    cropped = canvas.crop((crop_left, crop_top, crop_right, crop_bottom))
    return cropped.resize((out_size, out_size), Image.BILINEAR)


# --------------------------------------------------------------------- #
# OSM-rendered tile mosaic (cartographic map tiles, not satellite).     #
# --------------------------------------------------------------------- #

OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def osm_map_image_for_bbox(bbox, out_size: int = 1024,
                           max_workers: int = 4) -> Image.Image:
    """Return a PIL RGB image of size (out_size, out_size) covering bbox,
    rendered from openstreetmap.org tiles (cartographic style)."""
    w, s, e, n = bbox
    z = _pick_zoom(bbox, out_size)

    x0, y0 = _lonlat_to_tile(w, n, z)
    x1, y1 = _lonlat_to_tile(e, s, z)
    tx0 = int(math.floor(x0)); tx1 = int(math.floor(x1))
    ty0 = int(math.floor(y0)); ty1 = int(math.floor(y1))
    cols = list(range(tx0, tx1 + 1))
    rows = list(range(ty0, ty1 + 1))
    canvas = Image.new("RGB", (len(cols) * TILE, len(rows) * TILE),
                       (240, 240, 235))

    def _dl(z_, x_, y_):
        url = OSM_URL.format(z=z_, x=x_, y=y_)
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    jobs = [(z, x, y) for y in rows for x in cols]

    def fetch(job):
        try:
            return job, _dl(*job)
        except Exception as exc:  # noqa: BLE001
            return job, exc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for job, result in pool.map(fetch, jobs):
            _, x_, y_ = job
            if isinstance(result, Exception):
                continue
            px = (x_ - tx0) * TILE
            py = (y_ - ty0) * TILE
            canvas.paste(result, (px, py))

    crop_left = (x0 - tx0) * TILE
    crop_top = (y0 - ty0) * TILE
    crop_right = (x1 - tx0) * TILE
    crop_bottom = (y1 - ty0) * TILE
    cropped = canvas.crop((crop_left, crop_top, crop_right, crop_bottom))
    return cropped.resize((out_size, out_size), Image.BILINEAR)

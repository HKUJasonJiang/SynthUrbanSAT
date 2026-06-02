"""Real-world canopy-height grid fetcher / sampler.

Used to drive *realistic* tree placement and height in KR3:

  - density mask   : place more trees where canopy_height > 2 m
  - per-tree height: target h = canopy_height (with small jitter)

Two backends:

  - ``eth_10m`` : ETH Global Canopy Height 10 m (Lang et al. 2023).
                  Free, public; downloads 3deg COG tiles from
                  share.phys.ethz.ch.  Cached to ``cache/canopy/``.
  - ``local``   : user-supplied GeoTIFF (any CRS, any resolution).

Output of :func:`build_canopy_npz` is a small ``.npz`` with keys:

    heights : float32 (size, size) -- canopy height in metres,
              row 0 = north / col 0 = west (so local coords map as
              ``x_local = (col + 0.5)*gsd``,
              ``y_local = (size - 1 - row + 0.5)*gsd``).
    gsd     : float32 -- metres / pixel (matches render gsd).
    size    : int32   -- grid side in pixels.
    extent  : float32 -- side length in metres = gsd * size.

This is a tiny on-disk format so that KR3 (running in Blender's bundled
Python, which has numpy but typically *not* rasterio) can read the grid
with a single ``numpy.load``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- #
# ETH Global Canopy Height 10 m -- direct download                      #
# --------------------------------------------------------------------- #

# The original phys.ethz "share" URL is dead (now redirects to a DSpace
# handle that 500s). The dataset is mirrored on libdrive (ETH-hosted
# NextCloud) -- this URL serves the .tif directly and supports HEAD/GET.
ETH_LIBDRIVE_TOKEN = "cO8or7iOe5dT2Rt"


def _eth_tile_tag(lat: float, lon: float) -> str:
    """Return e.g. ``N39W099`` for the 3x3-deg ETH tile covering ``(lat, lon)``.

    ETH tiles are labelled by their **SW corner** (verified against
    ``ETH_GlobalCanopyHeight_10m_2020_N42W099_Map.tif`` which spans
    bounds ``left=-99, bottom=42, right=-96, top=45``):
    ``lat_sw = floor(lat/3)*3``, ``lon_sw = floor(lon/3)*3``.
    """
    lat_sw = int(math.floor(lat / 3.0) * 3)
    lon_sw = int(math.floor(lon / 3.0) * 3)
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    return f"{ns}{abs(lat_sw):02d}{ew}{abs(lon_sw):03d}"


def _eth_tile_url(tag: str) -> str:
    fname = f"ETH_GlobalCanopyHeight_10m_2020_{tag}_Map.tif"
    return (f"https://libdrive.ethz.ch/index.php/s/{ETH_LIBDRIVE_TOKEN}"
            f"/download?path=/3deg_cogs&files={fname}")


def _ensure_eth_tile(tag: str, cache_dir: Path,
                      timeout: float = 120.0) -> Path | None:
    """Download (if missing) the ETH 10m tile for ``tag``; return cached path.

    Tiles are ~150-300 MB (lossless COG). Subsequent calls hit the disk
    cache in ``cache_dir``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"ETH_{tag}.tif"
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    url = _eth_tile_url(tag)
    try:
        import urllib.request
        print(f"[canopy] downloading ETH tile {tag} ...", flush=True)
        print(f"[canopy]   from {url}", flush=True)
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 SynthUrbanSAT OSM canopy fetch"
        })
        bytes_written = 0
        last_log = 0
        with urllib.request.urlopen(req, timeout=timeout) as r, \
                open(out, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            while True:
                chunk = r.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                bytes_written += len(chunk)
                # progress log every ~20 MB
                if bytes_written - last_log >= 20 << 20:
                    last_log = bytes_written
                    if total:
                        print(f"[canopy]   {bytes_written/1e6:.0f} / "
                              f"{total/1e6:.0f} MB",
                              flush=True)
                    else:
                        print(f"[canopy]   {bytes_written/1e6:.0f} MB",
                              flush=True)
        if out.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"download too small ({out.stat().st_size} B) — "
                f"tile probably does not exist for {tag}"
            )
        print(f"[canopy] -> {out}  ({out.stat().st_size/1e6:.1f} MB)",
              flush=True)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[canopy] ETH download failed for {tag}: {e}", flush=True)
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        return None


# --------------------------------------------------------------------- #
# Sample to a local-UTM grid that matches KR3's render frustum          #
# --------------------------------------------------------------------- #

def _sample_to_local_grid(src_paths, *, utm_crs: str,
                           sw_x_utm: float, sw_y_utm: float,
                           gsd: float, size: int) -> np.ndarray:
    """Reproject + bilinear-resample one or more source tifs onto the
    target local-UTM grid covering ``[sw, sw + gsd*size]^2``.

    Multiple inputs are unioned via per-pixel ``max`` (handles tile
    overlaps and lets a local override extend the global mosaic).
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin

    out = np.zeros((size, size), dtype=np.float32)
    # rasterio's affine: row 0 is at the *top* (north), so origin = NW.
    nw_x = sw_x_utm
    nw_y = sw_y_utm + gsd * size
    dst_transform = from_origin(nw_x, nw_y, gsd, gsd)

    for sp in src_paths:
        try:
            with rasterio.open(sp) as src:
                src_raw = src.read(1)
                # ETH 10m canopy uses uint8 metres with 255 = nodata.
                # If src.nodata is unset, we still mask 255 explicitly,
                # otherwise bilinear leaks the sentinel into every cell.
                src_nodata = src.nodata if src.nodata is not None else 255
                src_data = src_raw.astype(np.float32)
                src_data[src_raw == src_nodata] = np.nan
                src_data = np.nan_to_num(
                    src_data, nan=0.0, posinf=0.0, neginf=0.0,
                )
                tmp = np.zeros_like(out)
                reproject(
                    source=src_data,
                    destination=tmp,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=0.0,
                    dst_nodata=0.0,
                )
                out = np.maximum(out, tmp)
        except Exception as e:  # noqa: BLE001
            print(f"[canopy] reproject failed for {sp}: {e}")
    out = np.clip(out, 0.0, 80.0)
    return out


def build_canopy_npz(*, lat: float, lon: float, gsd: float, size: int,
                      utm_crs: str, cx_utm: float, cy_utm: float,
                      cache_dir: Path,
                      out_npz: Path,
                      source: str = "eth_10m",
                      local_tif: str | None = None) -> dict:
    """Build a canopy-height ``.npz`` aligned with KR3's render frustum.

    Returns a small summary dict with stats for UI display. If sampling
    fails the npz is still written but full of zeros (so KR3 can detect
    "no canopy data" and skip canopy-driven scatter gracefully).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    half = 0.5 * gsd * size
    sw_x_utm = cx_utm - half
    sw_y_utm = cy_utm - half

    # Resolve source paths.
    src_paths: list[Path] = []
    src_kind = source
    if source == "local":
        if local_tif and Path(local_tif).exists():
            src_paths.append(Path(local_tif))
        else:
            print(f"[canopy] local tif not found: {local_tif}")
    elif source == "eth_10m":
        # Cover bbox corners + centre to handle tiles straddling boundaries.
        bbox_w = lon - (gsd * size) / 222000.0  # ~1deg lon ~ 111km*cos(lat)
        bbox_e = lon + (gsd * size) / 222000.0
        bbox_s = lat - (gsd * size) / 222000.0
        bbox_n = lat + (gsd * size) / 222000.0
        tags = set()
        for la in (bbox_s, lat, bbox_n):
            for lo in (bbox_w, lon, bbox_e):
                tags.add(_eth_tile_tag(la, lo))
        for tag in sorted(tags):
            p = _ensure_eth_tile(tag, cache_dir / "eth")
            if p is not None:
                src_paths.append(p)
    else:
        print(f"[canopy] unknown source '{source}'")

    if src_paths:
        heights = _sample_to_local_grid(
            src_paths, utm_crs=utm_crs,
            sw_x_utm=sw_x_utm, sw_y_utm=sw_y_utm,
            gsd=float(gsd), size=int(size),
        )
    else:
        heights = np.zeros((size, size), dtype=np.float32)

    np.savez_compressed(
        out_npz,
        heights=heights.astype(np.float32),
        gsd=np.float32(gsd),
        size=np.int32(size),
        extent=np.float32(gsd * size),
        sw_x_utm=np.float32(sw_x_utm),
        sw_y_utm=np.float32(sw_y_utm),
        utm_crs=np.array(str(utm_crs)),
        source=np.array(str(src_kind)),
    )

    nz = heights[heights > 1.0]
    summary = {
        "ok": bool(src_paths) and float(heights.max()) > 0.5,
        "source": src_kind,
        "n_sources": len(src_paths),
        "min_m": float(heights.min()),
        "max_m": float(heights.max()),
        "mean_canopy_m": float(nz.mean()) if nz.size else 0.0,
        "frac_treed": float((heights > 2.0).mean()),
        "npz_path": str(out_npz),
    }
    print(f"[canopy] built {out_npz}: max={summary['max_m']:.1f}m  "
          f"frac>2m={summary['frac_treed']:.3f}  "
          f"src={summary['n_sources']}")
    return summary


def render_canopy_preview(npz_path: Path):
    """Return a PIL.Image colourised view of the canopy grid (or None)."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    z = np.load(npz_path)
    h = z["heights"].astype(np.float32)
    if h.max() <= 0.0:
        rgb = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.uint8)
        return Image.fromarray(rgb)
    norm = np.clip(h / max(h.max(), 1e-3), 0.0, 1.0)
    # green ramp: low=brown-ish, high=dark green
    r = (255 * (0.55 - 0.35 * norm)).clip(0, 255).astype(np.uint8)
    g = (255 * (0.30 + 0.55 * norm)).clip(0, 255).astype(np.uint8)
    b = (255 * (0.15 + 0.10 * norm)).clip(0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[h < 0.5] = (40, 40, 40)
    return Image.fromarray(rgb)


# --------------------------------------------------------------------- #
# Vectorize canopy>threshold mask -> WGS84 foliage geojson.             #
# Same on-disk format as dataprep/sam3_to_polygons.write_foliage_geojson #
# so KR2 picks it up via the existing union mechanism.                  #
# --------------------------------------------------------------------- #

def canopy_npz_to_foliage_geojson(npz_path: Path, bbox_wgs, out_path: Path,
                                    *, height_threshold_m: float = 2.0,
                                    target_ratio: float | None = None,
                                    simplify_px: float = 1.5,
                                    min_area_px: int = 25) -> tuple[int, float, float]:
    """Vectorize the (canopy >= threshold) mask in a canopy npz to WGS84
    polygons and write them to ``out_path`` as a foliage geojson.

    Parameters
    ----------
    height_threshold_m
        Minimum canopy height to count as foliage. Default 2 m.
    target_ratio
        If given (e.g. ``0.4``), keep raising the threshold until the mask
        covers at most ``target_ratio`` of the tile. This is the easiest
        way to thin out heavily forested tiles ("only the tallest 40%").

    Returns
    -------
    (n_polys, used_threshold_m, achieved_ratio)
    """
    from dataprep.sam3_to_polygons import (
        mask_to_wgs84_polygons, write_foliage_geojson,
    )
    from scipy.ndimage import gaussian_filter
    z = np.load(str(npz_path), allow_pickle=False)
    h = z["heights"].astype(np.float32)
    # Apply Gaussian filter to smooth the coarse 10m ETH canopy grid
    # before thresholding. This converts blocky pixelation (方方正正) into
    # smooth, rounded organic forest shapes.
    h = gaussian_filter(h, sigma=4.0)
    thr = float(height_threshold_m)
    ratio = float((h >= thr).mean())

    # Auto-thresholding: keep only the tallest pixels until ratio <= target.
    # Works on the raster (cheap) BEFORE polygonisation.
    if target_ratio is not None and ratio > float(target_ratio) > 0.0:
        # use percentile to land near the target ratio in one shot, then
        # walk up by 0.5 m bumps if needed (tiles can have plateaus that
        # make the percentile imprecise).
        keep_frac = float(target_ratio)
        # percentile of values that ARE above floor (so background 0s
        # don't dominate); find threshold separating top keep_frac of tile
        q = 100.0 * (1.0 - keep_frac)
        thr_q = float(np.percentile(h, q))
        thr = max(thr, thr_q)
        ratio = float((h >= thr).mean())
        bumps = 0
        while ratio > float(target_ratio) and bumps < 60:
            thr += 0.5
            ratio = float((h >= thr).mean())
            bumps += 1
        print(f"[canopy] thinning to target_ratio={target_ratio:.2f}: "
              f"thr={thr:.1f}m -> ratio={ratio:.3f}", flush=True)

    if not (h >= thr).any():
        print(f"[canopy] vectorize: no pixel >= {thr:.1f}m")
        write_foliage_geojson([], out_path)
        return 0, thr, 0.0
    mask = (h >= thr).astype(np.uint8)
    polys = mask_to_wgs84_polygons(
        mask, tuple(bbox_wgs),
        simplify_px=float(simplify_px),
        min_area_px=int(min_area_px),
    )
    n = write_foliage_geojson(polys, out_path)
    print(f"[canopy] vectorize -> {out_path} ({n} polys, "
          f"thr={thr:.1f}m, ratio={ratio:.3f})")
    return n, thr, ratio

"""Generate full-frame off-nadir satellite depth tests.

This script does NOT change the production `4_seg.*` / `5_depth.*` logic. It
reads existing nadir seg/depth PNGs and produces A/B/C/D off-nadir maps in the
same 1024x1024 tile image frame. The tile is not rotated; only elevated
surfaces are relief-displaced and filled with simple facade ramps.

Usage:
    python scripts/render_sat_depth_tests.py \
        --depth-png output/ui_debug_city/tile_0002/5_depth.png \
        --seg-png output/ui_debug_city/tile_0002/4_seg.png \
        --out-dir output/sat_test/tile_0002
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SAT_CASES = [
    ("A", 5.0, 180.0),
    ("B", 15.0, 135.0),
    ("C", 25.0, 225.0),
    ("D", 20.0, 90.0),
]

SAT_PRESETS = [
    {
        "folder": "near-nadir-1",
        "name": "A",
        "camera_off_nadir_deg": 5.0,
        "camera_azimuth_deg": 180.0,
        "sun_elevation_deg": 60.0,
        "sun_azimuth_deg": 135.0,
    },
    {
        "folder": "near-nadir-2",
        "name": "B",
        "camera_off_nadir_deg": 15.0,
        "camera_azimuth_deg": 135.0,
        "sun_elevation_deg": 45.0,
        "sun_azimuth_deg": 180.0,
    },
    {
        "folder": "near-nadir-3",
        "name": "C",
        "camera_off_nadir_deg": 25.0,
        "camera_azimuth_deg": 225.0,
        "sun_elevation_deg": 35.0,
        "sun_azimuth_deg": 225.0,
    },
    {
        "folder": "near-nadir-4",
        "name": "D",
        "camera_off_nadir_deg": 20.0,
        "camera_azimuth_deg": 90.0,
        "sun_elevation_deg": 25.0,
        "sun_azimuth_deg": 270.0,
    },
]


def preset_folder_name(preset: dict) -> str:
    return str(preset["folder"])


def legacy_preset_folder_name(preset: dict) -> str:
    return (
        f"camera_off{int(round(preset['camera_off_nadir_deg'])):03d}"
        f"_az{int(round(preset['camera_azimuth_deg'])):03d}"
        f"_sun_el{int(round(preset['sun_elevation_deg'])):02d}"
        f"_az{int(round(preset['sun_azimuth_deg'])):03d}"
    )


def _preset_params(preset: dict, *, max_height_m: float, gsd_m: float) -> dict:
    return {
        "preset": preset["folder"],
        "case": preset["name"],
        "camera": {
            "off_nadir_deg": float(preset["camera_off_nadir_deg"]),
            "azimuth_deg": float(preset["camera_azimuth_deg"]),
        },
        "sun": {
            "elevation_deg": float(preset["sun_elevation_deg"]),
            "azimuth_deg": float(preset["sun_azimuth_deg"]),
        },
        "warp": {
            "method": "fixed_tile_frame_relief_displacement",
            "gsd_m_per_px": float(gsd_m),
            "max_height_m": float(max_height_m),
        },
        "files": {
            "seg": "1_seg.png",
            "depth": "2_depth.png",
            "shadow": "3_shadow.png",
            "params": "params.json",
        },
        "shadow": {
            "method": "height_ray_projection_from_depth",
            "encoding": "uint8 mask: 0=shadow, 255=lit_or_no_data",
            "caster_min_depth_u8": 20,
        },
    }


def _displacement_fields(
    depth_u8: np.ndarray,
    off_nadir_deg: float,
    azimuth_deg: float,
    *,
    max_height_m: float,
    gsd_m: float,
) -> tuple[np.ndarray, int, float, float]:
    src = depth_u8.astype(np.float32)
    height_m = (src / 255.0) * float(max_height_m)
    displacement = height_m * math.tan(math.radians(float(off_nadir_deg))) / float(gsd_m)
    max_step = int(math.ceil(float(np.nanmax(displacement))))

    az = math.radians(float(azimuth_deg))
    # Image coordinates: +x east/right, +y south/down. Compass azimuth is
    # clockwise from north, so northward displacement is negative image y.
    step_dx = math.sin(az)
    step_dy = -math.cos(az)
    return displacement, max_step, step_dx, step_dy


def _offnadir_depth_warp(
    depth_u8: np.ndarray,
    off_nadir_deg: float,
    azimuth_deg: float,
    *,
    max_height_m: float = 30.0,
    gsd_m: float = 0.5,
    min_height_u8: int = 2,
) -> np.ndarray:
    """Project a nadir height/depth raster into a fixed tile-frame view.

    The output keeps the same image frame as the input. Each elevated source
    pixel contributes a short line along the off-nadir look direction:
    low values near the source footprint approximate facade bases, and high
    values near the displaced roof/canopy location approximate visible tops.
    Overlaps are resolved with a max-height z-buffer.
    """
    if depth_u8.ndim != 2:
        raise ValueError("depth_u8 must be a single-channel array")
    h_px, w_px = depth_u8.shape
    src = depth_u8.astype(np.float32)
    displacement, max_step, step_dx, step_dy = _displacement_fields(
        depth_u8,
        off_nadir_deg,
        azimuth_deg,
        max_height_m=max_height_m,
        gsd_m=gsd_m,
    )

    yy, xx = np.indices((h_px, w_px), dtype=np.float32)
    out = np.zeros((h_px, w_px), dtype=np.float32)
    elevated = src >= float(min_height_u8)

    if max_step <= 0:
        return src.astype(np.uint8)

    # Sweep from footprint to displaced roof/canopy. Using maximum values gives
    # a simple image-space z-buffer and avoids holes in facade-like stretches.
    for step in range(max_step + 1):
        active = elevated & (displacement >= max(step - 0.5, 0.0))
        if not np.any(active):
            continue
        denom = np.maximum(displacement, 1.0)
        ramp = np.clip(step / denom, 0.0, 1.0)
        # Keep facades visible but lower than roof/canopy tops.
        value = src * (0.18 + 0.82 * ramp)
        x2 = np.rint(xx + step * step_dx).astype(np.int32)
        y2 = np.rint(yy + step * step_dy).astype(np.int32)
        valid = active & (x2 >= 0) & (x2 < w_px) & (y2 >= 0) & (y2 < h_px)
        if np.any(valid):
            np.maximum.at(out, (y2[valid], x2[valid]), value[valid])

    # Preserve tiny above-ground details that had subpixel displacement.
    out = np.maximum(out, src * (displacement < 0.5))
    return np.clip(out, 0, 255).astype(np.uint8)


def _offnadir_seg_warp(
    seg_rgb: np.ndarray,
    depth_u8: np.ndarray,
    off_nadir_deg: float,
    azimuth_deg: float,
    *,
    max_height_m: float = 30.0,
    gsd_m: float = 0.5,
    min_height_u8: int = 2,
) -> np.ndarray:
    """Warp semantic RGB labels with the same fixed-frame relief displacement."""
    if seg_rgb.ndim != 3 or seg_rgb.shape[2] != 3:
        raise ValueError("seg_rgb must be an RGB array")
    if seg_rgb.shape[:2] != depth_u8.shape:
        raise ValueError("seg_rgb and depth_u8 must have matching width/height")

    h_px, w_px = depth_u8.shape
    src_depth = depth_u8.astype(np.float32)
    displacement, max_step, step_dx, step_dy = _displacement_fields(
        depth_u8,
        off_nadir_deg,
        azimuth_deg,
        max_height_m=max_height_m,
        gsd_m=gsd_m,
    )
    if max_step <= 0:
        return seg_rgb.copy()

    yy, xx = np.indices((h_px, w_px), dtype=np.float32)
    out = seg_rgb.copy()
    zbuf = src_depth.copy()
    elevated = src_depth >= float(min_height_u8)

    for step in range(max_step + 1):
        active = elevated & (displacement >= max(step - 0.5, 0.0))
        if not np.any(active):
            continue
        denom = np.maximum(displacement, 1.0)
        ramp = np.clip(step / denom, 0.0, 1.0)
        value = src_depth * (0.18 + 0.82 * ramp)
        x2 = np.rint(xx + step * step_dx).astype(np.int32)
        y2 = np.rint(yy + step * step_dy).astype(np.int32)
        valid = active & (x2 >= 0) & (x2 < w_px) & (y2 >= 0) & (y2 < h_px)
        if not np.any(valid):
            continue
        target_y = y2[valid]
        target_x = x2[valid]
        incoming = value[valid]
        update = incoming >= zbuf[target_y, target_x]
        if np.any(update):
            yy_update = target_y[update]
            xx_update = target_x[update]
            out[yy_update, xx_update] = seg_rgb[valid][update]
            zbuf[yy_update, xx_update] = incoming[update]
    return out


def _shadow_from_depth(
    depth_u8: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    *,
    max_height_m: float = 30.0,
    gsd_m: float = 0.5,
    min_height_u8: int = 20,
) -> np.ndarray:
    """Create a fixed-frame shadow mask from height/depth and sun angles."""
    if depth_u8.ndim != 2:
        raise ValueError("depth_u8 must be a single-channel array")
    h_px, w_px = depth_u8.shape
    height_m = (depth_u8.astype(np.float32) / 255.0) * float(max_height_m)
    elevation = max(float(sun_elevation_deg), 1.0)
    shadow_len = height_m / math.tan(math.radians(elevation)) / float(gsd_m)
    max_step = int(math.ceil(float(np.nanmax(shadow_len))))
    if max_step <= 0:
        return np.zeros((h_px, w_px), dtype=np.uint8)

    az = math.radians(float(sun_azimuth_deg))
    # Sun azimuth is where light comes from; shadows go the opposite way.
    step_dx = -math.sin(az)
    step_dy = math.cos(az)
    yy, xx = np.indices((h_px, w_px), dtype=np.float32)
    elevated = depth_u8 >= int(min_height_u8)
    shadow = np.zeros((h_px, w_px), dtype=bool)

    for step in range(1, max_step + 1):
        active = elevated & (shadow_len >= max(step - 0.5, 0.0))
        if not np.any(active):
            continue
        x2 = np.rint(xx + step * step_dx).astype(np.int32)
        y2 = np.rint(yy + step * step_dy).astype(np.int32)
        valid = active & (x2 >= 0) & (x2 < w_px) & (y2 >= 0) & (y2 < h_px)
        if np.any(valid):
            shadow[y2[valid], x2[valid]] = True
    return np.where(shadow, 0, 255).astype(np.uint8)


def _write_contact_sheet(out_dir: Path, case_names: list[str], suffix: str) -> None:
    tiles = []
    for name in case_names:
        path = out_dir / f"{name}_{suffix}.png"
        if path.exists():
            tiles.append((name, Image.open(path).convert("RGB")))
    if not tiles:
        return
    size = tiles[0][1].size[0]
    canvas = Image.new("RGB", (size * 2, size * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (name, image) in enumerate(tiles[:4]):
        x = (idx % 2) * size
        y = (idx // 2) * size
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + 210, y + 44), fill=(255, 255, 255))
        draw.text((x + 16, y + 12), f"{name}_{suffix}", fill=(0, 0, 0))
    canvas.save(out_dir / f"sat_test_{suffix}_grid.png")


def write_offnadir_preset_folders(
    depth_png: Path | str,
    seg_png: Path | str,
    out_dir: Path | str,
    *,
    max_height_m: float = 30.0,
    gsd_m: float = 0.5,
) -> list[Path]:
    """Write `1_seg.png`, `2_depth.png`, and `3_shadow.png` under each preset folder."""
    depth_path = Path(depth_png)
    seg_path = Path(seg_png)
    base_dir = Path(out_dir)
    depth_u8 = np.asarray(Image.open(depth_path).convert("L"), dtype=np.uint8)
    seg_rgb = np.asarray(Image.open(seg_path).convert("RGB"), dtype=np.uint8)
    if seg_rgb.shape[:2] != depth_u8.shape:
        raise ValueError(f"seg size {seg_rgb.shape[:2]} does not match depth size {depth_u8.shape}")

    for old_dir in list(base_dir.glob("camera_off*")) + list(base_dir.glob("near-ndir-*")):
        if old_dir.is_dir():
            shutil.rmtree(old_dir, ignore_errors=True)

    written: list[Path] = []
    for preset in SAT_PRESETS:
        case_dir = base_dir / preset_folder_name(preset)
        case_dir.mkdir(parents=True, exist_ok=True)
        for old_name in ("4_seg.png", "5_depth.png"):
            old_path = case_dir / old_name
            if old_path.exists():
                old_path.unlink()
        depth_warped = _offnadir_depth_warp(
            depth_u8,
            float(preset["camera_off_nadir_deg"]),
            float(preset["camera_azimuth_deg"]),
            max_height_m=float(max_height_m),
            gsd_m=float(gsd_m),
        )
        seg_warped = _offnadir_seg_warp(
            seg_rgb,
            depth_u8,
            float(preset["camera_off_nadir_deg"]),
            float(preset["camera_azimuth_deg"]),
            max_height_m=float(max_height_m),
            gsd_m=float(gsd_m),
        )
        shadow = _shadow_from_depth(
            depth_warped,
            float(preset["sun_elevation_deg"]),
            float(preset["sun_azimuth_deg"]),
            max_height_m=float(max_height_m),
            gsd_m=float(gsd_m),
        )
        Image.fromarray(seg_warped).save(case_dir / "1_seg.png")
        Image.fromarray(depth_warped).save(case_dir / "2_depth.png")
        Image.fromarray(shadow).save(case_dir / "3_shadow.png")
        (case_dir / "params.json").write_text(
            json.dumps(
                _preset_params(preset, max_height_m=max_height_m, gsd_m=gsd_m),
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(case_dir)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-png", required=True)
    parser.add_argument("--seg-png", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-height-m", type=float, default=30.0)
    parser.add_argument("--gsd", type=float, default=0.5)
    args = parser.parse_args()

    depth_path = Path(args.depth_png)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_u8 = np.asarray(Image.open(depth_path).convert("L"), dtype=np.uint8)
    seg_path = Path(args.seg_png) if args.seg_png else depth_path.with_name("4_seg.png")
    seg_rgb = None
    if seg_path.exists():
        seg_rgb = np.asarray(Image.open(seg_path).convert("RGB"), dtype=np.uint8)
        if seg_rgb.shape[:2] != depth_u8.shape:
            raise ValueError(f"seg size {seg_rgb.shape[:2]} does not match depth size {depth_u8.shape}")
    else:
        print(f"[sat_test] seg skipped: {seg_path} not found")

    for name, off_nadir, azimuth in SAT_CASES:
        warped = _offnadir_depth_warp(
            depth_u8,
            off_nadir,
            azimuth,
            max_height_m=float(args.max_height_m),
            gsd_m=float(args.gsd),
        )
        out_path = out_dir / f"{name}_depth.png"
        Image.fromarray(warped).save(out_path)
        print(
            f"[sat_test] {name}: off_nadir={off_nadir:.1f}, "
            f"azimuth={azimuth:.1f}, saved={out_path}"
        )
        if seg_rgb is not None:
            seg_warped = _offnadir_seg_warp(
                seg_rgb,
                depth_u8,
                off_nadir,
                azimuth,
                max_height_m=float(args.max_height_m),
                gsd_m=float(args.gsd),
            )
            seg_out_path = out_dir / f"{name}_seg.png"
            Image.fromarray(seg_warped).save(seg_out_path)
            print(f"[sat_test] {name}: seg saved={seg_out_path}")
    _write_contact_sheet(out_dir, [case[0] for case in SAT_CASES], "depth")
    print(f"[sat_test] grid saved={out_dir / 'sat_test_depth_grid.png'}")
    if seg_rgb is not None:
        _write_contact_sheet(out_dir, [case[0] for case in SAT_CASES], "seg")
        print(f"[sat_test] grid saved={out_dir / 'sat_test_seg_grid.png'}")
        folders = write_offnadir_preset_folders(
            depth_path,
            seg_path,
            out_dir,
            max_height_m=float(args.max_height_m),
            gsd_m=float(args.gsd),
        )
        print(f"[sat_test] preset folders saved={len(folders)}")


if __name__ == "__main__":
    main()

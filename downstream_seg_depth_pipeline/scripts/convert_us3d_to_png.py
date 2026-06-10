"""Convert a pre-split US3D root to PNG while preserving height metres.

The output keeps the same split layout:

    <out>/<split>/{rgb,seg,depth}/<stem>.png

RGB is saved as uint8 PNG. Segmentation is copied/converted to PNG. Depth is
encoded as uint16 PNG with a small JSON sidecar so downstream loaders can decode
it back to metres:

    metres = encoded / scale + offset_m
    nodata  = 65535 -> NaN
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


RGB_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
LABEL_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
DEPTH_EXTS = (".tif", ".tiff", ".png")
DEPTH_ENCODING = {
    "format": "uint16_png_linear_metres",
    "scale": 100.0,
    "offset_m": -10.0,
    "nodata": 65535,
}


def _find(base: Path, stem: str, exts: tuple[str, ...]) -> Path | None:
    for try_stem in (stem, stem.replace("_RGB_", "_")):
        for ext in exts:
            cand = base / f"{try_stem}{ext}"
            if cand.exists():
                return cand
    return None


def _read_depth(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _write_depth_png(src: Path, dst: Path) -> None:
    arr = _read_depth(src)
    scale = float(DEPTH_ENCODING["scale"])
    offset_m = float(DEPTH_ENCODING["offset_m"])
    nodata = int(DEPTH_ENCODING["nodata"])
    finite = np.isfinite(arr)
    enc = np.full(arr.shape, nodata, dtype=np.uint16)
    vals = np.rint((arr[finite] - offset_m) * scale)
    vals = np.clip(vals, 0, nodata - 1).astype(np.uint16)
    enc[finite] = vals
    Image.fromarray(enc, mode="I;16").save(dst)


def _write_rgb_png(src: Path, dst: Path) -> None:
    Image.open(src).convert("RGB").save(dst)


def _write_seg_png(src: Path, dst: Path) -> None:
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
    else:
        Image.open(src).save(dst)


def convert_split(src_root: Path, out_root: Path, split: str) -> int:
    src_split = src_root / split
    out_split = out_root / split
    for sub in ("rgb", "seg", "depth"):
        (out_split / sub).mkdir(parents=True, exist_ok=True)

    count = 0
    for rgb_src in sorted((src_split / "rgb").iterdir()):
        if rgb_src.suffix.lower() not in RGB_EXTS:
            continue
        stem = rgb_src.stem
        seg_src = _find(src_split / "seg", stem, LABEL_EXTS)
        depth_src = _find(src_split / "depth", stem, DEPTH_EXTS)
        if not (seg_src and depth_src):
            print(f"[skip] {split}/{stem}: missing seg/depth")
            continue
        _write_rgb_png(rgb_src, out_split / "rgb" / f"{stem}.png")
        _write_seg_png(seg_src, out_split / "seg" / f"{stem}.png")
        _write_depth_png(depth_src, out_split / "depth" / f"{stem}.png")
        count += 1
    print(f"[convert] {split}: {count} tiles")
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pre-split US3D root")
    ap.add_argument("--out", required=True, help="output PNG root")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "depth_encoding.json").write_text(json.dumps(DEPTH_ENCODING, indent=2) + "\n")
    prompt_src = src_root / "prompt.json"
    if prompt_src.exists():
        shutil.copy2(prompt_src, out_root / "prompt.json")

    totals = {split: convert_split(src_root, out_root, split) for split in ("train", "val", "test")}
    print("[done]", out_root, totals)


if __name__ == "__main__":
    main()
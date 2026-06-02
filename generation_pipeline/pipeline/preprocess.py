"""Image preprocessing for seg / depth / rgb inputs.

Lifted (essentially verbatim) from the original training notebook so the
generation pipeline is self-contained. Reads:
- Palette PNG / greyscale / RGB / TIFF segs (-> long class index map)
- Float TIFF / PNG depth (-> [0,1] tensor)
- uint8 / uint16 / float RGB (-> [0,1] tensor)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

from . import COLOR_MAP_PATH

SUPPORTED_EXTS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
SIBLING_DIRS = ('rgb', 'seg', 'depth')


def _is_tiff(path: str) -> bool:
    return Path(path).suffix.lower() in ('.tif', '.tiff')


def _load_array_tiff(path: str) -> np.ndarray:
    return tifffile.imread(path)


def _pil_open_preserve_mode(path: str) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


def preprocess_seg(path: str, size: int, num_classes: int) -> torch.Tensor:
    """Load a seg image -> [H, W] long class-index tensor."""
    if not _is_tiff(path):
        img = _pil_open_preserve_mode(path)
        if img.size != (size, size):
            img = img.resize((size, size), Image.NEAREST)
        if img.mode == 'P':
            arr = np.array(img, dtype=np.int64)
            arr = np.clip(arr - 1, 0, num_classes - 1)
            return torch.from_numpy(arr)
        raw = np.array(img.convert('RGB') if img.mode not in ('L', 'I', 'F') else img,
                       dtype=np.int64)
    else:
        raw = _load_array_tiff(path)
        while raw.ndim > 3:
            raw = raw[0]
        if raw.shape[:2] != (size, size):
            img_tmp = Image.fromarray(
                raw.astype(np.uint8) if raw.dtype != np.uint8 else raw
            ).resize((size, size), Image.NEAREST)
            raw = np.array(img_tmp, dtype=np.int64)
        else:
            raw = raw.astype(np.int64)

    if raw.ndim == 2:
        arr = np.clip(raw, 0, num_classes - 1)
        return torch.from_numpy(arr)

    with open(COLOR_MAP_PATH) as f:
        cmap = json.load(f)
    rgb_to_cls = {tuple(int(c) for c in v['rgb']): int(k) for k, v in cmap.items()}
    rgb = raw[..., :3].astype(np.uint8)
    arr = np.full(rgb.shape[:2], num_classes - 1, dtype=np.int64)
    for rgb_tuple, cid in rgb_to_cls.items():
        mask = (rgb == np.array(rgb_tuple, dtype=np.uint8)).all(axis=-1)
        arr[mask] = cid
    return torch.from_numpy(arr)


def preprocess_depth(path: str, size: int) -> torch.Tensor:
    """Load a depth image -> [1, H, W] float tensor in [0, 1]."""
    if _is_tiff(path):
        arr = _load_array_tiff(path).astype(np.float32)
    else:
        img = Image.open(path)
        if img.mode != 'F':
            img = img.convert('L') if img.mode != 'L' else img
        arr = np.array(img, dtype=np.float32)

    while arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] <= arr.shape[0] else arr[0]

    if arr.shape != (size, size):
        arr_pil = Image.fromarray(arr).resize((size, size), Image.LANCZOS)
        arr = np.array(arr_pil, dtype=np.float32)

    mn, mx = float(arr.min()), float(arr.max())
    arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
    return torch.from_numpy(arr).unsqueeze(0)


def preprocess_rgb(path: str, size: int) -> torch.Tensor:
    """Load an RGB image -> [3, H, W] float tensor in [0, 1]."""
    if _is_tiff(path):
        arr = _load_array_tiff(path)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        elif np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            arr = (arr - mn) / (mx - mn + 1e-8)
        else:
            arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0

    if arr.shape[:2] != (size, size):
        pil_tmp = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        arr = np.array(pil_tmp.resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0

    return torch.from_numpy(arr).permute(2, 0, 1)


# ─── Colourised previews ─────────────────────────────────────────────────────

def seg_to_rgb(seg_tensor: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    return palette[seg_tensor.cpu().numpy()]


def depth_to_rgb(depth_tensor: torch.Tensor) -> np.ndarray:
    import matplotlib.cm as cm
    d = depth_tensor[0].cpu().numpy()
    return (cm.viridis(d)[..., :3] * 255).astype(np.uint8)


# ─── Sibling resolver / folder scan ──────────────────────────────────────────

def _find_sibling(src_path: str, target_dir: str) -> str | None:
    if not src_path:
        return None
    p = Path(src_path)
    if not p.exists():
        return None
    parent_root = p.parent.parent
    target = parent_root / target_dir
    if not target.is_dir():
        return None
    preferred = [p.suffix.lower()] + [e for e in SUPPORTED_EXTS if e != p.suffix.lower()]
    for ext in preferred:
        for cand in (target / f'{p.stem}{ext}', target / f'{p.stem}{ext.upper()}'):
            if cand.exists():
                return str(cand)
    return None


def resolve_siblings(src_path: str, src_kind: str) -> dict:
    out = {'seg': None, 'depth': None, 'rgb': None}
    out[src_kind] = src_path
    if not src_path:
        return out
    p = Path(src_path)
    if p.parent.name.lower() != src_kind:
        return out
    for other in SIBLING_DIRS:
        if other == src_kind:
            continue
        out[other] = _find_sibling(src_path, other)
    return out


def scan_root_folder(root: str) -> dict:
    out = {'stems': [], 'by_stem': {}, 'status': ''}
    if not root or not root.strip():
        out['status'] = '_(enter a folder path and click Scan)_'
        return out
    root_p = Path(root.strip()).expanduser()
    if not root_p.is_dir():
        out['status'] = f'❌ Not a directory: `{root_p}`'
        return out

    by_kind = {}
    missing_dirs = []
    for kind in SIBLING_DIRS:
        d = root_p / kind
        if not d.is_dir():
            missing_dirs.append(kind)
            by_kind[kind] = {}
            continue
        entries = {}
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS and f.stem not in entries:
                entries[f.stem] = str(f)
        by_kind[kind] = entries

    if missing_dirs:
        out['status'] = (f'❌ Missing sub-folder(s) under `{root_p}`: {", ".join(missing_dirs)} '
                         f'(expected all of {", ".join(SIBLING_DIRS)})')
        return out

    common = sorted(set(by_kind['seg']) & set(by_kind['depth']) & set(by_kind['rgb']))
    out['stems'] = common
    for s in common:
        out['by_stem'][s] = {k: by_kind[k][s] for k in SIBLING_DIRS}
    out['status'] = (
        f'✅ Scanned `{root_p}`: {len(common)} common stems.' if common
        else f'⚠️ Scanned `{root_p}` but found no common stems across all three sub-folders.'
    )
    return out

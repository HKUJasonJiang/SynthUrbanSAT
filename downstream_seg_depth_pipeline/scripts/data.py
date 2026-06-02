"""Dataset and mixture loaders for downstream probing.

Expected layout for each data root (real US3D and synthetic share it):

    <root>/<split>/
        rgb/    *.tif|*.png   satellite RGB (real or generated), uint8
        seg/    *.png         6-class palette/RGB segmentation
        depth/  *.tif|*.exr   AGL height in METRES (US3D .tif / OSM .exr)

Companion files are matched by stem (with the same ``_RGB_`` -> ``_`` fallback
used in train_pipeline/dataprep.py). RGB is ImageNet-normalised for DINOv2.

A :class:`MixtureDataset` concatenates a fraction of real tiles with a capped
number of synthetic tiles, enabling the R / S(TSTR) / R+S conditions and the
data-scaling curves from a single config.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from scripts.labels import LabelSpace

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_RGB_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
_DEPTH_EXTS = (".tif", ".tiff", ".exr", ".png")


def _read_height_metres(path: str) -> np.ndarray:
    """Read an AGL height raster (metres) as float32 [H, W]. Supports tif/exr/png."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".exr":
        arr = _read_exr(path)
    else:
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    # Non-finite (sky / no-data) -> NaN so metrics ignore it.
    arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
    return arr


def _read_exr(path: str) -> np.ndarray:
    try:
        import imageio.v2 as imageio

        return np.asarray(imageio.imread(path), dtype=np.float32)
    except Exception:
        pass
    try:
        import cv2

        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if arr is None:
            raise IOError(f"cv2 failed to read {path}")
        return np.asarray(arr, dtype=np.float32)
    except Exception as e:  # pragma: no cover - depends on optional libs
        raise IOError(
            f"Could not read EXR {path!r}. Install 'imageio[freeimage]' or opencv-python."
        ) from e


class TileDataset(Dataset):
    """One data root/split: yields (rgb, seg, height) aligned tensors."""

    def __init__(self, split_dir: str, label_space: LabelSpace, image_size: int = 518,
                 task: str = "segmentation"):
        self.split_dir = split_dir
        self.ls = label_space
        self.image_size = image_size
        self.task = task

        rgb_dir = os.path.join(split_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            self.stems = []
            return
        self.stems = []
        for f in sorted(os.listdir(rgb_dir)):
            if f.lower().endswith(_RGB_EXTS):
                stem = os.path.splitext(f)[0]
                if self._find("seg", stem) and self._find("depth", stem):
                    self.stems.append((f, stem))

    def _find(self, sub, stem):
        base = os.path.join(self.split_dir, sub)
        exts = _DEPTH_EXTS if sub == "depth" else (".png", ".tif", ".tiff", ".jpg", ".jpeg")
        for try_stem in (stem, stem.replace("_RGB_", "_")):
            for ext in exts:
                cand = os.path.join(base, try_stem + ext)
                if os.path.exists(cand):
                    return cand
        return None

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        fname, stem = self.stems[idx]
        s = self.image_size
        rgb_path = os.path.join(self.split_dir, "rgb", fname)
        seg_path = self._find("seg", stem)
        depth_path = self._find("depth", stem)

        rgb_img = Image.open(rgb_path).convert("RGB").resize((s, s), Image.BILINEAR)
        rgb = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

        seg = self.ls.decode_seg(Image.open(seg_path), target_hw=(s, s))
        seg = torch.from_numpy(seg).long()

        height = _read_height_metres(depth_path)
        if height.shape != (s, s):
            h_img = Image.fromarray(height).resize((s, s), Image.BILINEAR)
            height = np.array(h_img, dtype=np.float32)
        height = torch.from_numpy(height).unsqueeze(0)

        return {"rgb": rgb, "seg": seg, "height": height, "stem": stem}


def resolve_synth_root(cfg: dict) -> str:
    """Pick the synthetic data root for the current run.

    Prefers ``data.synth_sources[data.synth_source]`` (named source, e.g.
    'us3d_paired' or 'osm'); falls back to the legacy ``data.synth_root``.
    """
    d = cfg["data"]
    sources = d.get("synth_sources")
    name = d.get("synth_source")
    if sources and name:
        if name not in sources:
            raise KeyError(
                f"synth_source {name!r} not in synth_sources {list(sources)}. "
                "Set data.synth_source to a defined key or extend data.synth_sources."
            )
        return sources[name]
    return d.get("synth_root", "./dataset_synth")


def _subset(ds: Dataset, count: int, seed: int = 0) -> Dataset:
    """Deterministically take ``count`` items (count<=0 -> empty, > len -> all)."""
    n = len(ds)
    if count <= 0:
        return torch.utils.data.Subset(ds, [])
    if count >= n:
        return ds
    rng = random.Random(seed)
    idx = rng.sample(range(n), count)
    return torch.utils.data.Subset(ds, idx)


def build_mixture(real_split_dir, synth_split_dir, label_space, image_size, task,
                  real_fraction=1.0, synth_count=0, seed=0):
    """Build a training dataset mixing a fraction of real with capped synthetic.

    real_fraction in [0, 1]; synth_count is an absolute tile cap. Returns a
    torch Dataset (possibly a ConcatDataset).
    """
    parts = []
    if real_split_dir and os.path.isdir(real_split_dir) and real_fraction > 0:
        real = TileDataset(real_split_dir, label_space, image_size, task)
        n_real = max(1, int(round(len(real) * real_fraction))) if len(real) else 0
        parts.append(_subset(real, n_real, seed))
    if synth_split_dir and os.path.isdir(synth_split_dir) and synth_count > 0:
        synth = TileDataset(synth_split_dir, label_space, image_size, task)
        parts.append(_subset(synth, synth_count, seed))
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        raise ValueError("Empty training mixture: check real_fraction / synth_count / paths.")
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)

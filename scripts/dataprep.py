"""Data preparation and loading for HDC²A training.

Classes:
  HDC2ADataset(Dataset)  - Loads (rgb, seg, depth) triplets + optional text embeddings.

Functions:
  create_dataloaders(dataset_dir, color_map_path, ..., embeddings_dict=None)
      -> (train_loader, val_loader)  or  (None, None) if data is missing

Dataset structure expected:
    dataset/{train,val}/
        rgb/            *.tif  (target RGB images, uint8)
        seg/            *.png  (segmentation maps, palette-mode P, values 0-5)
                               value 0 = background, values 1-5 = class IDs 0-4
        depth/          *.tif  (depth maps, float32 single-channel, in metres)
    dataset/
        prompt.json     single global text prompt JSON (all samples share one prompt)

Filenames must have the same stem across rgb/, seg/, depth/ within each split.
Extensions may differ: rgb=.tif, seg=.png, depth=.tif.

Text embeddings:
  embeddings_dict may contain:
    'global'              -> [seq_len, text_dim]  shared by all samples (preferred)
    'split/filename.ext'  -> per-sample embedding (legacy fallback)
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from scripts.augment import HDC2AAugment


class HDC2ADataset(Dataset):
    """Dataset for HDC²A training: loads aligned (RGB, Seg, Depth, Text) tuples.

    Args:
        data_dir: path to split directory (e.g. dataset/train)
        color_map_path: path to color_map.json  (kept for API compatibility; unused)
        image_size: resize target (square)
        num_classes: number of segmentation classes (5, mapping palette idx 1-5 → 0-4)
        split: 'train' or 'val' (used as key prefix for per-sample embeddings)
        embeddings_dict: dict with 'global' or 'split/filename' → tensor
    """
    def __init__(self, data_dir, color_map_path, image_size=512, num_classes=5,
                 split='train', embeddings_dict=None, augment=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.split = split
        self.embeddings_dict = embeddings_dict
        self.augment = augment  # HDC2AAugment instance or None

        # Load color map for RGB seg → class index conversion
        self._rgb_to_class = {}  # (R, G, B) -> class_id
        if color_map_path and os.path.exists(color_map_path):
            with open(color_map_path) as f:
                cmap = json.load(f)
            for cls_str, info in cmap.items():
                rgb_tuple = tuple(info['rgb'])
                self._rgb_to_class[rgb_tuple] = int(cls_str)

        # Discover files from rgb/ directory (source of truth for sample list)
        # Filter to only those that have matching seg and depth companions.
        rgb_dir = os.path.join(data_dir, 'rgb')
        if os.path.isdir(rgb_dir):
            candidates = sorted([
                f for f in os.listdir(rgb_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
            ])
            self.filenames = []
            skipped = 0
            for fname in candidates:
                stem = os.path.splitext(fname)[0]
                try:
                    self._find_companion('seg', stem)
                    self._find_companion('depth', stem)
                    self.filenames.append(fname)
                except FileNotFoundError:
                    skipped += 1
            if skipped:
                print(f'[HDC2ADataset] {split}: skipped {skipped} RGB files with no seg/depth companion')
        else:
            self.filenames = []
            print(f'[HDC2ADataset] Warning: no rgb/ directory found at {rgb_dir}')

    def _companion_stem(self, stem):
        """Return the stem used to find seg/depth companions.

        RGB files may carry '_RGB_' in the name (e.g. JAX_Tile_018_RGB_007_1)
        while seg/depth files use the short form (JAX_Tile_018_007_1).
        We try the original stem first, then strip '_RGB_' → '_'.
        """
        return stem.replace('_RGB_', '_')

    def _find_companion(self, subdir, stem):
        """Return path to companion file matching stem in subdir, trying multiple extensions.

        Tries original stem first, then the _RGB_-stripped variant.
        """
        base_dir = os.path.join(self.data_dir, subdir)
        for try_stem in (stem, self._companion_stem(stem)):
            for ext in ('.png', '.tif', '.tiff', '.jpg', '.jpeg'):
                candidate = os.path.join(base_dir, try_stem + ext)
                if os.path.exists(candidate):
                    return candidate
        raise FileNotFoundError(
            f'No companion file found for stem {stem!r} in {base_dir} '
            f'(tried original and _RGB_-stripped, exts .png/.tif/.tiff/.jpg/.jpeg)')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        stem = os.path.splitext(fname)[0]
        rgb_path = os.path.join(self.data_dir, 'rgb', fname)
        seg_path = self._find_companion('seg', stem)
        depth_path = self._find_companion('depth', stem)

        # ── RGB: uint8 tif → [3, H, W] float [0, 1] ────────────────────────
        rgb_img = Image.open(rgb_path).convert('RGB')
        if rgb_img.size != (self.image_size, self.image_size):
            rgb_img = rgb_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        rgb = torch.from_numpy(np.array(rgb_img, dtype=np.float32)).permute(2, 0, 1) / 255.0

        # ── Seg: palette PNG → [H, W] long (class indices 0 .. num_classes-1) ──
        # Supports both palette (mode P) and RGB seg maps.
        seg_img = Image.open(seg_path)
        if seg_img.size != (self.image_size, self.image_size):
            seg_img = seg_img.resize((self.image_size, self.image_size), Image.NEAREST)
        if seg_img.mode == 'P':
            # Palette format: idx 0 = background, idx 1-5 = semantic classes 0-4.
            # Map: class_id = clamp(palette_idx - 1, 0, num_classes-1)
            seg_arr = np.array(seg_img, dtype=np.int64)
            seg_arr = np.clip(seg_arr - 1, 0, self.num_classes - 1)
        else:
            # RGB seg map → convert via color_map.json lookup
            seg_rgb = np.array(seg_img.convert('RGB'), dtype=np.uint8)  # [H, W, 3]
            seg_arr = np.full(seg_rgb.shape[:2], self.num_classes - 1, dtype=np.int64)
            for rgb_tuple, cls_id in self._rgb_to_class.items():
                mask = np.all(seg_rgb == np.array(rgb_tuple, dtype=np.uint8), axis=-1)
                seg_arr[mask] = cls_id
        seg_tensor = torch.from_numpy(seg_arr)

        # ── Depth: float32 tif (metres) → [1, H, W] float [0, 1] ───────────
        # Per-image min-max normalisation so the encoder always sees full range.
        depth_img = Image.open(depth_path)      # mode F (float32)
        if depth_img.size != (self.image_size, self.image_size):
            depth_img = depth_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        depth_arr = np.array(depth_img, dtype=np.float32)
        d_min, d_max = float(depth_arr.min()), float(depth_arr.max())
        if d_max > d_min:
            depth_arr = (depth_arr - d_min) / (d_max - d_min)
        else:
            depth_arr = np.zeros_like(depth_arr)
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)

        # ── Augmentation (training only; caller sets augment=None for val) ──
        if self.augment is not None:
            rgb, seg_tensor, depth_tensor = self.augment(rgb, seg_tensor, depth_tensor)

        result = {
            'rgb': rgb,             # [3, H, W] float [0, 1]
            'seg': seg_tensor,      # [H, W] long [0, num_classes-1]
            'depth': depth_tensor,  # [1, H, W] float [0, 1]
            'filename': fname,
        }

        # ── Text embedding lookup (global key preferred, then per-sample) ────
        if self.embeddings_dict is not None:
            embed_key = f'{self.split}/{fname}'
            if 'global' in self.embeddings_dict:
                result['prompt_embeds'] = self.embeddings_dict['global']
            elif embed_key in self.embeddings_dict:
                result['prompt_embeds'] = self.embeddings_dict[embed_key]

        return result


def create_dataloaders(dataset_dir, color_map_path, image_size=512,
                       batch_size=1, num_workers=0, num_classes=5,
                       embeddings_dict=None, shuffle_train=True,
                       use_augment=False, augment_kwargs=None,
                       include_test=False):
    """Create train, validation, and optionally test dataloaders.

    Args:
        dataset_dir: root dataset directory containing train/ val/ [test/]
        color_map_path: path to color_map.json
        image_size: resize target
        batch_size: batch size
        num_workers: dataloader workers (0 for main thread)
        num_classes: number of segmentation classes
        embeddings_dict: precomputed text embeddings {"split/file.png": tensor}
        shuffle_train: whether to shuffle the training set
        use_augment: enable data augmentation on the training split
        augment_kwargs: dict of keyword arguments forwarded to HDC2AAugment
        include_test: if True, also create and return a test_loader

    Returns:
        (train_loader, val_loader)            when include_test=False
        (train_loader, val_loader, test_loader) when include_test=True
        Any loader is None if the corresponding split directory is missing / empty.
    """
    train_augment = None
    if use_augment:
        kwargs = augment_kwargs or {}
        train_augment = HDC2AAugment(**kwargs)
        print(f'[Data] Augmentation enabled: {train_augment}')
    train_loader = None
    val_loader = None
    test_loader = None

    def _make_loader(split_name, shuffle=False, augment=None):
        split_dir = os.path.join(dataset_dir, split_name)
        if not (os.path.isdir(split_dir) and os.path.isdir(os.path.join(split_dir, 'rgb'))):
            print(f'[Data] No {split_name}/ directory at {split_dir} — skipping.')
            return None
        ds = HDC2ADataset(split_dir, color_map_path, image_size, num_classes,
                          split=split_name, embeddings_dict=embeddings_dict,
                          augment=augment)
        if len(ds) == 0:
            print(f'[Data] {split_name.capitalize()} directory exists but no images found.')
            return None
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
            drop_last=(split_name == 'train'),
        )
        print(f'[Data] {split_name.capitalize()}: {len(ds)} samples, batch_size={batch_size}')
        if split_name == 'train' and embeddings_dict:
            has_global = 'global' in embeddings_dict
            print(f'[Data] Train: using {"global" if has_global else "per-sample"} text embeddings')
        return loader

    train_loader = _make_loader('train', shuffle=shuffle_train, augment=train_augment)
    val_loader   = _make_loader('val',   shuffle=False)
    if include_test:
        test_loader = _make_loader('test', shuffle=False)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader

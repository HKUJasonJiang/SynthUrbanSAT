"""Unified label-space utilities for the SynthUrbanSAT downstream tasks.

Both US3D-Enhanced and OSM-synthetic segmentation maps use the same 6-class
palette (see configs/label_map.json, mirrors train_pipeline color_map.json):

    0 road | 1 water | 2 foliage | 3 building | 4 grass | 5 ground

Two on-disk encodings are supported, matching train_pipeline/dataprep.py:
  * Palette-mode PNG (mode 'P'): index 0 = background/void, 1..5 = classes 0..4.
  * RGB PNG: decoded via the per-class RGB colours in label_map.json.
"""

import json
import os

import numpy as np

DEFAULT_LABEL_MAP = os.path.join(os.path.dirname(__file__), os.pardir, "configs", "label_map.json")


class LabelSpace:
    """Holds the unified class list and RGB<->index lookups."""

    def __init__(self, label_map_path: str = DEFAULT_LABEL_MAP):
        with open(label_map_path) as f:
            cfg = json.load(f)
        self.ignore_index = int(cfg.get("ignore_index", 255))
        classes = cfg["classes"]
        self.num_classes = len(classes)
        # index -> name, index -> rgb tuple
        self.names = [classes[str(i)]["name"] for i in range(self.num_classes)]
        self.rgb = [tuple(classes[str(i)]["rgb"]) for i in range(self.num_classes)]
        self._rgb_to_idx = {rgb: i for i, rgb in enumerate(self.rgb)}

    def decode_seg(self, pil_image, target_hw=None) -> np.ndarray:
        """Convert a PIL seg image to an int64 [H, W] class-index array.

        Palette ('P') maps idx 1..5 -> 0..4 (idx 0 -> ground/background = class 5
        by convention here, kept consistent with downstream ignore handling).
        RGB images are matched against the class palette; unmatched pixels are
        set to ``ignore_index``.
        """
        from PIL import Image

        if target_hw is not None and pil_image.size != (target_hw[1], target_hw[0]):
            pil_image = pil_image.resize((target_hw[1], target_hw[0]), Image.NEAREST)

        if pil_image.mode == "P":
            arr = np.array(pil_image, dtype=np.int64)
            # Mirror train_pipeline: class_id = clamp(palette_idx - 1, 0, K-1).
            arr = np.clip(arr - 1, 0, self.num_classes - 1)
            return arr

        rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)
        out = np.full(rgb.shape[:2], self.ignore_index, dtype=np.int64)
        for color, idx in self._rgb_to_idx.items():
            mask = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
            out[mask] = idx
        return out

"""Data augmentation for HDC²A training (remote-sensing paired data).

All spatial transforms are applied **jointly** to (rgb, seg, depth) so they
stay perfectly aligned.  Photometric transforms apply to **RGB only**.

The dataset contains satellite/aerial imagery (JAX tiles) where:
  - There is no canonical "up" direction → rotation by 90°/180°/270° is safe.
  - Multiple sensors / acquisition dates → colour + brightness jitter helps.
  - Cloud / shadow occlusions → random cutout helps.
  - Different GSD / atmospheric conditions → Gaussian blur + noise helps.

Usage::

    from scripts.augment import HDC2AAugment
    aug = HDC2AAugment()                     # all defaults
    aug = HDC2AAugment(p_scale_crop=0.0)     # disable scale-crop
    rgb, seg, depth = aug(rgb, seg, depth)

Tensor formats expected:
    rgb:   [3, H, W]  float32  in [0, 1]
    seg:   [H, W]     int64    in [0, num_classes-1]
    depth: [1, H, W]  float32  in [0, 1]

Returns the same formats.
"""

import math
import random

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hflip(rgb, seg, depth):
    return rgb.flip(-1), seg.flip(-1), depth.flip(-1)


def _vflip(rgb, seg, depth):
    return rgb.flip(-2), seg.flip(-2), depth.flip(-2)


def _rot90(rgb, seg, depth, k):
    """Rotate by k*90 degrees CCW (k in {1,2,3})."""
    return (torch.rot90(rgb,   k, [-2, -1]),
            torch.rot90(seg,   k, [-2, -1]),
            torch.rot90(depth, k, [-2, -1]))


def _scale_crop(rgb, seg, depth, scale_lo, scale_hi):
    """Random scale-crop: crop a random patch and resize back to original H×W.

    The crop size is sampled uniformly from [scale_lo, scale_hi] of the image
    side length.  This simulates multi-scale / zoom variation.

    seg uses nearest-neighbour; rgb and depth use bilinear.
    """
    _, H, W = rgb.shape
    scale = random.uniform(scale_lo, scale_hi)
    crop_h = max(1, int(round(H * scale)))
    crop_w = max(1, int(round(W * scale)))

    top  = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    rgb_c   = rgb  [:, top:top+crop_h, left:left+crop_w]
    seg_c   = seg  [   top:top+crop_h, left:left+crop_w]
    depth_c = depth[:, top:top+crop_h, left:left+crop_w]

    # Resize back
    rgb_r   = F.interpolate(rgb_c.unsqueeze(0), size=(H, W),
                            mode='bilinear', align_corners=False).squeeze(0)
    seg_r   = F.interpolate(seg_c.unsqueeze(0).unsqueeze(0).float(), size=(H, W),
                            mode='nearest').squeeze(0).squeeze(0).long()
    depth_r = F.interpolate(depth_c.unsqueeze(0), size=(H, W),
                            mode='bilinear', align_corners=False).squeeze(0)
    return rgb_r, seg_r, depth_r


def _color_jitter(rgb, brightness, contrast, saturation, hue):
    """Random colour jitter applied to rgb tensor [3,H,W] in [0,1].

    Applies in random order: brightness → contrast → saturation → hue.
    All factors are sampled uniformly from [1-x, 1+x] (or [-x,x] for hue).
    """
    ops = []
    if brightness > 0:
        ops.append('brightness')
    if contrast > 0:
        ops.append('contrast')
    if saturation > 0:
        ops.append('saturation')
    if hue > 0:
        ops.append('hue')
    random.shuffle(ops)

    out = rgb.clone()
    for op in ops:
        if op == 'brightness':
            factor = random.uniform(max(0.0, 1 - brightness), 1 + brightness)
            out = (out * factor).clamp(0, 1)
        elif op == 'contrast':
            factor = random.uniform(max(0.0, 1 - contrast), 1 + contrast)
            mean = out.mean(dim=(-2, -1), keepdim=True)
            out = ((out - mean) * factor + mean).clamp(0, 1)
        elif op == 'saturation':
            factor = random.uniform(max(0.0, 1 - saturation), 1 + saturation)
            gray = 0.2126 * out[0:1] + 0.7152 * out[1:2] + 0.0722 * out[2:3]
            out = ((out - gray) * factor + gray).clamp(0, 1)
        elif op == 'hue':
            # Rotate hue channel in HSV space approximated via RGB rotation matrix
            angle = random.uniform(-hue, hue) * 2 * math.pi  # hue shift in radians
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            # Rodrigues rotation around (1,1,1)/sqrt(3)
            k = 1.0 / math.sqrt(3.0)
            R = torch.tensor([
                [cos_a + k*k*(1-cos_a),   k*k*(1-cos_a)-k*sin_a, k*k*(1-cos_a)+k*sin_a],
                [k*k*(1-cos_a)+k*sin_a,   cos_a + k*k*(1-cos_a), k*k*(1-cos_a)-k*sin_a],
                [k*k*(1-cos_a)-k*sin_a,   k*k*(1-cos_a)+k*sin_a, cos_a + k*k*(1-cos_a)],
            ], dtype=out.dtype, device=out.device)
            flat = out.reshape(3, -1)
            out = (R @ flat).reshape_as(out).clamp(0, 1)
    return out


def _gaussian_blur(rgb, kernel_size, sigma_lo, sigma_hi):
    """Apply Gaussian blur with random sigma to rgb tensor.

    kernel_size must be odd.
    """
    sigma = random.uniform(sigma_lo, sigma_hi)
    ks = kernel_size
    # Build 1-D Gaussian kernel and apply as separable 2-D filter
    coords = torch.arange(ks, dtype=rgb.dtype, device=rgb.device) - ks // 2
    kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # [ks, ks]
    kernel_2d = kernel_2d.view(1, 1, ks, ks).expand(3, 1, ks, ks)
    pad = ks // 2
    rgb_b = F.conv2d(rgb.unsqueeze(0), kernel_2d, padding=pad, groups=3).squeeze(0)
    return rgb_b.clamp(0, 1)


def _gaussian_noise(rgb, std_lo, std_hi):
    """Add Gaussian noise with random std to rgb tensor."""
    std = random.uniform(std_lo, std_hi)
    noise = torch.randn_like(rgb) * std
    return (rgb + noise).clamp(0, 1)


def _cutout(rgb, n_holes, hole_fraction):
    """Zero out n_holes random rectangular patches in rgb (H, W fraction of each side)."""
    _, H, W = rgb.shape
    out = rgb.clone()
    for _ in range(n_holes):
        h_cut = int(H * hole_fraction)
        w_cut = int(W * hole_fraction)
        cy = random.randint(0, H - 1)
        cx = random.randint(0, W - 1)
        y1 = max(0, cy - h_cut // 2)
        y2 = min(H, cy + h_cut // 2)
        x1 = max(0, cx - w_cut // 2)
        x2 = min(W, cx + w_cut // 2)
        out[:, y1:y2, x1:x2] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class HDC2AAugment:
    """Joint augmentation for (rgb, seg, depth) paired remote-sensing data.

    Parameters
    ----------
    p_hflip : float
        Probability of random horizontal flip (default 0.5).
    p_vflip : float
        Probability of random vertical flip (default 0.5).
    p_rotate : float
        Probability of random 90°/180°/270° rotation (default 0.5).
    p_scale_crop : float
        Probability of random scale-crop (default 0.5).
    scale_crop_range : tuple(float, float)
        (min_scale, max_scale) for scale-crop; e.g. (0.7, 1.0) means crop
        between 70 % and 100 % of the image side (default (0.7, 1.0)).
    p_color_jitter : float
        Probability of colour jitter (RGB only, default 0.8).
    brightness : float
        Max brightness change fraction (default 0.3).
    contrast : float
        Max contrast change fraction (default 0.3).
    saturation : float
        Max saturation change fraction (default 0.2).
    hue : float
        Max hue rotation in turns, range [0, 0.5] (default 0.05).
    p_blur : float
        Probability of Gaussian blur (RGB only, default 0.3).
    blur_kernel_size : int
        Gaussian blur kernel size, must be odd (default 7).
    blur_sigma : tuple(float, float)
        (min, max) sigma for blur (default (0.1, 2.0)).
    p_noise : float
        Probability of Gaussian noise (RGB only, default 0.3).
    noise_std : tuple(float, float)
        (min, max) noise std (default (0.005, 0.03)).
    p_cutout : float
        Probability of random cutout (RGB only, default 0.3).
    n_holes : int
        Number of cutout holes (default 1).
    hole_fraction : float
        Each hole's side as fraction of image side (default 0.2).
    """

    def __init__(
        self,
        # Geometric
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rotate: float = 0.5,
        p_scale_crop: float = 0.5,
        scale_crop_range: tuple = (0.7, 1.0),
        # Photometric
        p_color_jitter: float = 0.8,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        hue: float = 0.05,
        p_blur: float = 0.3,
        blur_kernel_size: int = 7,
        blur_sigma: tuple = (0.1, 2.0),
        p_noise: float = 0.3,
        noise_std: tuple = (0.005, 0.03),
        # Occlusion
        p_cutout: float = 0.3,
        n_holes: int = 1,
        hole_fraction: float = 0.2,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rotate = p_rotate
        self.p_scale_crop = p_scale_crop
        self.scale_crop_range = scale_crop_range

        self.p_color_jitter = p_color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.p_blur = p_blur
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_sigma = blur_sigma

        self.p_noise = p_noise
        self.noise_std = noise_std

        self.p_cutout = p_cutout
        self.n_holes = n_holes
        self.hole_fraction = hole_fraction

    def __call__(self, rgb, seg, depth):
        """Apply augmentations.

        Args:
            rgb:   [3, H, W] float32 [0, 1]
            seg:   [H, W]    int64   [0, num_classes-1]
            depth: [1, H, W] float32 [0, 1]

        Returns:
            Augmented (rgb, seg, depth) in the same formats.
        """
        # ── Geometric (rgb + seg + depth jointly) ───────────────────────────
        if random.random() < self.p_hflip:
            rgb, seg, depth = _hflip(rgb, seg, depth)

        if random.random() < self.p_vflip:
            rgb, seg, depth = _vflip(rgb, seg, depth)

        if random.random() < self.p_rotate:
            k = random.randint(1, 3)  # 90, 180, 270
            rgb, seg, depth = _rot90(rgb, seg, depth, k)

        if random.random() < self.p_scale_crop:
            rgb, seg, depth = _scale_crop(
                rgb, seg, depth,
                self.scale_crop_range[0], self.scale_crop_range[1],
            )

        # ── Photometric (rgb only) ───────────────────────────────────────────
        if random.random() < self.p_color_jitter:
            rgb = _color_jitter(
                rgb,
                self.brightness, self.contrast,
                self.saturation, self.hue,
            )

        if random.random() < self.p_blur:
            rgb = _gaussian_blur(rgb, self.blur_kernel_size,
                                 self.blur_sigma[0], self.blur_sigma[1])

        if random.random() < self.p_noise:
            rgb = _gaussian_noise(rgb, self.noise_std[0], self.noise_std[1])

        # ── Occlusion (rgb only) ─────────────────────────────────────────────
        if random.random() < self.p_cutout:
            rgb = _cutout(rgb, self.n_holes, self.hole_fraction)

        return rgb, seg, depth

    def __repr__(self):
        return (
            f'HDC2AAugment('
            f'hflip={self.p_hflip}, vflip={self.p_vflip}, rotate={self.p_rotate}, '
            f'scale_crop={self.p_scale_crop}@{self.scale_crop_range}, '
            f'color_jitter={self.p_color_jitter}(b={self.brightness},c={self.contrast},'
            f's={self.saturation},h={self.hue}), '
            f'blur={self.p_blur}, noise={self.p_noise}, cutout={self.p_cutout})'
        )

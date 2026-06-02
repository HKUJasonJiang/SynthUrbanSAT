"""Task heads for downstream probing.

  * LinearHead : 1x1 conv on the last feature map, bilinearly upsampled.
                 Cheapest probe; segmentation only (DINOv2 linear protocol).
  * DPTHead    : compact Dense Prediction Transformer decoder that fuses 4
                 backbone feature maps. Used for both segmentation (out_ch =
                 num_classes) and height regression (out_ch = 1).

Both heads take a list of [B, C, h, w] feature maps and return a dense
prediction at the input image resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_ch, kernel_size=1)

    def forward(self, feats, out_hw):
        x = self.proj(feats[-1])
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


class _ResidualConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class _FusionBlock(nn.Module):
    """Upsample lower-res fused features and add the lateral skip."""

    def __init__(self, dim):
        super().__init__()
        self.res = _ResidualConv(dim)
        self.out_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.res(skip)
        x = self.res(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class DPTHead(nn.Module):
    """Compact DPT decoder over 4 backbone feature maps (coarse->fine order)."""

    def __init__(self, feature_dims, out_ch: int, hidden_dim: int = 256):
        super().__init__()
        assert len(feature_dims) == 4, "DPTHead expects exactly 4 feature maps"
        # Per-stage reassembly to a common channel width at different scales.
        # Stages from shallow(fine) -> deep(coarse) get progressively downsampled,
        # matching the standard DPT reassemble (x4, x2, x1, x0.5).
        self.proj = nn.ModuleList(nn.Conv2d(d, hidden_dim, 1) for d in feature_dims)
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=4),  # finest
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2),
            nn.Identity(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),  # coarsest
        ])
        self.fusion = nn.ModuleList(_FusionBlock(hidden_dim) for _ in range(4))
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, out_ch, 1),
        )

    def forward(self, feats, out_hw):
        # Bring all stages to a common spatial size before fusion.
        laterals = [self.resample[i](self.proj[i](feats[i])) for i in range(4)]
        ref_hw = laterals[0].shape[-2:]
        laterals = [F.interpolate(l, size=ref_hw, mode="bilinear", align_corners=False)
                    for l in laterals]
        # Top-down fusion from coarse (idx 3) to fine (idx 0).
        x = self.fusion[3](laterals[3])
        x = F.interpolate(x, size=ref_hw, mode="bilinear", align_corners=False)
        for i in (2, 1, 0):
            x = self.fusion[i](x, laterals[i])
            x = F.interpolate(x, size=ref_hw, mode="bilinear", align_corners=False)
        out = self.output(x)
        return F.interpolate(out, size=out_hw, mode="bilinear", align_corners=False)

"""Frozen DINOv2 backbone wrapper for downstream probing.

Loads a DINOv2 ViT via ``torch.hub`` and exposes multi-layer dense patch
features as 2D feature maps suitable for a DPT decoder. The backbone is frozen
by default (ICLR main protocol): only the task head is trained, so the only
variable across experiments is the training-data mixture.
"""

import torch
import torch.nn as nn

_EMBED_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}
_NUM_BLOCKS = {
    "dinov2_vits14": 12,
    "dinov2_vitb14": 12,
    "dinov2_vitl14": 24,
    "dinov2_vitg14": 40,
}


class DINOv2Backbone(nn.Module):
    """Wrap a DINOv2 ViT and return multi-scale [B, C, h, w] feature maps."""

    def __init__(self, name: str = "dinov2_vitl14", out_indices=(4, 11, 17, 23),
                 freeze: bool = True):
        super().__init__()
        if name not in _EMBED_DIM:
            raise ValueError(f"Unknown DINOv2 variant {name!r}; choose from {list(_EMBED_DIM)}")
        self.name = name
        self.patch_size = 14
        self.embed_dim = _EMBED_DIM[name]
        self.out_indices = tuple(out_indices)
        n_blocks = _NUM_BLOCKS[name]
        if any(i >= n_blocks for i in self.out_indices):
            raise ValueError(f"out_indices {self.out_indices} exceed {n_blocks} blocks for {name}")

        self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.frozen = freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        # Keep the frozen backbone in eval mode regardless of the parent's state.
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    @property
    def feature_dims(self):
        return [self.embed_dim] * len(self.out_indices)

    def forward(self, x):
        """x: [B, 3, H, W] with H, W divisible by 14. Returns list of feature maps."""
        b, _, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f"Input {h}x{w} must be divisible by patch size {self.patch_size}")
        hp, wp = h // self.patch_size, w // self.patch_size

        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            tokens = self.model.get_intermediate_layers(
                x, n=self.out_indices, reshape=True, norm=True
            )
        # reshape=True already yields [B, C, hp, wp]; normalise to a list.
        feats = [t if t.dim() == 4 else t.transpose(1, 2).reshape(b, -1, hp, wp)
                 for t in tokens]
        return feats

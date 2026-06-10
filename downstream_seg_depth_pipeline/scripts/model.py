"""Probe model = frozen DINOv2 backbone + a single task head.

Segmentation -> per-pixel class logits  [B, num_classes, H, W]
Height (AGL) -> per-pixel height in metres [B, 1, H, W] (ReLU, >= 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.backbone import DINOv2Backbone
from scripts.heads import DPTHead, LinearHead


class ProbeModel(nn.Module):
    def __init__(self, task: str = "segmentation", num_classes: int = 6,
                 backbone_name: str = "dinov2_vitl14", out_indices=(4, 11, 17, 23),
                 head_type: str = "dpt", hidden_dim: int = 256, freeze_backbone: bool = True,
                 ndsm_max_m: float = 25.0):
        super().__init__()
        if task not in ("segmentation", "height"):
            raise ValueError(f"task must be 'segmentation' or 'height', got {task!r}")
        self.task = task
        self.ndsm_max_m = float(ndsm_max_m)
        out_ch = num_classes if task == "segmentation" else 1

        self.backbone = DINOv2Backbone(backbone_name, out_indices, freeze=freeze_backbone)
        dims = self.backbone.feature_dims
        if head_type == "linear":
            if task == "height":
                # Linear probe still works for regression; uses the last map only.
                self.head = LinearHead(dims[-1], out_ch)
            else:
                self.head = LinearHead(dims[-1], out_ch)
        elif head_type == "dpt":
            self.head = DPTHead(dims, out_ch, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"head_type must be 'linear' or 'dpt', got {head_type!r}")

    def forward(self, x):
        out_hw = x.shape[-2:]
        feats = self.backbone(x)
        out = self.head(feats, out_hw)
        if self.task == "height":
            # Positive metre-scale regression without saturating at zero or capping tall objects.
            out = F.softplus(out)
        return out

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

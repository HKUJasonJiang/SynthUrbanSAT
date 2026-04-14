"""HDC²A Model Architecture + FP8 Utilities

Classes:
  RotaryEmbedding        - 1D rotary position embedding for spatial tokens
  SemanticEncoder         - 5-class seg map  -> [B, 4096, 768] tokens
  DepthEncoder            - Depth map        -> [B, 4096, 768] tokens
  DoubleStreamFusionBlock - Joint cross-attention between seg & depth streams
  HDC2AAdapter            - Full adapter: encoders + 3x fusion + gated merge
                            Output: [B, 1024, 3072] control tokens

  FP8FrozenLinear         - Drop-in Linear replacement storing weights in FP8
  convert_frozen_linears_to_fp8(model) - Recursively swap frozen nn.Linear -> FP8

Config-sensitive:
  num_classes, fusion_dim, output_dim, num_heads, num_fusion_blocks,
  num_fourier_bands, boundary_threshold  (all set in train_script.py CONFIG)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# HDC²A: Heterogeneous Dual-Condition Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """1D Rotary Position Embedding for spatial token sequences."""
    def __init__(self, dim, max_seq_len=8192, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, x):
        """Apply RoPE to x of shape [B, N, H, D]."""
        seq_len = x.shape[1]
        D = x.shape[-1]
        cos = self.cos_cached[:seq_len][None, :, None, :]
        sin = self.sin_cached[:seq_len][None, :, None, :]
        x1 = x[..., :D // 2]
        x2 = x[..., D // 2:]
        x_rotated = torch.cat([-x2, x1], dim=-1)
        cos = cos.expand(-1, -1, -1, D // 2).repeat(1, 1, 1, 2)
        sin = sin.expand(-1, -1, -1, D // 2).repeat(1, 1, 1, 2)
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class SemanticEncoder(nn.Module):
    """Encodes discrete segmentation maps into semantic tokens.

    seg_map [B, H, W] (long, 5 classes)
      → class embedding [B, 64, H, W]
      → boundary enhancement
      → ConvStem (stride=8) [B, 512, H/8, W/8]
      → project → [B, N, 768]
    """
    def __init__(self, num_classes=5, embed_dim=64, out_dim=768, boundary_threshold=0.1):
        super().__init__()
        self.boundary_threshold = boundary_threshold
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.boundary_conv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512), nn.GELU(),
        )
        self.proj = nn.Linear(512, out_dim)

    def forward(self, seg_map):
        x = self.class_embed(seg_map)
        x = x.permute(0, 3, 1, 2)

        C = x.shape[1]
        kernel = torch.ones(C, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0
        x_smooth = F.conv2d(x, kernel, padding=1, groups=C)
        boundary_mask = (x - x_smooth).abs().sum(dim=1, keepdim=True)
        boundary_mask = (boundary_mask > self.boundary_threshold).to(x.dtype)

        boundary_feat = self.boundary_conv(x)
        x = x + boundary_mask * boundary_feat

        x = self.conv_stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DepthEncoder(nn.Module):
    """Encodes continuous depth maps into depth tokens.

    depth_map [B, 1, H, W] (float [0,1])
      → Fourier encoding [B, 65, H, W]
      → ConvStem (stride=8) [B, 512, H/8, W/8]
      → project → [B, N, 768]
    """
    def __init__(self, num_fourier_bands=32, out_dim=768):
        super().__init__()
        in_ch = 1 + num_fourier_bands * 2
        freq_bands = 2.0 ** torch.linspace(0, num_fourier_bands - 1, num_fourier_bands)
        self.register_buffer('freq_bands', freq_bands)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512), nn.GELU(),
        )
        self.proj = nn.Linear(512, out_dim)

    def forward(self, depth_map):
        depth_exp = depth_map.unsqueeze(-1) * self.freq_bands.view(1, 1, 1, 1, -1)
        depth_sin = torch.sin(2 * math.pi * depth_exp)
        depth_cos = torch.cos(2 * math.pi * depth_exp)
        depth_enc = torch.cat([depth_sin, depth_cos], dim=-1)
        depth_enc = depth_enc.squeeze(1).permute(0, 3, 1, 2)
        x = torch.cat([depth_map, depth_enc], dim=1)

        x = self.conv_stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DoubleStreamFusionBlock(nn.Module):
    """Joint attention fusion block for semantic and depth token streams.

    Each stream has its own Q,K,V projections and FFN.
    K and V are concatenated from both streams for cross-modal attention.
    RoPE is applied to Q, K before attention.
    """
    def __init__(self, dim=768, num_heads=12, max_seq_len=8192):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_sem = nn.Linear(dim, dim * 3)
        self.qkv_dep = nn.Linear(dim, dim * 3)
        self.proj_sem = nn.Linear(dim, dim)
        self.proj_dep = nn.Linear(dim, dim)

        self.ffn_sem = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ffn_dep = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        self.norm1_sem = nn.LayerNorm(dim)
        self.norm1_dep = nn.LayerNorm(dim)
        self.norm2_sem = nn.LayerNorm(dim)
        self.norm2_dep = nn.LayerNorm(dim)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, T_s, T_d):
        B, N, D = T_s.shape
        H, HD = self.num_heads, self.head_dim

        T_s_norm = self.norm1_sem(T_s)
        T_d_norm = self.norm1_dep(T_d)

        qkv_s = self.qkv_sem(T_s_norm).reshape(B, N, 3, H, HD)
        qkv_d = self.qkv_dep(T_d_norm).reshape(B, N, 3, H, HD)
        q_s, k_s, v_s = qkv_s.unbind(2)
        q_d, k_d, v_d = qkv_d.unbind(2)

        q_s = self.rope(q_s)
        k_s = self.rope(k_s)
        q_d = self.rope(q_d)
        k_d = self.rope(k_d)

        k_joint = torch.cat([k_s, k_d], dim=1)
        v_joint = torch.cat([v_s, v_d], dim=1)

        q_s_t = q_s.transpose(1, 2)
        q_d_t = q_d.transpose(1, 2)
        k_joint_t = k_joint.transpose(1, 2)
        v_joint_t = v_joint.transpose(1, 2)

        out_s = F.scaled_dot_product_attention(q_s_t, k_joint_t, v_joint_t)
        out_d = F.scaled_dot_product_attention(q_d_t, k_joint_t, v_joint_t)

        out_s = out_s.transpose(1, 2).reshape(B, N, D)
        out_d = out_d.transpose(1, 2).reshape(B, N, D)

        out_s = self.proj_sem(out_s)
        out_d = self.proj_dep(out_d)

        T_s = T_s + out_s
        T_d = T_d + out_d
        T_s = T_s + self.ffn_sem(self.norm2_sem(T_s))
        T_d = T_d + self.ffn_dep(self.norm2_dep(T_d))
        return T_s, T_d


class HDC2AAdapter(nn.Module):
    """Heterogeneous Dual-Condition Adapter.

    Input:
        seg_map:   [B, H, W]    long, 5 classes
        depth_map: [B, 1, H, W] float [0, 1]

    Output:
        control_tokens: [B, N_ctrl, output_dim]
        where N_ctrl = (H/16)*(W/16) = 1024 for 512×512 input
    """
    def __init__(self, num_classes=5, fusion_dim=768, output_dim=3072,
                 num_heads=12, num_fusion_blocks=3, num_fourier_bands=32,
                 boundary_threshold=0.1, image_size=512):
        super().__init__()
        self.output_dim = output_dim

        # Encoders use stride=8, so spatial tokens = (image_size/8)^2
        max_seq_len = (image_size // 8) ** 2

        self.semantic_encoder = SemanticEncoder(
            num_classes=num_classes, embed_dim=64, out_dim=fusion_dim,
            boundary_threshold=boundary_threshold)
        self.depth_encoder = DepthEncoder(
            num_fourier_bands=num_fourier_bands, out_dim=fusion_dim)

        self.fusion_blocks = nn.ModuleList([
            DoubleStreamFusionBlock(dim=fusion_dim, num_heads=num_heads,
                                   max_seq_len=max_seq_len)
            for _ in range(num_fusion_blocks)])

        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), nn.GELU(),
            nn.Linear(fusion_dim, 1), nn.Sigmoid())
        self.W_s = nn.Linear(fusion_dim, output_dim)
        self.W_d = nn.Linear(fusion_dim, output_dim)

        self.output_scale = nn.Parameter(torch.tensor(-5.0))
        self.spatial_downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, seg_map, depth_map):
        T_s = self.semantic_encoder(seg_map)
        T_d = self.depth_encoder(depth_map)

        for block in self.fusion_blocks:
            T_s, T_d = block(T_s, T_d)

        T_concat = torch.cat([T_s, T_d], dim=-1)
        g = self.gate_net(T_concat)

        T_s_proj = self.W_s(T_s)
        T_d_proj = self.W_d(T_d)
        T_merged = g * T_s_proj + (1 - g) * T_d_proj

        T_ctrl = T_merged * self.output_scale.sigmoid()

        B, N, D = T_ctrl.shape
        H = W = int(math.sqrt(N))
        T_ctrl = T_ctrl.transpose(1, 2).reshape(B, D, H, W)
        T_ctrl = self.spatial_downsample(T_ctrl)
        T_ctrl = T_ctrl.flatten(2).transpose(1, 2)

        return T_ctrl


# ═══════════════════════════════════════════════════════════════════════════════
# FP8 Frozen Storage (saves ~50% VRAM for frozen weights)
# ═══════════════════════════════════════════════════════════════════════════════

class _FP8LinearFunc(torch.autograd.Function):
    """Custom autograd: saves FP8 weight (not BF16) for backward, halving saved tensor VRAM."""
    @staticmethod
    def forward(ctx, x, weight_fp8, weight_scale):
        ctx.save_for_backward(weight_fp8, weight_scale)
        weight = weight_fp8.to(x.dtype) * weight_scale.to(x.dtype)
        return x @ weight.T

    @staticmethod
    def backward(ctx, grad_output):
        weight_fp8, weight_scale = ctx.saved_tensors
        weight = weight_fp8.to(grad_output.dtype) * weight_scale.to(grad_output.dtype)
        return grad_output @ weight, None, None


class FP8FrozenLinear(nn.Module):
    """Drop-in replacement for frozen nn.Linear that stores weight in FP8."""
    def __init__(self, weight_bf16, bias=None):
        super().__init__()
        amax = weight_bf16.abs().amax()
        scale = (amax / torch.finfo(torch.float8_e4m3fn).max).clamp(min=1e-12)
        self.register_buffer('weight_fp8', (weight_bf16 / scale).to(torch.float8_e4m3fn))
        self.register_buffer('weight_scale', scale.view(1))
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x):
        out = _FP8LinearFunc.apply(x, self.weight_fp8, self.weight_scale)
        if self.bias is not None:
            out = out + self.bias
        return out


def convert_frozen_linears_to_fp8(model):
    """Replace frozen nn.Linear layers with FP8FrozenLinear to halve VRAM."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not any(p.requires_grad for p in module.parameters()):
                replacements.append(name)
    for name in replacements:
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        new_mod = FP8FrozenLinear(old.weight.data,
                                  old.bias.data if old.bias is not None else None)
        setattr(parent, parts[-1], new_mod)
        del old
    return len(replacements)

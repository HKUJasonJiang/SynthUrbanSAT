"""Utility functions for GPU memory management and model loading.

Functions:
  check_memory(stage)            - Print RAM/VRAM usage, abort if over threshold
  clear_cache()                  - gc.collect + torch.cuda.empty_cache
  dequant_fp8_state_dict(sd)     - Dequantise FP8 safetensors to BF16 via scales
  load_vae(path, device, dtype)  - Load AutoencoderKLFlux2, return (vae, bn_mean, bn_std)
  load_transformer(...)          - Load Flux2 transformer: FP8 dequant -> diffusers key
                                   conversion -> controlnet merge -> re-compress FP8
  patchify_latents(z)            - 2x2 spatial patch grouping for Flux2 tokens
  pack_latents(z)                - Flatten patched latents to [B, N, C]
  prepare_latent_ids(h, w)       - Position IDs for transformer
  prepare_text_ids(seq_len)      - Position IDs for text tokens
  encode_rgb_to_latent(vae, rgb, bn_mean, bn_std) - RGB image -> normalised latent

External deps loaded inside functions: diffusers (for key converter).
"""

import gc
import os
import sys

import psutil
import torch
import torch.nn as nn
from safetensors.torch import load_file

from scripts.colors import cyan, yellow, bold_red, gray


MEM_THRESHOLD = 0.90


def check_memory(stage: str, threshold: float = MEM_THRESHOLD, abort: bool = True):
    """Print RAM/VRAM usage and optionally abort if over threshold."""
    vm = psutil.virtual_memory()
    ram_pct = vm.percent / 100.0
    _GiB = 1024 ** 3
    ram_gb, ram_tot = vm.used / _GiB, vm.total / _GiB
    vram_gb = torch.cuda.memory_allocated() / _GiB
    vram_tot = torch.cuda.get_device_properties(0).total_memory / _GiB if torch.cuda.is_available() else 0
    vram_pct = vram_gb / vram_tot if vram_tot > 0 else 0.0
    ram_str = f'RAM: {ram_gb:.1f}/{ram_tot:.1f} GiB ({ram_pct*100:.1f}%)'
    vram_str = f'VRAM: {vram_gb:.1f}/{vram_tot:.1f} GiB ({vram_pct*100:.1f}%)'
    if vram_pct >= 0.8:
        vram_str = yellow(vram_str)
    elif vram_pct >= threshold:
        vram_str = bold_red(vram_str)
    print(f'{gray("[MEM @ " + stage + "]")} {ram_str}  |  {vram_str}')
    if abort:
        for name, pct, used, total in [("RAM", ram_pct, ram_gb, ram_tot),
                                        ("VRAM", vram_pct, vram_gb, vram_tot)]:
            if pct >= threshold:
                print(f'ABORT: {name} {pct*100:.1f}% >= {threshold*100:.0f}% at "{stage}"',
                      file=sys.stderr)
                sys.exit(1)


def clear_cache():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dequant_fp8_state_dict(safetensors_path, device='cuda', dtype=torch.bfloat16):
    """Load FP8-mixed safetensors and dequantize to target dtype on device.

    Returns:
        dequant_sd: dict of dequantized tensors on device
        fp8_count: number of FP8 tensors that were dequantized
    """
    raw_sd = load_file(safetensors_path)
    dequant_sd = {}
    fp8_count = 0
    keys = list(raw_sd.keys())
    for key in keys:
        if key.endswith('.input_scale') or key.endswith('.weight_scale'):
            continue
        tensor = raw_sd.pop(key)
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key.replace('.weight', '.weight_scale')
            if scale_key in raw_sd:
                scale = raw_sd[scale_key].float()
                dequant_sd[key] = (tensor.to(torch.float32) * scale).to(dtype).to(device)
                fp8_count += 1
            else:
                dequant_sd[key] = tensor.to(dtype).to(device)
        else:
            dequant_sd[key] = (tensor.to(dtype) if tensor.is_floating_point() else tensor).to(device)
        del tensor
    del raw_sd
    clear_cache()
    return dequant_sd, fp8_count


def load_vae(vae_path, device='cuda', dtype=torch.bfloat16):
    """Load and configure the Flux2 VAE encoder/decoder.

    Returns:
        vae: frozen VAE model on device
        bn_mean: batch norm running mean [1, C, 1, 1]
        bn_std: batch norm running std [1, C, 1, 1]
    """
    # Import from local models copy
    from models.videox_fun.models import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2(
        in_channels=3, out_channels=3,
        down_block_types=('DownEncoderBlock2D',) * 4,
        up_block_types=('UpDecoderBlock2D',) * 4,
        block_out_channels=(128, 256, 512, 512),
        latent_channels=32, sample_size=1024, patch_size=(2, 2),
    )
    vae_sd = load_file(vae_path)
    m, u = vae.load_state_dict(vae_sd, strict=False)
    assert len(m) == 0 and len(u) == 0, f'VAE load failed — Missing: {m}, Unexpected: {u}'

    vae = vae.to(device, dtype=dtype).eval()
    vae.requires_grad_(False)

    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device, dtype)

    del vae_sd
    clear_cache()
    return vae, bn_mean, bn_std


def load_transformer(transformer_path, controlnet_path, control_in_dim,
                     device='cuda', dtype=torch.bfloat16):
    """Load Flux2ControlTransformer with FP8 dequant and FP8 frozen compression.

    Returns:
        transformer: model with control params trainable, backbone frozen in FP8
    """
    from models.videox_fun.models import Flux2ControlTransformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
    from scripts.models import convert_frozen_linears_to_fp8

    # Step A: Dequantize FP8 transformer weights
    print('  Dequantizing FP8 transformer weights...')
    dequant_sd, fp8_count = dequant_fp8_state_dict(transformer_path, device=device, dtype=dtype)
    print(f'  Dequantized {fp8_count} FP8 tensors')

    # Step B: Convert ComfyUI → diffusers key format
    print('  Converting ComfyUI → diffusers keys...')
    diffusers_sd = convert_flux2_transformer_checkpoint_to_diffusers(dequant_sd)
    print(f'  Converted: {len(diffusers_sd)} diffusers keys')
    del dequant_sd
    clear_cache()

    # Step C: Load ControlNet weights
    print('  Loading ControlNet weights...')
    controlnet_sd = load_file(controlnet_path)
    for k in list(controlnet_sd.keys()):
        controlnet_sd[k] = (controlnet_sd[k].to(dtype).to(device)
                            if controlnet_sd[k].is_floating_point()
                            else controlnet_sd[k].to(device))
    print(f'  ControlNet: {len(controlnet_sd)} keys')

    # Step D: Create model on meta device
    print(f'  Creating Flux2ControlTransformer2DModel (control_in_dim={control_in_dim})...')
    with torch.device('meta'):
        transformer = Flux2ControlTransformer2DModel(
            control_layers=[0, 2, 4, 6],
            control_in_dim=control_in_dim,
            patch_size=1,
            in_channels=128,
            num_layers=8,
            num_single_layers=48,
            attention_head_dim=128,
            num_attention_heads=48,
            joint_attention_dim=15360,
            timestep_guidance_channels=256,
            mlp_ratio=3.0,
            axes_dims_rope=(32, 32, 32, 32),
        )

    # Step E: Merge weights — skip control_img_in (pretrained 260 vs ours control_in_dim)
    merged_sd = {}
    merged_sd.update(diffusers_sd)
    skipped = []
    for k, v in controlnet_sd.items():
        if 'control_img_in' in k:
            skipped.append(f'{k} {list(v.shape)}')
            continue
        merged_sd[k] = v
    if skipped:
        print(f'  Skipped {len(skipped)} control_img_in keys (dim mismatch):')
        for s in skipped:
            print(f'    {s}')

    m, u = transformer.load_state_dict(merged_sd, strict=False, assign=True)
    print(f'  Missing: {len(m)}, Unexpected: {len(u)}')

    # Initialize missing params (e.g. control_img_in with new dim)
    for key in m:
        parts = key.split('.')
        mod = transformer
        for p in parts[:-1]:
            mod = getattr(mod, p)
        param = getattr(mod, parts[-1])
        new_param = torch.empty_like(param, device=device, dtype=dtype)
        if new_param.dim() >= 2:
            nn.init.kaiming_uniform_(new_param)
        else:
            nn.init.zeros_(new_param)
        setattr(mod, parts[-1], nn.Parameter(new_param))
        print(f'  Initialized {key} {list(new_param.shape)} on {device}')

    del diffusers_sd, controlnet_sd, merged_sd
    clear_cache()

    # Move to device and freeze backbone
    transformer = transformer.to(device, dtype=dtype)
    clear_cache()

    for name, param in transformer.named_parameters():
        param.requires_grad = 'control' in name

    # Convert frozen layers to FP8 for VRAM savings
    _GiB = 1024 ** 3
    vram_before = torch.cuda.memory_allocated() / _GiB
    num_fp8 = convert_frozen_linears_to_fp8(transformer)
    clear_cache()
    vram_after = torch.cuda.memory_allocated() / _GiB
    print(f'  FP8 compression: {num_fp8} frozen Linears, '
          f'{vram_before:.1f} → {vram_after:.1f} GiB (saved {vram_before - vram_after:.1f} GiB)')

    return transformer


# ═══════════════════════════════════════════════════════════════════════════════
# Latent helpers (for target RGB processing)
# ═══════════════════════════════════════════════════════════════════════════════

def patchify_latents(latents):
    """(B, C, H, W) → (B, C*4, H/2, W/2) via 2×2 patch grouping."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(B, C * 4, H // 2, W // 2)


def pack_latents(latents):
    """(B, C, H, W) → (B, H*W, C)."""
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)


def prepare_latent_ids(latents, device='cuda'):
    """Position IDs for latent tokens. (B, H*W, 4)."""
    B, _, H, W = latents.shape
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(H),
                               torch.arange(W), torch.arange(1))
    return ids.unsqueeze(0).expand(B, -1, -1).to(device)


def prepare_text_ids(prompt_embeds, device='cuda'):
    """Position IDs for text tokens. (B, L, 4)."""
    B, L, _ = prompt_embeds.shape
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(1),
                               torch.arange(1), torch.arange(L))
    return ids.unsqueeze(0).expand(B, -1, -1).to(device)


def encode_rgb_to_latent(vae, rgb_tensor, bn_mean, bn_std, dtype=torch.bfloat16):
    """Encode RGB tensor [B, 3, H, W] in [0,1] → packed latent tokens.

    Returns:
        packed: [B, N, C] packed normalized latent tokens
        patchified: [B, C*4, H/2, W/2] patchified latent (for position IDs)
    """
    rgb_norm = rgb_tensor * 2.0 - 1.0  # [0,1] → [-1,1]
    with torch.no_grad():
        latent = vae.encode(rgb_norm)[0].mode()
        patchified = patchify_latents(latent)
        normalized = ((patchified - bn_mean) / bn_std).to(dtype)
        packed = pack_latents(normalized)
    return packed, patchified

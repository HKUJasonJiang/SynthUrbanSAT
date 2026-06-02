"""Sampling — HDC²A (ours) and vanilla baseline (Flux2 + Union ControlNet, 260-dim).

Both use Euler flow-matching with the same noise seeds so outputs are
seed-aligned for direct comparison.
"""

from __future__ import annotations

import math

import torch

from .state import DEVICE, DTYPE, STATE


# ─── Lazy import helpers (defer torch-heavy modules until first call) ────────

def _train_helpers():
    from scripts.utility import prepare_latent_ids, prepare_text_ids
    from scripts.overfit import _decode_packed_latent
    return prepare_latent_ids, prepare_text_ids, _decode_packed_latent


def _patchify_latents(latents):
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(B, C * 4, H // 2, W // 2)


def _pack_latents(latents):
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)


# ─── HDC²A (ours) ───────────────────────────────────────────────────────────

@torch.no_grad()
def hdc2a_control_context(seg_B: torch.Tensor, depth_B: torch.Tensor) -> torch.Tensor:
    """Run the HDC²A adapter; returns [B, N_ctrl, output_dim]."""
    ctx = STATE.hdc2a(seg_B, depth_B)
    return ctx.to(STATE.ours_transformer.dtype)


@torch.no_grad()
def sample_ours(seg_b: torch.Tensor, depth_b: torch.Tensor,
                prompt_embed_B: torch.Tensor, *,
                num_steps: int, guidance_scale: float,
                seeds: list[int], progress=None):
    """Run our HDC²A + LoRA pipeline. Returns (rgb [B,3,H,W] on CPU, ctrl_ctx tensor)."""
    prepare_latent_ids, prepare_text_ids, decode = _train_helpers()
    img_size = STATE.image_size
    B = len(seeds)
    seg_B   = seg_b.expand(B, -1, -1).contiguous().to(DEVICE)
    depth_B = depth_b.expand(B, -1, -1, -1).contiguous().to(DEVICE, DTYPE)

    text_ids = prepare_text_ids(prompt_embed_B, DEVICE)
    H2 = W2 = img_size // 16
    N, C = H2 * W2, 128
    dummy = torch.zeros(B, C, H2, W2, device=DEVICE)
    latent_ids = prepare_latent_ids(dummy, DEVICE)
    guidance = torch.full((B,), float(guidance_scale), device=DEVICE, dtype=DTYPE)

    xs = []
    for s in seeds:
        g = torch.Generator(device=DEVICE).manual_seed(int(s))
        xs.append(torch.randn(N, C, device=DEVICE, dtype=DTYPE, generator=g))
    x = torch.stack(xs, dim=0)

    ctrl_ctx = hdc2a_control_context(seg_B, depth_B)

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=DEVICE)
    for i in range(num_steps):
        t_curr, t_next = timesteps[i], timesteps[i + 1]
        dt = t_next - t_curr
        t_batch = t_curr.expand(B).to(DTYPE)
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = STATE.ours_transformer(
                hidden_states         = x,
                encoder_hidden_states = prompt_embed_B,
                timestep              = t_batch,
                img_ids               = latent_ids,
                txt_ids               = text_ids,
                guidance              = guidance,
                control_context       = ctrl_ctx,
                return_dict           = False,
            )
        v_pred = out[0].to(DTYPE)
        x = x + dt * v_pred
        if progress is not None:
            progress((i + 1) / num_steps, desc=f'Ours sampling {i+1}/{num_steps}')

    rgb = decode(x.float(), STATE.bn_mean.float(), STATE.bn_std.float(), STATE.vae)
    return rgb.cpu(), ctrl_ctx


# ─── Vanilla baseline (Flux2 + Union ControlNet, 260-dim) ───────────────────

@torch.no_grad()
def _vanilla_control_context(modality_rgb: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Build the 260-dim ``control_context`` for the original Union ControlNet
    from a single colourised modality image (the original ControlNet operates
    on RGB control images: a seg colourised PNG or a depth-colourised PNG).

    ``modality_rgb``: [1,3,H,W] in [0,1].
    Returns: [B, N, 260] on DEVICE with the transformer's dtype.
    """
    vae = STATE.vae
    norm = (modality_rgb * 2.0 - 1.0).to(DEVICE, DTYPE)
    latent_raw = vae.encode(norm)[0].mode()                         # [1, 32, h, w]
    latents = _patchify_latents(latent_raw)
    latents = ((latents - STATE.bn_mean) / STATE.bn_std).to(DTYPE)
    latents = _pack_latents(latents)                                # [1, N, 128]
    h, w = latent_raw.shape[2], latent_raw.shape[3]

    mask = torch.zeros(1, 1, h, w, device=DEVICE, dtype=DTYPE)
    mask = _pack_latents(_patchify_latents(mask))                   # [1, N, 4]
    inp = torch.zeros(1, 32, h, w, device=DEVICE, dtype=DTYPE)
    inp = _pack_latents(_patchify_latents(inp))                     # [1, N, 128]

    ctx = torch.cat([latents, mask, inp], dim=2)                    # [1, N, 260]
    ctx = ctx.expand(batch_size, -1, -1).contiguous()
    return ctx.to(STATE.baseline_transformer.dtype)


@torch.no_grad()
def sample_baseline(control_rgb: torch.Tensor, prompt_embed_B: torch.Tensor, *,
                    num_steps: int, guidance_scale: float, seeds: list[int],
                    progress=None, label: str = 'baseline'):
    """Run vanilla Flux2 + Union ControlNet with a single RGB control image.

    ``control_rgb``: [1,3,H,W] in [0,1] — either the seg colourised image or
    a viridis-colourised depth map.
    """
    assert STATE.baseline_transformer is not None, 'Call STATE.load_baseline() first'
    prepare_latent_ids, prepare_text_ids, decode = _train_helpers()
    img_size = STATE.image_size
    B = len(seeds)

    text_ids = prepare_text_ids(prompt_embed_B, DEVICE)
    H2 = W2 = img_size // 16
    N, C = H2 * W2, 128
    dummy = torch.zeros(B, C, H2, W2, device=DEVICE)
    latent_ids = prepare_latent_ids(dummy, DEVICE)
    guidance = torch.full((B,), float(guidance_scale), device=DEVICE, dtype=DTYPE)

    xs = []
    for s in seeds:
        g = torch.Generator(device=DEVICE).manual_seed(int(s))
        xs.append(torch.randn(N, C, device=DEVICE, dtype=DTYPE, generator=g))
    x = torch.stack(xs, dim=0)

    ctrl_ctx = _vanilla_control_context(control_rgb, B)

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=DEVICE)
    for i in range(num_steps):
        t_curr, t_next = timesteps[i], timesteps[i + 1]
        dt = t_next - t_curr
        t_batch = t_curr.expand(B).to(DTYPE)
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = STATE.baseline_transformer(
                hidden_states         = x,
                encoder_hidden_states = prompt_embed_B,
                timestep              = t_batch,
                img_ids               = latent_ids,
                txt_ids               = text_ids,
                guidance              = guidance,
                control_context       = ctrl_ctx,
                return_dict           = False,
            )
        v_pred = out[0].to(DTYPE)
        x = x + dt * v_pred
        if progress is not None:
            progress((i + 1) / num_steps, desc=f'{label} {i+1}/{num_steps}')

    rgb = decode(x.float(), STATE.bn_mean.float(), STATE.bn_std.float(), STATE.vae)
    return rgb.cpu()

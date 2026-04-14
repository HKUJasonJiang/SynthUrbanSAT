"""Training, validation, and testing loops for HDC²A + Flux2 ControlNet.

Functions:
  sample_timestep(B, device)   - Uniform [0,1] for flow matching
  flow_matching_forward(z, t)  - Noisy latent + velocity target
  train_one_epoch(...)         - One epoch; uses precomputed prompt_embeds if in batch
  validate(...)                - Validation loss; same interface as train_one_epoch
  save_checkpoint(...)         - Saves hdc2a.pt, control_params.pt, optimizer.pt, meta.pt
  load_checkpoint(...)         - Restores model + optimizer; returns last epoch
  test_forward_pass(...)       - Sanity check with random data before training starts

All functions take a `config` dict (from train_script.py CONFIG).
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from scripts.models import HDC2AAdapter, FP8FrozenLinear
from scripts.utility import (
    check_memory, clear_cache, encode_rgb_to_latent,
    patchify_latents, pack_latents, prepare_latent_ids, prepare_text_ids,
)
from scripts.colors import (
    bold_cyan, bold_green, bold_red, bold_yellow, bold_magenta,
    cyan, green, yellow, magenta, gray, bold,
)


def sample_timestep(batch_size, device):
    """Sample random timestep in [0, 1] for flow matching.
    Uses logit-normal distribution (biased toward t=0.5) for better gradient signal.
    Fallback to uniform if config['timestep_sampling'] is not set or is 'uniform'.
    """
    return torch.rand(batch_size, device=device)


def sample_timestep_logit_normal(batch_size, device, mean=0.0, std=1.0):
    """Logit-normal timestep sampling — concentrates samples near t=0.5.

    Used by Flux2 / SD3 training to focus gradient signal on the
    mid-noise region where learning is most informative.
    """
    u = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(u)
    return t


def flow_matching_forward(target_packed, t):
    """Create noisy latent and target for flow matching.

    Args:
        target_packed: [B, N, C] clean latent tokens
        t: [B] or scalar timestep in [0, 1]

    Returns:
        noisy_latent: [B, N, C]
        noise: [B, N, C]
        flow_target: [B, N, C]  (noise - target = velocity)
    """
    noise = torch.randn_like(target_packed)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    t_expanded = t.view(-1, 1, 1)  # [B, 1, 1]
    noisy_latent = (1.0 - t_expanded) * target_packed + t_expanded * noise
    flow_target = noise - target_packed
    return noisy_latent, noise, flow_target


def train_one_epoch(epoch, hdc2a, transformer, vae, bn_mean, bn_std,
                    train_loader, optimizer, scaler, config, scheduler=None,
                    step_vis_callback=None, step_vis_interval=10):
    """Run one training epoch.

    Args:
        epoch: current epoch number
        hdc2a: HDC2AAdapter model (trainable)
        transformer: Flux2ControlTransformer2DModel (control params trainable)
        vae: frozen VAE for RGB encoding
        bn_mean, bn_std: VAE batch norm stats
        train_loader: DataLoader
        optimizer: optimizer
        scaler: GradScaler (or None for bf16)
        config: dict with training hyperparameters

    Returns:
        (avg_loss, opt_steps): average training loss and number of optimizer steps taken
    """
    hdc2a.train()
    # If freeze_backbone is set, only HDC2A trains; transformer stays in eval
    freeze_backbone = config.get('freeze_controlnet_backbone', False)
    if freeze_backbone:
        transformer.eval()
    else:
        transformer.train()

    # Choose timestep sampling strategy
    use_logit_normal = config.get('logit_normal_timestep', True)

    device = config['device']
    dtype = config['dtype']
    grad_accum_steps = config.get('grad_accum_steps', 1)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    log_interval = config.get('log_interval', 10)

    total_loss = 0.0
    num_batches = 0
    opt_steps = 0
    optimizer.zero_grad()

    total_epochs = config.get('num_epochs', 1)
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    epoch_t0 = time.time()

    for step, batch in enumerate(train_loader):
        rgb = batch['rgb'].to(device, dtype=dtype)        # [B, 3, H, W]
        seg = batch['seg'].to(device)                      # [B, H, W] long
        depth = batch['depth'].to(device, dtype=dtype)     # [B, 1, H, W]
        B = rgb.shape[0]

        # Encode RGB → latent tokens
        target_packed, patchified = encode_rgb_to_latent(vae, rgb, bn_mean, bn_std, dtype)
        latent_ids = prepare_latent_ids(patchified, device)

        # Sample timestep (logit-normal biases toward t=0.5 for better signal)
        if use_logit_normal:
            t = sample_timestep_logit_normal(B, device)
        else:
            t = sample_timestep(B, device)
        noisy_latent, noise, flow_target = flow_matching_forward(target_packed, t)

        # Text embedding: use precomputed if available, else random placeholder
        seq_len = config.get('text_seq_len', 512)
        text_dim = config.get('text_dim', 15360)
        if 'prompt_embeds' in batch:
            prompt_embeds = batch['prompt_embeds'].to(device, dtype=dtype)  # [B, 512, 15360]
        else:
            prompt_embeds = torch.randn(B, seq_len, text_dim, device=device, dtype=dtype)
        text_ids = prepare_text_ids(prompt_embeds, device)

        timestep = t.to(dtype)
        guidance = torch.full((B,), config.get('guidance_scale', 3.5),
                              device=device, dtype=dtype)

        # Forward pass
        with torch.amp.autocast('cuda', dtype=dtype):
            control_context = hdc2a(seg, depth)
            output = transformer(
                hidden_states=noisy_latent,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=guidance,
                control_context=control_context.to(transformer.dtype),
                return_dict=False,
            )
            noise_pred = output[0]
            # Min-SNR-style loss weighting for flow matching:
            # w(t) = 1 / (t*(1-t) + eps)  — up-weights extreme timesteps (t≈0/1)
            # to compensate for their inherently lower learning signal.
            # Normalised so mean(w)≈1, making effective gradient uniform across t.
            if config.get('minsnr_loss_weight', True):
                eps = 1e-3
                w = 1.0 / (t * (1.0 - t) + eps)          # [B]
                w = w / w.mean()                           # normalise
                w = w.view(-1, 1, 1).to(dtype)
                loss = (F.mse_loss(noise_pred, flow_target, reduction='none') * w
                        ).mean() / grad_accum_steps
            else:
                loss = F.mse_loss(noise_pred, flow_target) / grad_accum_steps

        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            # Clip gradients
            all_params = list(hdc2a.parameters()) + [
                p for p in transformer.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

            optimizer.step()
            opt_steps += 1
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        batch_loss = loss.item() * grad_accum_steps
        total_loss += batch_loss
        num_batches += 1

        # Logging
        global_step = epoch * steps_per_epoch + step
        if (step + 1) % log_interval == 0:
            avg = total_loss / num_batches
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            progress_pct = (global_step + 1) / total_steps * 100
            elapsed = time.time() - epoch_t0
            step_rate = (step + 1) / max(elapsed, 0.01)
            eta_epoch = (steps_per_epoch - step - 1) / max(step_rate, 0.001)
            pct_str = bold_cyan(f'{progress_pct:.1f}%')
            print(f'  {gray(f"[Epoch {epoch}][{step+1}/{steps_per_epoch}]")} '
                  f'loss={yellow(f"{batch_loss:.6f}")} avg={avg:.6f} '
                  f'VRAM={vram:.1f}GiB | '
                  f'{pct_str} done | '
                  f'ETA(epoch): {eta_epoch:.0f}s')
            wandb.log({
                "step_loss": batch_loss,
                "gpu_mem_gib": vram,
                "global_step": global_step,
            })

        # === OVERFIT TEST === 逐步可视化：每 step_vis_interval 步保存一张对比图
        if step_vis_callback is not None and global_step % step_vis_interval == 0:
            step_vis_callback(global_step, batch)
            # 恢复训练模式（callback 内部会切换 eval，这里统一恢复）
            hdc2a.train()
            freeze_backbone = config.get('freeze_controlnet_backbone', False)
            if not freeze_backbone:
                transformer.train()
        # === END OVERFIT TEST ===

    # Flush remaining accumulated gradients (when steps not divisible by grad_accum)
    if num_batches > 0 and num_batches % grad_accum_steps != 0:
        all_params = list(hdc2a.parameters()) + [
            p for p in transformer.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()
        opt_steps += 1
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, opt_steps


@torch.no_grad()
def validate(epoch, hdc2a, transformer, vae, bn_mean, bn_std,
             val_loader, config):
    """Run validation.

    Returns:
        avg_loss: average validation loss
    """
    hdc2a.eval()
    transformer.eval()

    device = config['device']
    dtype = config['dtype']

    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        rgb = batch['rgb'].to(device, dtype=dtype)
        seg = batch['seg'].to(device)
        depth = batch['depth'].to(device, dtype=dtype)
        B = rgb.shape[0]

        target_packed, patchified = encode_rgb_to_latent(vae, rgb, bn_mean, bn_std, dtype)
        latent_ids = prepare_latent_ids(patchified, device)

        t = torch.full((B,), 0.5, device=device)  # fixed t=0.5 for val consistency
        noisy_latent, noise, flow_target = flow_matching_forward(target_packed, t)

        seq_len = config.get('text_seq_len', 512)
        text_dim = config.get('text_dim', 15360)
        if 'prompt_embeds' in batch:
            prompt_embeds = batch['prompt_embeds'].to(device, dtype=dtype)
        else:
            prompt_embeds = torch.randn(B, seq_len, text_dim, device=device, dtype=dtype)
        text_ids = prepare_text_ids(prompt_embeds, device)

        timestep = t.to(dtype)
        guidance = torch.full((B,), config.get('guidance_scale', 3.5),
                              device=device, dtype=dtype)

        with torch.amp.autocast('cuda', dtype=dtype):
            control_context = hdc2a(seg, depth)
            output = transformer(
                hidden_states=noisy_latent,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=guidance,
                control_context=control_context.to(transformer.dtype),
                return_dict=False,
            )
            noise_pred = output[0]
            loss = F.mse_loss(noise_pred, flow_target)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(epoch, hdc2a, transformer, optimizer, loss, output_dir, config,
                    keep_last_n=3, best_loss=None, best_ckpt_path=None,
                    scheduler=None, global_step=None):
    """Save checkpoint with HDC2A + control params + optimizer + scheduler state.

    Keeps only the last `keep_last_n` regular checkpoints to prevent disk full.
    Separately tracks the single best-val-loss checkpoint.

    Args:
        keep_last_n: how many most-recent checkpoints to retain
        best_loss: best val loss seen so far (float or None)
        best_ckpt_path: path to the current best checkpoint (str or None)
        scheduler: LR scheduler (state saved for correct resume)
        global_step: current optimizer step count

    Returns:
        (ckpt_dir, is_best, new_best_loss, new_best_ckpt_path)
    """
    import shutil

    ckpt_dir = os.path.join(output_dir, f'checkpoint_epoch_{epoch:04d}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Check free disk space before writing (warn if < 50 GB)
    import shutil as _shutil
    stat = _shutil.disk_usage(output_dir)
    free_gb = stat.free / 1e9
    if free_gb < 50:
        print(f'  {bold_red("WARNING:")} Only {free_gb:.1f} GB free on disk. '
              f'Skipping optimizer state to save space.')
        save_optimizer = False
    else:
        save_optimizer = True

    # Save HDC2A adapter
    torch.save(hdc2a.state_dict(), os.path.join(ckpt_dir, 'hdc2a.pt'))

    # Save only trainable transformer params (control blocks)
    # Note: use named_parameters to get reliable requires_grad status;
    # state_dict() values may not preserve requires_grad correctly.
    _trainable_keys = {n for n, p in transformer.named_parameters() if p.requires_grad}
    ctrl_sd = {k: v for k, v in transformer.state_dict().items()
               if k in _trainable_keys or 'control' in k}
    torch.save(ctrl_sd, os.path.join(ckpt_dir, 'control_params.pt'))

    # Save optimizer (large — skip if low disk)
    if save_optimizer:
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, 'optimizer.pt'))

    # Save scheduler state for correct resume
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, 'scheduler.pt'))

    # Save metadata
    is_best = (best_loss is None) or (loss < best_loss)
    meta = {
        'epoch': epoch,
        'loss': loss,
        'is_best': is_best,
        'best_val_loss': best_loss if not is_best else loss,
        'global_step': global_step,
        'config': {k: str(v) for k, v in config.items()},
    }
    torch.save(meta, os.path.join(ckpt_dir, 'meta.pt'))
    print(f'  Checkpoint saved: {ckpt_dir}  {bold_green("(BEST)") if is_best else ""}')

    # Update best checkpoint tracking
    new_best_loss = loss if is_best else best_loss
    new_best_ckpt_path = ckpt_dir if is_best else best_ckpt_path

    # Rotate: delete old checkpoints, preserve best and last N
    all_ckpts = sorted([
        d for d in os.listdir(output_dir)
        if d.startswith('checkpoint_epoch_')
        and os.path.isdir(os.path.join(output_dir, d))
    ])
    to_keep = set(all_ckpts[-keep_last_n:])  # always keep latest N
    if new_best_ckpt_path:
        to_keep.add(os.path.basename(new_best_ckpt_path))  # always keep best

    for old_ckpt in all_ckpts:
        if old_ckpt not in to_keep:
            old_path = os.path.join(output_dir, old_ckpt)
            shutil.rmtree(old_path)
            print(f'  Deleted old checkpoint: {old_ckpt}')

    return ckpt_dir, is_best, new_best_loss, new_best_ckpt_path


def load_checkpoint(ckpt_dir, hdc2a, transformer, optimizer=None, device='cuda',
                    scheduler=None):
    """Load checkpoint.

    Returns:
        dict with keys: epoch, best_val_loss, global_step
    """
    hdc2a.load_state_dict(
        torch.load(os.path.join(ckpt_dir, 'hdc2a.pt'), map_location=device,
                   weights_only=True))

    ctrl_sd = torch.load(os.path.join(ckpt_dir, 'control_params.pt'), map_location=device,
                         weights_only=True)
    m, u = transformer.load_state_dict(ctrl_sd, strict=False)
    if u:
        print(f'  Warning: {len(u)} unexpected keys in control_params.pt (ignored): {u[:3]}...')
    print(f'  Loaded control_params.pt: {len(ctrl_sd)} keys, {len(m)} missing from ckpt')

    if optimizer is not None:
        opt_path = os.path.join(ckpt_dir, 'optimizer.pt')
        if os.path.exists(opt_path):
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location=device,
                                                          weights_only=True))
            except Exception as e:
                print(f'  Warning: optimizer.pt is corrupt or incompatible ({e})')
                print(f'  Optimizer state not restored — starting optimizer from scratch')
        else:
            print(f'  optimizer.pt not found in {ckpt_dir} — optimizer state not restored')

    # Restore scheduler state (critical for correct LR on resume)
    sched_path = os.path.join(ckpt_dir, 'scheduler.pt')
    if scheduler is not None and os.path.exists(sched_path):
        try:
            scheduler.load_state_dict(torch.load(sched_path, map_location='cpu',
                                                      weights_only=True))
            print(f'  Restored LR scheduler state from scheduler.pt')
        except Exception as e:
            print(f'  Warning: scheduler.pt incompatible ({e}) — scheduler starts fresh')
    elif scheduler is not None:
        print(f'  scheduler.pt not found — scheduler starts fresh (warmup re-triggers!)')

    meta_path = os.path.join(ckpt_dir, 'meta.pt')
    result = {'epoch': 0, 'best_val_loss': float('inf'), 'global_step': 0}
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location='cpu', weights_only=True)
        result['epoch'] = meta['epoch']
        result['best_val_loss'] = meta.get('best_val_loss', float('inf'))
        result['global_step'] = meta.get('global_step', 0)
        loss = meta.get('loss', float('nan'))
        print(f'  Loaded checkpoint from epoch {result["epoch"]}, loss={loss:.6f}')
        print(f'  best_val_loss={result["best_val_loss"]:.6f}, global_step={result["global_step"]}')
    else:
        # Old-format checkpoint: parse epoch from directory name
        import re as _re
        m = _re.search(r'checkpoint_epoch_(\d+)', os.path.basename(ckpt_dir))
        result['epoch'] = int(m.group(1)) if m else 0
        print(f'  meta.pt not found — inferred epoch {result["epoch"]} from directory name')
    return result


def test_forward_pass(hdc2a, transformer, vae, bn_mean, bn_std, config):
    """Quick forward + backward test to verify the pipeline works.

    Uses random data — no dataset needed.
    Tests: (1) forward pass, (2) loss computation, (3) backward pass, (4) gradient flow.
    """
    device = config['device']
    dtype = config['dtype']
    img_size = config['image_size']

    # --- 1. Forward pass (eval mode) ---
    print(f'  {bold_cyan("[test 1/4]")} Forward pass (eval mode)...')
    rgb = torch.rand(1, 3, img_size, img_size, device=device, dtype=dtype)
    seg = torch.randint(0, config.get('num_classes', 5),
                        (1, img_size, img_size), device=device)
    depth = torch.rand(1, 1, img_size, img_size, device=device, dtype=dtype)

    target_packed, patchified = encode_rgb_to_latent(vae, rgb, bn_mean, bn_std, dtype)
    latent_ids = prepare_latent_ids(patchified, device)

    t = torch.tensor([0.5], device=device)
    noisy_latent, noise, flow_target = flow_matching_forward(target_packed, t)

    seq_len = config.get('text_seq_len', 512)
    text_dim = config.get('text_dim', 15360)
    prompt_embeds = torch.randn(1, seq_len, text_dim, device=device, dtype=dtype)
    text_ids = prepare_text_ids(prompt_embeds, device)
    guidance = torch.tensor([3.5], device=device, dtype=dtype)

    hdc2a.eval()
    transformer.eval()

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        control_context = hdc2a(seg, depth)
        output = transformer(
            hidden_states=noisy_latent,
            encoder_hidden_states=prompt_embeds,
            timestep=t.to(dtype),
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=guidance,
            control_context=control_context.to(transformer.dtype),
            return_dict=False,
        )
    noise_pred = output[0]
    print(f'    Output shape: {noise_pred.shape}')
    print(f'    Output stats: mean={noise_pred.mean():.4f}, std={noise_pred.std():.4f}')
    vram_forward = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f'    VRAM peak (forward): {cyan(f"{vram_forward:.2f} GiB")}')
    del noise_pred, output, control_context
    clear_cache()

    # --- 2. Loss computation ---
    print(f'  {bold_cyan("[test 2/4]")} Loss computation (train mode)...')
    hdc2a.train()
    transformer.train()

    with torch.amp.autocast('cuda', dtype=dtype):
        control_context = hdc2a(seg, depth)
        output = transformer(
            hidden_states=noisy_latent,
            encoder_hidden_states=prompt_embeds,
            timestep=t.to(dtype),
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=guidance,
            control_context=control_context.to(transformer.dtype),
            return_dict=False,
        )
        noise_pred = output[0]
        loss = F.mse_loss(noise_pred, flow_target)
    print(f'    Loss value: {yellow(f"{loss.item():.6f}")}')

    # --- 3. Backward pass ---
    print(f'  {bold_cyan("[test 3/4]")} Backward pass...')
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    vram_backward = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f'    Backward completed. VRAM peak (backward): {cyan(f"{vram_backward:.2f} GiB")}')

    # --- 4. Gradient check ---
    print(f'  {bold_cyan("[test 4/4]")} Gradient flow check...')
    hdc2a_params = [(n, p) for n, p in hdc2a.named_parameters() if p.requires_grad]
    hdc2a_with_grad = sum(1 for _, p in hdc2a_params if p.grad is not None and p.grad.abs().max() > 0)
    hdc2a_total = len(hdc2a_params)

    ctrl_params = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
    ctrl_with_grad = sum(1 for _, p in ctrl_params if p.grad is not None and p.grad.abs().max() > 0)
    ctrl_total = len(ctrl_params)

    print(f'    HDC²A:   {green(f"{hdc2a_with_grad}/{hdc2a_total}")} params have non-zero grad')
    print(f'    Control: {green(f"{ctrl_with_grad}/{ctrl_total}")} params have non-zero grad')

    if hdc2a_with_grad == 0:
        print(f'    {bold_red("⚠ WARNING: No gradients flowing to HDC²A!")}')
    freeze_backbone = config.get('freeze_controlnet_backbone', False)
    if ctrl_with_grad == 0 and not freeze_backbone:
        print(f'    {bold_red("⚠ WARNING: No gradients flowing to control blocks!")}')
    elif ctrl_with_grad == 0 and freeze_backbone:
        print(f'    {bold_cyan("Control blocks frozen (requires_grad=False) — expected.")}')

    # Print a few grad norms for diagnostics
    print(f'    {magenta("Top grad norms (HDC²A):")}')
    grad_norms = [(n, p.grad.norm().item()) for n, p in hdc2a_params
                  if p.grad is not None]
    grad_norms.sort(key=lambda x: -x[1])
    for name, norm in grad_norms[:5]:
        print(f'      {name}: {norm:.6f}')

    # Zero out grads so we start clean
    hdc2a.zero_grad()
    transformer.zero_grad()

    backbone_frozen_ok = (ctrl_with_grad == 0 and freeze_backbone)
    status = 'PASSED' if (hdc2a_with_grad > 0 and (ctrl_with_grad > 0 or backbone_frozen_ok)) else 'WARN'
    status_str = bold_green(f'Test result: {status}') if status == 'PASSED' else bold_red(f'Test result: {status}')
    print(f'  {status_str}')

    del noise_pred, output, control_context, noisy_latent, loss
    clear_cache()
    return status == 'PASSED'

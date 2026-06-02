"""Global model state — HDC²A ours and (optional) vanilla baseline.

Both pipelines share the same VAE + text encoder + tokenizer (loaded once).
They each hold their own `Flux2ControlTransformer2DModel`:

- `ours_transformer`: built with `control_in_dim = cfg['control_in_dim']`
  (typically 3072), then LoRA-injected and fine-tuned weights from
  `control_params.pt` loaded on top.
- `baseline_transformer` (lazy): built with `control_in_dim = 260`, raw
  ControlNet-Union weights from the safetensors. No HDC²A, no LoRA.

For the baseline, `control_context` is constructed at sampling time from a
single modality (seg-only or depth-only) using the original VideoX-Fun
`[control_latent(128) | mask_cond(4) | inpaint_latent(128)]` recipe.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import (
    BASE_DIR, COLOR_MAP_PATH, CONTROLNET_PATH, LORA_DIR,
    TEXT_ENCODER_PATH, TOKENIZER_DIR, TRANSFORMER_PATH, VAE_PATH,
    TRAIN_PIPELINE_ROOT, VIDEOX_FUN_ROOT, list_lora_checkpoints,
)


# Inject training scripts and VideoX-Fun on sys.path so we can reuse code
# without copy-pasting. The training repo already has battle-tested loaders.
for _p in (str(TRAIN_PIPELINE_ROOT), str(VIDEOX_FUN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Imported lazily inside .load() so importing this module is cheap.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_gb() -> float:
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def _coerce(v):
    if isinstance(v, str):
        if v == 'True':  return True
        if v == 'False': return False
        if v == 'None':  return None
        try:    return int(v)
        except Exception: pass
        try:    return float(v)
        except Exception: pass
    return v


class AppState:
    """Lazy-loaded model holder. Call .load(ckpt) before generating."""

    cfg: dict
    use_mlp_fusion: bool
    lora_rank: int
    num_classes: int
    image_size: int
    seg_palette: np.ndarray

    vae: Optional[torch.nn.Module] = None
    bn_mean: Optional[torch.Tensor] = None
    bn_std: Optional[torch.Tensor] = None
    text_encoder: Optional[torch.nn.Module] = None
    tokenizer: object = None

    ours_transformer: Optional[torch.nn.Module] = None
    hdc2a: Optional[torch.nn.Module] = None
    lora_modules: dict
    ckpt_dir: Optional[Path] = None

    baseline_transformer: Optional[torch.nn.Module] = None

    def __init__(self):
        self.cfg = {}
        self.lora_modules = {}

    # ─── HDC²A ours ──────────────────────────────────────────────────────

    def load(self, ckpt_dir: Path, persistent_text_encoder: bool = True):
        """Load (or reload) the HDC²A + LoRA transformer + VAE."""
        from scripts.utility import load_vae, load_transformer
        from scripts.overfit import apply_lora_to_control_blocks
        from scripts.models import HDC2AAdapter
        from scripts.text_encoder import load_text_encoder

        if self.ours_transformer is not None:
            self._unload_ours()

        ckpt = Path(ckpt_dir)
        self.ckpt_dir = ckpt
        print(f'\n=== Loading HDC²A checkpoint: {ckpt.name} ===')

        meta = torch.load(ckpt / 'meta.pt', map_location='cpu', weights_only=False)
        self.cfg = {k: _coerce(v) for k, v in meta.get('config', {}).items()}

        # Override training-time absolute weight paths with our self-contained ones.
        self.cfg['vae_path'] = str(VAE_PATH)
        self.cfg['transformer_path'] = str(TRANSFORMER_PATH)
        self.cfg['controlnet_path'] = str(CONTROLNET_PATH)
        self.cfg['text_encoder_path'] = str(TEXT_ENCODER_PATH)
        self.cfg['tokenizer_path'] = str(TOKENIZER_DIR)
        print(f'  epoch={meta.get("epoch")}  loss={meta.get("loss")}')

        hdc_keys = list(torch.load(ckpt / 'hdc2a.pt', map_location='cpu', weights_only=True).keys())
        self.use_mlp_fusion = any(k.startswith('mlp_fusion.') for k in hdc_keys)

        ctrl_peek = torch.load(ckpt / 'control_params.pt', map_location='cpu', weights_only=True)
        lora_A_shapes = [v.shape for k, v in ctrl_peek.items() if k.endswith('.lora_A')]
        assert lora_A_shapes, 'No lora_A keys in control_params.pt'
        self.lora_rank = int(lora_A_shapes[0][0])
        self.lora_alpha = float(self.lora_rank)
        del ctrl_peek

        self.num_classes = int(self.cfg['num_classes'])
        self.image_size = int(self.cfg['image_size'])
        print(f'  fusion={"MLP" if self.use_mlp_fusion else "DoubleStream"}  '
              f'lora_rank={self.lora_rank}  image_size={self.image_size}  '
              f'num_classes={self.num_classes}')

        # ── VAE (shared with baseline) ───────────────────────────────────
        if self.vae is None:
            print('[1/4] VAE')
            self.vae, self.bn_mean, self.bn_std = load_vae(
                self.cfg['vae_path'], device=DEVICE, dtype=DTYPE)

        # ── Ours transformer (HDC²A control_in_dim, with LoRA) ───────────
        print('[2/4] Ours transformer (+ LoRA)')
        self.ours_transformer = load_transformer(
            self.cfg['transformer_path'], self.cfg['controlnet_path'],
            int(self.cfg['control_in_dim']),
            device=DEVICE, dtype=DTYPE,
        ).eval()
        for p in self.ours_transformer.parameters():
            p.requires_grad_(False)

        self.lora_modules = apply_lora_to_control_blocks(
            self.ours_transformer, rank=self.lora_rank, alpha=self.lora_alpha)
        for m in self.lora_modules.values():
            m.to(DEVICE, DTYPE)
            for p in m.parameters():
                p.requires_grad_(False)

        ctrl_sd = torch.load(ckpt / 'control_params.pt', map_location=DEVICE, weights_only=True)
        missing, unexpected = self.ours_transformer.load_state_dict(ctrl_sd, strict=False)
        bad = [k for k in unexpected if '.lora_' in k]
        assert not bad, f'LoRA keys not loaded: {bad[:3]}'
        print(f'  loaded {len(ctrl_sd)} keys ({len(missing)} missing frozen / {len(unexpected)} unexpected)')
        del ctrl_sd
        _clear_cuda()

        # ── HDC²A adapter ────────────────────────────────────────────────
        print('[3/4] HDC²A adapter')
        self.hdc2a = HDC2AAdapter(
            num_classes        = self.num_classes,
            fusion_dim         = int(self.cfg['fusion_dim']),
            output_dim         = int(self.cfg['control_in_dim']),
            num_heads          = int(self.cfg['num_heads']),
            num_fusion_blocks  = int(self.cfg['num_fusion_blocks']),
            num_fourier_bands  = int(self.cfg['num_fourier_bands']),
            boundary_threshold = float(self.cfg['boundary_threshold']),
            image_size         = self.image_size,
            use_mlp_fusion     = self.use_mlp_fusion,
        ).to(DEVICE, DTYPE).eval()
        hdc_sd = torch.load(ckpt / 'hdc2a.pt', map_location=DEVICE, weights_only=True)
        self.hdc2a.load_state_dict(hdc_sd, strict=True)
        for p in self.hdc2a.parameters():
            p.requires_grad_(False)

        with open(COLOR_MAP_PATH) as f:
            cmap_json = json.load(f)
        self.seg_palette = np.array(
            [cmap_json[str(i)]['rgb'] for i in range(self.num_classes)], dtype=np.uint8)

        # ── Persistent text encoder ─────────────────────────────────────
        if persistent_text_encoder and self.text_encoder is None:
            print('[4/4] Mistral text encoder (persistent)')
            self.text_encoder, self.tokenizer = load_text_encoder(
                self.cfg['text_encoder_path'], device=DEVICE, dtype=DTYPE)
        elif not persistent_text_encoder:
            self.text_encoder, self.tokenizer = None, None
        print(f'=== Ours ready. VRAM={vram_gb():.1f} GiB ===\n')

    def _unload_ours(self):
        # Drop the ours transformer + adapter + LoRA wrappers to free ~38 GB.
        self.ours_transformer = None
        self.hdc2a = None
        self.lora_modules = {}
        _clear_cuda()

    # ─── LoRA hot-swap ───────────────────────────────────────────────────

    def lora_enable(self, enabled: bool):
        for key, lora in self.lora_modules.items():
            parts = key.split('.')
            parent = self.ours_transformer
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            leaf = parts[-1]
            target = lora if enabled else lora.linear
            if leaf.isdigit():
                parent[int(leaf)] = target
            else:
                setattr(parent, leaf, target)

    # ─── Vanilla baseline (lazy) ─────────────────────────────────────────

    def unload_text_encoder(self):
        """Drop the resident Mistral text encoder to free ~35 GB VRAM."""
        if self.text_encoder is None:
            return
        # Drop STATE refs *first* so the only remaining refs are the locals
        # we hand to `unload_text_encoder()` (which deletes them).
        te, tk = self.text_encoder, self.tokenizer
        self.text_encoder, self.tokenizer = None, None
        from scripts.text_encoder import unload_text_encoder as _u
        try:
            _u(te, tk)
        except Exception:
            pass
        del te, tk
        _clear_cuda()
        print(f'  VRAM after Mistral unload: {vram_gb():.1f} GiB')

    def load_baseline(self, drop_ours: bool = True):
        """Load the vanilla Flux2 + Union ControlNet (260-dim) transformer.

        Idempotent: returns immediately if already loaded.

        Because the FP8 dequant peaks at ~68 GiB before re-compression to
        ~38 GiB, and the ours transformer alone holds another ~38 GiB, the
        two cannot coexist on a 95 GiB H200. We therefore drop the ours
        transformer + Mistral text encoder first. The caller can reload
        ours later via :meth:`reload_ours_if_dropped`.
        """
        if self.baseline_transformer is not None:
            return
        from scripts.utility import load_transformer
        if self.text_encoder is not None:
            print('  (Unloading Mistral)')
            self.unload_text_encoder()
        if drop_ours and self.ours_transformer is not None:
            print('  (Unloading ours transformer + HDC²A to fit baseline FP8 dequant peak)')
            self._unload_ours()
        print('\n=== Loading vanilla baseline (Flux2 + Union ControlNet 260-dim) ===')
        self.baseline_transformer = load_transformer(
            str(TRANSFORMER_PATH), str(CONTROLNET_PATH),
            control_in_dim=260, device=DEVICE, dtype=DTYPE,
        ).eval()
        # ``load_transformer`` deliberately skips ``control_img_in.*`` (it's
        # built for fine-tuning where the projection is freshly trained). For
        # the true vanilla baseline we *do* want the original ControlNet
        # projection — copy those two tensors over from the safetensors.
        self._load_real_control_img_in(self.baseline_transformer)
        for p in self.baseline_transformer.parameters():
            p.requires_grad_(False)
        _clear_cuda()
        print(f'=== Baseline ready. VRAM={vram_gb():.1f} GiB ===\n')

    @staticmethod
    def _load_real_control_img_in(transformer):
        """Copy the real ``control_img_in.{weight,bias}`` from the ControlNet
        safetensors into *transformer* (replacing the Kaiming init).
        """
        from safetensors.torch import load_file
        sd = load_file(str(CONTROLNET_PATH))
        w_key = 'control_img_in.weight'
        b_key = 'control_img_in.bias'
        layer = transformer.control_img_in
        with torch.no_grad():
            if w_key in sd:
                w = sd[w_key].to(layer.weight.device, layer.weight.dtype)
                assert w.shape == layer.weight.shape, \
                    f'control_img_in.weight shape mismatch: {w.shape} vs {layer.weight.shape}'
                layer.weight.copy_(w)
            if b_key in sd and layer.bias is not None:
                b = sd[b_key].to(layer.bias.device, layer.bias.dtype)
                assert b.shape == layer.bias.shape
                layer.bias.copy_(b)
        del sd
        _clear_cuda()
        print('  Loaded real Union-ControlNet control_img_in.{weight,bias} (260-dim)')

    def unload_baseline(self):
        self.baseline_transformer = None
        _clear_cuda()

    def reload_ours_if_dropped(self, persistent_text_encoder: bool = True):
        """If the ours transformer was unloaded to make room for the baseline,
        rebuild it from the previously-loaded checkpoint."""
        if self.ours_transformer is not None:
            return
        if self.ckpt_dir is None:
            return
        # Free the baseline first — same reason as in load_baseline().
        if self.baseline_transformer is not None:
            print('  (Unloading baseline to make room for ours)')
            self.unload_baseline()
        print('  (Reloading ours from', self.ckpt_dir.name + ')')
        self.load(self.ckpt_dir, persistent_text_encoder=persistent_text_encoder)


STATE = AppState()

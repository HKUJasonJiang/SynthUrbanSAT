"""Gradio web app — HDC²A + Flux2 + ControlNet(LoRA) interactive inference.

Launch:
    cd /home/xg_wang_group/SynthUrbanSAT
    conda activate flux_train
    python notebooks/app.py
    # open http://<host>:7860

Pipeline
--------
1. Load VAE + Flux2 transformer (bf16, FP8 dequant) + HDC²A once at startup.
2. User uploads seg + depth + (optional) GT RGB, edits the JSON prompt,
   picks 4 seeds, toggles LoRA on/off and adapter on/off-proxy.
3. Click **"Encode prompt"** — loads Mistral (bf16), encodes the flattened
   prompt, unloads Mistral immediately. Result cached until prompt changes.
4. Click **"Generate"** — runs Euler flow-matching sampling for all 4 seeds
   in a single batch; optionally re-runs with LoRA hot-swapped out; optionally
   re-runs with the HDC²A adapter bypassed by a raw seg+depth patch-embed
   (ControlNet attention still receives the inputs, just unprocessed).

Adapter ablation caveat
-----------------------
Truly "removing" HDC²A would need a model retrained without it (or with a
different control head). As a proxy we replace its output with a deterministic
patch embedding of the raw one-hot seg + depth at the SAME token shape the
transformer expects. The backbone + LoRA still run with real seg/depth
information, just without the adapter's learned fusion. This visualises the
**adapter's contribution** on top of raw control signals. Clearly labelled
in the UI.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models import HDC2AAdapter
from scripts.utility import (
    load_vae, load_transformer, prepare_latent_ids, prepare_text_ids,
)
from scripts.overfit import (
    apply_lora_to_control_blocks, _decode_packed_latent, LoRALinear,
)
from scripts.text_encoder import (
    load_text_encoder, encode_prompts, unload_text_encoder,
)

import gradio as gr


# ═════════════════════════════════════════════════════════════════════════════
# Configuration — edit these if your checkpoint is elsewhere
# ═════════════════════════════════════════════════════════════════════════════
CKPT_DIR = PROJECT_ROOT / 'output/lora_rank_128_mlp_H200/checkpoint_epoch_0499'
COLOR_MAP_PATH = PROJECT_ROOT / 'configs/color_map.json'
DEFAULT_PROMPT_JSON = PROJECT_ROOT / 'dataset/prompt.json'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.bfloat16        # inference dtype for VAE/transformer/HDC²A
TEXT_DTYPE = torch.bfloat16    # Mistral dtype (bf16 is identical to fp16 for us)

# Set at __main__ from --no-textencoder CLI flag.
# True  -> Mistral stays resident (default; faster per-request).
# False -> Mistral is loaded+unloaded for each Encode call (old behaviour).
PERSISTENT_TEXT_ENCODER = True

# Filename resolution roots (parent folder must contain any of these sibling dirs).
SIBLING_DIRS = ('rgb', 'seg', 'depth')
SUPPORTED_EXTS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')

# ─── Prompt diversity axes ───────────────────────────────────────────────────
# Dropdown options for the "diversity" overlay on the prompt JSON. The literal
# string ``'none'`` is the *first* option on every axis — it means "do not
# constrain this axis, let the model pick freely" and is **NOT** emitted into
# the composed prompt. Keep each option short enough that the full diversity
# clause stays under ~25 tokens (safely inside the 512-token Mistral budget).
NONE_OPT = 'none'   # sentinel: drop this axis from the flat prompt
DIVERSITY_OPTIONS = {
    'season': [NONE_OPT, 'spring', 'summer', 'autumn', 'winter'],
    'time_of_day': [NONE_OPT, 'early morning', 'morning', 'midday',
                    'afternoon', 'golden hour', 'dusk'],
    'weather': [NONE_OPT, 'clear sky', 'partly cloudy', 'overcast',
                'light haze', 'post-rain wet ground'],
    # Region drives architectural style + ground colour / vegetation priors.
    # Kept to 6 visually distinct clusters; finer granularity would be
    # redundant given seg+depth already encode layout and density.
    'region': [NONE_OPT,
               'North American suburban',
               'North American downtown',
               'European historic city',
               'East Asian dense urban',
               'Arid / Middle-Eastern city',
               'Tropical city'],
    # Lighting mood — orthogonal to time_of_day (shadow HARDNESS rather than
    # sun elevation). Useful for matching/escaping the GT lighting vibe.
    'lighting_mood': [NONE_OPT, 'strong directional shadows', 'soft diffuse light', 'low-angle sunlight', 'flat cloudy light', 'crisp high-contrast light'],
    # Vegetation state — decouples foliage/colour from season (e.g. dry
    # summer, lush winter-tropical, leafless mild-autumn).
    'vegetation_state': [NONE_OPT, 'leafless bare trees', 'lush green vegetation', 'dry brown vegetation', 'autumn foliage', 'sparse vegetation'],
    # Snow is covered by the weather axis ('light snow cover', 'fresh snow')
    # so a separate snow_cover axis would be redundant.
}
DIVERSITY_KEYS = tuple(DIVERSITY_OPTIONS.keys())

# VLM captioner for the Dataset Expansion block. Default:
# mistralai/Mistral-Small-3.1-24B-Instruct-2503.
# Alternative overrides via ``HDC2A_CAPTIONER``:
#   • ``Qwen/Qwen2-VL-2B-Instruct``                 ~2B,  ~5 GB  (faster, decent)
#   • ``HuggingFaceTB/SmolVLM-Instruct``            ~2B,  ~5 GB  (very fast)
#   • ``microsoft/Florence-2-large``                0.8B, ~2 GB  (need task prompt)
#   • ``Salesforce/blip-image-captioning-large``    ~500MB fp16  (lightest fallback)
#   • ``mistralai/Mistral-Small-3.1-24B-Instruct-2503`` 24B, ~50 GB (best quality, needs multi-GPU)
CAPTIONER_MODEL_ID = os.environ.get(
    'HDC2A_CAPTIONER', 'mistralai/Mistral-Small-3.1-24B-Instruct-2503')
# Hard cap on the caption length inserted into the prompt so the final text
# stays well inside Mistral's 512-token window. ~220 chars ≈ 50 tokens, enough
# for one ~30-word sentence covering the whole-image RGB attributes (surface
# material, vegetation type, vehicle level, water, architectural character)
# that seg + depth controls cannot encode.
CAPTION_MAX_CHARS = 220

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

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


def compose_prompt_from_json(json_str):
    """Flatten a structured prompt JSON into a single comma-joined string.
    Mirrors ``compose_prompt_from_json`` in train_script.py.

    Also understands two *optional* extension fields added for UI-side
    diversification (ignored gracefully if absent, so training-time JSON stays
    compatible):

    - ``diversity``: dict with any of
      ``season``, ``time_of_day``, ``weather``, ``region``. Values equal to
      ``"(any)"`` or empty are skipped. Emitted as a compact clause like
      ``"summer season, midday, clear sky, North American suburb context"``.
    - ``reference_caption``: str — a short scene description (typically from a
      VLM run on a reference RGB). Truncated to ``CAPTION_MAX_CHARS`` and
      appended as ``"scene reference: ..."``.
    """
    parts = []
    if 'scene' in json_str:    parts.append(json_str['scene'])
    if 'style' in json_str:    parts.append(json_str['style'])
    if 'elements' in json_str:
        for k, v in json_str['elements'].items():
            parts.append(f'{k}: {v}')
    if 'lighting' in json_str: parts.append(json_str['lighting'])
    if 'quality' in json_str:  parts.append(json_str['quality'])

    # ---- Diversity clause -------------------------------------------------
    div = json_str.get('diversity') or {}
    if isinstance(div, dict):
        div_parts = []

        def _take(key: str) -> str:
            v = (div.get(key) or '').strip()
            # Treat legacy ``(any)`` the same as the new ``none`` sentinel so
            # older JSON written by previous UI versions still parses cleanly.
            if not v or v.lower() in ('none', '(any)'):
                return ''
            return v

        season = _take('season')
        if season:
            if ',' in season or 'season' in season.lower():
                div_parts.append(season)
            else:
                div_parts.append(f'{season} season')
        tod = _take('time_of_day')
        if tod:
            div_parts.append(tod)
        weather = _take('weather')
        if weather:
            div_parts.append(weather)
        lighting = _take('lighting_mood')
        if lighting:
            div_parts.append(lighting)

        veg = _take('vegetation_state')
        if veg:
            div_parts.append(veg)

        region = _take('region')
        if region:
            div_parts.append(region)

        if div_parts:
            parts.append(', '.join(div_parts))

    # ---- Reference caption (from VLM) -------------------------------------
    cap = (json_str.get('reference_caption') or '').strip()
    if cap:
        cap = cap.replace('\n', ' ').strip()
        if len(cap) > CAPTION_MAX_CHARS:
            cap = cap[:CAPTION_MAX_CHARS].rstrip() + '…'
        parts.append(f'scene reference: {cap}')

    return ', '.join(parts)


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vram_gb() -> float:
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def _normalize_user_path(s: str) -> str:
    """Normalize a user-typed path so it works cross-platform.

    - strips surrounding whitespace and matched quotes (single/double)
    - on POSIX, converts Windows-style backslashes to forward slashes so a
      path like ``C:\\Users\\foo`` typed by a Windows user still parses; the
      drive prefix will simply not exist on the server, producing a clean
      ``Not a directory`` error rather than a single literal filename.
    - leaves the raw string alone on Windows (``Path`` handles both seps).
    """
    if not s:
        return ''
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    if os.sep == '/' and '\\' in s:
        s = s.replace('\\', '/')
    return s


# ═════════════════════════════════════════════════════════════════════════════
# Sibling file resolver
#
# Given an uploaded file inside a folder named ``rgb``, ``seg`` or ``depth``,
# look for files with the same stem in the two sibling folders (any supported
# extension). Used by the UI to auto-fill the other two uploads so the user
# only has to pick one file.
# ═════════════════════════════════════════════════════════════════════════════

def _find_sibling(src_path: str, target_dir: str) -> Optional[str]:
    """Return the path of a file with the same stem as *src_path* inside
    ``<src parent's parent>/<target_dir>/``. Returns None if not found."""
    if not src_path:
        return None
    p = Path(src_path)
    if not p.exists():
        return None
    stem = p.stem
    # sibling folder = p.parent.parent / target_dir
    parent_root = p.parent.parent
    target = parent_root / target_dir
    if not target.is_dir():
        return None
    # Preserve the same extension if it exists, else try all supported.
    preferred = [p.suffix.lower()] + [e for e in SUPPORTED_EXTS if e != p.suffix.lower()]
    for ext in preferred:
        cand = target / f'{stem}{ext}'
        if cand.exists():
            return str(cand)
        # also try upper-case variant (just in case)
        cand_u = target / f'{stem}{ext.upper()}'
        if cand_u.exists():
            return str(cand_u)
    return None


def resolve_siblings(src_path: str, src_kind: str) -> dict:
    """Given an uploaded file and its kind ('seg'|'depth'|'rgb'), return a dict
    with resolved paths for the other two kinds (or None if not found)."""
    out = {'seg': None, 'depth': None, 'rgb': None}
    out[src_kind] = src_path
    if not src_path:
        return out
    # If the uploaded file isn't sitting inside a folder named like its kind,
    # we can't resolve; just return what we have.
    p = Path(src_path)
    if p.parent.name.lower() != src_kind:
        return out
    for other in SIBLING_DIRS:
        if other == src_kind:
            continue
        out[other] = _find_sibling(src_path, other)
    return out


def scan_root_folder(root: str) -> dict:
    """Scan ``<root>/{seg,depth,rgb}/`` and return a dict of common file stems
    that exist in all three (or missing reported).

    Returns
    -------
    dict with:
      - 'stems':    list of stems present in all three dirs (sorted)
      - 'by_stem':  {stem: {'seg': path, 'depth': path, 'rgb': path}}
      - 'status':   human-readable message
    """
    out = {'stems': [], 'by_stem': {}, 'status': ''}
    root = _normalize_user_path(root or '')
    if not root:
        out['status'] = '_(enter a folder path and click Scan)_'
        return out
    root_p = Path(root).expanduser()
    if not root_p.is_dir():
        hint = ''
        if os.sep == '/' and (len(root) >= 2 and root[1] == ':'):
            hint = (' (looks like a Windows path — this server is Linux; '
                    'the folder must exist on the server filesystem)')
        out['status'] = f'❌ Not a directory: `{root_p}`{hint}'
        return out

    # Build {kind: {stem: path}}
    by_kind = {}
    missing_dirs = []
    for kind in SIBLING_DIRS:
        d = root_p / kind
        if not d.is_dir():
            missing_dirs.append(kind)
            by_kind[kind] = {}
            continue
        entries = {}
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                # Prefer .png > .tif > .tiff > .jpg if duplicate stems
                if f.stem not in entries:
                    entries[f.stem] = str(f)
        by_kind[kind] = entries

    if missing_dirs:
        out['status'] = (f'❌ Missing sub-folder(s) under `{root_p}`: '
                         f'{", ".join(missing_dirs)} '
                         f'(expected all of {", ".join(SIBLING_DIRS)})')
        return out

    # Stems common to all three
    common = set(by_kind['seg']) & set(by_kind['depth']) & set(by_kind['rgb'])
    stems = sorted(common)
    out['stems'] = stems
    for s in stems:
        out['by_stem'][s] = {k: by_kind[k][s] for k in SIBLING_DIRS}
    if not stems:
        out['status'] = (f'⚠️ Scanned `{root_p}` but found no common stems '
                         f'across all three sub-folders.')
    else:
        out['status'] = (f'✅ Scanned `{root_p}`: found **{len(stems)}** common stems. '
                         f'Pick one from the dropdown.')
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Model state — load once, keep resident
# ═════════════════════════════════════════════════════════════════════════════

def list_checkpoints() -> list[str]:
    """Return all `output/<run>/checkpoint_epoch_*` dirs (sorted, most-recent first)."""
    root = PROJECT_ROOT / 'output'
    if not root.is_dir():
        return []
    ckpts = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        for c in sorted(run_dir.glob('checkpoint_epoch_*')):
            if c.is_dir() and (c / 'meta.pt').is_file():
                ckpts.append(str(c.relative_to(PROJECT_ROOT)))
    # Most-recent-looking last → reverse for newest-first
    return list(reversed(ckpts))


class AppState:
    cfg: dict                # recovered training config
    use_mlp_fusion: bool
    lora_rank: int
    lora_alpha: float
    num_classes: int
    image_size: int
    vae: torch.nn.Module
    bn_mean: torch.Tensor
    bn_std: torch.Tensor
    transformer: torch.nn.Module
    hdc2a: torch.nn.Module
    lora_modules: dict
    seg_palette: np.ndarray
    text_encoder: Optional[torch.nn.Module] = None
    tokenizer: object = None
    ckpt_dir: Optional[Path] = None

    def _unload_models(self):
        """Drop references to heavy modules and free GPU memory before reload."""
        for name in ('transformer', 'hdc2a', 'vae', 'lora_modules',
                     'bn_mean', 'bn_std', 'seg_palette'):
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:
                    setattr(self, name, None)
        _clear_cuda()

    def load(self, ckpt_dir: Optional[Path] = None):
        # If reloading, free old models first; keep Mistral resident if already loaded.
        if getattr(self, 'transformer', None) is not None:
            self._unload_models()
        ckpt = Path(ckpt_dir) if ckpt_dir is not None else CKPT_DIR
        self.ckpt_dir = ckpt
        print(f'\n=== Loading checkpoint: {ckpt} ===')
        meta = torch.load(ckpt / 'meta.pt', map_location='cpu', weights_only=False)
        self.cfg = {k: _coerce(v) for k, v in meta.get('config', {}).items()}
        print(f'  epoch={meta.get("epoch")}  loss={meta.get("loss")}')

        # Auto-detect fusion mode from HDC²A keys
        hdc_keys = list(torch.load(ckpt / 'hdc2a.pt', map_location='cpu', weights_only=True).keys())
        self.use_mlp_fusion = any(k.startswith('mlp_fusion.') for k in hdc_keys)

        # Auto-detect LoRA rank from control_params
        ctrl = torch.load(ckpt / 'control_params.pt', map_location='cpu', weights_only=True)
        lora_A_shapes = [v.shape for k, v in ctrl.items() if k.endswith('.lora_A')]
        assert lora_A_shapes, 'No lora_A keys in control_params.pt'
        self.lora_rank = int(lora_A_shapes[0][0])
        self.lora_alpha = float(self.lora_rank)
        del ctrl

        self.num_classes = int(self.cfg['num_classes'])
        self.image_size = int(self.cfg['image_size'])
        print(f'  use_mlp_fusion={self.use_mlp_fusion}  lora_rank={self.lora_rank}  '
              f'image_size={self.image_size}  num_classes={self.num_classes}')

        # --- VAE ---------------------------------------------------------------
        print('[1/3] VAE')
        self.vae, self.bn_mean, self.bn_std = load_vae(self.cfg['vae_path'], device=DEVICE, dtype=DTYPE)

        # --- Transformer + LoRA ------------------------------------------------
        print('[2/3] Transformer (+ LoRA)')
        self.transformer = load_transformer(
            self.cfg['transformer_path'], self.cfg['controlnet_path'],
            int(self.cfg['control_in_dim']),
            device=DEVICE, dtype=DTYPE,
        ).eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.lora_modules = apply_lora_to_control_blocks(
            self.transformer, rank=self.lora_rank, alpha=self.lora_alpha)
        for m in self.lora_modules.values():
            m.to(DEVICE, DTYPE)
            for p in m.parameters():
                p.requires_grad_(False)

        ctrl_sd = torch.load(ckpt / 'control_params.pt', map_location=DEVICE, weights_only=True)
        missing, unexpected = self.transformer.load_state_dict(ctrl_sd, strict=False)
        lora_unexpected = [k for k in unexpected if '.lora_' in k]
        assert not lora_unexpected, f'LoRA keys not loaded: {lora_unexpected[:3]}'
        print(f'  control_params.pt: {len(ctrl_sd)} keys, '
              f'{len(missing)} missing (frozen backbone), {len(unexpected)} unexpected')
        del ctrl_sd
        _clear_cuda()

        # --- HDC²A adapter -----------------------------------------------------
        print('[3/3] HDC²A')
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

        # Seg palette
        with open(COLOR_MAP_PATH) as f:
            cmap_json = json.load(f)
        self.seg_palette = np.array(
            [cmap_json[str(i)]['rgb'] for i in range(self.num_classes)], dtype=np.uint8)

        # --- (Optional) persistent text encoder --------------------------------
        if PERSISTENT_TEXT_ENCODER and getattr(self, 'text_encoder', None) is None:
            print('[4/4] Mistral text encoder (persistent)')
            text_encoder_path = self.cfg['text_encoder_path']
            self.text_encoder, self.tokenizer = load_text_encoder(
                text_encoder_path, device=DEVICE, dtype=TEXT_DTYPE)
            print(f'  Mistral resident. VRAM={_vram_gb():.1f} GiB')
        elif not PERSISTENT_TEXT_ENCODER:
            self.text_encoder, self.tokenizer = None, None

        print(f'=== Models ready. VRAM={_vram_gb():.1f} GiB ===\n')

    # ── LoRA hot-swap ────────────────────────────────────────────────────────

    def lora_enable(self, enabled: bool):
        """Set whether LoRALinear wrappers are active in the transformer tree."""
        for key, lora in self.lora_modules.items():
            parts = key.split('.')
            parent = self.transformer
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            leaf = parts[-1]
            target = lora if enabled else lora.linear
            if leaf.isdigit():
                parent[int(leaf)] = target
            else:
                setattr(parent, leaf, target)


STATE = AppState()


# ═════════════════════════════════════════════════════════════════════════════
# Text encoding — load Mistral, encode, unload
# ═════════════════════════════════════════════════════════════════════════════

def encode_prompt_ui(prompt_json_text: str):
    """Return (prompt_embed_tensor_on_gpu, flat_prompt_str, status_message)."""
    try:
        prompt_obj = json.loads(prompt_json_text)
    except Exception as e:
        return None, '', f'❌ Invalid JSON: {e}'

    flat = compose_prompt_from_json(prompt_obj) if isinstance(prompt_obj, dict) else str(prompt_obj)
    if not flat.strip():
        return None, '', '❌ Empty prompt.'

    persistent = STATE.text_encoder is not None
    text_encoder_path = STATE.cfg['text_encoder_path']
    vram_before = _vram_gb()

    if persistent:
        text_encoder, tokenizer = STATE.text_encoder, STATE.tokenizer
        print(f'[Encode] Using resident Mistral. VRAM: {vram_before:.1f} GiB')
    else:
        print(f'\n[Encode] Loading Mistral text encoder ({text_encoder_path})...')
        try:
            text_encoder, tokenizer = load_text_encoder(text_encoder_path, device=DEVICE, dtype=TEXT_DTYPE)
        except Exception as e:
            _clear_cuda()
            return None, flat, f'❌ Text encoder load failed: {e}'
        print(f'[Encode] VRAM after load: {_vram_gb():.1f} GiB (was {vram_before:.1f})')

    try:
        embed = encode_prompts(
            text_encoder, tokenizer, [flat],
            max_sequence_length=int(STATE.cfg.get('text_seq_len', 512)),
            device=DEVICE, dtype=DTYPE,
        )   # [1, 512, 15360]
    except Exception as e:
        if not persistent:
            unload_text_encoder(text_encoder, tokenizer)
            _clear_cuda()
        return None, flat, f'❌ Encoding failed: {e}'

    embed = embed[0].detach().clone()   # [512, 15360]
    if not persistent:
        unload_text_encoder(text_encoder, tokenizer)
        _clear_cuda()
    msg = (f'✅ Encoded prompt ({len(flat)} chars, {embed.shape[0]} tokens). '
           f'VRAM now: {_vram_gb():.1f} GiB'
           f'{" (Mistral resident)" if persistent else ""}.\n'
           f'Flat: "{flat[:200]}{"..." if len(flat) > 200 else ""}"')
    print(f'[Encode] Done. VRAM: {_vram_gb():.1f} GiB')
    return embed, flat, msg


# ═════════════════════════════════════════════════════════════════════════════
# Robust file loading (tifffile for .tif/.tiff, PIL fallback for everything else)
# ═════════════════════════════════════════════════════════════════════════════

def _is_tiff(path: str) -> bool:
    return Path(path).suffix.lower() in ('.tif', '.tiff')


def _load_array_tiff(path: str) -> np.ndarray:
    """Read a TIFF as a raw numpy array using tifffile (no scaling, no mode conversion)."""
    import tifffile
    return tifffile.imread(path)


def _pil_open_preserve_mode(path: str) -> Image.Image:
    """Open image with PIL preserving the original mode (including palette 'P')."""
    img = Image.open(path)
    img.load()   # force decode so we have the data before the file is closed
    return img


# ─── Seg ─────────────────────────────────────────────────────────────────────

def preprocess_seg(path: str, size: int, num_classes: int) -> torch.Tensor:
    """Load a seg image from *path* → [H, W] long class-index tensor.

    Supported formats
    -----------------
    - Palette PNG (mode 'P'):  values are raw palette indices; class_id = clip(idx-1, 0, C-1).
    - Greyscale PNG/TIF:       single-channel uint8/uint16; values treated as class indices directly
                                (0-based; values > C-1 are clamped).
    - RGB PNG/TIF:             colour-to-class mapping via color_map.json.
    """
    ext = Path(path).suffix.lower()

    if not _is_tiff(path):
        # PIL for non-TIFF: preserves palette mode 'P' for PNG
        img = _pil_open_preserve_mode(path)
        if img.size != (size, size):
            img = img.resize((size, size), Image.NEAREST)
        if img.mode == 'P':
            arr = np.array(img, dtype=np.int64)
            arr = np.clip(arr - 1, 0, num_classes - 1)
            return torch.from_numpy(arr)
        # Convert to array for further processing
        raw = np.array(img.convert('RGB') if img.mode not in ('L', 'I', 'F') else img,
                       dtype=np.int64)
    else:
        # tifffile for TIFF: raw pixel values
        raw = _load_array_tiff(path)
        # Drop extra dimensions (e.g. extra channel or Z axis)
        while raw.ndim > 3:
            raw = raw[0]
        # Resize if needed
        if raw.shape[:2] != (size, size):
            img_tmp = Image.fromarray(
                raw.astype(np.uint8) if raw.dtype != np.uint8 else raw
            ).resize((size, size), Image.NEAREST)
            raw = np.array(img_tmp, dtype=np.int64)
        else:
            raw = raw.astype(np.int64)

    # At this point raw is [H, W] (greyscale → class indices) or [H, W, 3] (RGB colour map)
    if raw.ndim == 2:
        arr = np.clip(raw, 0, num_classes - 1)
        return torch.from_numpy(arr)

    # RGB → class via color_map.json
    with open(COLOR_MAP_PATH) as f:
        cmap = json.load(f)
    rgb_to_cls = {tuple(int(c) for c in v['rgb']): int(k) for k, v in cmap.items()}
    rgb = raw[..., :3].astype(np.uint8)
    arr = np.full(rgb.shape[:2], num_classes - 1, dtype=np.int64)
    for rgb_tuple, cid in rgb_to_cls.items():
        m = np.all(rgb == np.array(rgb_tuple, dtype=np.uint8), axis=-1)
        arr[m] = cid
    return torch.from_numpy(arr)   # [H, W] long


# ─── Depth ───────────────────────────────────────────────────────────────────

def preprocess_depth(path: str, size: int) -> torch.Tensor:
    """Load a depth image from *path* → [1, H, W] float tensor in [0, 1].

    tifffile is used for .tif/.tiff to preserve float32/float64 precision.
    PIL is used as a fallback for PNG or other formats.
    """
    if _is_tiff(path):
        arr = _load_array_tiff(path).astype(np.float32)
    else:
        img = Image.open(path)
        img.load()
        arr = np.array(img, dtype=np.float32)

    # Collapse extra dims: [D, H, W, C] → [H, W]
    while arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] <= arr.shape[0] else arr[0]

    if arr.shape != (size, size):
        arr_pil = Image.fromarray(arr).resize((size, size), Image.LANCZOS)
        arr = np.array(arr_pil, dtype=np.float32)

    mn, mx = float(arr.min()), float(arr.max())
    arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W] float


# ─── RGB ─────────────────────────────────────────────────────────────────────

def preprocess_rgb(path: str, size: int) -> torch.Tensor:
    """Load an RGB image from *path* → [3, H, W] float tensor in [0, 1].

    Handles uint8, uint16 (satellite imagery), and float TIFFs correctly.
    """
    if _is_tiff(path):
        arr = _load_array_tiff(path)
        # Drop extra dims
        while arr.ndim > 3:
            arr = arr[0]
        # Ensure [H, W, 3]
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        # Normalise to [0, 1] based on dtype
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        elif np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            arr = (arr - mn) / (mx - mn + 1e-8)
        else:
            arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0

    # Resize
    if arr.shape[:2] != (size, size):
        pil_tmp = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        arr = np.array(pil_tmp.resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0

    return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]


def seg_to_rgb(seg_tensor: torch.Tensor) -> np.ndarray:
    s = seg_tensor.cpu().numpy()
    palette = getattr(STATE, 'seg_palette', None)
    if palette is None:
        # Fallback: build palette directly from color map (STATE not yet fully loaded)
        with open(COLOR_MAP_PATH) as _f:
            _cmap = json.load(_f)
        nc = getattr(STATE, 'num_classes', 6)
        palette = np.array([_cmap[str(i)]['rgb'] for i in range(nc)], dtype=np.uint8)
    return palette[s]


def depth_to_rgb(depth_tensor: torch.Tensor) -> np.ndarray:
    import matplotlib.cm as cm
    d = depth_tensor[0].cpu().numpy()
    return (cm.viridis(d)[..., :3] * 255).astype(np.uint8)


def make_preview(path, kind: str) -> Optional[np.ndarray]:
    """Render a preview thumbnail for a seg/depth/rgb file. Returns uint8 HxWx3 or None.

    Handles both plain string paths and Gradio-6-style dicts
    ``{"path": ..., "orig_name": ..., "mime_type": ...}`` that are emitted for
    image/png files by gr.File's .change() event."""
    # Unwrap Gradio 6 dict format (PNG files get special treatment by Gradio)
    if isinstance(path, dict):
        path = path.get('path') or path.get('name') or path.get('tmp_path')
    if not path:
        return None
    try:
        size = STATE.image_size
        if kind == 'seg':
            lbl = preprocess_seg(str(path), size, STATE.num_classes)   # [H, W] long
            return seg_to_rgb(lbl)
        if kind == 'depth':
            d = preprocess_depth(str(path), size)                       # [1, H, W] float
            return depth_to_rgb(d)
        # rgb
        r = preprocess_rgb(str(path), size)                             # [3, H, W] in [0,1]
        return (r.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    except Exception as e:
        import traceback
        msg = f'[preview:{kind}] path={path!r} err={e}\n{traceback.format_exc()}'
        print(msg, flush=True)
        # Also write to a temp log so we can inspect from shell
        try:
            Path('/tmp/preview_debug.log').write_text(msg)
        except Exception:
            pass
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Sampler
# ═════════════════════════════════════════════════════════════════════════════

def _raw_patchify_control_context(seg_B: torch.Tensor, depth_B: torch.Tensor,
                                   num_classes: int, output_dim: int,
                                   patch: int = 16) -> torch.Tensor:
    """Deterministic patch embedding of **raw** seg+depth with NO learned
    HDC²A weights. Produces a tensor of the same shape HDC²A would output,
    ``[B, (H/patch)*(W/patch), output_dim]``, so the transformer's control
    attention blocks receive a legal control_context carrying the raw
    seg/depth information — just not refined by the adapter.

    Construction:
      1. one-hot seg  -> [B, num_classes, H, W]
      2. concat depth -> [B, num_classes+1, H, W]
      3. unfold into non-overlapping ``patch×patch`` tokens
         -> [B, N, (num_classes+1)*patch*patch]
      4. right-pad (or truncate) last dim to ``output_dim``

    Used only as the "adapter-OFF" ablation: shows what the ControlNet
    attention layers (still LoRA-tuned) can do with unprocessed inputs.
    """
    B, H, W = seg_B.shape
    assert H % patch == 0 and W % patch == 0, f'image size {H}×{W} not divisible by patch {patch}'
    seg_oh = F.one_hot(seg_B.long().clamp(0, num_classes - 1), num_classes)   # [B,H,W,C]
    seg_oh = seg_oh.permute(0, 3, 1, 2).to(depth_B.dtype)                      # [B,C,H,W]
    x = torch.cat([seg_oh, depth_B], dim=1)                                    # [B,C+1,H,W]
    Cch = x.shape[1]
    x = x.unfold(2, patch, patch).unfold(3, patch, patch)                      # [B,C,H/p,W/p,p,p]
    x = x.contiguous().permute(0, 2, 3, 1, 4, 5)                               # [B,H/p,W/p,C,p,p]
    x = x.reshape(B, -1, Cch * patch * patch)                                  # [B,N,raw_dim]
    raw_dim = x.shape[-1]
    if raw_dim < output_dim:
        pad = torch.zeros(B, x.shape[1], output_dim - raw_dim,
                          device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=-1)
    elif raw_dim > output_dim:
        x = x[..., :output_dim]
    return x


@torch.no_grad()
def sample_batch(seg_b, depth_b, prompt_embed_B, *, num_steps: int,
                 guidance_scale: float, seeds: list, bypass_adapter: bool):
    """Run Euler flow-matching for ``len(seeds)`` samples in one batch.

    seg_b:   [1, H, W] long  -> expanded to [B, H, W]
    depth_b: [1, 1, H, W] -> expanded to [B, 1, H, W]
    prompt_embed_B: [B, L, 15360]  (already broadcasted to batch size B)
    seeds:   list of ints, length B
    bypass_adapter: if True, skip HDC²A and feed a deterministic patch
        embedding of raw (seg, depth) instead — the ControlNet attention
        blocks still receive seg/depth information, just unprocessed.

    Returns: decoded RGB tensor [B, 3, H_img, W_img] in [0, 1], on CPU.
    """
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

    # Per-seed noise, stacked into one batch
    xs = []
    for s in seeds:
        g = torch.Generator(device=DEVICE).manual_seed(int(s))
        xs.append(torch.randn(N, C, device=DEVICE, dtype=DTYPE, generator=g))
    x = torch.stack(xs, dim=0)   # [B, N, C]

    ctrl_ctx = STATE.hdc2a(seg_B, depth_B).to(STATE.transformer.dtype)
    if bypass_adapter:
        # Replace HDC²A output with a deterministic patch embedding of raw
        # (seg, depth) at the SAME shape the transformer expects. No learned
        # adapter weights involved — pure raw signal → ControlNet attention.
        ctrl_ctx = _raw_patchify_control_context(
            seg_B, depth_b,
            num_classes=STATE.num_classes,
            output_dim=ctrl_ctx.shape[-1],
            patch=16,
        ).to(STATE.transformer.dtype)

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=DEVICE)

    for i in range(num_steps):
        t_curr, t_next = timesteps[i], timesteps[i + 1]
        dt = t_next - t_curr
        t_batch = t_curr.expand(B).to(DTYPE)
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = STATE.transformer(
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

    rgb = _decode_packed_latent(x.float(), STATE.bn_mean.float(), STATE.bn_std.float(), STATE.vae)
    return rgb.cpu()


@torch.no_grad()
def sample_batch_chained(seg_b, depth_b, prompt_embed_B, *, num_steps: int,
                         guidance_scale: float, seeds: list,
                         seg_strength: float, seg_start: float, seg_end: float,
                         depth_strength: float, depth_start: float, depth_end: float):
    """Simulate ComfyUI's two-ControlNet chain (seg + depth) with **NO LoRA
    and NO HDC²A adapter** — i.e. pure Flux2 + base ControlNet weights.

    Strategy
    --------
    - LoRA is hot-swapped OFF for the duration of this call (restored on exit).
    - Each branch's control context is a **deterministic raw patch embedding**
      of a single modality (seg-only with depth=0; depth-only with seg=0),
      built by ``_raw_patchify_control_context`` — no learned parameters.
    - At each Euler step:

          ctx(t) = w_seg(p)·raw_patch(seg,0) + w_depth(p)·raw_patch(0,depth)

      where ``p = 1 - t`` is denoising progress and each weight is the branch
      strength when ``p ∈ [start, end]`` else 0 — matching ComfyUI semantics.

    This is the closest apples-to-apples analogue of ComfyUI's two-ControlNet
    chain for this codebase: the trained ControlNet attention receives only
    unprocessed seg/depth, with no adapter fusion and no LoRA finetune.
    """
    img_size = STATE.image_size
    B = len(seeds)
    seg_B   = seg_b.expand(B, -1, -1).contiguous().to(DEVICE)
    depth_b = depth_b.expand(B, -1, -1, -1).contiguous().to(DEVICE, DTYPE)

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

    # Per-branch raw patch embeddings (seg-only, depth-only). We need the
    # adapter's output_dim just to know the target channel count — but we
    # don't invoke the adapter's forward pass, so no learned weights are used.
    out_dim = int(STATE.cfg['control_in_dim'])
    zero_seg   = torch.zeros_like(seg_B)
    zero_depth = torch.zeros_like(depth_b)
    ctx_seg = _raw_patchify_control_context(
        seg_B, zero_depth, num_classes=STATE.num_classes,
        output_dim=out_dim, patch=16,
    ).to(STATE.transformer.dtype)
    ctx_depth = _raw_patchify_control_context(
        zero_seg, depth_b, num_classes=STATE.num_classes,
        output_dim=out_dim, patch=16,
    ).to(STATE.transformer.dtype)

    # LoRA OFF for the whole chained run; restore on exit.
    STATE.lora_enable(False)
    try:
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=DEVICE)
        for i in range(num_steps):
            t_curr, t_next = timesteps[i], timesteps[i + 1]
            dt = t_next - t_curr
            t_batch = t_curr.expand(B).to(DTYPE)

            # Denoising progress in [0, 1] — matches ComfyUI's percent.
            p = 1.0 - float(t_curr.item())
            w_s = seg_strength   if (seg_start   <= p <= seg_end  ) else 0.0
            w_d = depth_strength if (depth_start <= p <= depth_end) else 0.0
            ctrl_ctx = w_s * ctx_seg + w_d * ctx_depth

            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = STATE.transformer(
                    hidden_states         = x,
                    encoder_hidden_states = prompt_embed_b,
                    timestep              = t_batch,
                    img_ids               = latent_ids,
                    txt_ids               = text_ids,
                    guidance              = guidance,
                    control_context       = ctrl_ctx,
                    return_dict           = False,
                )
            v_pred = out[0].to(DTYPE)
            x = x + dt * v_pred
    finally:
        STATE.lora_enable(True)

    rgb = _decode_packed_latent(x.float(), STATE.bn_mean.float(), STATE.bn_std.float(), STATE.vae)
    return rgb.cpu()


# ═════════════════════════════════════════════════════════════════════════════
# Grid builder
# ═════════════════════════════════════════════════════════════════════════════

def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def _get_font(size: int = 16):
    """Best-effort load of Times New Roman-like serif fonts; fallback to readable sans/default."""
    from PIL import ImageFont
    for path in (
        '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/times.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _resize_to(arr, size: int) -> np.ndarray:
    """Resize any uint8 image (or placeholder None) to a square ``size``."""
    if arr is None:
        return np.full((size, size, 3), 40, dtype=np.uint8)
    im = Image.fromarray(arr)
    if im.size != (size, size):
        # Use NEAREST for very small feature maps so we don't blur them.
        interp = Image.NEAREST if max(im.size) <= 64 else Image.LANCZOS
        im = im.resize((size, size), interp)
    return np.array(im)


def build_row_grid(
    panels: list,           # list of (label, image_np_or_None)
    *,
    title: Optional[str] = None,
    thumb: int = 384,
    gap: int = 12,
    sub_title_h: int = 40,
    title_h: int = 52,
    bg=(255, 255, 255),
    panel_bg=(20, 20, 20),
    title_bg=(32, 32, 32),
    text_fg=(235, 235, 235),
) -> np.ndarray:
    """Compose panels in a 1×N row with per-panel sub-titles and an optional
    main title. Panels are separated by a white ``gap``."""
    from PIL import ImageDraw
    n = len(panels)
    if n == 0:
        return np.full((thumb, thumb, 3), 40, dtype=np.uint8)

    # Geometry
    total_w = n * thumb + (n - 1) * gap
    top_title = title_h if title else 0
    total_h = top_title + sub_title_h + thumb

    out = np.full((total_h, total_w, 3), bg, dtype=np.uint8)

    # Main title bar
    if title:
        bar = Image.new('RGB', (total_w, title_h), color=title_bg)
        d = ImageDraw.Draw(bar)
        font = _get_font(30)
        try:
            bbox = d.textbbox((0, 0), title, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = d.textsize(title, font=font)
        d.text(((total_w - tw) // 2, max(4, (title_h - th) // 2)),
               title, fill=text_fg, font=font)
        out[:title_h] = np.array(bar)

    # Sub-titles + panels
    sub_font = _get_font(22)
    for i, (label, arr) in enumerate(panels):
        x0 = i * (thumb + gap)
        # sub-title
        sub = Image.new('RGB', (thumb, sub_title_h), color=panel_bg)
        d = ImageDraw.Draw(sub)
        label = '' if label is None else str(label)
        try:
            bbox = d.textbbox((0, 0), label, font=sub_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = d.textsize(label, font=sub_font)
        d.text(((thumb - tw) // 2, max(4, (sub_title_h - th) // 2)),
               label, fill=text_fg, font=sub_font)
        out[top_title:top_title + sub_title_h, x0:x0 + thumb] = np.array(sub)
        # image tile
        tile = _resize_to(arr, thumb)
        out[top_title + sub_title_h:top_title + sub_title_h + thumb,
            x0:x0 + thumb] = tile
    return out


def build_summary_grid(seg_np, depth_np, feature_np, gen_np, gt_np_or_none,
                       *, title: Optional[str] = None,
                       thumb: int = 384) -> np.ndarray:
    """1×5 summary: Seg | Depth | HDC²A feature | Generated | GT."""
    panels = [
        ('Seg',                 seg_np),
        ('Depth',               depth_np),
        ('HDC²A feature',       feature_np),
        ('Generated (seed 0)',  gen_np),
        ('GT RGB' if gt_np_or_none is not None else 'GT (n/a)',
         gt_np_or_none),
    ]
    return build_row_grid(panels, title=title, thumb=thumb)


def build_seed_grid(images_np: list, labels: list,
                    *, title: Optional[str] = None,
                    thumb: int = 384) -> np.ndarray:
    """1×N row of seed tiles, one sub-title per tile."""
    panels = list(zip(labels, images_np))
    return build_row_grid(panels, title=title, thumb=thumb)


# ═════════════════════════════════════════════════════════════════════════════
# Main generate callback
# ═════════════════════════════════════════════════════════════════════════════

def generate_ui(seg_path, depth_path, gt_path, prompt_embed_state,
                seeds_str, num_steps, guidance_scale,
                run_lora_off, run_adapter_off,
                run_chained, seg_strength, seg_start, seg_end,
                depth_strength, depth_start, depth_end,
                progress=gr.Progress(track_tqdm=False)):
    """Wrapper that traps exceptions so errors are surfaced to the UI instead
    of silently swallowed by the Gradio queue."""
    import traceback
    _empty = (None, None, None, None, None, None)
    print(f'\n[Generate] seg={seg_path} depth={depth_path} gt={gt_path} '
          f'seeds={seeds_str!r} steps={num_steps} guidance={guidance_scale} '
          f'loraOff={run_lora_off} adaptOff={run_adapter_off}')
    try:
        return _generate_impl(seg_path, depth_path, gt_path, prompt_embed_state,
                              seeds_str, num_steps, guidance_scale,
                              run_lora_off, run_adapter_off,
                              run_chained, seg_strength, seg_start, seg_end,
                              depth_strength, depth_start, depth_end, progress)
    except Exception as e:
        tb = traceback.format_exc()
        print('[Generate] ERROR:\n' + tb)
        # Truncate traceback for UI
        short_tb = '\n'.join(tb.strip().splitlines()[-6:])
        return (*_empty, f'❌ **Generation failed** — `{type(e).__name__}: {e}`\n\n```\n{short_tb}\n```')


def _generate_impl(seg_path, depth_path, gt_path, prompt_embed_state,
                   seeds_str, num_steps, guidance_scale,
                   run_lora_off, run_adapter_off,
                   run_chained, seg_strength, seg_start, seg_end,
                   depth_strength, depth_start, depth_end, progress):
    _empty = (None, None, None, None, None, None)
    if not getattr(STATE, 'hdc2a', None) or not getattr(STATE, 'transformer', None):
        return (*_empty, '❌ Model not loaded — please wait for startup or reload the checkpoint.')
    if prompt_embed_state is None:
        return (*_empty, '❌ Please encode the prompt first.')
    if not seg_path or not depth_path:
        return (*_empty, '❌ Please upload both seg and depth files.')

    try:
        # Validate files are readable; actual loading happens inside preprocess_*
        if not Path(seg_path).exists():
            raise FileNotFoundError(f'seg file not found: {seg_path}')
        if not Path(depth_path).exists():
            raise FileNotFoundError(f'depth file not found: {depth_path}')
        if gt_path and not Path(gt_path).exists():
            raise FileNotFoundError(f'GT file not found: {gt_path}')
    except Exception as e:
        return (*_empty, f'❌ Could not open uploaded file: {e}')

    # --- Parse seeds ------------------------------------------------------
    try:
        seeds = [int(s.strip()) for s in seeds_str.split(',') if s.strip()]
        assert 1 <= len(seeds) <= 8
    except Exception:
        return (*_empty, f'❌ Invalid seeds: "{seeds_str}" (expect e.g. "0,1,2,3")')

    size = STATE.image_size
    num_classes = STATE.num_classes

    # --- Preprocess inputs ------------------------------------------------
    progress(0.05, desc='Preprocessing inputs')
    try:
        seg = preprocess_seg(seg_path, size, num_classes)     # [H,W]
        depth = preprocess_depth(depth_path, size)            # [1,H,W]
    except Exception as e:
        return (*_empty, f'❌ Preprocessing failed: {e}')
    seg_b   = seg.unsqueeze(0)                                # [1,H,W]
    depth_b = depth.unsqueeze(0)                              # [1,1,H,W]

    gt_np = None
    if gt_path:
        try:
            gt_np = _to_uint8_rgb(preprocess_rgb(gt_path, size))
        except Exception as e:
            gt_np = None  # GT is optional; don't abort on failure

    seg_rgb = seg_to_rgb(seg)
    depth_rgb = depth_to_rgb(depth)

    # --- HDC²A control context (sanity heatmap) ---------------------------
    progress(0.15, desc='Running HDC²A')
    with torch.no_grad():
        ctrl = STATE.hdc2a(seg_b.to(DEVICE), depth_b.to(DEVICE, DTYPE))
    N = ctrl.shape[1]
    side = int(math.sqrt(N))
    hm = ctrl[0].float().mean(dim=-1).reshape(side, side).cpu().numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    import matplotlib.cm as cm
    feature_small = (cm.magma(hm)[..., :3] * 255).astype(np.uint8)
    # Upscale the feature map for standalone display so the panel isn't a tiny tile.
    feature_rgb = np.array(
        Image.fromarray(feature_small).resize((size, size), Image.NEAREST)
    )

    # --- Broadcast prompt embedding to batch ------------------------------
    B = len(seeds)
    prompt_embed_B = prompt_embed_state.to(DEVICE, DTYPE).unsqueeze(0).expand(B, -1, -1).contiguous()

    # --- Generation: LoRA ON, adapter ON (main) ---------------------------
    progress(0.2, desc=f'Sampling ({num_steps} steps, {B} seeds) — LoRA ON')
    STATE.lora_enable(True)
    main_rgb = sample_batch(seg_b, depth_b, prompt_embed_B,
                            num_steps=int(num_steps),
                            guidance_scale=float(guidance_scale),
                            seeds=seeds, bypass_adapter=False)   # [B, 3, H, W]

    seed_tiles = [_to_uint8_rgb(main_rgb[i]) for i in range(B)]
    seed_labels = [f'seed={s}' for s in seeds]
    seed_grid = build_seed_grid(
        seed_tiles, seed_labels,
        title=f'Generated samples — {B} seed(s), {num_steps} steps, guidance={guidance_scale}',
    )

    # --- Comparison: seg | depth | feature | generated(seed0) | [chain] | GT --------
    # Use the uploaded filename stem as the summary title.
    stem = Path(seg_path).stem

    # Optional: ComfyUI-style chained ControlNet (seg → depth) on seed[0].
    chained_rgb_np = None
    if run_chained:
        progress(0.6, desc='Sampling — chained ControlNet (seg + depth, ComfyUI-style)')
        STATE.lora_enable(True)
        chained = sample_batch_chained(
            seg_b, depth_b, prompt_embed_b[:1],
            num_steps=int(num_steps), guidance_scale=float(guidance_scale),
            seeds=seeds[:1],
            seg_strength=float(seg_strength),
            seg_start=float(seg_start), seg_end=float(seg_end),
            depth_strength=float(depth_strength),
            depth_start=float(depth_start), depth_end=float(depth_end),
        )
        chained_rgb_np = _to_uint8_rgb(chained[0])

    summary_panels = [
        ('Seg',                seg_rgb),
        ('Depth',              depth_rgb),
        ('HDC²A feature',      feature_rgb),
        ('Generated (seed 0)', seed_tiles[0]),
    ]
    if chained_rgb_np is not None:
        summary_panels.append((
            f'Chained CN — no LoRA, no adapter (seg s={seg_strength:.2f} '
            f'[{seg_start:.2f},{seg_end:.2f}] · depth s={depth_strength:.2f} '
            f'[{depth_start:.2f},{depth_end:.2f}])',
            chained_rgb_np,
        ))
    summary_panels.append(
        ('GT RGB' if gt_np is not None else 'GT (n/a)', gt_np)
    )
    summary = build_row_grid(summary_panels, title=stem)

    # --- Optional: LoRA OFF / adapter zero-ctx on seed[0] -----------------
    ablation = None
    if run_lora_off or run_adapter_off:
        panels = [('LoRA ON + adapter ON', seed_tiles[0])]

        if run_lora_off:
            progress(0.75, desc='Sampling — LoRA OFF')
            STATE.lora_enable(False)
            off_rgb = sample_batch(seg_b, depth_b, prompt_embed_b[:1],
                                   num_steps=int(num_steps),
                                   guidance_scale=float(guidance_scale),
                                   seeds=seeds[:1], bypass_adapter=False)
            panels.append(('LoRA OFF', _to_uint8_rgb(off_rgb[0])))

        if run_adapter_off:
            progress(0.9, desc='Sampling — adapter OFF (raw seg+depth bypass)')
            STATE.lora_enable(True)
            zc_rgb = sample_batch(seg_b, depth_b, prompt_embed_b[:1],
                                  num_steps=int(num_steps),
                                  guidance_scale=float(guidance_scale),
                                  seeds=seeds[:1], bypass_adapter=True)
            panels.append(('Adapter OFF (raw bypass)', _to_uint8_rgb(zc_rgb[0])))

        STATE.lora_enable(True)   # restore default state
        ablation = build_row_grid(
            panels,
            title=f'Ablation on seed={seeds[0]} — {stem}',
        )

    status = (f'✅ Generated {B} samples @ {size}×{size}, '
              f'{num_steps} Euler steps, guidance={guidance_scale}. '
              f'VRAM={_vram_gb():.1f} GiB.')
    return seg_rgb, depth_rgb, feature_rgb, seed_grid, summary, ablation, status


# ═════════════════════════════════════════════════════════════════════════════
# Prompt-diversity helpers + VLM captioner (Dataset Expansion block)
# ═════════════════════════════════════════════════════════════════════════════

def _diversity_block(values: tuple) -> dict:
    """Pair a tuple of dropdown values (in ``DIVERSITY_KEYS`` order) with their
    axis names, normalising missing / empty entries to ``NONE_OPT``."""
    out = {}
    for i, k in enumerate(DIVERSITY_KEYS):
        v = values[i] if i < len(values) else NONE_OPT
        out[k] = (v or NONE_OPT).strip() or NONE_OPT
    return out


def _merge_diversity_into_prompt_json(prompt_text: str, *values) -> tuple:
    """Parse *prompt_text*, inject a ``diversity`` block with the dropdown
    values (positional, in ``DIVERSITY_KEYS`` order), and return
    ``(updated_json_str, status_msg)``.

    Values equal to ``NONE_OPT`` (``'none'``) are stored verbatim in the JSON
    so the UI state round-trips, but `compose_prompt_from_json` skips them
    at encode time. All pre-existing fields are left untouched."""
    try:
        obj = json.loads(prompt_text) if prompt_text.strip() else {}
        if not isinstance(obj, dict):
            return prompt_text, '❌ Prompt must be a JSON object, not an array/scalar.'
    except Exception as e:
        return prompt_text, f'❌ Could not parse prompt JSON: {e}'

    obj['diversity'] = _diversity_block(values)
    new_text = json.dumps(obj, indent=2, ensure_ascii=False)

    active = [f'{k}={v}' for k, v in obj['diversity'].items()
              if v and v.lower() not in ('none', '(any)')]
    msg = ('✅ Injected `diversity` into prompt JSON — '
           + (', '.join(active) if active
              else 'all *none*, no clause will be emitted')
           + '. Click **Encode prompt** to refresh the embedding.')
    return new_text, msg


# ── VLM captioner (lazy load + unload each call) ────────────────────────────
# The captioner in this block is **independent** of the main Flux2 text
# encoder (also Mistral, but the text-only pruned variant). We treat it as an
# ephemeral "describer" loaded on demand, auto-unloaded after each call so it
# doesn't compete with the transformer / VAE / HDC²A for GPU memory.
_CAPTIONER_CACHE: dict = {
    'processor': None, 'model': None, 'model_id': None, 'family': None,
}

# Aerial-tuned instruction for the VLM. Deliberately asks for *one short
# sentence*: we want a detail anchor for consistency, NOT a long description
# that would blow the prompt budget or override the diversity axes.
AERIAL_VLM_INSTRUCTION = (
    'You are describing an aerial top-down satellite photograph for a '
    'generative model. Building layout, road network, and depth/height are '
    'already provided by separate segmentation + depth controls, and '
    '**season, time of day, weather, region/architectural style, lighting '
    'mood, and vegetation state are controlled by separate diversity tags '
    'set by the user**, so your caption must NOT mention or imply ANY of '
    'those. Focus ONLY on whole-image RGB appearance attributes that are '
    'invariant under those tags. In ONE concise sentence (~25 words) cover: '
    '(1) **overall area type at the functional level** (e.g. suburban '
    'residential, dense urban core, industrial / warehouse zone, commercial '
    'strip, institutional campus, mixed-use) — describe the FUNCTION, NOT '
    'the architectural style; '
    '(2) **dominant ground / surface materials** (asphalt, concrete, gravel, '
    'dirt, paved plazas, packed earth); '
    '(3) **vehicle presence** as a coarse level only — none / sparse / '
    'moderate / many — without giving a number; '
    '(4) **water presence** (visible river / pond / pool / none). '
    'Stay at the whole-image level. STRICT EXCLUSIONS — do NOT mention or '
    'describe: weather, clouds, sky, season, time of day, sun angle, '
    'shadows, lighting, image quality, foliage color or greenness, leaf '
    'state (lush / leafless / dry / autumn-coloured), grass color, snow, '
    'wetness, regional architectural style (e.g. "European", "American '
    'suburb", "East Asian", "Mediterranean", "tropical"), building '
    'materials/style descriptors (e.g. "brick rowhouses", "glass towers", '
    '"detached suburban homes", "stone buildings"), atmosphere or mood '
    '(e.g. "serene", "peaceful", "vibrant"), object positions ("on the '
    'left"), or specific landmarks / counts.'
)


def _captioner_family(model_id: str) -> str:
    """Classify *model_id* into a loader family. Extend this switch when
    adding new VLM backends."""
    mid = model_id.lower()
    if 'blip' in mid:
        return 'blip'
    if 'mistral' in mid and ('vl' in mid or '3.1' in mid or 'small-3' in mid):
        return 'mistral3'
    # Default to HuggingFace `AutoModelForImageTextToText` + `AutoProcessor`,
    # which covers most recent VLMs (Qwen2-VL, Llava, Llama-3.2-Vision, etc.).
    return 'auto'


def _load_captioner(model_id: str):
    """Lazy-load a VLM captioner on demand. Returns ``(processor, model, family)``.

    Three loader paths:

    - ``blip``     : classic ``BlipForConditionalGeneration`` (~500MB, fp16).
    - ``mistral3`` : Mistral-Small-3.1 24B VL via ``Mistral3ForConditionalGeneration``
                     (bf16, ``device_map='auto'`` so it shards if needed).
    - ``auto``     : generic ``AutoModelForImageTextToText`` + ``AutoProcessor``.

    The first two are ~500MB and ~50GB respectively; Mistral-VL on a single
    24GB card will OOM, use BLIP as the fallback via the ``HDC2A_CAPTIONER``
    env var.
    """
    if (_CAPTIONER_CACHE['model'] is not None
            and _CAPTIONER_CACHE['model_id'] == model_id):
        return (_CAPTIONER_CACHE['processor'], _CAPTIONER_CACHE['model'],
                _CAPTIONER_CACHE['family'])

    family = _captioner_family(model_id)
    print(f'[Caption] Loading VLM `{model_id}` (family={family}) ...')

    if family == 'blip':
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(DEVICE).eval()
    elif family == 'mistral3':
        # Mistral-Small-3.1 VL — 24B params, bf16 + device_map='auto'.
        # Class was introduced in transformers ≥4.49.
        from transformers import AutoProcessor
        try:
            from transformers import Mistral3ForConditionalGeneration as _Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as _Cls
        processor = AutoProcessor.from_pretrained(model_id)
        model = _Cls.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map='auto',
        ).eval()
    else:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Qwen2-VL / Qwen2.5-VL honour ``min_pixels`` / ``max_pixels`` to cap
        # the number of vision tokens. Aerial top-down scenes don't need 4K
        # resolution; clamp to ~512²–768² worth of tokens to keep VRAM sane.
        proc_kwargs = {}
        if 'qwen' in model_id.lower() and 'vl' in model_id.lower():
            proc_kwargs.update(min_pixels=256 * 28 * 28,
                               max_pixels=768 * 28 * 28)
        processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map='auto',
        ).eval()

    _CAPTIONER_CACHE.update(processor=processor, model=model,
                            model_id=model_id, family=family)
    print(f'[Caption] Loaded. VRAM={_vram_gb():.1f} GiB')
    return processor, model, family


def _unload_captioner():
    if _CAPTIONER_CACHE['model'] is not None:
        _CAPTIONER_CACHE['model'] = None
        _CAPTIONER_CACHE['processor'] = None
        _CAPTIONER_CACHE['model_id'] = None
        _CAPTIONER_CACHE['family'] = None
        _clear_cuda()


def _run_caption(processor, model, family: str, img) -> str:
    """Dispatch one captioning forward pass per family. Returns raw text."""
    if family == 'blip':
        prompt_prefix = 'an aerial satellite photograph showing'
        inputs = processor(img, prompt_prefix, return_tensors='pt').to(
            DEVICE, torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, num_beams=3,
                                 repetition_penalty=1.15)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    # Chat-style (mistral3 / generic auto) — build a messages list with the
    # image + instruction and let the processor's chat template do the work.
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': img},
            {'type': 'text',  'text':  AERIAL_VLM_INSTRUCTION},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors='pt',
    )
    # Route tensors to the primary device of the (possibly sharded) model.
    primary_device = next(model.parameters()).device
    inputs = {k: (v.to(primary_device) if hasattr(v, 'to') else v)
              for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                             repetition_penalty=1.1)
    gen_ids = out[0][input_len:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def caption_rgb_ui(rgb_path, keep_resident: bool = False):
    """Generate a short VLM caption for *rgb_path*. Returns (caption, status).

    By default the model is unloaded after each call so it doesn't compete for
    VRAM with the main Flux2/HDC²A pipeline. Set *keep_resident* True for
    batch-captioning workflows where you plan to call this repeatedly."""
    # Unwrap Gradio 6 dict (PNG files passed as dicts by gr.File .change())
    if isinstance(rgb_path, dict):
        rgb_path = rgb_path.get('path') or rgb_path.get('name') or rgb_path.get('tmp_path')
    if not rgb_path or not Path(rgb_path).exists():
        return '', '❌ Please upload an RGB reference image first.'
    try:
        processor, model, family = _load_captioner(CAPTIONER_MODEL_ID)
    except Exception as e:
        _unload_captioner()
        return '', (f'❌ Could not load captioner `{CAPTIONER_MODEL_ID}`: {e}\n\n'
                    f'Override via `HDC2A_CAPTIONER=<model-id>` env var '
                    f'(e.g. `Salesforce/blip-image-captioning-large` for a '
                    f'~500MB lightweight fallback).')

    try:
        img = Image.open(rgb_path).convert('RGB')
        caption = _run_caption(processor, model, family, img)
        caption = ' '.join(caption.split())   # collapse whitespace
    except Exception as e:
        if not keep_resident:
            _unload_captioner()
        return '', f'❌ Captioning failed: {e}'

    if not keep_resident:
        _unload_captioner()

    msg = (f'✅ Captioned via `{CAPTIONER_MODEL_ID}` ({len(caption)} chars). '
           f'Edit freely below — it will be truncated to {CAPTION_MAX_CHARS} '
           f'chars when inserted into the prompt.')
    return caption, msg


def build_finalized_prompt(base_prompt_text: str, caption: str,
                           *values) -> tuple:
    """Compose a full prompt JSON by overlaying *caption* + diversity on top
    of *base_prompt_text*. *values* are the dropdown values in
    ``DIVERSITY_KEYS`` order. Returns ``(json_text, flat_preview, status)``."""
    try:
        obj = json.loads(base_prompt_text) if base_prompt_text.strip() else {}
        if not isinstance(obj, dict):
            obj = {}
    except Exception:
        obj = {}

    # Fall back to a minimal skeleton if user's base is unparseable.
    obj.setdefault('scene', 'Aerial top-down satellite view of an urban area')
    obj.setdefault('style', 'Google Earth style, photorealistic satellite imagery')
    obj.setdefault('quality', 'masterpiece, raw photo, crisp details')

    obj['diversity'] = _diversity_block(values)
    cap = (caption or '').strip()
    if cap:
        obj['reference_caption'] = cap
    elif 'reference_caption' in obj:
        # User cleared the caption → drop the stale field
        obj.pop('reference_caption', None)

    json_text = json.dumps(obj, indent=2, ensure_ascii=False)
    flat = compose_prompt_from_json(obj)
    msg = (f'✅ Finalized prompt — {len(flat)} chars flat, '
           f'{len(json_text.splitlines())} lines JSON.')
    return json_text, flat, msg


# ═════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_PROMPT_TEXT = (
    DEFAULT_PROMPT_JSON.read_text()
    if DEFAULT_PROMPT_JSON.is_file()
    else '{"scene": "", "style": "", "elements": {}, "lighting": "", "quality": ""}'
)


def _header_md() -> str:
    ckpt_disp = (STATE.ckpt_dir.relative_to(PROJECT_ROOT)
                 if STATE.ckpt_dir else '(none)')
    return (
        f"# HDC²A + Flux2 + ControlNet (LoRA) — Inference\n"
        f"Checkpoint: `{ckpt_disp}` &nbsp;·&nbsp; "
        f"Image size: **{STATE.image_size}×{STATE.image_size}** &nbsp;·&nbsp; "
        f"LoRA rank: **{STATE.lora_rank}** &nbsp;·&nbsp; "
        f"Fusion: **{'MLP' if STATE.use_mlp_fusion else 'DoubleStream'}**\n\n"
        f"**Workflow**: (0) pick a checkpoint and click *Load*.  "
        f"(1) edit the JSON prompt and click *Encode prompt*.  "
        f"(2) upload seg + depth (+ optional GT RGB) or scan a folder.  "
        f"(3) click *Generate*."
    )


def build_ui():
    with gr.Blocks(title='HDC²A + Flux2 ControlNet — Inference') as demo:
        header_md = gr.Markdown(_header_md())

        prompt_embed_state = gr.State(value=None)
        # Hidden mirror of scan results so the dropdown callback can resolve stems.
        scan_state = gr.State(value={'stems': [], 'by_stem': {}})

        # ── 0. Checkpoint ────────────────────────────────────────────────
        gr.Markdown('### 0. Checkpoint')
        with gr.Row():
            ckpt_dd = gr.Dropdown(
                label='Checkpoint (output/*/checkpoint_epoch_*)',
                choices=list_checkpoints(),
                value=str(STATE.ckpt_dir.relative_to(PROJECT_ROOT)) if STATE.ckpt_dir else None,
                interactive=True, scale=6,
            )
            ckpt_refresh_btn = gr.Button('🔄 Rescan', variant='secondary', scale=1)
            ckpt_load_btn = gr.Button('📦 Load', variant='primary', scale=1)
        ckpt_status = gr.Markdown(
            f'_(current: `{STATE.ckpt_dir.relative_to(PROJECT_ROOT) if STATE.ckpt_dir else "n/a"}`)_'
        )

        # ── 1. Inputs ────────────────────────────────────────────────────
        gr.Markdown('### 1. Inputs')
        # Row 1: path | select
        with gr.Row():
            with gr.Column(scale=1):
                root_box = gr.Textbox(
                    label='📂 Path — parent folder containing seg/ depth/ rgb/',
                    placeholder='e.g. /home/xg_wang_group/SynthUrbanSAT/dataset/test',
                )
                scan_btn = gr.Button('🔍 Scan', variant='secondary')
            with gr.Column(scale=1):
                stem_dd = gr.Dropdown(
                    label='Select — scene stem (same name across seg/depth/rgb)',
                    choices=[], interactive=True, allow_custom_value=False,
                )
                scan_status = gr.Markdown('_(enter a path and click Scan)_')

        # Row 2: seg | depth | rgb (each = file + preview)
        with gr.Row():
            with gr.Column(scale=1):
                seg_in = gr.File(label='Seg (palette PNG / RGB colour-mapped)',
                                 file_types=['.png', '.tif', '.tiff', '.jpg', '.jpeg'],
                                 type='filepath')
                seg_preview = gr.Image(label='Seg preview', interactive=False, height=240)
            with gr.Column(scale=1):
                depth_in = gr.File(label='Depth (float TIF recommended)',
                                   file_types=['.tif', '.tiff', '.png'],
                                   type='filepath')
                depth_preview = gr.Image(label='Depth preview', interactive=False, height=240)
            with gr.Column(scale=1):
                gt_in = gr.File(label='RGB (optional, GT reference)',
                                file_types=['.png', '.tif', '.tiff', '.jpg', '.jpeg'],
                                type='filepath')
                gt_preview = gr.Image(label='RGB preview', interactive=False, height=240)
        sibling_status = gr.Markdown('_(upload any file above to auto-fill siblings)_')

        # ── 2. Settings ──────────────────────────────────────────────────
        gr.Markdown('### 2. Settings')
        # Row 1: text prompt | diversity overlay
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('**Text prompt (JSON)**')
                prompt_box = gr.Code(
                    value=DEFAULT_PROMPT_TEXT, language='json', label='Prompt JSON',
                    lines=14,
                )
                encode_btn = gr.Button('🔤 Encode prompt', variant='secondary')
                prompt_status = gr.Markdown('_(prompt not yet encoded)_')
            with gr.Column(scale=1):
                # ── Diversity overlay + (optional) Dataset Expansion ──────
                gr.Markdown(
                    '**Diversity overlay** — pick axes below and click '
                    '*Apply* to inject a `diversity` block into the JSON. '
                    'Axes left on **`none`** are recorded in the JSON but '
                    'NOT emitted into the flat prompt (the model is free to '
                    'pick).\n\n'
                    '**Dataset Expansion** (checkbox below): also caption the '
                    'GT RGB from **Section 1** with a VLM and inject it as a '
                    '`reference_caption` so the generated RGB stays consistent '
                    f'with that scene. Captioner: `{CAPTIONER_MODEL_ID}` '
                    '(override via `HDC2A_CAPTIONER` env var).'
                )
                expand_chk = gr.Checkbox(
                    value=False,
                    label='🎨 Use Dataset Expansion (caption GT RGB + inject as reference)',
                )
                with gr.Row():
                    caption_btn = gr.Button('🔍 Caption GT RGB with VLM',
                                            variant='secondary')
                caption_box = gr.Textbox(
                    label='VLM caption (editable — only used when Dataset Expansion is on)',
                    lines=3,
                    placeholder='Click "Caption GT RGB with VLM" above (uses '
                                'the RGB from Section 1), or type a short '
                                'scene description yourself.',
                )
                caption_status = gr.Markdown('_(no caption yet)_')

                # Build one gr.Dropdown per axis, laid out two-per-row. Stored
                # in a dict keyed by axis name so the wiring can iterate
                # ``DIVERSITY_KEYS`` and stay in sync with the config above.
                div_dd = {}
                _axis_labels = {
                    'season':           'Season',
                    'time_of_day':      'Time of day',
                    'weather':          'Weather',
                    'region':           'Region / construction style',
                    'lighting_mood':    'Lighting mood',
                    'vegetation_state': 'Vegetation state',
                }
                _keys = list(DIVERSITY_KEYS)
                for i in range(0, len(_keys), 2):
                    with gr.Row():
                        for k in _keys[i:i + 2]:
                            div_dd[k] = gr.Dropdown(
                                label=_axis_labels.get(k, k),
                                choices=DIVERSITY_OPTIONS[k],
                                value=NONE_OPT, interactive=True, scale=1)
                apply_div_btn = gr.Button(
                    '🎨 Apply diversity (+caption if expansion on) → JSON',
                    variant='secondary')

        # Row 2: sampling | seg branch | depth branch  (ControlNet section)
        gr.Markdown(
            '**ControlNet (ComfyUI-style chained — seg → depth)**  \n'
            '_Pure Flux2 + base ControlNet weights: **LoRA OFF** and **HDC²A adapter OFF**. '
            'Each branch feeds a deterministic raw patch embedding of a single modality '
            '(seg-only / depth-only) into the ControlNet, blended per Euler step with the '
            'strength + start%/end% window below. Result appears as an extra panel in the '
            'summary (seed[0])._'
        )
        chained_chk = gr.Checkbox(
            value=True,
            label='Also run chained ControlNet (seg + depth scheduled)')
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('**Sampling**')
                seeds_box = gr.Textbox(value='0, 1, 2, 3', label='Seeds (comma-separated, up to 8)')
                steps_slider = gr.Slider(4, 60, value=28, step=1, label='Euler steps')
                cfg_slider = gr.Slider(1.0, 10.0,
                                       value=float(STATE.cfg.get('guidance_scale', 3.5)),
                                       step=0.5, label='Guidance scale')
                lora_off_chk = gr.Checkbox(
                    value=True,
                    label='Also run LoRA OFF (ablation on seed[0])')
                adapt_off_chk = gr.Checkbox(
                    value=True,
                    label='Also run adapter OFF — raw seg+depth bypass (ablation on seed[0])')
            with gr.Column(scale=1):
                gr.Markdown('**Seg branch**')
                seg_strength_sl = gr.Slider(0.0, 2.0, value=0.70, step=0.05, label='Seg strength')
                seg_start_sl    = gr.Slider(0.0, 1.0, value=0.00, step=0.01, label='Seg start %')
                seg_end_sl      = gr.Slider(0.0, 1.0, value=0.70, step=0.01, label='Seg end %')
            with gr.Column(scale=1):
                gr.Markdown('**Depth branch**')
                depth_strength_sl = gr.Slider(0.0, 2.0, value=0.85, step=0.05, label='Depth strength')
                depth_start_sl    = gr.Slider(0.0, 1.0, value=0.30, step=0.01, label='Depth start %')
                depth_end_sl      = gr.Slider(0.0, 1.0, value=0.70, step=0.01, label='Depth end %')

        generate_btn = gr.Button('🚀 Generate', variant='primary')
        status = gr.Markdown('_(ready)_')

        # ── 3. Results ───────────────────────────────────────────────────
        gr.Markdown('### 3. Results')
        with gr.Row():
            seg_view = gr.Image(label='Seg (colorized)', interactive=False)
            depth_view = gr.Image(label='Depth (normalized)', interactive=False)
            feat_view = gr.Image(
                label=f'HDC²A feature (native {STATE.image_size//32}×{STATE.image_size//32}, upscaled)',
                interactive=False,
            )

        seed_view = gr.Image(label='Generated grid (1×N, one tile per seed)', interactive=False)
        summary_view = gr.Image(label='Summary: Seg | Depth | Feature | Generated | [Chained CN] | GT',
                                interactive=False)
        ablation_view = gr.Image(label='Ablation: LoRA on/off + adapter OFF (raw seg+depth bypass) on seed[0]',
                                 interactive=False)

        # --- Wiring -------------------------------------------------------
        def _encode_cb(prompt_text):
            embed, flat, msg = encode_prompt_ui(prompt_text)
            return embed, msg

        encode_btn.click(
            _encode_cb,
            inputs=[prompt_box],
            outputs=[prompt_embed_state, prompt_status],
        )

        # Diversity dropdowns → merge into prompt JSON (does not auto-encode;
        # user still clicks Encode prompt). When the *Dataset Expansion*
        # checkbox is enabled we additionally inject the VLM caption as
        # ``reference_caption`` via ``build_finalized_prompt``. Otherwise we
        # only merge the diversity block via ``_merge_diversity_into_prompt_json``.
        def _apply_overlay_cb(prompt_text, use_expansion, caption, *values):
            if use_expansion:
                json_text, _flat, msg = build_finalized_prompt(
                    prompt_text, caption, *values)
                return json_text, msg + ' (diversity + reference_caption injected)'
            return _merge_diversity_into_prompt_json(prompt_text, *values)

        apply_div_btn.click(
            _apply_overlay_cb,
            inputs=[prompt_box, expand_chk, caption_box]
                   + [div_dd[k] for k in DIVERSITY_KEYS],
            outputs=[prompt_box, prompt_status],
        )

        # Caption button → uses GT RGB from Section 1 (gt_in)
        caption_btn.click(
            caption_rgb_ui,
            inputs=[gt_in],
            outputs=[caption_box, caption_status],
        )

        # --- Auto-resolve siblings on upload ------------------------------
        # Use .upload() (fires only on real user upload, NOT on programmatic set)
        # and only write to the OTHER two File slots to avoid re-triggering cascades.
        def _make_upload_resolver(kind: str):
            """Returns a callback (path, scan_state) -> (other1, other2, msg, prev1, prev2).

            Resolution strategy:
              1. If the uploaded file's parent dir is literally named `kind`
                 (server-side path), look at sibling folders `../seg`, `../depth`,
                 `../rgb` for a file with the same stem.
              2. Otherwise (browser upload → temp dir), look up the stem in
                 `scan_state` populated by the *Scan* button above.
              3. Otherwise, just show a helpful message.

            Returns 5 values: (sibling1_path, sibling2_path, status_msg, sibling1_preview, sibling2_preview)
            Previews are computed here because gr.File.change() does NOT fire on programmatic
            value updates in Gradio 6 — only on user-initiated uploads.
            """
            others = [k for k in ('seg', 'depth', 'rgb') if k != kind]
            def _cb(path, state):
                # Unwrap Gradio 6 dict (PNG files passed as dicts by gr.File .upload())
                if isinstance(path, dict):
                    path = path.get('path') or path.get('name') or path.get('tmp_path')
                if not path:
                    return gr.update(), gr.update(), '_(cleared)_', None, None
                p = Path(path)
                # --- 1. server-side path with sibling folder layout ---
                if p.parent.name.lower() == kind:
                    resolved = resolve_siblings(str(path), kind)
                    vals = [resolved[k] for k in others]
                    found = [k for k, v in zip(others, vals) if v is not None]
                    missing = [k for k, v in zip(others, vals) if v is None]
                    parts = [f'✅ **{kind}** from `{p.parent.parent}`']
                    if found:
                        parts.append('→ auto-filled: ' + ', '.join(f'**{k}**' for k in found))
                    if missing:
                        parts.append('(missing: ' + ', '.join(missing) + ')')
                    prev0 = make_preview(vals[0], others[0]) if vals[0] is not None else None
                    prev1 = make_preview(vals[1], others[1]) if vals[1] is not None else None
                    return (
                        vals[0] if vals[0] is not None else gr.update(),
                        vals[1] if vals[1] is not None else gr.update(),
                        ' '.join(parts),
                        prev0,
                        prev1,
                    )
                # --- 2. browser upload → match by stem against scanned root ---
                stem = p.stem
                by_stem = (state or {}).get('by_stem', {})
                if stem in by_stem:
                    paths = by_stem[stem]
                    vals = [paths.get(k) for k in others]
                    found = [k for k, v in zip(others, vals) if v]
                    msg = (f'✅ Browser-uploaded **{kind}** (`{stem}`) matched stem in '
                           f'scanned root → auto-filled: ' + ', '.join(f'**{k}**' for k in found))
                    prev0 = make_preview(vals[0], others[0]) if vals[0] else None
                    prev1 = make_preview(vals[1], others[1]) if vals[1] else None
                    return (
                        vals[0] if vals[0] else gr.update(),
                        vals[1] if vals[1] else gr.update(),
                        msg,
                        prev0,
                        prev1,
                    )
                # --- 3. no match ---
                msg = (f'_Uploaded **{kind}** (`{p.name}`) — cannot auto-fill. '
                       f'Either (a) place files server-side under `…/{kind}/` with siblings '
                       f'`…/seg/`, `…/depth/`, `…/rgb/`, or (b) set the Parent folder above, '
                       f'click **Scan**, then uploads with matching stem will auto-fill._')
                return gr.update(), gr.update(), msg, None, None
            return _cb

        # Wire upload -> fill OTHER TWO components (+ their previews, since .change() won't fire)
        # kind='seg'   → others=['depth','rgb']  → fill depth_in, gt_in, depth_preview, gt_preview
        # kind='depth' → others=['seg','rgb']    → fill seg_in,   gt_in, seg_preview,   gt_preview
        # kind='rgb'   → others=['seg','depth']  → fill seg_in,   depth_in, seg_preview, depth_preview
        seg_in.upload(
            _make_upload_resolver('seg'),
            inputs=[seg_in, scan_state],
            outputs=[depth_in, gt_in, sibling_status, depth_preview, gt_preview],
        )
        depth_in.upload(
            _make_upload_resolver('depth'),
            inputs=[depth_in, scan_state],
            outputs=[seg_in, gt_in, sibling_status, seg_preview, gt_preview],
        )
        gt_in.upload(
            _make_upload_resolver('rgb'),
            inputs=[gt_in, scan_state],
            outputs=[seg_in, depth_in, sibling_status, seg_preview, depth_preview],
        )

        # --- File change -> update preview thumbnail (fires on both upload & programmatic set) ---
        seg_in.change(lambda p: make_preview(p, 'seg'),   inputs=[seg_in],   outputs=[seg_preview])
        depth_in.change(lambda p: make_preview(p, 'depth'), inputs=[depth_in], outputs=[depth_preview])
        gt_in.change(lambda p: make_preview(p, 'rgb'),    inputs=[gt_in],    outputs=[gt_preview])

        # --- Folder scan + dropdown --------------------------------------
        def _scan_cb(root):
            result = scan_root_folder(root)
            return (
                result,                                           # scan_state
                gr.update(choices=result['stems'], value=None),   # stem_dd
                result['status'],                                 # scan_status
            )

        scan_btn.click(
            _scan_cb,
            inputs=[root_box],
            outputs=[scan_state, stem_dd, scan_status],
        )

        def _pick_stem_cb(stem, state):
            if not stem or not state or stem not in state.get('by_stem', {}):
                return gr.update(), gr.update(), gr.update(), '_(pick a stem)_', None, None, None
            paths = state['by_stem'][stem]
            msg = f'✅ Loaded stem **`{stem}`** from folder scan.'
            # Compute previews here because gr.File.change() does NOT fire on
            # programmatic value updates in Gradio 6 — only on user uploads.
            seg_prev = make_preview(paths['seg'], 'seg')
            dep_prev = make_preview(paths['depth'], 'depth')
            rgb_prev = make_preview(paths['rgb'], 'rgb')
            return paths['seg'], paths['depth'], paths['rgb'], msg, seg_prev, dep_prev, rgb_prev

        stem_dd.change(
            _pick_stem_cb,
            inputs=[stem_dd, scan_state],
            outputs=[seg_in, depth_in, gt_in, sibling_status, seg_preview, depth_preview, gt_preview],
        )

        # --- Checkpoint selector ------------------------------------------
        def _ckpt_refresh_cb():
            choices = list_checkpoints()
            cur = str(STATE.ckpt_dir.relative_to(PROJECT_ROOT)) if STATE.ckpt_dir else None
            return gr.update(choices=choices, value=cur if cur in choices else None)

        def _ckpt_load_cb(rel_path):
            if not rel_path:
                return gr.update(), '_(pick a checkpoint first)_', gr.update()
            abs_path = PROJECT_ROOT / rel_path
            if not (abs_path / 'meta.pt').is_file():
                return gr.update(), f'❌ `{rel_path}` is not a valid checkpoint dir', gr.update()
            try:
                STATE.load(abs_path)
            except Exception as e:
                return gr.update(), f'❌ load failed: `{e}`', gr.update()
            msg = (f'✅ Loaded `{rel_path}` — image {STATE.image_size}×{STATE.image_size}, '
                   f'LoRA rank {STATE.lora_rank}, fusion '
                   f"{'MLP' if STATE.use_mlp_fusion else 'DoubleStream'}.")
            # Invalidate cached prompt embedding (num_classes / config may have changed)
            return None, msg, _header_md()

        ckpt_refresh_btn.click(_ckpt_refresh_cb, inputs=[], outputs=[ckpt_dd])
        ckpt_load_btn.click(
            _ckpt_load_cb,
            inputs=[ckpt_dd],
            outputs=[prompt_embed_state, ckpt_status, header_md],
        )

        generate_btn.click(
            generate_ui,
            inputs=[seg_in, depth_in, gt_in, prompt_embed_state,
                    seeds_box, steps_slider, cfg_slider,
                    lora_off_chk, adapt_off_chk,
                    chained_chk, seg_strength_sl, seg_start_sl, seg_end_sl,
                    depth_strength_sl, depth_start_sl, depth_end_sl],
            outputs=[seg_view, depth_view, feat_view,
                     seed_view, summary_view, ablation_view, status],
        )

        # ── Notes ─────────────────────────────────────────────────────────
        gr.Markdown("""
        ---
        **Notes**

        - **Adapter OFF is a proxy**: instead of HDC²A's learned output we feed a deterministic
          patch embedding of the **raw** one-hot seg + depth into the control attention blocks
          (same token shape, no learned params). The ControlNet + LoRA still see seg/depth, just
          unprocessed — this isolates the adapter's contribution. A truly adapter-free baseline
          would require retraining without HDC²A, which is out of scope.
        - **LoRA ON/OFF** swaps `LoRALinear` wrappers for their wrapped `nn.Linear` (no weights
          reloaded). Same seed → identical noise path, so differences are purely from LoRA.
        - **Text encoder**: by default Mistral stays resident for instant encoding. Launch with
          `python notebooks/app.py --no-textencoder` to load+unload per request (saves ~6 GiB VRAM).
        - **Dtype**: bf16 throughout. Frozen transformer backbone is re-compressed to FP8 on load
          (VRAM savings only; math still bf16).
        """)

    return demo


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDC²A + Flux2 Gradio app')
    parser.add_argument(
        '--no-textencoder', action='store_true',
        help='Do NOT keep Mistral resident; load+unload per Encode call '
             '(saves ~6 GiB VRAM at the cost of ~10s per prompt change).'
    )
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 7860)))
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    PERSISTENT_TEXT_ENCODER = not args.no_textencoder
    print(f'[Launch] persistent text encoder = {PERSISTENT_TEXT_ENCODER}')

    STATE.load()
    demo = build_ui()
    demo.queue(max_size=8).launch(
        server_name=args.host, server_port=args.port,
        show_error=True, share=args.share,
        theme=gr.themes.Soft(),
    )

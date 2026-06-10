"""Self-contained Gradio webui for HDC²A + Flux2 ControlNet inference & comparison.

Launch:
    cd generation_pipeline
    bash setup.sh          # one-shot weights setup (only needed first time)
    python app.py          # opens http://<host>:7860

This pipeline is designed to be self-contained inside ``generation_pipeline/``:
  - All weights live under ``weights/`` (symlinked or copied by setup.sh).
  - Saved runs live under ``output/<custom_name>/``.
    - The training pipeline at ``../train_pipeline`` is reused via sys.path (model
    classes only — no path leakage into the GUI configuration).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw

# Ensure ``pipeline`` package is importable when launching from any cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline import (
    COLOR_MAP_PATH, DEFAULT_PROMPT_JSON, OUTPUT_DIR, PIPELINE_ROOT,
    list_lora_checkpoints, verify_base_weights,
)
from pipeline.preprocess import (
    SIBLING_DIRS, SUPPORTED_EXTS, depth_to_rgb, preprocess_depth,
    preprocess_rgb, preprocess_seg, resolve_siblings, scan_root_folder,
    seg_to_rgb,
)
from pipeline.save import save_run
from pipeline.state import DEVICE, DTYPE, STATE, vram_gb

# ─── Diversity overlay (lifted unchanged from training/notebooks/app.py) ────
NONE_OPT = 'none'
DIVERSITY_OPTIONS = {
    'season': [NONE_OPT, 'spring', 'summer', 'autumn', 'winter'],
    'time_of_day': [NONE_OPT, 'early morning', 'morning', 'midday',
                    'afternoon', 'golden hour', 'dusk'],
    'weather': [NONE_OPT, 'clear sky', 'partly cloudy', 'overcast',
                'light haze', 'post-rain wet ground'],
    'region': [NONE_OPT,
               'North American suburban', 'North American downtown',
               'European historic city', 'East Asian dense urban',
               'Arid / Middle-Eastern city', 'Tropical city'],
    'lighting_mood': [NONE_OPT, 'strong directional shadows',
                      'soft diffuse light', 'low-angle sunlight',
                      'flat cloudy light', 'crisp high-contrast light'],
    'vegetation_state': [NONE_OPT, 'leafless bare trees',
                         'lush green vegetation', 'dry brown vegetation',
                         'autumn foliage', 'sparse vegetation'],
}
DIVERSITY_KEYS = tuple(DIVERSITY_OPTIONS.keys())
CAPTION_MAX_CHARS = 220


# ═════════════════════════════════════════════════════════════════════════════
# Prompt composition
# ═════════════════════════════════════════════════════════════════════════════

def compose_prompt_from_json(obj: dict) -> str:
    parts = []
    if 'scene' in obj:    parts.append(obj['scene'])
    if 'style' in obj:    parts.append(obj['style'])
    if 'elements' in obj:
        for k, v in (obj.get('elements') or {}).items():
            parts.append(f'{k}: {v}')
    if 'lighting' in obj: parts.append(obj['lighting'])
    if 'quality' in obj:  parts.append(obj['quality'])

    div = obj.get('diversity') or {}
    if isinstance(div, dict):
        div_parts = []
        def _take(k):
            v = (div.get(k) or '').strip()
            return '' if not v or v.lower() in ('none', '(any)') else v
        season = _take('season')
        if season:
            div_parts.append(season if ('season' in season.lower() or ',' in season)
                             else f'{season} season')
        for k in ('time_of_day', 'weather', 'lighting_mood', 'vegetation_state', 'region'):
            v = _take(k)
            if v: div_parts.append(v)
        if div_parts:
            parts.append(', '.join(div_parts))

    cap = (obj.get('reference_caption') or '').strip()
    if cap:
        cap = cap.replace('\n', ' ').strip()
        if len(cap) > CAPTION_MAX_CHARS:
            cap = cap[:CAPTION_MAX_CHARS].rstrip() + '…'
        parts.append(f'scene reference: {cap}')

    return ', '.join(p for p in parts if p)


def merge_diversity(prompt_text: str, *values) -> tuple[str, str]:
    try:
        obj = json.loads(prompt_text) if prompt_text.strip() else {}
        if not isinstance(obj, dict):
            return prompt_text, '❌ Prompt must be a JSON object.'
    except Exception as e:
        return prompt_text, f'❌ Could not parse prompt JSON: {e}'
    block = {}
    for i, k in enumerate(DIVERSITY_KEYS):
        v = (values[i] if i < len(values) else NONE_OPT) or NONE_OPT
        block[k] = v.strip() or NONE_OPT
    obj['diversity'] = block
    active = [f'{k}={v}' for k, v in block.items() if v.lower() not in ('none', '(any)')]
    msg = '✅ Diversity injected: ' + (', '.join(active) if active else '(all none)')
    return json.dumps(obj, indent=2, ensure_ascii=False), msg


# ═════════════════════════════════════════════════════════════════════════════
# Encode prompt
# ═════════════════════════════════════════════════════════════════════════════

def encode_prompt_ui(prompt_json_text: str):
    """Returns (embed_tensor_or_None, flat_str, status_msg)."""
    from scripts.text_encoder import encode_prompts, load_text_encoder, unload_text_encoder

    try:
        obj = json.loads(prompt_json_text)
    except Exception as e:
        return None, '', f'❌ Invalid JSON: {e}'
    flat = compose_prompt_from_json(obj) if isinstance(obj, dict) else str(obj)
    if not flat.strip():
        return None, '', '❌ Empty prompt.'

    persistent = STATE.text_encoder is not None
    if persistent:
        te, tk = STATE.text_encoder, STATE.tokenizer
    else:
        print('[Encode] Loading Mistral on demand ...')
        te, tk = load_text_encoder(STATE.cfg['text_encoder_path'], device=DEVICE, dtype=DTYPE)

    try:
        embed = encode_prompts(
            te, tk, [flat],
            max_sequence_length=int(STATE.cfg.get('text_seq_len', 512)),
            device=DEVICE, dtype=DTYPE,
        )
    except Exception as e:
        if not persistent:
            unload_text_encoder(te, tk)
            gc.collect(); torch.cuda.empty_cache()
        return None, flat, f'❌ Encoding failed: {e}'

    embed = embed[0].detach().clone()
    if not persistent:
        unload_text_encoder(te, tk)
        gc.collect(); torch.cuda.empty_cache()

    msg = (f'✅ Encoded ({len(flat)} chars, {embed.shape[0]} tokens). '
           f'VRAM={vram_gb():.1f} GiB. Flat: "{flat[:160]}{"…" if len(flat) > 160 else ""}"')
    return embed, flat, msg


# ═════════════════════════════════════════════════════════════════════════════
# Image helpers
# ═════════════════════════════════════════════════════════════════════════════

def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _get_font(size=18):
    from PIL import ImageFont
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ):
        try: return ImageFont.truetype(path, size=size)
        except Exception: pass
    return ImageFont.load_default()


def build_panel_grid(panels: list[tuple[str, np.ndarray | None]],
                     *, thumb: int = 320, gap: int = 12,
                     sub_h: int = 32, title: str | None = None,
                     title_h: int = 48) -> np.ndarray:
    """Row grid: (label, image) pairs side-by-side with sub-titles."""
    if not panels:
        return np.full((thumb, thumb, 3), 255, dtype=np.uint8)
    n = len(panels)
    W = n * thumb + (n - 1) * gap
    top = title_h if title else 0
    H = top + sub_h + thumb
    out = np.full((H, W, 3), 255, dtype=np.uint8)

    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    if title:
        draw.rectangle([0, 0, W, title_h], fill=(35, 35, 38))
        font = _get_font(24)
        bbox = draw.textbbox((0, 0), title, font=font)
        tx = (W - (bbox[2] - bbox[0])) // 2
        ty = (title_h - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), title, fill=(235, 235, 240), font=font)

    sub_font = _get_font(18)
    for i, (label, arr) in enumerate(panels):
        x0 = i * (thumb + gap)
        # Subtitle bar
        draw.rectangle([x0, top, x0 + thumb, top + sub_h], fill=(50, 50, 55))
        bbox = draw.textbbox((0, 0), label, font=sub_font)
        tx = x0 + (thumb - (bbox[2] - bbox[0])) // 2
        ty = top + (sub_h - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), label, fill=(230, 230, 235), font=sub_font)
        # Image
        if arr is None:
            tile = np.full((thumb, thumb, 3), 200, dtype=np.uint8)
        else:
            im = Image.fromarray(arr if arr.dtype == np.uint8 else arr.clip(0, 255).astype(np.uint8))
            if im.size != (thumb, thumb):
                im = im.resize((thumb, thumb), Image.LANCZOS)
            tile = np.array(im.convert('RGB'))
        pil.paste(Image.fromarray(tile), (x0, top + sub_h))

    return np.array(pil)


def make_preview(path, kind: str, image_size: int = 512,
                 num_classes: int = 6, palette=None) -> Optional[np.ndarray]:
    if isinstance(path, dict):
        path = path.get('path') or path.get('name') or path.get('tmp_path')
    if not path or not Path(path).exists():
        return None
    try:
        if kind == 'seg':
            if palette is None:
                return None
            t = preprocess_seg(path, image_size, num_classes)
            return seg_to_rgb(t, palette)
        if kind == 'depth':
            t = preprocess_depth(path, image_size)
            return depth_to_rgb(t)
        return _to_uint8(preprocess_rgb(path, image_size))
    except Exception as e:
        print(f'[preview {kind}] {e}')
        return None


def feature_heatmap(ctrl: torch.Tensor, size: int) -> np.ndarray:
    """[B, N, C] → upscaled magma heatmap of token-mean activations."""
    N = ctrl.shape[1]
    side = int(math.sqrt(N))
    hm = ctrl[0].float().mean(dim=-1).reshape(side, side).cpu().numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    rgb = (cm.magma(hm)[..., :3] * 255).astype(np.uint8)
    return np.array(Image.fromarray(rgb).resize((size, size), Image.NEAREST))


# ═════════════════════════════════════════════════════════════════════════════
# Generate
# ═════════════════════════════════════════════════════════════════════════════

def _parse_seeds(s: str) -> list[int]:
    seeds = []
    for tok in (s or '').replace(';', ',').split(','):
        tok = tok.strip()
        if tok:
            seeds.append(int(tok))
    return seeds[:8] or [0]


def generate(seg_path, depth_path, gt_path, prompt_embed_state,
             seeds_str, num_steps, guidance_scale,
             run_baseline_seg, run_baseline_depth,
             progress=gr.Progress(track_tqdm=False)):
    """Main generation callback. Returns dict-like outputs for each gr.Image."""
    from pipeline.inference import sample_ours, sample_baseline

    empty_state = {
        'seg_rgb': None, 'depth_rgb': None, 'feature_rgb': None,
        'ours_tiles': [], 'baseline_seg_tiles': [], 'baseline_depth_tiles': [],
        'summary': None, 'seeds': [], 'flat_prompt': '',
        'gt_rgb': None,
    }

    def _ret(state, msg, *imgs):
        # Pack into UI outputs: seg, depth, feature, ours_grid, summary,
        # baseline_seg, baseline_depth, status, run_state
        return (*imgs, msg, state)

    if STATE.ours_transformer is None:
        if STATE.ckpt_dir is not None:
            progress(0.01, desc='Reloading ours transformer from last checkpoint')
            try:
                STATE.reload_ours_if_dropped(persistent_text_encoder=PERSISTENT_TEXT_ENCODER)
            except Exception as e:
                return _ret(empty_state, f'❌ Reload-ours failed: {e}',
                            None, None, None, None, None, None, None)
        else:
            return _ret(empty_state, '❌ Load a checkpoint first.',
                        None, None, None, None, None, None, None)
    if prompt_embed_state is None:
        return _ret(empty_state, '❌ Encode the prompt first.', None, None, None, None, None, None, None)
    if not seg_path or not depth_path:
        return _ret(empty_state, '❌ Need both seg and depth.', None, None, None, None, None, None, None)
    seg_path = seg_path['path'] if isinstance(seg_path, dict) else seg_path
    depth_path = depth_path['path'] if isinstance(depth_path, dict) else depth_path
    gt_path = (gt_path['path'] if isinstance(gt_path, dict) else gt_path) if gt_path else None

    seeds = _parse_seeds(seeds_str)
    size = STATE.image_size
    nc = STATE.num_classes

    progress(0.02, desc='Preprocessing')
    seg = preprocess_seg(seg_path, size, nc)
    depth = preprocess_depth(depth_path, size)
    seg_rgb = seg_to_rgb(seg, STATE.seg_palette)
    depth_rgb = depth_to_rgb(depth)
    gt_rgb = _to_uint8(preprocess_rgb(gt_path, size)) if gt_path else None

    B = len(seeds)
    prompt_embed_B = prompt_embed_state.to(DEVICE, DTYPE).unsqueeze(0).expand(B, -1, -1).contiguous()

    # Ours
    progress(0.05, desc=f'Ours: sampling {B} seeds × {num_steps} steps')
    STATE.lora_enable(True)
    t0 = time.time()
    ours_rgb, ctrl_ctx = sample_ours(
        seg.unsqueeze(0), depth.unsqueeze(0), prompt_embed_B,
        num_steps=int(num_steps), guidance_scale=float(guidance_scale),
        seeds=seeds, progress=None,
    )
    print(f'  ours: {time.time()-t0:.1f}s, VRAM={vram_gb():.1f}GB')
    ours_tiles = [_to_uint8(ours_rgb[i]) for i in range(B)]
    feature_rgb = feature_heatmap(ctrl_ctx, size)

    # Baselines (lazy load shared baseline transformer)
    baseline_seg_tiles = []
    baseline_depth_tiles = []
    if run_baseline_seg or run_baseline_depth:
        progress(0.4, desc='Loading vanilla baseline transformer (one-time)')
        STATE.load_baseline()

    if run_baseline_seg:
        progress(0.5, desc='Baseline (seg-only)')
        seg_rgb_tensor = torch.from_numpy(seg_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t0 = time.time()
        b_seg_rgb = sample_baseline(
            seg_rgb_tensor, prompt_embed_B,
            num_steps=int(num_steps), guidance_scale=float(guidance_scale),
            seeds=seeds, label='baseline-seg',
        )
        print(f'  baseline-seg: {time.time()-t0:.1f}s, VRAM={vram_gb():.1f}GB')
        baseline_seg_tiles = [_to_uint8(b_seg_rgb[i]) for i in range(B)]

    if run_baseline_depth:
        progress(0.75, desc='Baseline (depth-only)')
        depth_rgb_tensor = torch.from_numpy(depth_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t0 = time.time()
        b_dep_rgb = sample_baseline(
            depth_rgb_tensor, prompt_embed_B,
            num_steps=int(num_steps), guidance_scale=float(guidance_scale),
            seeds=seeds, label='baseline-depth',
        )
        print(f'  baseline-depth: {time.time()-t0:.1f}s, VRAM={vram_gb():.1f}GB')
        baseline_depth_tiles = [_to_uint8(b_dep_rgb[i]) for i in range(B)]

    # Per-seed grid (ours)
    progress(0.95, desc='Composing grids')
    ours_grid = build_panel_grid(
        [(f'seed={s}', tile) for s, tile in zip(seeds, ours_tiles)],
        title=f'Ours — {B} seed(s), {num_steps} steps, cfg={guidance_scale}',
    )

    # Summary: Seg | Depth | Feature | Ours[0] | BaseSeg[0]? | BaseDepth[0]? | GT
    panels = [
        ('Seg',          seg_rgb),
        ('Depth',        depth_rgb),
        ('HDC²A feat.',  feature_rgb),
        ('Ours (s0)',    ours_tiles[0]),
    ]
    if baseline_seg_tiles:
        panels.append(('Vanilla seg-only (s0)', baseline_seg_tiles[0]))
    if baseline_depth_tiles:
        panels.append(('Vanilla depth-only (s0)', baseline_depth_tiles[0]))
    panels.append(('GT RGB' if gt_rgb is not None else 'GT (n/a)', gt_rgb))
    summary = build_panel_grid(panels, title=Path(seg_path).stem, thumb=300)

    flat_prompt = ''
    try:
        flat_prompt = compose_prompt_from_json(json.loads(_LAST_PROMPT_JSON['text']))
    except Exception:
        pass

    state = {
        'seg_rgb': seg_rgb, 'depth_rgb': depth_rgb, 'feature_rgb': feature_rgb,
        'ours_tiles': ours_tiles,
        'baseline_seg_tiles': baseline_seg_tiles,
        'baseline_depth_tiles': baseline_depth_tiles,
        'summary': summary, 'seeds': seeds, 'flat_prompt': flat_prompt,
        'gt_rgb': gt_rgb,
        'seg_src': seg_path, 'depth_src': depth_path, 'rgb_src': gt_path,
    }
    msg = (f'✅ {B} seeds @ {size}×{size}, {num_steps} steps, cfg={guidance_scale}. '
           f'VRAM={vram_gb():.1f}GB.')
    return (
        seg_rgb, depth_rgb, feature_rgb,
        ours_grid,
        baseline_seg_tiles[0] if baseline_seg_tiles else None,
        baseline_depth_tiles[0] if baseline_depth_tiles else None,
        summary,
        msg, state,
    )


# Track most-recently encoded prompt JSON text so save_run can persist it.
_LAST_PROMPT_JSON = {'text': ''}


# ═════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_PROMPT_TEXT = (
    DEFAULT_PROMPT_JSON.read_text() if DEFAULT_PROMPT_JSON.is_file()
    else json.dumps({
        'scene': 'Aerial top-down satellite view of an urban area',
        'style': 'Google Earth style, photorealistic satellite imagery',
        'elements': {'buildings': 'mixed', 'roads': 'paved', 'vegetation': 'mixed'},
        'lighting': 'natural daylight',
        'quality': 'masterpiece, crisp details',
    }, indent=2)
)


def _header(ckpt_dir):
    name = ckpt_dir.name if ckpt_dir else '(none)'
    sz = getattr(STATE, 'image_size', '?')
    rank = getattr(STATE, 'lora_rank', '?')
    fusion = ('MLP' if getattr(STATE, 'use_mlp_fusion', False) else 'DoubleStream') \
        if hasattr(STATE, 'use_mlp_fusion') else '?'
    return (
        f'# HDC²A + Flux2 ControlNet — Self-contained Pipeline\n'
        f'**Checkpoint:** `{name}` · **Image:** {sz}×{sz} · '
        f'**LoRA rank:** {rank} · **Fusion:** {fusion} · '
        f'**VRAM:** {vram_gb():.1f}GB'
    )


def build_ui():
    ckpt_choices = [str(p.name) for p in list_lora_checkpoints()]
    with gr.Blocks(title='HDC²A + Flux2 — Generation Pipeline') as demo:
        prompt_embed_state = gr.State(None)
        scan_state = gr.State({'stems': [], 'by_stem': {}})
        run_state = gr.State({})
        header = gr.Markdown(_header(STATE.ckpt_dir))

        # ── 0. Checkpoint ──
        gr.Markdown('### 0. Checkpoint (under `weights/lora/`)')
        with gr.Row():
            ckpt_dd = gr.Dropdown(label='LoRA checkpoint', choices=ckpt_choices,
                                  value=STATE.ckpt_dir.name if STATE.ckpt_dir else (ckpt_choices[0] if ckpt_choices else None),
                                  interactive=True, scale=4)
            ckpt_refresh = gr.Button('🔄 Rescan', scale=1)
            ckpt_load = gr.Button('📦 Load', variant='primary', scale=1)
        ckpt_status = gr.Markdown('_(pick a checkpoint and click Load)_')

        # ── 1. Inputs ──
        gr.Markdown('### 1. Inputs — Seg, Depth, RGB reference')
        with gr.Row():
            with gr.Column(scale=1):
                root_box = gr.Textbox(label='Folder scan (parent with seg/ depth/ rgb/)',
                                      placeholder='../osm_pipeline/output/<city>')
                scan_btn = gr.Button('🔍 Scan')
            with gr.Column(scale=1):
                stem_dd = gr.Dropdown(label='Stem', choices=[], interactive=True,
                                      allow_custom_value=False)
                scan_status = gr.Markdown('_(enter a folder + click Scan, OR upload manually below)_')

        with gr.Row():
            with gr.Column():
                seg_in = gr.File(label='Seg', file_types=list(SUPPORTED_EXTS), type='filepath')
                seg_preview = gr.Image(label='Seg preview', interactive=False, height=240)
            with gr.Column():
                depth_in = gr.File(label='Depth', file_types=['.tif', '.tiff', '.png'], type='filepath')
                depth_preview = gr.Image(label='Depth preview', interactive=False, height=240)
            with gr.Column():
                gt_in = gr.File(label='RGB reference (optional)',
                                file_types=list(SUPPORTED_EXTS), type='filepath')
                gt_preview = gr.Image(label='RGB preview', interactive=False, height=240)
        sibling_status = gr.Markdown('_(upload one file → siblings auto-fill)_')

        # ── 2. Prompt ──
        gr.Markdown('### 2. Prompt')
        with gr.Row():
            with gr.Column():
                prompt_box = gr.Code(value=DEFAULT_PROMPT_TEXT, language='json',
                                     label='Prompt JSON', lines=14)
                encode_btn = gr.Button('🔤 Encode prompt', variant='secondary')
                prompt_status = gr.Markdown('_(not yet encoded)_')
            with gr.Column():
                gr.Markdown('**Diversity overlay** — pick axes then Apply to inject `diversity` into the JSON.')
                div_dd = {}
                keys = list(DIVERSITY_KEYS)
                for i in range(0, len(keys), 2):
                    with gr.Row():
                        for k in keys[i:i+2]:
                            div_dd[k] = gr.Dropdown(label=k, choices=DIVERSITY_OPTIONS[k],
                                                    value=NONE_OPT, interactive=True)
                apply_div = gr.Button('🎨 Apply diversity → JSON')

        # ── 3. Sampling + Comparison toggles ──
        gr.Markdown('### 3. Sampling & comparison')
        with gr.Row():
            seeds_box = gr.Textbox(value='0, 1', label='Seeds (comma-separated)')
            steps_sl = gr.Slider(4, 60, value=28, step=1, label='Euler steps')
            cfg_sl = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label='Guidance scale')
        with gr.Row():
            base_seg_chk = gr.Checkbox(value=True,
                label='Run vanilla Flux2 + Union ControlNet baseline — Seg-only (260-dim)')
            base_depth_chk = gr.Checkbox(value=True,
                label='Run vanilla baseline — Depth-only (260-dim)')

        gen_btn = gr.Button('🚀 Generate', variant='primary')
        gen_status = gr.Markdown('_(ready)_')

        # ── 4. Results ──
        gr.Markdown('### 4. Results — every image is independently downloadable')
        with gr.Row():
            seg_view = gr.Image(label='Seg (colorized)', interactive=False)
            depth_view = gr.Image(label='Depth (normalized)', interactive=False)
            feat_view = gr.Image(label='HDC²A feature heatmap', interactive=False)
        ours_view = gr.Image(label='Ours — per-seed grid', interactive=False)
        with gr.Row():
            base_seg_view = gr.Image(label='Vanilla Flux2 + Union CN — Seg-only (seed 0)', interactive=False)
            base_depth_view = gr.Image(label='Vanilla Flux2 + Union CN — Depth-only (seed 0)', interactive=False)
        summary_view = gr.Image(label='Side-by-side summary', interactive=False)

        # ── 5. Save ──
        gr.Markdown('### 5. Save run → `output/<folder_name>/`')
        with gr.Row():
            run_name_box = gr.Textbox(label='Folder name', value=f'run_{time.strftime("%Y%m%d_%H%M%S")}',
                                      placeholder='my_experiment_01')
            save_btn = gr.Button('💾 Save all results to output/', variant='primary')
        save_status = gr.Markdown('_(generate first, then save)_')

        # ─── Wiring ───────────────────────────────────────────────────────

        def _ckpt_refresh_cb():
            return gr.update(choices=[p.name for p in list_lora_checkpoints()])

        def _ckpt_load_cb(name):
            if not name:
                return gr.update(), '❌ pick a checkpoint', _header(STATE.ckpt_dir)
            from pipeline import LORA_DIR
            path = LORA_DIR / name
            if not (path / 'meta.pt').is_file():
                return gr.update(), f'❌ `{name}` is not a valid checkpoint', _header(STATE.ckpt_dir)
            try:
                STATE.load(path, persistent_text_encoder=PERSISTENT_TEXT_ENCODER)
            except Exception as e:
                import traceback; traceback.print_exc()
                return gr.update(), f'❌ load failed: {e}', _header(STATE.ckpt_dir)
            return None, f'✅ Loaded `{name}`. VRAM={vram_gb():.1f}GB', _header(STATE.ckpt_dir)

        ckpt_refresh.click(_ckpt_refresh_cb, outputs=[ckpt_dd])
        ckpt_load.click(_ckpt_load_cb, inputs=[ckpt_dd],
                        outputs=[prompt_embed_state, ckpt_status, header])

        # Prompt encoding
        def _encode_cb(text):
            _LAST_PROMPT_JSON['text'] = text
            embed, _flat, msg = encode_prompt_ui(text)
            return embed, msg
        encode_btn.click(_encode_cb, inputs=[prompt_box],
                         outputs=[prompt_embed_state, prompt_status])
        apply_div.click(merge_diversity,
                        inputs=[prompt_box] + [div_dd[k] for k in DIVERSITY_KEYS],
                        outputs=[prompt_box, prompt_status])

        # File previews
        def _preview(p, kind):
            return make_preview(p, kind, STATE.image_size, STATE.num_classes,
                                getattr(STATE, 'seg_palette', None))
        seg_in.change(lambda p: _preview(p, 'seg'),   inputs=[seg_in],   outputs=[seg_preview])
        depth_in.change(lambda p: _preview(p, 'depth'), inputs=[depth_in], outputs=[depth_preview])
        gt_in.change(lambda p: _preview(p, 'rgb'),    inputs=[gt_in],    outputs=[gt_preview])

        # Folder scan
        def _scan_cb(root):
            res = scan_root_folder(root)
            return res, gr.update(choices=res['stems'], value=None), res['status']
        scan_btn.click(_scan_cb, inputs=[root_box],
                       outputs=[scan_state, stem_dd, scan_status])

        def _pick_cb(stem, state):
            if not stem or stem not in (state or {}).get('by_stem', {}):
                return gr.update(), gr.update(), gr.update(), '_(pick a stem)_', None, None, None
            paths = state['by_stem'][stem]
            return (paths['seg'], paths['depth'], paths['rgb'],
                    f'✅ loaded `{stem}`',
                    _preview(paths['seg'], 'seg'),
                    _preview(paths['depth'], 'depth'),
                    _preview(paths['rgb'], 'rgb'))
        stem_dd.change(_pick_cb, inputs=[stem_dd, scan_state],
                       outputs=[seg_in, depth_in, gt_in, sibling_status,
                                seg_preview, depth_preview, gt_preview])

        # Sibling auto-resolve on upload (only for server-side paths)
        def _make_upload(kind):
            others = [k for k in ('seg', 'depth', 'rgb') if k != kind]
            def _cb(path, state):
                if isinstance(path, dict):
                    path = path.get('path') or path.get('name') or path.get('tmp_path')
                if not path:
                    return gr.update(), gr.update(), '_(cleared)_', None, None
                p = Path(path)
                if p.parent.name.lower() == kind:
                    res = resolve_siblings(str(path), kind)
                    vals = [res[k] for k in others]
                    prev = [_preview(v, k) if v else None for v, k in zip(vals, others)]
                    return (vals[0] or gr.update(), vals[1] or gr.update(),
                            f'✅ Auto-filled siblings from `{p.parent.parent}`',
                            prev[0], prev[1])
                stem = p.stem
                by_stem = (state or {}).get('by_stem', {})
                if stem in by_stem:
                    paths = by_stem[stem]
                    vals = [paths.get(k) for k in others]
                    prev = [_preview(v, k) if v else None for v, k in zip(vals, others)]
                    return (vals[0] or gr.update(), vals[1] or gr.update(),
                            f'✅ Matched stem `{stem}` from scanned root',
                            prev[0], prev[1])
                return gr.update(), gr.update(), '_(no siblings found)_', None, None
            return _cb
        seg_in.upload(_make_upload('seg'),     inputs=[seg_in, scan_state],
                      outputs=[depth_in, gt_in, sibling_status, depth_preview, gt_preview])
        depth_in.upload(_make_upload('depth'), inputs=[depth_in, scan_state],
                        outputs=[seg_in, gt_in, sibling_status, seg_preview, gt_preview])
        gt_in.upload(_make_upload('rgb'),      inputs=[gt_in, scan_state],
                     outputs=[seg_in, depth_in, sibling_status, seg_preview, depth_preview])

        # Generate
        gen_btn.click(
            generate,
            inputs=[seg_in, depth_in, gt_in, prompt_embed_state,
                    seeds_box, steps_sl, cfg_sl, base_seg_chk, base_depth_chk],
            outputs=[seg_view, depth_view, feat_view,
                     ours_view, base_seg_view, base_depth_view, summary_view,
                     gen_status, run_state],
        )

        # Save
        def _save_cb(name, state, prompt_text, num_steps, cfg, base_seg, base_depth):
            if not state or not state.get('ours_tiles'):
                return '❌ Nothing to save — generate first.'
            try:
                run_dir, files = save_run(
                    name,
                    seg_src_path=state.get('seg_src'),
                    depth_src_path=state.get('depth_src'),
                    rgb_src_path=state.get('rgb_src'),
                    seg_preview=state.get('seg_rgb'),
                    depth_preview=state.get('depth_rgb'),
                    rgb_preview=state.get('gt_rgb'),
                    feature_preview=state.get('feature_rgb'),
                    ours_tiles=state.get('ours_tiles'),
                    baseline_seg_tiles=state.get('baseline_seg_tiles'),
                    baseline_depth_tiles=state.get('baseline_depth_tiles'),
                    summary_grid=state.get('summary'),
                    prompt_json_text=prompt_text,
                    flat_prompt=state.get('flat_prompt', ''),
                    seeds=state.get('seeds', []),
                    metadata_extra={
                        'checkpoint': STATE.ckpt_dir.name if STATE.ckpt_dir else None,
                        'image_size': STATE.image_size,
                        'lora_rank': STATE.lora_rank,
                        'fusion': 'MLP' if STATE.use_mlp_fusion else 'DoubleStream',
                        'num_steps': int(num_steps),
                        'guidance_scale': float(cfg),
                        'baseline_seg_only': bool(base_seg),
                        'baseline_depth_only': bool(base_depth),
                    },
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                return f'❌ save failed: {e}'
            return (f'✅ Saved {len(files)} files to `output/{run_dir.name}/`\n\n'
                    f'`{run_dir}`')
        save_btn.click(
            _save_cb,
            inputs=[run_name_box, run_state, prompt_box, steps_sl, cfg_sl,
                    base_seg_chk, base_depth_chk],
            outputs=[save_status],
        )

    return demo



# ═════════════════════════════════════════════════════════════════════════════
# CLI compare mode for paper figures
# ═════════════════════════════════════════════════════════════════════════════

NEAR_NADIR_RE = __import__('re').compile(r'^near[-_]nadir[-_](\d+)$', __import__('re').IGNORECASE)


def _tile_sort_key(path: Path):
    import math as _math
    import re as _re
    m = _re.search(r'(\d+)$', path.name)
    return (int(m.group(1)) if m else _math.inf, path.name)


def _discover_tile_dirs(input_dir: Path) -> list[Path]:
    return sorted(
        (p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith('tile_')),
        key=_tile_sort_key,
    )


def _discover_near_dirs(tile_dir: Path) -> list[Path]:
    dirs = [p for p in tile_dir.iterdir() if p.is_dir() and NEAR_NADIR_RE.match(p.name)]
    return sorted(dirs, key=lambda p: int(NEAR_NADIR_RE.match(p.name).group(1)))


def _parse_batch_arg(batch: str | None) -> tuple[int | None, int | None]:
    if not batch:
        return None, None
    parts = [x.strip() for x in batch.split(',')]
    if len(parts) != 2 or not all(parts):
        raise ValueError('--compare-batch must look like START,END, e.g. 0,10')
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end < start:
        raise ValueError('--compare-batch expects 0 <= START <= END')
    return start, end


def _selected_compare_tiles(input_dir: Path, args) -> list[Path]:
    tiles = _discover_tile_dirs(input_dir)
    if args.compare_tile_names:
        wanted = set(args.compare_tile_names)
        tiles = [t for t in tiles if t.name in wanted]
    start, end = _parse_batch_arg(args.compare_batch)
    if start is not None:
        tiles = tiles[start:end]
    if args.compare_random_tiles:
        import random as _random
        rng = _random.Random(args.compare_random_seed)
        n = min(int(args.compare_random_tiles), len(tiles))
        tiles = sorted(rng.sample(tiles, n), key=_tile_sort_key)
    return tiles


def _replace_suffix(name: str, suffix: str) -> str:
    return str(Path(name).with_suffix(suffix))


def _compare_paths(tile_dir: Path, args) -> tuple[str, Path, Path, Path | None]:
    depth_suffix = '.exr' if args.compare_depth_exr else '.png'
    if args.compare_near_nadir is not None:
        view = f'near-nadir-{args.compare_near_nadir}'
        view_dir = tile_dir / view
        seg_path = view_dir / args.compare_sub_seg_name
        depth_path = view_dir / _replace_suffix(args.compare_sub_depth_name, depth_suffix)
    elif args.compare_near_nadir_random:
        import random as _random
        near_dirs = _discover_near_dirs(tile_dir)
        if not near_dirs:
            raise FileNotFoundError(f'no near-nadir-* dirs under {tile_dir}')
        tile_key = _tile_sort_key(tile_dir)[0]
        tile_seed = int(tile_key) if tile_key != float('inf') else abs(hash(tile_dir.name))
        view_dir = _random.Random(args.compare_random_seed + tile_seed).choice(near_dirs)
        view = view_dir.name
        seg_path = view_dir / args.compare_sub_seg_name
        depth_path = view_dir / _replace_suffix(args.compare_sub_depth_name, depth_suffix)
    else:
        view = 'root'
        seg_path = tile_dir / args.compare_seg_name
        depth_path = tile_dir / _replace_suffix(args.compare_depth_name, depth_suffix)
    rgb_path = tile_dir / args.compare_rgb_name
    return view, seg_path, depth_path, rgb_path if rgb_path.exists() else None


def _save_compare_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    Image.fromarray(arr[..., :3]).save(path)


@torch.no_grad()
def run_compare_one(tile_dir: Path, args, prompt_embed: torch.Tensor, out_root: Path) -> dict:
    from pipeline.inference import sample_ours, sample_baseline

    view, seg_path, depth_path, rgb_path = _compare_paths(tile_dir, args)
    for label, path in (('seg', seg_path), ('depth', depth_path)):
        if not path.exists():
            raise FileNotFoundError(f'missing {label}: {path}')

    STATE.reload_ours_if_dropped(persistent_text_encoder=True)

    size = STATE.image_size
    seed = int(args.compare_seed)
    seg = preprocess_seg(str(seg_path), size, STATE.num_classes)
    depth = preprocess_depth(str(depth_path), size) if depth_path.suffix.lower() != '.exr' else _load_compare_exr(depth_path, size)
    seg_rgb = seg_to_rgb(seg, STATE.seg_palette)
    depth_rgb = depth_to_rgb(depth)
    osm_rgb = _to_uint8(preprocess_rgb(str(rgb_path), size)) if rgb_path else None

    prompt_B = prompt_embed.to(DEVICE, DTYPE).unsqueeze(0)

    STATE.lora_enable(True)
    ours_rgb, ctrl_ctx = sample_ours(
        seg.unsqueeze(0), depth.unsqueeze(0), prompt_B,
        num_steps=int(args.compare_steps), guidance_scale=float(args.compare_cfg),
        seeds=[seed], progress=None,
    )
    synth_rgb = _to_uint8(ours_rgb[0])
    feature_rgb = feature_heatmap(ctrl_ctx, size)

    STATE.lora_enable(False)
    nolora_rgb, _ = sample_ours(
        seg.unsqueeze(0), depth.unsqueeze(0), prompt_B,
        num_steps=int(args.compare_steps), guidance_scale=float(args.compare_cfg),
        seeds=[seed], progress=None,
    )
    nolora_rgb = _to_uint8(nolora_rgb[0])
    STATE.lora_enable(True)

    STATE.load_baseline()
    seg_rgb_tensor = torch.from_numpy(seg_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    depth_rgb_tensor = torch.from_numpy(depth_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    base_seg_rgb = sample_baseline(
        seg_rgb_tensor, prompt_B,
        num_steps=int(args.compare_steps), guidance_scale=float(args.compare_cfg),
        seeds=[seed], label='compare-seg-only',
    )
    base_depth_rgb = sample_baseline(
        depth_rgb_tensor, prompt_B,
        num_steps=int(args.compare_steps), guidance_scale=float(args.compare_cfg),
        seeds=[seed], label='compare-depth-only',
    )
    base_seg_rgb = _to_uint8(base_seg_rgb[0])
    base_depth_rgb = _to_uint8(base_depth_rgb[0])

    depth_tag = 'exr' if args.compare_depth_exr else 'png'
    out_dir = out_root / tile_dir.name / view / f'depth_{depth_tag}' / f'seed_{seed:04d}'
    images = {
        'osm_rgb': osm_rgb,
        'seg': seg_rgb,
        'depth': depth_rgb,
        'hdc2a_feature': feature_rgb,
        'synth_rgb': synth_rgb,
        'without_lora': nolora_rgb,
        'seg_only': base_seg_rgb,
        'depth_only': base_depth_rgb,
    }
    for name, arr in images.items():
        if arr is not None:
            _save_compare_png(arr, out_dir / f'{name}.png')

    grid = build_panel_grid(
        [('OSM sat RGB', osm_rgb), ('Seg', seg_rgb), ('Depth', depth_rgb),
         ('HDC2A feature', feature_rgb), ('Synth RGB', synth_rgb),
         ('Without LoRA', nolora_rgb), ('Seg only', base_seg_rgb), ('Depth only', base_depth_rgb)],
        thumb=280,
        title=f'{tile_dir.name}/{view} seed={seed}',
    )
    _save_compare_png(grid, out_dir / 'grid.png')

    metadata = {
        'tile': tile_dir.name,
        'view': view,
        'seed': seed,
        'depth_tag': depth_tag,
        'seg_path': str(seg_path),
        'depth_path': str(depth_path),
        'osm_rgb_path': str(rgb_path) if rgb_path else None,
        'outputs': {name: str(out_dir / f'{name}.png') for name, arr in images.items() if arr is not None},
        'grid': str(out_dir / 'grid.png'),
        'num_steps': int(args.compare_steps),
        'cfg': float(args.compare_cfg),
        'checkpoint': STATE.ckpt_dir.name if STATE.ckpt_dir else None,
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def _load_compare_exr(path: Path, size: int) -> torch.Tensor:
    import OpenEXR
    with OpenEXR.File(str(path)) as f:
        channels = f.channels()
        key = 'V' if 'V' in channels else next(iter(channels))
        arr = channels[key].pixels.astype(np.float32)
    while arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] <= arr.shape[0] else arr[0]
    if np.isinf(arr).any():
        finite = arr[np.isfinite(arr)]
        arr = np.where(np.isinf(arr), float(finite.max()) if finite.size else 0.0, arr)
    arr = np.nan_to_num(arr, nan=0.0)
    if arr.shape != (size, size):
        arr = np.array(Image.fromarray(arr).resize((size, size), Image.LANCZOS), dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def run_compare_cli(args) -> int:
    input_dir = Path(args.compare_osm_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f'--compare-osm-dir not found: {input_dir}')
    if args.compare_near_nadir is not None and args.compare_near_nadir_random:
        raise ValueError('Use only one of --compare-near-nadir and --compare-near-nadir-random')

    missing = verify_base_weights()
    if missing:
        raise FileNotFoundError(f'Missing base weights: {missing}')

    if args.compare_ckpt:
        ckpt_dir = (_HERE / 'weights' / 'lora' / args.compare_ckpt).resolve()
        if not (ckpt_dir / 'meta.pt').is_file():
            raise FileNotFoundError(f'checkpoint not found: {ckpt_dir}')
    else:
        ckpts = list_lora_checkpoints()
        if not ckpts:
            raise FileNotFoundError('No checkpoints under weights/lora')
        ckpt_dir = ckpts[0]

    out_root = (Path(args.compare_out).expanduser().resolve() if args.compare_out
                else OUTPUT_DIR / f'{input_dir.name}-compare')
    out_root.mkdir(parents=True, exist_ok=True)

    STATE.load(ckpt_dir, persistent_text_encoder=True)
    prompt_text_json = Path(args.compare_prompt_json).expanduser().read_text()
    _LAST_PROMPT_JSON['text'] = prompt_text_json
    prompt_embed, flat_prompt, msg = encode_prompt_ui(prompt_text_json)
    if prompt_embed is None:
        raise RuntimeError(msg)
    print(msg)

    if not args.compare_tile_names and not args.compare_batch and not args.compare_random_tiles:
        raise ValueError('Compare mode needs a tile selection: use --compare-tile-names, --compare-batch, or --compare-random-tiles')

    tiles = _selected_compare_tiles(input_dir, args)
    if not tiles:
        raise RuntimeError('No selected tile folders')

    results = []
    for i, tile_dir in enumerate(tiles, start=1):
        print(f'[{i}/{len(tiles)}] compare {tile_dir.name}')
        try:
            results.append(run_compare_one(tile_dir, args, prompt_embed, out_root))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [ERR] {tile_dir.name}: {e}')

    manifest = {
        'input_dir': str(input_dir),
        'checkpoint': str(ckpt_dir),
        'flat_prompt': flat_prompt,
        'out_root': str(out_root),
        'tile_names': args.compare_tile_names,
        'batch': args.compare_batch,
        'random_tiles': int(args.compare_random_tiles),
        'seed': int(args.compare_seed),
        'num_steps': int(args.compare_steps),
        'cfg': float(args.compare_cfg),
        'items': results,
    }
    (out_root / 'manifest_compare.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f'Done. Compare outputs: {out_root}')
    return 0


# ═════════════════════════════════════════════════════════════════════════════
PERSISTENT_TEXT_ENCODER = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-textencoder', action='store_true',
                        help='Load/unload Mistral per encode call instead of keeping resident')
    parser.add_argument('--no-preload', action='store_true',
                        help='Do not preload a checkpoint on startup')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 7860)))
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--share', action='store_true')

    # Non-WebUI paper-figure compare mode.
    parser.add_argument('--compare-osm-dir', default=None,
                        help='Run compare mode on an OSM output folder instead of launching WebUI')
    parser.add_argument('--compare-tile-names', nargs='+', default=[],
                        help='Specific tile folders for compare mode, e.g. tile_0001 tile_0002')
    parser.add_argument('--compare-batch', default=None,
                        help='Half-open sorted tile range START,END for compare mode')
    parser.add_argument('--compare-random-tiles', type=int, default=0,
                        help='Randomly sample this many tiles for compare mode after tile/batch filtering')
    parser.add_argument('--compare-near-nadir', type=int, default=None,
                        help='Use fixed near-nadir-N view for compare mode')
    parser.add_argument('--compare-near-nadir-random', action='store_true',
                        help='Choose one near-nadir-* view per tile for compare mode')
    parser.add_argument('--compare-random-seed', type=int, default=0)
    parser.add_argument('--compare-depth-exr', action='store_true')
    parser.add_argument('--compare-seed', type=int, default=0,
                        help='Single seed to use for paper compare figures')
    parser.add_argument('--compare-steps', type=int, default=28)
    parser.add_argument('--compare-cfg', type=float, default=3.5)
    parser.add_argument('--compare-ckpt', default=None)
    parser.add_argument('--compare-prompt-json', default=str(DEFAULT_PROMPT_JSON))
    parser.add_argument('--compare-out', default=None)
    parser.add_argument('--compare-rgb-name', default='2_rgb.png')
    parser.add_argument('--compare-seg-name', default='4_seg.png')
    parser.add_argument('--compare-depth-name', default='5_depth.png')
    parser.add_argument('--compare-sub-seg-name', default='1_seg.png')
    parser.add_argument('--compare-sub-depth-name', default='2_depth.png')
    args = parser.parse_args()

    PERSISTENT_TEXT_ENCODER = not args.no_textencoder

    if args.compare_osm_dir:
        try:
            sys.exit(run_compare_cli(args))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'❌ compare failed: {e}')
            sys.exit(1)

    missing = verify_base_weights()
    if missing:
        print('⚠️  Missing base weights — please run setup.sh first:')
        for m in missing:
            print(f'    - {m}')
        print('  Continuing anyway (UI will report errors when you try to load).')

    if not args.no_preload:
        ckpts = list_lora_checkpoints()
        if ckpts:
            try:
                STATE.load(ckpts[0], persistent_text_encoder=PERSISTENT_TEXT_ENCODER)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f'⚠️  Preload failed: {e}')
                print('  Launching UI anyway; pick a checkpoint manually.')
        else:
            print('⚠️  No checkpoints under weights/lora/. Run setup.sh to fetch one.')

    demo = build_ui()
    demo.queue(max_size=4).launch(
        server_name=args.host, server_port=args.port,
        show_error=True, share=args.share, theme=gr.themes.Soft(),
    )

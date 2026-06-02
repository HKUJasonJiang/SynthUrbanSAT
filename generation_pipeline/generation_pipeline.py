"""Folder-of-tiles inference driver.

Given a root directory like:

    <input-dir>/
        tile_0001/
            2_rgb.png        # GT RGB (reference, not used as input)
            4_seg.png        # 6-class colored segmentation
            5_depth.png      # uint8 grayscale depth
            5_depth.exr      # float32 metric depth (single 'V' channel)
            ...
        tile_0002/
            ...

run each tile through the same HDC²A + Flux2 ControlNet pipeline used by
``app.py`` / ``batch_eval.py`` (STATE.load + sample_ours), once per depth
source requested via ``--depth-ext`` (``png``, ``exr``, or ``both``), and
save one PNG per tile×depth-ext combo containing:

    Seg | Depth | seed_0 | seed_1 | ... | GT RGB

Defaults to ``--depth-ext both`` so 20 tiles × 2 depth versions = 40 grids.

Example:
    cd generation_pipeline
    python -u generation_pipeline.py \\
        --input-dir ../osm_pipeline/output/omaha_rich_test \\
        --depth-ext both --seeds 0 42 2>&1 | tee /tmp/gen_pipeline.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import DEFAULT_PROMPT_JSON, list_lora_checkpoints, verify_base_weights
from pipeline.preprocess import preprocess_depth, preprocess_rgb, preprocess_seg, depth_to_rgb, seg_to_rgb
from pipeline.state import DEVICE, DTYPE, STATE, vram_gb
from pipeline.inference import sample_ours

# Reuse batch_eval helpers (compose_prompt, encode_prompt, build_row, to_uint8).
from batch_eval import build_row, compose_prompt_from_json, encode_prompt, to_uint8


# ─── EXR depth loader (mirrors preprocess_depth normalization) ───────────────
def load_depth_exr(path: str, size: int) -> torch.Tensor:
    """Load a single-channel float32 EXR depth -> [1, H, W] tensor in [0, 1]."""
    import OpenEXR

    with OpenEXR.File(path) as f:
        channels = f.channels()
        # Prefer 'V' (Blender's depth output); else first channel.
        key = 'V' if 'V' in channels else next(iter(channels))
        arr = channels[key].pixels.astype(np.float32)

    while arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] <= arr.shape[0] else arr[0]

    # Replace inf (Blender sky) with finite max before normalization.
    if np.isinf(arr).any():
        finite_max = float(arr[np.isfinite(arr)].max()) if np.isfinite(arr).any() else 0.0
        arr = np.where(np.isinf(arr), finite_max, arr)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    if arr.shape != (size, size):
        arr_pil = Image.fromarray(arr).resize((size, size), Image.LANCZOS)
        arr = np.array(arr_pil, dtype=np.float32)

    mn, mx = float(arr.min()), float(arr.max())
    arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
    return torch.from_numpy(arr).unsqueeze(0)


def load_depth(path: Path, size: int) -> torch.Tensor:
    if path.suffix.lower() == '.exr':
        return load_depth_exr(str(path), size)
    return preprocess_depth(str(path), size)


# ─── Per-tile run ────────────────────────────────────────────────────────────
@torch.no_grad()
def run_tile(tile_dir: Path, rgb_name: str, seg_name: str, depth_name: str,
             depth_tag: str, prompt_embed: torch.Tensor,
             seeds: list[int], num_steps: int, cfg: float, out_dir: Path,
             sub_dir: str | None = None):
    # When sub_dir is given, seg + depth come from <tile_dir>/<sub_dir>/
    input_root = tile_dir / sub_dir if sub_dir else tile_dir
    rgb_path = tile_dir / rgb_name      # GT always from tile root
    seg_path = input_root / seg_name
    depth_path = input_root / depth_name
    for p, label in ((rgb_path, 'rgb'), (seg_path, 'seg'), (depth_path, 'depth')):
        if not p.exists():
            label_extra = f' (in {sub_dir})' if sub_dir and label != 'rgb' else ''
            print(f'  [SKIP] {tile_dir.name} ({depth_tag}): missing {label}{label_extra}: {p}')
            return

    size = STATE.image_size
    nc = STATE.num_classes
    seg = preprocess_seg(str(seg_path), size, nc)
    depth = load_depth(depth_path, size)
    gt_rgb_t = preprocess_rgb(str(rgb_path), size)

    seg_rgb = seg_to_rgb(seg, STATE.seg_palette)
    depth_rgb = depth_to_rgb(depth)
    gt_rgb = to_uint8(gt_rgb_t)

    B = len(seeds)
    prompt_B = prompt_embed.to(DEVICE, DTYPE).unsqueeze(0).expand(B, -1, -1).contiguous()
    STATE.lora_enable(True)
    ours_rgb, _ = sample_ours(
        seg.unsqueeze(0), depth.unsqueeze(0), prompt_B,
        num_steps=int(num_steps), guidance_scale=float(cfg),
        seeds=seeds, progress=None,
    )
    tiles = [to_uint8(ours_rgb[i]) for i in range(B)]

    panels = [('Seg', seg_rgb), (f'Depth ({depth_tag})', depth_rgb)]
    panels += [(f'seed={s}', tiles[i]) for i, s in enumerate(seeds)]
    panels.append(('GT RGB', gt_rgb))

    sub_label = f'{sub_dir}__' if sub_dir else ''
    img = build_row(
        panels, thumb=320, gap=8,
        title=(f'{tile_dir.name}/{sub_dir if sub_dir else ""} — depth={depth_tag} — '
               f'{num_steps} steps, cfg={cfg}, VRAM={vram_gb():.1f} GiB'),
    )
    out_path = out_dir / f'{tile_dir.name}__{sub_label}depth{depth_tag}.png'
    img.save(out_path)
    print(f'  ✓ {out_path}')


# ─── Main ────────────────────────────────────────────────────────────────────
def discover_tiles(input_dir: Path) -> list[Path]:
    tiles = sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith('tile_'))
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True,
                    help='Root directory containing tile_XXXX subfolders')
    ap.add_argument('--depth-ext', choices=['png', 'exr', 'both'], default='both',
                    help='Which depth file(s) to use per tile')
    ap.add_argument('--rgb-name', default='2_rgb.png')
    ap.add_argument('--seg-name', default='4_seg.png')
    ap.add_argument('--depth-png-name', default='5_depth.png')
    ap.add_argument('--depth-exr-name', default='5_depth.exr')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 42])
    ap.add_argument('--num-steps', type=int, default=28)
    ap.add_argument('--cfg', type=float, default=3.5)
    ap.add_argument('--ckpt', default=None,
                    help='Checkpoint dir name under weights/lora (default: latest)')
    ap.add_argument('--prompt-json', default=str(DEFAULT_PROMPT_JSON),
                    help='Path to prompt JSON; default uses pipeline default')
    ap.add_argument('--out', default=None,
                    help='Output directory (default: output/genpipe__<input-name>__<ckpt-name>)')
    ap.add_argument('--limit', type=int, default=0,
                    help='If >0, only process the first N tiles (for testing)')
    ap.add_argument('--tile-names', nargs='+', default=[],
                    help='If given, only process these tile folder names (e.g. tile_0019 tile_0020)')
    ap.add_argument('--sub-dirs', nargs='*', default=[],
                    help='If given, read seg+depth from these subdirs of each tile '
                         '(e.g. near-nadir-1 near-nadir-2). GT RGB still from tile root.')
    ap.add_argument('--sub-seg-name', default='1_seg.png',
                    help='Seg filename inside each --sub-dir (default: 1_seg.png)')
    ap.add_argument('--sub-depth-name', default='2_depth.png',
                    help='Depth filename inside each --sub-dir (default: 2_depth.png)')
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f'❌ --input-dir not found: {input_dir}')
        sys.exit(1)

    tiles = discover_tiles(input_dir)
    if args.tile_names:
        tiles = [t for t in tiles if t.name in args.tile_names]
    if args.limit > 0:
        tiles = tiles[:args.limit]
    if not tiles:
        print(f'❌ no tile_XXXX subfolders under {input_dir}')
        sys.exit(1)
    print(f'Found {len(tiles)} tile folder(s) under {input_dir}')

    missing = verify_base_weights()
    if missing:
        print('MISSING base weights:', missing)
        sys.exit(1)

    if args.ckpt:
        ckpt_dir = (HERE / 'weights' / 'lora' / args.ckpt).resolve()
    else:
        cks = list_lora_checkpoints()
        if not cks:
            print('No checkpoints in weights/lora')
            sys.exit(1)
        ckpt_dir = cks[0]
    print(f'Using checkpoint: {ckpt_dir.name}')

    out_dir = (Path(args.out) if args.out
               else HERE / 'output' / f'genpipe__{input_dir.name}__{ckpt_dir.name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {out_dir}\n')

    STATE.load(ckpt_dir, persistent_text_encoder=True)

    prompt_obj = json.loads(Path(args.prompt_json).read_text())
    prompt_text = compose_prompt_from_json(prompt_obj)
    print(f'\nPrompt ({len(prompt_text)} chars): {prompt_text[:160]}...\n')
    prompt_embed = encode_prompt(prompt_text)
    print(f'Encoded prompt: {tuple(prompt_embed.shape)}, VRAM={vram_gb():.1f} GiB\n')

    # Build run specs: list of (sub_dir_or_None, seg_name, depth_tag, depth_name)
    if args.sub_dirs:
        run_specs = [
            (sub, args.sub_seg_name, 'png', args.sub_depth_name)
            for sub in args.sub_dirs
        ]
    else:
        if args.depth_ext == 'both':
            depth_specs = [('png', args.depth_png_name), ('exr', args.depth_exr_name)]
        elif args.depth_ext == 'png':
            depth_specs = [('png', args.depth_png_name)]
        else:
            depth_specs = [('exr', args.depth_exr_name)]
        run_specs = [(None, args.seg_name, tag, dname) for tag, dname in depth_specs]

    total = len(tiles) * len(run_specs)
    done = 0
    for tile_dir in tiles:
        for sub_dir, seg_name, depth_tag, depth_name in run_specs:
            done += 1
            label = f'{tile_dir.name}/{sub_dir}' if sub_dir else tile_dir.name
            print(f'\n[{done}/{total}] === {label} (depth={depth_tag}) ===')
            try:
                run_tile(tile_dir, args.rgb_name, seg_name, depth_name,
                         depth_tag, prompt_embed, args.seeds,
                         args.num_steps, args.cfg, out_dir, sub_dir=sub_dir)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f'  [ERR] {label} ({depth_tag}): {e}')

    print(f'\nAll done. PNGs in {out_dir}')


if __name__ == '__main__':
    main()

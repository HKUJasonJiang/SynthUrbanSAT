"""Batch sanity-check inference: 5 train + 5 val samples × multiple seeds.

Reuses the existing pipeline (STATE.load + sample_ours) so behaviour is
identical to the Gradio app. Writes one PNG per stem with the layout:

    Seg | Depth | seed0 | seed1 | ... | seedN | GT
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import (
    COLOR_MAP_PATH, DEFAULT_PROMPT_JSON, list_lora_checkpoints,
    verify_base_weights,
)
from pipeline.preprocess import (
    preprocess_depth, preprocess_rgb, preprocess_seg, depth_to_rgb, seg_to_rgb,
)
from pipeline.state import DEVICE, DTYPE, STATE, vram_gb
from pipeline.inference import sample_ours


# ─── train_script compose_prompt_from_json equivalent ────────────────────────
def compose_prompt_from_json(obj: dict) -> str:
    parts = []
    if 'scene' in obj:    parts.append(obj['scene'])
    if 'style' in obj:    parts.append(obj['style'])
    if 'elements' in obj:
        for k, v in (obj.get('elements') or {}).items():
            parts.append(f'{k}: {v}')
    if 'lighting' in obj: parts.append(obj['lighting'])
    if 'quality' in obj:  parts.append(obj['quality'])
    return ', '.join(p for p in parts if p)


def encode_prompt(prompt_text: str) -> torch.Tensor:
    from scripts.text_encoder import encode_prompts
    embed = encode_prompts(
        STATE.text_encoder, STATE.tokenizer, [prompt_text],
        max_sequence_length=int(STATE.cfg.get('text_seq_len', 512)),
        device=DEVICE, dtype=DTYPE,
    )
    return embed[0].detach().clone()


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def get_font(size: int = 20):
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def build_row(panels, *, thumb=320, gap=10, sub_h=32, title=None, title_h=44):
    n = len(panels)
    W = n * thumb + (n - 1) * gap
    top = title_h if title else 0
    H = top + sub_h + thumb
    img = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if title:
        draw.rectangle([0, 0, W, title_h], fill=(35, 35, 38))
        f = get_font(22)
        bbox = draw.textbbox((0, 0), title, font=f)
        draw.text(((W - (bbox[2] - bbox[0])) // 2,
                   (title_h - (bbox[3] - bbox[1])) // 2),
                  title, fill=(235, 235, 240), font=f)
    f_sub = get_font(18)
    for i, (label, arr) in enumerate(panels):
        x0 = i * (thumb + gap)
        draw.rectangle([x0, top, x0 + thumb, top + sub_h], fill=(50, 50, 55))
        bbox = draw.textbbox((0, 0), label, font=f_sub)
        draw.text((x0 + (thumb - (bbox[2] - bbox[0])) // 2,
                   top + (sub_h - (bbox[3] - bbox[1])) // 2),
                  label, fill=(230, 230, 235), font=f_sub)
        if arr is None:
            arr = np.full((thumb, thumb, 3), 80, dtype=np.uint8)
        tile = Image.fromarray(arr).resize((thumb, thumb), Image.LANCZOS)
        img.paste(tile, (x0, top + sub_h))
    return img


def find_companion(stem: str, split_dir: Path, subdir: str) -> Path:
    base = split_dir / subdir
    for try_stem in (stem, stem.replace('_RGB_', '_')):
        for ext in ('.png', '.tif', '.tiff', '.jpg', '.jpeg'):
            p = base / f'{try_stem}{ext}'
            if p.exists():
                return p
    raise FileNotFoundError(f'no {subdir} companion for {stem} in {base}')


@torch.no_grad()
def run_one(stem: str, split_dir: Path, prompt_embed: torch.Tensor,
            seeds: list[int], num_steps: int, cfg: float, out_dir: Path):
    rgb_path = None
    for ext in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
        c = split_dir / 'rgb' / f'{stem}{ext}'
        if c.exists():
            rgb_path = c
            break
    if rgb_path is None:
        print(f'  [SKIP] {stem}: no RGB found')
        return
    seg_path = find_companion(stem, split_dir, 'seg')
    depth_path = find_companion(stem, split_dir, 'depth')

    size = STATE.image_size
    nc = STATE.num_classes
    seg = preprocess_seg(str(seg_path), size, nc)
    depth = preprocess_depth(str(depth_path), size)
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

    panels = [('Seg', seg_rgb), ('Depth', depth_rgb)]
    panels += [(f'seed={s}', tiles[i]) for i, s in enumerate(seeds)]
    panels.append(('GT', gt_rgb))

    img = build_row(panels, thumb=320, gap=8,
                    title=f'{split_dir.name}/{stem} — {num_steps} steps, cfg={cfg}, VRAM={vram_gb():.1f} GiB')
    out_path = out_dir / f'{split_dir.name}__{stem}.png'
    img.save(out_path)
    print(f'  ✓ {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir',
                    default=str((HERE.parent / 'train_pipeline' / 'dataset').resolve()))
    ap.add_argument('--train-stems', nargs='*', default=[
        'JAX_Tile_018_007_1',
        'JAX_Tile_166_026_3',
        'JAX_Tile_284_015_4',
        'JAX_Tile_416_009_1',
        'JAX_Tile_556_023_2',
    ])
    ap.add_argument('--val-stems', nargs='*', default=[
        'JAX_Tile_018_007_4',
        'JAX_Tile_161_021_1',
        'JAX_Tile_276_011_4',
        'JAX_Tile_409_008_3',
        'JAX_Tile_505_020_3',
    ])
    ap.add_argument('--test-stems', nargs='*', default=[])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 42, 1234])
    ap.add_argument('--num-steps', type=int, default=28)
    ap.add_argument('--cfg', type=float, default=3.5)
    ap.add_argument('--ckpt', default=None,
                    help='Checkpoint dir name under weights/lora (default: latest)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

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
    print(f'\nUsing checkpoint: {ckpt_dir.name}')

    out_dir = Path(args.out) if args.out else (HERE / 'output' / f'batch_eval__{ckpt_dir.name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {out_dir}\n')

    STATE.load(ckpt_dir, persistent_text_encoder=True)

    prompt_obj = json.loads(Path(DEFAULT_PROMPT_JSON).read_text())
    prompt_text = compose_prompt_from_json(prompt_obj)
    print(f'\nPrompt ({len(prompt_text)} chars): {prompt_text[:160]}...\n')
    prompt_embed = encode_prompt(prompt_text)
    print(f'Encoded prompt: {tuple(prompt_embed.shape)}, VRAM={vram_gb():.1f} GiB\n')

    dataset_dir = Path(args.dataset_dir)
    splits = [('train', args.train_stems), ('val', args.val_stems)]
    if args.test_stems:
        splits.append(('test', args.test_stems))
    for split, stems in splits:
        print(f'\n=== {split} ===')
        split_dir = dataset_dir / split
        for stem in stems:
            try:
                run_one(stem, split_dir, prompt_embed,
                        args.seeds, args.num_steps, args.cfg, out_dir)
            except Exception as e:
                print(f'  [ERR] {stem}: {e}')

    print(f'\nAll done. PNGs in {out_dir}')


if __name__ == '__main__':
    main()

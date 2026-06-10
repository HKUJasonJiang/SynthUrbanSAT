"""Batch inference for OSM pipeline outputs.

Default behavior is intentionally simple: one generated view per tile.

    python generation_pipeline.py --input-dir ../osm_pipeline/output/omaha-984

View selection:
- default: root/top view, using ``tile_XXXX/4_seg.png`` + ``5_depth.png``
- ``--near-nadir 2``: use ``tile_XXXX/near-nadir-2/1_seg.png`` + ``2_depth.png``
- ``--near-nadir-random``: choose one discovered ``near-nadir-*`` folder per tile
- ``--depth-exr``: switch the selected view's depth file suffix to ``.exr``

Batching and GPUs:
- ``--batch 0,10`` runs tiles in the half-open range [0, 10)
- ``--gpus 0,1`` launches one worker process per GPU and shards selected tiles
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import DEFAULT_PROMPT_JSON, list_lora_checkpoints, verify_base_weights
from pipeline.inference import sample_ours
from pipeline.preprocess import depth_to_rgb, preprocess_depth, preprocess_rgb, preprocess_seg, seg_to_rgb
from pipeline.state import DEVICE, DTYPE, STATE, vram_gb


ROOT_VIEW = 'root'
NEAR_NADIR_RE = re.compile(r'^near[-_]nadir[-_](\d+)$', re.IGNORECASE)


@dataclass(frozen=True)
class RunSpec:
    tile_dir: Path
    view_name: str
    seg_path: Path
    depth_path: Path
    depth_tag: str
    rgb_path: Path | None

    @property
    def label(self) -> str:
        return self.tile_dir.name if self.view_name == ROOT_VIEW else f'{self.tile_dir.name}/{self.view_name}'

    @property
    def safe_name(self) -> str:
        return f'{self.tile_dir.name}__{self.view_name}__depth-{self.depth_tag}'


# ---------------------------------------------------------------------------
# Prompt and image helpers
# ---------------------------------------------------------------------------

def compose_prompt_from_json(obj: dict) -> str:
    parts = []
    if 'scene' in obj:
        parts.append(obj['scene'])
    if 'style' in obj:
        parts.append(obj['style'])
    if 'elements' in obj:
        for k, v in (obj.get('elements') or {}).items():
            parts.append(f'{k}: {v}')
    if 'lighting' in obj:
        parts.append(obj['lighting'])
    if 'quality' in obj:
        parts.append(obj['quality'])

    div = obj.get('diversity') or {}
    if isinstance(div, dict):
        for k in ('season', 'time_of_day', 'weather', 'lighting_mood', 'vegetation_state', 'region'):
            v = str(div.get(k) or '').strip()
            if v and v.lower() not in ('none', '(any)'):
                parts.append(v)
    return ', '.join(p for p in parts if p)


def encode_prompt(prompt_text: str) -> torch.Tensor:
    from scripts.text_encoder import encode_prompts

    embed = encode_prompts(
        STATE.text_encoder,
        STATE.tokenizer,
        [prompt_text],
        max_sequence_length=int(STATE.cfg.get('text_seq_len', 512)),
        device=DEVICE,
        dtype=DTYPE,
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
    width = n * thumb + (n - 1) * gap
    top = title_h if title else 0
    height = top + sub_h + thumb
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if title:
        draw.rectangle([0, 0, width, title_h], fill=(35, 35, 38))
        font = get_font(22)
        bbox = draw.textbbox((0, 0), title, font=font)
        draw.text(
            ((width - (bbox[2] - bbox[0])) // 2, (title_h - (bbox[3] - bbox[1])) // 2),
            title,
            fill=(235, 235, 240),
            font=font,
        )

    sub_font = get_font(18)
    for i, (label, arr) in enumerate(panels):
        x0 = i * (thumb + gap)
        draw.rectangle([x0, top, x0 + thumb, top + sub_h], fill=(50, 50, 55))
        bbox = draw.textbbox((0, 0), label, font=sub_font)
        draw.text(
            (x0 + (thumb - (bbox[2] - bbox[0])) // 2, top + (sub_h - (bbox[3] - bbox[1])) // 2),
            label,
            fill=(230, 230, 235),
            font=sub_font,
        )
        if arr is None:
            arr = np.full((thumb, thumb, 3), 80, dtype=np.uint8)
        tile = Image.fromarray(arr).resize((thumb, thumb), Image.LANCZOS)
        img.paste(tile, (x0, top + sub_h))
    return img


def feature_heatmap(ctrl: torch.Tensor, size: int) -> np.ndarray:
    """Convert [B, N, C] HDC2A context to an RGB heatmap preview."""
    import matplotlib.cm as cm

    n_tokens = ctrl.shape[1]
    side = int(math.sqrt(n_tokens))
    hm = ctrl[0].float().mean(dim=-1).reshape(side, side).cpu().numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    rgb = (cm.magma(hm)[..., :3] * 255).astype(np.uint8)
    return np.array(Image.fromarray(rgb).resize((size, size), Image.NEAREST))


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    Image.fromarray(arr[..., :3]).save(path)


# ---------------------------------------------------------------------------
# Depth loading
# ---------------------------------------------------------------------------

def load_depth_exr(path: str, size: int) -> torch.Tensor:
    """Load a single-channel float32 EXR depth to [1, H, W] in [0, 1]."""
    import OpenEXR

    with OpenEXR.File(path) as f:
        channels = f.channels()
        key = 'V' if 'V' in channels else next(iter(channels))
        arr = channels[key].pixels.astype(np.float32)

    while arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] <= arr.shape[0] else arr[0]

    if np.isinf(arr).any():
        finite = arr[np.isfinite(arr)]
        finite_max = float(finite.max()) if finite.size else 0.0
        arr = np.where(np.isinf(arr), finite_max, arr)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    if arr.shape != (size, size):
        arr = np.array(Image.fromarray(arr).resize((size, size), Image.LANCZOS), dtype=np.float32)

    mn, mx = float(arr.min()), float(arr.max())
    arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def load_depth(path: Path, size: int) -> torch.Tensor:
    if path.suffix.lower() == '.exr':
        return load_depth_exr(str(path), size)
    return preprocess_depth(str(path), size)


# ---------------------------------------------------------------------------
# OSM folder discovery
# ---------------------------------------------------------------------------

def _tile_sort_key(path: Path):
    m = re.search(r'(\d+)$', path.name)
    return (int(m.group(1)) if m else math.inf, path.name)


def discover_tile_dirs(input_dir: Path) -> list[Path]:
    return sorted(
        (p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith('tile_')),
        key=_tile_sort_key,
    )


def discover_near_nadir_dirs(tile_dir: Path) -> list[Path]:
    dirs = []
    for p in tile_dir.iterdir():
        if p.is_dir() and NEAR_NADIR_RE.match(p.name):
            dirs.append(p)
    return sorted(dirs, key=lambda p: int(NEAR_NADIR_RE.match(p.name).group(1)))


def optional_rgb(tile_dir: Path, rgb_name: str) -> Path | None:
    p = tile_dir / rgb_name
    return p if p.exists() else None


def _replace_suffix(name: str, suffix: str) -> str:
    return str(Path(name).with_suffix(suffix))


def build_run_specs(tile_dir: Path, args) -> list[RunSpec]:
    rgb_path = optional_rgb(tile_dir, args.rgb_name)
    depth_suffix = '.exr' if args.depth_exr else '.png'
    depth_tag = 'exr' if args.depth_exr else 'png'

    if args.near_nadir_random:
        near_dirs = discover_near_nadir_dirs(tile_dir)
        if not near_dirs:
            return [RunSpec(
                tile_dir=tile_dir,
                view_name='near-nadir-random',
                seg_path=tile_dir / 'near-nadir-random' / args.sub_seg_name,
                depth_path=tile_dir / 'near-nadir-random' / _replace_suffix(args.sub_depth_name, depth_suffix),
                depth_tag=depth_tag,
                rgb_path=rgb_path,
            )]
        tile_key = _tile_sort_key(tile_dir)[0]
        tile_seed = int(tile_key) if tile_key is not math.inf else abs(hash(tile_dir.name))
        near_dir = random.Random(args.random_seed + tile_seed).choice(near_dirs)
        return [RunSpec(
            tile_dir=tile_dir,
            view_name=near_dir.name,
            seg_path=near_dir / args.sub_seg_name,
            depth_path=near_dir / _replace_suffix(args.sub_depth_name, depth_suffix),
            depth_tag=depth_tag,
            rgb_path=rgb_path,
        )]

    if args.near_nadir is not None:
        near_dir = tile_dir / f'near-nadir-{args.near_nadir}'
        return [RunSpec(
            tile_dir=tile_dir,
            view_name=near_dir.name,
            seg_path=near_dir / args.sub_seg_name,
            depth_path=near_dir / _replace_suffix(args.sub_depth_name, depth_suffix),
            depth_tag=depth_tag,
            rgb_path=rgb_path,
        )]

    return [RunSpec(
        tile_dir=tile_dir,
        view_name=ROOT_VIEW,
        seg_path=tile_dir / args.seg_name,
        depth_path=tile_dir / _replace_suffix(args.depth_name, depth_suffix),
        depth_tag=depth_tag,
        rgb_path=rgb_path,
    )]


def validate_spec(spec: RunSpec) -> list[str]:
    missing = []
    if not spec.seg_path.exists():
        missing.append(f'seg={spec.seg_path}')
    if not spec.depth_path.exists():
        missing.append(f'depth={spec.depth_path}')
    return missing


def spec_output_dir(out_dir: Path, spec: RunSpec) -> Path:
    return out_dir / spec.tile_dir.name / spec.view_name / f'depth_{spec.depth_tag}'


def spec_rgb_path(out_dir: Path, spec: RunSpec, seed: int) -> Path:
    return spec_output_dir(out_dir, spec) / f'rgb_seed_{int(seed):04d}.png'


def spec_feature_path(out_dir: Path, spec: RunSpec) -> Path:
    return spec_output_dir(out_dir, spec) / 'hdc2a_feature.png'


def spec_grid_path(out_dir: Path, spec: RunSpec) -> Path:
    return spec_output_dir(out_dir, spec) / 'grid.png'


def spec_metadata_path(out_dir: Path, spec: RunSpec) -> Path:
    return spec_output_dir(out_dir, spec) / 'metadata.json'


def parse_batch(batch: str | None) -> tuple[int | None, int | None]:
    if not batch:
        return None, None
    parts = [x.strip() for x in batch.split(',')]
    if len(parts) != 2 or not all(parts):
        raise ValueError('--batch must look like START,END, e.g. --batch 0,10')
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end < start:
        raise ValueError('--batch expects 0 <= START <= END')
    return start, end


def parse_gpus(gpus: str) -> list[str]:
    return [g.strip() for g in (gpus or '').split(',') if g.strip()]


def selected_tiles(input_dir: Path, args) -> list[Path]:
    tiles = discover_tile_dirs(input_dir)
    if args.tile_names:
        wanted = set(args.tile_names)
        tiles = [t for t in tiles if t.name in wanted]
    start, end = parse_batch(args.batch)
    if start is not None:
        tiles = tiles[start:end]
    elif args.limit > 0:
        tiles = tiles[:args.limit]
    if args.num_shards > 1:
        tiles = [t for i, t in enumerate(tiles) if i % args.num_shards == args.shard_index]
    return tiles


# ---------------------------------------------------------------------------
# Per-spec inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_spec(spec: RunSpec, prompt_embed: torch.Tensor, seeds: list[int],
             num_steps: int, cfg: float, out_dir: Path, save_grids: bool,
             save_tiles: bool, skip_existing: bool) -> bool:
    missing = validate_spec(spec)
    if missing:
        print(f'  [SKIP] {spec.label}: missing {", ".join(missing)}')
        return False

    expected_outputs = [spec_feature_path(out_dir, spec), spec_metadata_path(out_dir, spec)]
    if save_tiles:
        expected_outputs.extend(spec_rgb_path(out_dir, spec, seed) for seed in seeds)
    if save_grids:
        expected_outputs.append(spec_grid_path(out_dir, spec))
    if skip_existing and expected_outputs and all(p.exists() for p in expected_outputs):
        print(f'  SKIP existing {spec.label} depth={spec.depth_tag}')
        return True

    size = STATE.image_size
    num_classes = STATE.num_classes
    seg = preprocess_seg(str(spec.seg_path), size, num_classes)
    depth = load_depth(spec.depth_path, size)
    gt_rgb = to_uint8(preprocess_rgb(str(spec.rgb_path), size)) if spec.rgb_path else None

    seg_rgb = seg_to_rgb(seg, STATE.seg_palette)
    depth_rgb = depth_to_rgb(depth)

    batch = len(seeds)
    prompt_batch = prompt_embed.to(DEVICE, DTYPE).unsqueeze(0).expand(batch, -1, -1).contiguous()
    STATE.lora_enable(True)
    ours_rgb, ctrl_ctx = sample_ours(
        seg.unsqueeze(0),
        depth.unsqueeze(0),
        prompt_batch,
        num_steps=int(num_steps),
        guidance_scale=float(cfg),
        seeds=seeds,
        progress=None,
    )
    generated = [to_uint8(ours_rgb[i]) for i in range(batch)]
    feature_rgb = feature_heatmap(ctrl_ctx, size)

    view_dir = spec_output_dir(out_dir, spec)
    view_dir.mkdir(parents=True, exist_ok=True)

    seed_outputs = {}
    if save_tiles:
        for seed, tile in zip(seeds, generated):
            rgb_path = spec_rgb_path(out_dir, spec, seed)
            save_png(tile, rgb_path)
            seed_outputs[str(int(seed))] = str(rgb_path)

    feature_path = spec_feature_path(out_dir, spec)
    save_png(feature_rgb, feature_path)

    grid_path = None
    if save_grids:
        panels = [('Seg', seg_rgb), (f'Depth ({spec.depth_tag})', depth_rgb), ('HDC2A feature', feature_rgb)]
        panels += [(f'seed={seed}', tile) for seed, tile in zip(seeds, generated)]
        panels.append(('RGB ref' if gt_rgb is not None else 'RGB ref n/a', gt_rgb))
        grid = build_row(
            panels,
            thumb=320,
            gap=8,
            title=f'{spec.label} - depth={spec.depth_tag} - {num_steps} steps, cfg={cfg}, VRAM={vram_gb():.1f} GiB',
        )
        grid_path = spec_grid_path(out_dir, spec)
        grid.save(grid_path)

    metadata = {
        'tile': spec.tile_dir.name,
        'view': spec.view_name,
        'depth_tag': spec.depth_tag,
        'seg_path': str(spec.seg_path),
        'depth_path': str(spec.depth_path),
        'rgb_ref_path': str(spec.rgb_path) if spec.rgb_path else None,
        'seeds': [int(seed) for seed in seeds],
        'rgb_outputs': seed_outputs,
        'feature_output': str(feature_path),
        'grid_output': str(grid_path) if grid_path else None,
        'num_steps': int(num_steps),
        'cfg': float(cfg),
    }
    spec_metadata_path(out_dir, spec).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f'  OK {spec.label} depth={spec.depth_tag} -> {view_dir}')
    return True

def write_manifest(out_dir: Path, *, input_dir: Path, ckpt_dir: Path, prompt_json: dict,
                   prompt_text: str, args, specs: list[RunSpec], completed: int, skipped: int):
    manifest = {
        'input_dir': str(input_dir),
        'checkpoint': str(ckpt_dir),
        'prompt_json': prompt_json,
        'flat_prompt': prompt_text,
        'seeds': args.seeds,
        'num_steps': args.num_steps,
        'cfg': args.cfg,
        'near_nadir': args.near_nadir,
        'near_nadir_random': args.near_nadir_random,
        'random_seed': args.random_seed,
        'depth_exr': args.depth_exr,
        'batch': args.batch,
        'gpus': args.gpus,
        'dry_run': args.dry_run,
        'skip_existing': args.skip_existing,
        'num_shards': args.num_shards,
        'shard_index': args.shard_index,
        'total_specs': len(specs),
        'completed': completed,
        'skipped': skipped,
        'items': [
            {
                'tile': s.tile_dir.name,
                'view': s.view_name,
                'seg': str(s.seg_path),
                'depth': str(s.depth_path),
                'depth_tag': s.depth_tag,
                'rgb_ref': str(s.rgb_path) if s.rgb_path else None,
                'safe_name': s.safe_name,
                'output_dir': str(spec_output_dir(out_dir, s)),
                'feature_output': str(spec_feature_path(out_dir, s)),
                'grid_output': str(spec_grid_path(out_dir, s)),
                'rgb_outputs': {str(int(seed)): str(spec_rgb_path(out_dir, s, int(seed))) for seed in args.seeds},
            }
            for s in specs
        ],
    }
    suffix = f'_shard{args.shard_index:02d}' if args.num_shards > 1 else ''
    (out_dir / f'manifest{suffix}.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def parse_args():
    ap = argparse.ArgumentParser(description='Run generation over an OSM pipeline output folder.')
    ap.add_argument('--input-dir', required=True, help='Root directory containing tile_XXXX subfolders')
    ap.add_argument('--near-nadir', type=int, default=None,
                    help='Use a fixed near-nadir view number, e.g. --near-nadir 2 uses near-nadir-2')
    ap.add_argument('--near-nadir-random', action='store_true',
                    help='Choose one discovered near-nadir-* view per tile')
    ap.add_argument('--random-seed', type=int, default=0,
                    help='Seed for --near-nadir-random so view choices are reproducible')
    ap.add_argument('--depth-exr', action='store_true',
                    help='Use .exr depth for the selected view instead of .png')
    ap.add_argument('--batch', default=None,
                    help='Half-open tile range START,END after sorting and tile-name filtering, e.g. --batch 0,10')
    ap.add_argument('--gpus', default='',
                    help='Comma-separated GPU IDs. With multiple IDs, launches one worker per GPU, e.g. --gpus 0,1')
    ap.add_argument('--rgb-name', default='2_rgb.png')
    ap.add_argument('--seg-name', default='4_seg.png')
    ap.add_argument('--depth-name', default='5_depth.png')
    ap.add_argument('--sub-seg-name', default='1_seg.png')
    ap.add_argument('--sub-depth-name', default='2_depth.png')
    ap.add_argument('--tile-names', nargs='+', default=[],
                    help='If given, process only these tile folders, e.g. tile_0019 tile_0020')
    ap.add_argument('--limit', type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument('--seed', default=None, help='Comma-separated seeds, e.g. --seed 1,2,42')
    ap.add_argument('--seeds', nargs='+', type=int, default=None, help='Space-separated seeds; kept for compatibility')
    ap.add_argument('--num-steps', type=int, default=28)
    ap.add_argument('--cfg', type=float, default=3.5)
    ap.add_argument('--ckpt', default=None, help='Checkpoint dir name under weights/lora. Default: latest')
    ap.add_argument('--prompt-json', default=str(DEFAULT_PROMPT_JSON), help='Path to prompt JSON')
    ap.add_argument('--out', default=None,
                    help='Output directory. Default: output/osm_batch__<input-name>__<ckpt-name>')
    ap.add_argument('--no-grids', action='store_true', help='Do not save comparison grid PNGs')
    ap.add_argument('--no-tiles', action='store_true', help='Do not save per-seed generated RGB PNGs')
    ap.add_argument('--dry-run', action='store_true', help='Print selected work items without loading models')
    ap.add_argument('--skip-existing', action='store_true', help='Skip a tile/view if all requested output files already exist')
    ap.add_argument('--num-shards', type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument('--shard-index', type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument('--gpu-worker', action='store_true', help=argparse.SUPPRESS)
    return ap.parse_args()


def parse_seed_list(args) -> list[int]:
    if args.seed:
        return [int(x.strip()) for x in args.seed.split(',') if x.strip()]
    if args.seeds:
        return [int(x) for x in args.seeds]
    return [0, 42]


def validate_args(args):
    args.seeds = parse_seed_list(args)
    if not args.seeds:
        raise ValueError('At least one seed is required')
    if args.near_nadir is not None and args.near_nadir_random:
        raise ValueError('Use only one of --near-nadir and --near-nadir-random')
    if args.near_nadir is not None and args.near_nadir < 1:
        raise ValueError('--near-nadir must be >= 1')
    if args.num_shards < 1:
        raise ValueError('--num-shards must be >= 1')
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError('--shard-index must satisfy 0 <= shard-index < num-shards')
    parse_batch(args.batch)


def resolve_ckpt(args) -> Path:
    if args.ckpt:
        ckpt_dir = (HERE / 'weights' / 'lora' / args.ckpt).resolve()
        if not (ckpt_dir / 'meta.pt').is_file():
            raise FileNotFoundError(f'checkpoint not found or missing meta.pt: {ckpt_dir}')
        return ckpt_dir
    checkpoints = list_lora_checkpoints()
    if not checkpoints:
        raise FileNotFoundError('no checkpoints in weights/lora')
    return checkpoints[0]


def default_out_dir(input_dir: Path, ckpt_dir: Path, args) -> Path:
    if args.out:
        return Path(args.out).expanduser().resolve()
    view_tag = ROOT_VIEW
    if args.near_nadir is not None:
        view_tag = f'near-nadir-{args.near_nadir}'
    elif args.near_nadir_random:
        view_tag = f'near-nadir-random-s{args.random_seed}'
    depth_tag = 'exr' if args.depth_exr else 'png'
    return HERE / 'output' / f'osm_batch__{input_dir.name}__{view_tag}__depth-{depth_tag}__{ckpt_dir.name}'


def launch_gpu_workers(args, input_dir: Path, ckpt_dir: Path, out_dir: Path, gpus: list[str]) -> int:
    print(f'Launching {len(gpus)} GPU worker(s): {", ".join(gpus)}')
    procs = []
    for shard_index, gpu in enumerate(gpus):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--input-dir', str(input_dir),
            '--num-shards', str(len(gpus)),
            '--shard-index', str(shard_index),
            '--gpu-worker',
            '--out', str(out_dir),
            '--seed', ','.join(str(s) for s in args.seeds),
            '--num-steps', str(args.num_steps),
            '--cfg', str(args.cfg),
            '--prompt-json', str(args.prompt_json),
        ]
        if args.ckpt:
            cmd.extend(['--ckpt', args.ckpt])
        if args.near_nadir is not None:
            cmd.extend(['--near-nadir', str(args.near_nadir)])
        if args.near_nadir_random:
            cmd.append('--near-nadir-random')
            cmd.extend(['--random-seed', str(args.random_seed)])
        if args.depth_exr:
            cmd.append('--depth-exr')
        if args.batch:
            cmd.extend(['--batch', args.batch])
        if args.tile_names:
            cmd.append('--tile-names')
            cmd.extend(args.tile_names)
        if args.rgb_name != '2_rgb.png':
            cmd.extend(['--rgb-name', args.rgb_name])
        if args.seg_name != '4_seg.png':
            cmd.extend(['--seg-name', args.seg_name])
        if args.depth_name != '5_depth.png':
            cmd.extend(['--depth-name', args.depth_name])
        if args.sub_seg_name != '1_seg.png':
            cmd.extend(['--sub-seg-name', args.sub_seg_name])
        if args.sub_depth_name != '2_depth.png':
            cmd.extend(['--sub-depth-name', args.sub_depth_name])
        if args.no_grids:
            cmd.append('--no-grids')
        if args.no_tiles:
            cmd.append('--no-tiles')
        if args.skip_existing:
            cmd.append('--skip-existing')

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        print(f'  worker {shard_index}: CUDA_VISIBLE_DEVICES={gpu}')
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(HERE)))

    rc = 0
    for proc in procs:
        rc = max(rc, proc.wait())
    if rc == 0:
        controller_manifest = {
            'input_dir': str(input_dir),
            'checkpoint': str(ckpt_dir),
            'out_dir': str(out_dir),
            'gpus': gpus,
            'skip_existing': args.skip_existing,
            'num_shards': len(gpus),
            'batch': args.batch,
            'near_nadir': args.near_nadir,
            'near_nadir_random': args.near_nadir_random,
            'random_seed': args.random_seed,
            'depth_exr': args.depth_exr,
        }
        (out_dir / 'manifest_controller.json').write_text(json.dumps(controller_manifest, indent=2, ensure_ascii=False))
    return rc


def run_generation(args, input_dir: Path, ckpt_dir: Path, out_dir: Path) -> None:
    tiles = selected_tiles(input_dir, args)
    if not tiles:
        raise RuntimeError(f'no selected tile_XXXX folders under {input_dir}')

    specs = []
    for tile_dir in tiles:
        specs.extend(build_run_specs(tile_dir, args))
    if not specs:
        raise RuntimeError('selected tiles produced no run specs')

    print(f'Input dir: {input_dir}')
    print(f'Found {len(tiles)} selected tile folder(s), {len(specs)} run(s)')
    print(f'Using checkpoint: {ckpt_dir.name}')
    print(f'Output dir: {out_dir}')
    if args.num_shards > 1:
        print(f'Shard: {args.shard_index + 1}/{args.num_shards}')
    print()

    if args.dry_run:
        print('Dry run items:')
        for spec in specs[:50]:
            missing = validate_spec(spec)
            status = 'missing ' + ', '.join(missing) if missing else 'ok'
            print(f'  {spec.label} depth={spec.depth_tag} seg={spec.seg_path.name} depth_file={spec.depth_path.name} [{status}]')
        if len(specs) > 50:
            print(f'  ... {len(specs) - 50} more')
        return

    STATE.load(ckpt_dir, persistent_text_encoder=True)

    prompt_json_path = Path(args.prompt_json).expanduser().resolve()
    prompt_obj = json.loads(prompt_json_path.read_text())
    prompt_text = compose_prompt_from_json(prompt_obj)
    if not prompt_text.strip():
        raise ValueError(f'prompt JSON did not produce a non-empty prompt: {prompt_json_path}')
    print(f'Prompt ({len(prompt_text)} chars): {prompt_text[:180]}{"..." if len(prompt_text) > 180 else ""}\n')
    prompt_embed = encode_prompt(prompt_text)
    print(f'Encoded prompt: {tuple(prompt_embed.shape)}, VRAM={vram_gb():.1f} GiB\n')

    completed = 0
    skipped = 0
    for idx, spec in enumerate(specs, start=1):
        print(f'[{idx}/{len(specs)}] {spec.label} depth={spec.depth_tag}')
        try:
            ok = run_spec(
                spec,
                prompt_embed,
                args.seeds,
                args.num_steps,
                args.cfg,
                out_dir,
                save_grids=not args.no_grids,
                save_tiles=not args.no_tiles,
                skip_existing=args.skip_existing,
            )
            completed += int(ok)
            skipped += int(not ok)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f'  [ERR] {spec.label} depth={spec.depth_tag}: {exc}')
            skipped += 1

    write_manifest(
        out_dir,
        input_dir=input_dir,
        ckpt_dir=ckpt_dir,
        prompt_json=prompt_obj,
        prompt_text=prompt_text,
        args=args,
        specs=specs,
        completed=completed,
        skipped=skipped,
    )
    print(f'\nAll done. completed={completed}, skipped={skipped}. Output: {out_dir}')


def main():
    args = parse_args()
    try:
        validate_args(args)
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f'--input-dir not found: {input_dir}')

        if args.dry_run:
            ckpt_dir = Path(args.ckpt or 'dry-run')
        else:
            missing = verify_base_weights()
            if missing:
                raise FileNotFoundError(f'MISSING base weights: {missing}')
            ckpt_dir = resolve_ckpt(args)

        out_dir = default_out_dir(input_dir, ckpt_dir, args)
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        gpus = parse_gpus(args.gpus)
        if gpus and not args.gpu_worker and not args.dry_run:
            rc = launch_gpu_workers(args, input_dir, ckpt_dir, out_dir, gpus)
            sys.exit(rc)

        run_generation(args, input_dir, ckpt_dir, out_dir)
    except Exception as exc:
        print(f'ERROR: {exc}')
        sys.exit(1)


if __name__ == '__main__':
    main()

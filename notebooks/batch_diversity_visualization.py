from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / 'notebooks'


def _load_hdc2a_app():
    module_path = NOTEBOOK_DIR / 'app.py'
    spec = importlib.util.spec_from_file_location('hdc2a_app', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load app module from {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hdc2a_app = _load_hdc2a_app()


AXIS_VALUES = {
    'season': ['spring', 'summer', 'autumn', 'winter'],
    'time_of_day': ['morning', 'midday', 'golden hour', 'dusk'],
    'weather': ['clear sky', 'partly cloudy', 'overcast', 'post-rain wet ground'],
    'region': [
        'North American suburban',
        'European historic city',
        'East Asian dense urban',
        'Tropical city',
    ],
    'lighting_mood': [
        'strong directional shadows',
        'soft diffuse light',
        'low-angle sunlight',
        'flat cloudy light',
    ],
    'vegetation_state': [
        'leafless bare trees',
        'lush green vegetation',
        'autumn foliage',
        'dry brown vegetation',
    ],
}

DEFAULT_AXES = list(AXIS_VALUES.keys())
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'notebooks' / 'diversity_visualization_papers'
SUPPORTED_EXTS = tuple(ext.lower() for ext in hdc2a_app.SUPPORTED_EXTS)
EXT_PRIORITY = {'.png': 0, '.tif': 1, '.tiff': 2, '.jpg': 3, '.jpeg': 4}


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r'[^a-z0-9]+', '_', value)
    return value.strip('_') or 'value'


def _load_text(path: Path) -> str:
    if not path.is_file():
        return ''
    return path.read_text(encoding='utf-8')


def _load_base_prompt(prompt_json: Path) -> str:
    text = _load_text(prompt_json)
    if text.strip():
        return text
    fallback = {
        'scene': 'Aerial top-down satellite view of an urban area',
        'style': 'Google Earth style, photorealistic satellite imagery',
        'elements': {},
        'lighting': '',
        'quality': 'masterpiece, raw photo, crisp details',
    }
    return json.dumps(fallback, indent=2, ensure_ascii=False)


def _parse_prompt_obj(prompt_text: str) -> dict:
    try:
        obj = json.loads(prompt_text) if prompt_text.strip() else {}
    except Exception:
        obj = {}
    return obj if isinstance(obj, dict) else {}


def _scan_dir_by_stem(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    entries = {}
    for file_path in sorted(
        folder.iterdir(),
        key=lambda p: (p.stem.lower(), EXT_PRIORITY.get(p.suffix.lower(), 99), p.name.lower()),
    ):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        entries.setdefault(file_path.stem, file_path)
    return entries


def _resolve_dataset(input_root: Path) -> dict[str, dict[str, Path | None]]:
    seg_dir = input_root / 'seg'
    depth_dir = input_root / 'depth'
    rgb_dir = input_root / 'rgb'

    if not seg_dir.is_dir():
        raise FileNotFoundError(f'missing required folder: {seg_dir}')
    if not depth_dir.is_dir():
        raise FileNotFoundError(f'missing required folder: {depth_dir}')

    seg_map = _scan_dir_by_stem(seg_dir)
    depth_map = _scan_dir_by_stem(depth_dir)
    rgb_map = _scan_dir_by_stem(rgb_dir) if rgb_dir.is_dir() else {}

    stems = sorted(set(seg_map) & set(depth_map))
    dataset = {}
    for stem in stems:
        dataset[stem] = {
            'seg': seg_map[stem],
            'depth': depth_map[stem],
            'rgb': rgb_map.get(stem),
        }
    return dataset


def _parse_stems_arg(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return []
    stems: list[str] = []
    for value in raw_values:
        for part in value.split(','):
            part = part.strip()
            if part:
                stems.append(part)
    seen = set()
    deduped = []
    for stem in stems:
        if stem not in seen:
            seen.add(stem)
            deduped.append(stem)
    return deduped


def _select_stems(all_stems: list[str], requested: list[str], num_samples: int) -> list[str]:
    if requested:
        missing = [stem for stem in requested if stem not in all_stems]
        if missing:
            print(f'[warn] ignoring missing stems: {", ".join(missing)}')
        return [stem for stem in requested if stem in all_stems]
    return all_stems[: max(0, num_samples)]


def _to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def _save_np_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _build_prompt_for_axis(base_prompt_text: str, caption: str | None, axis: str, value: str) -> tuple[str, str]:
    diversity = {name: hdc2a_app.NONE_OPT for name in hdc2a_app.DIVERSITY_KEYS}
    diversity[axis] = value
    prompt_obj = _parse_prompt_obj(base_prompt_text)
    prompt_obj['diversity'] = diversity
    if caption:
        prompt_obj['reference_caption'] = caption
    elif 'reference_caption' in prompt_obj:
        prompt_obj.pop('reference_caption', None)
    json_text = json.dumps(prompt_obj, indent=2, ensure_ascii=False)
    flat_prompt = hdc2a_app.compose_prompt_from_json(prompt_obj)
    return json_text, flat_prompt


def _make_fixed_inputs(stem_data: dict[str, Path | None], size: int, num_classes: int) -> dict[str, np.ndarray | None]:
    seg = hdc2a_app.preprocess_seg(str(stem_data['seg']), size, num_classes)
    depth = hdc2a_app.preprocess_depth(str(stem_data['depth']), size)
    fixed = {
        'seg': hdc2a_app.seg_to_rgb(seg),
        'depth': hdc2a_app.depth_to_rgb(depth),
        'rgb': None,
    }
    rgb_path = stem_data.get('rgb')
    if rgb_path is not None:
        fixed['rgb'] = _to_uint8_rgb(hdc2a_app.preprocess_rgb(str(rgb_path), size))
    return fixed


def _caption_reference(rgb_path: Path | None) -> str | None:
    if rgb_path is None:
        return None
    caption, status = hdc2a_app.caption_rgb_ui(str(rgb_path), keep_resident=False)
    if caption:
        print(f'[caption] {rgb_path.name}: {status}')
        return caption
    print(f'[warn] caption failed for {rgb_path.name}: {status}')
    return None


def _generate_variant(
    seg_path: Path,
    depth_path: Path,
    prompt_json_text: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
) -> tuple[np.ndarray, str, str]:
    prompt_embed, flat_prompt, _status = hdc2a_app.encode_prompt_ui(prompt_json_text)
    if prompt_embed is None:
        raise RuntimeError('prompt encoding failed')

    size = hdc2a_app.STATE.image_size
    seg = hdc2a_app.preprocess_seg(str(seg_path), size, hdc2a_app.STATE.num_classes)
    depth = hdc2a_app.preprocess_depth(str(depth_path), size)
    seg_b = seg.unsqueeze(0)
    depth_b = depth.unsqueeze(0)
    prompt_embed_b = prompt_embed.to(hdc2a_app.DEVICE, hdc2a_app.DTYPE).unsqueeze(0)

    hdc2a_app.STATE.lora_enable(True)
    rgb_batch = hdc2a_app.sample_batch(
        seg_b,
        depth_b,
        prompt_embed_b,
        num_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        seeds=[int(seed)],
        bypass_adapter=False,
    )
    image_np = _to_uint8_rgb(rgb_batch[0])

    del prompt_embed, prompt_embed_b, rgb_batch
    hdc2a_app._clear_cuda()
    return image_np, prompt_json_text, flat_prompt


def _build_grid(fixed_inputs: dict[str, np.ndarray | None], variants: list[tuple[str, np.ndarray]], title: str) -> np.ndarray:
    panels = [('Seg', fixed_inputs['seg']), ('Depth', fixed_inputs['depth'])]
    if fixed_inputs.get('rgb') is not None:
        panels.append(('Reference RGB', fixed_inputs['rgb']))
    panels.extend(variants)
    return hdc2a_app.build_row_grid(panels, title=title, thumb=384)


def _write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate paper-style diversity visualizations by varying one diversity axis at a time.',
    )
    parser.add_argument('--input-root', required=True, help='Dataset root containing seg/, depth/, and optional rgb/.')
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT), help='Directory to write diversity_visualization_papers outputs.')
    parser.add_argument('--prompt-json', default=str(hdc2a_app.DEFAULT_PROMPT_JSON), help='Base prompt JSON file.')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of stems to process when --stems is not set.')
    parser.add_argument('--stems', nargs='*', default=None, help='Optional explicit stems to process; accepts space-separated or comma-separated values.')
    parser.add_argument('--seed', type=int, default=0, help='Base seed for generation.')
    parser.add_argument('--num-steps', type=int, default=28, help='Euler sampling steps.')
    parser.add_argument('--guidance-scale', type=float, default=3.5, help='Classifier-free guidance scale.')
    parser.add_argument('--use-reference-caption', action='store_true', help='Caption reference RGB, if available, and inject it into the prompt.')
    parser.add_argument('--checkpoint', default=str(hdc2a_app.CKPT_DIR), help='Checkpoint directory to load.')
    parser.add_argument('--axes', nargs='*', default=DEFAULT_AXES, help='Subset of diversity axes to render.')
    parser.add_argument('--make-grid', action='store_true', help='Write a simple grid.png alongside the per-value renders.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_root = Path(args.input_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    prompt_json_path = Path(args.prompt_json).expanduser()
    checkpoint_dir = Path(args.checkpoint).expanduser()

    requested_axes = [axis.strip() for axis in args.axes if axis.strip()]
    invalid_axes = [axis for axis in requested_axes if axis not in AXIS_VALUES]
    if invalid_axes:
        raise ValueError(f'Unsupported axes: {", ".join(invalid_axes)}')

    base_prompt_text = _load_base_prompt(prompt_json_path)
    dataset = _resolve_dataset(input_root)
    requested_stems = _parse_stems_arg(args.stems)
    selected_stems = _select_stems(list(dataset.keys()), requested_stems, args.num_samples)
    if not selected_stems:
        raise RuntimeError(f'No valid stems found under {input_root}')

    print(f'[input] {len(dataset)} available stems, processing {len(selected_stems)}: {", ".join(selected_stems)}')
    print(f'[output] {output_root}')

    captions: dict[str, str | None] = {stem: None for stem in selected_stems}
    if args.use_reference_caption:
        for stem in selected_stems:
            rgb_path = dataset[stem].get('rgb')
            if rgb_path is not None:
                captions[stem] = _caption_reference(rgb_path)

    hdc2a_app.PERSISTENT_TEXT_ENCODER = True
    hdc2a_app.STATE.load(checkpoint_dir)

    try:
        size = hdc2a_app.STATE.image_size
        num_classes = hdc2a_app.STATE.num_classes

        for stem_index, stem in enumerate(selected_stems):
            stem_data = dataset[stem]
            stem_seed = int(args.seed) + stem_index
            stem_caption = captions.get(stem)

            print(f'[stem] {stem} (seed={stem_seed})')
            try:
                fixed_inputs = _make_fixed_inputs(stem_data, size, num_classes)
            except Exception as exc:
                print(f'[error] failed to preprocess fixed inputs for {stem}: {exc}')
                continue

            for axis in requested_axes:
                axis_dir = output_root / axis / stem
                axis_dir.mkdir(parents=True, exist_ok=True)
                fixed_dir = axis_dir / 'fixed_inputs'
                fixed_dir.mkdir(parents=True, exist_ok=True)

                _save_np_rgb(fixed_inputs['seg'], fixed_dir / 'seg.png')
                _save_np_rgb(fixed_inputs['depth'], fixed_dir / 'depth.png')
                if fixed_inputs.get('rgb') is not None:
                    _save_np_rgb(fixed_inputs['rgb'], fixed_dir / 'reference_rgb.png')

                axis_meta = {
                    'stem': stem,
                    'seed': stem_seed,
                    'checkpoint': str(checkpoint_dir),
                    'input_root': str(input_root),
                    'output_root': str(output_root),
                    'prompt_json_path': str(prompt_json_path),
                    'base_prompt_json': json.loads(base_prompt_text) if base_prompt_text.strip() else {},
                    'reference_caption': stem_caption,
                    'use_reference_caption': bool(args.use_reference_caption),
                    'axis': axis,
                    'values': AXIS_VALUES[axis],
                    'num_steps': int(args.num_steps),
                    'guidance_scale': float(args.guidance_scale),
                    'files': {
                        'seg': str(stem_data['seg']),
                        'depth': str(stem_data['depth']),
                        'rgb': str(stem_data['rgb']) if stem_data.get('rgb') is not None else None,
                    },
                    'fixed_inputs': {
                        'seg': 'fixed_inputs/seg.png',
                        'depth': 'fixed_inputs/depth.png',
                        'reference_rgb': 'fixed_inputs/reference_rgb.png' if fixed_inputs.get('rgb') is not None else None,
                    },
                    'variants': [],
                    'errors': [],
                }

                variant_panels: list[tuple[str, np.ndarray]] = []
                for value_index, value in enumerate(AXIS_VALUES[axis]):
                    print(f'  [axis] {axis} -> {value}')
                    try:
                        prompt_json_text, flat_prompt = _build_prompt_for_axis(
                            base_prompt_text,
                            stem_caption,
                            axis,
                            value,
                        )
                        image_np, _, _ = _generate_variant(
                            stem_data['seg'],
                            stem_data['depth'],
                            prompt_json_text,
                            seed=stem_seed,
                            num_steps=int(args.num_steps),
                            guidance_scale=float(args.guidance_scale),
                        )
                        filename = f'{value_index + 1:02d}_{_slugify(value)}.png'
                        image_path = axis_dir / filename
                        _save_np_rgb(image_np, image_path)
                        variant_panels.append((value, image_np))
                        axis_meta['variants'].append({
                            'value': value,
                            'seed': stem_seed,
                            'image': filename,
                            'prompt_json': prompt_json_text,
                            'flat_prompt': flat_prompt,
                        })
                    except Exception as exc:
                        print(f'  [error] {stem} / {axis} / {value}: {exc}')
                        axis_meta['errors'].append({
                            'value': value,
                            'error': f'{type(exc).__name__}: {exc}',
                        })
                        continue

                if args.make_grid:
                    try:
                        grid_np = _build_grid(
                            fixed_inputs,
                            variant_panels,
                            title=f'{stem} | {axis}',
                        )
                        _save_np_rgb(grid_np, axis_dir / 'grid.png')
                        axis_meta['grid'] = 'grid.png'
                    except Exception as exc:
                        print(f'[warn] grid failed for {stem} / {axis}: {exc}')
                        axis_meta['grid_error'] = f'{type(exc).__name__}: {exc}'

                _write_metadata(axis_dir / 'metadata.json', axis_meta)

    finally:
        hdc2a_app._clear_cuda()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
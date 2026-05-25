"""Batch dataset extension — apply the app's "Dataset Expansion" pipeline
to every (rgb, seg, depth) triple in dataset/{train,val} and write a
diversified RGB pool to dataset_add_diveristy/.

Workflow per source stem
------------------------
1. VLM-caption the source RGB once (Qwen2.5-VL by default).
2. Build N variation dicts spanning the 4 seasons (n // 4 each) with the
   other 5 diversity axes randomly sampled. Soft season ↔ vegetation
   coupling avoids "winter + lush green" nonsense; toggle off with
   ``--full-random``.
3. For each variation:
    - inject diversity + reference_caption into a copy of dataset/prompt.json
    - encode with Mistral (resident across the run)
    - sample 1 image with sample_batch (seed fixed, default 0)
    - save RGB to dst/{split}/rgb/STEM__vNN.tif
4. Replicate seg.png and depth.tif under the same STEM__vNN names (default
   symlinks to the original file → no disk waste, guaranteed identical
   across all variations of the same stem).
5. Append entries to manifest.json (flushed every K stems for resume safety).

Reuses ``notebooks/app.py`` functions read-only — no edits to app.py or
train_script.py.

Usage examples
--------------
Smoke test on 2 val stems:
    python notebooks/dataset_extension.py --limit 2 --splits val \
        --n-per-image 4

Full run, both splits, 12 variations each:
    python notebooks/dataset_extension.py

Captions only (e.g. on a small captioner GPU first):
    python notebooks/dataset_extension.py --captioner-only

Generation only (after captions are cached):
    python notebooks/dataset_extension.py --generate-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ─── Project paths (mirror app.py) ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the running pieces of the app. The launcher in app.py is guarded by
# ``if __name__ == '__main__'`` so importing it does NOT start Gradio.
import notebooks.app as app  # noqa: E402
from notebooks.app import (  # noqa: E402
    STATE,
    DEVICE,
    DTYPE,
    TEXT_DTYPE,
    CKPT_DIR,
    DEFAULT_PROMPT_JSON,
    DIVERSITY_OPTIONS,
    DIVERSITY_KEYS,
    NONE_OPT,
    CAPTIONER_MODEL_ID,
    compose_prompt_from_json,
    preprocess_seg,
    preprocess_depth,
    sample_batch,
    _to_uint8_rgb,
    _load_captioner,
    _unload_captioner,
    _run_caption,
)
from scripts.text_encoder import (  # noqa: E402
    load_text_encoder, encode_prompts, unload_text_encoder,
)


# ─── Diversity sampling ─────────────────────────────────────────────────────

# Soft season ↔ vegetation_state coupling. None of these are hard constraints
# — the season axis still goes into the prompt unchanged, this just avoids
# obvious nonsense like "winter + lush green vegetation" at sampling time.
_SEASON_VEG_POOL = {
    'spring': ['lush green', 'sparse vegetation'],
    'summer': ['lush green', 'sparse vegetation', 'dry brown'],
    'autumn': ['autumn foliage', 'sparse vegetation', 'dry brown'],
    'winter': ['leafless bare trees', 'sparse vegetation'],
}

# Soft season ↔ weather coupling. The base ``weather`` axis in app.py does
# NOT include any snow option (its in-code comment about 'fresh snow' /
# 'light snow cover' was never actually wired up). To make winter look like
# winter we inject snow weather strings here — ``compose_prompt_from_json``
# emits whatever string is present without validation, so this works without
# touching app.py. Non-winter seasons use the default non-`none` weather pool.
_SEASON_WEATHER_POOL = {
    'spring': None,    # use default non-none weather list
    'summer': None,
    'autumn': None,
    # Winter: bias heavily toward snow. ~3/4 of winter variations get snow,
    # the rest fall back to overcast/cloudy for variety (e.g. mild-climate
    # winter scenes without snow cover).
    'winter': [
        'fresh snow on rooftops and ground',
        'fresh snow on rooftops and ground',
        'light snow cover',
        'light snow cover',
        'snow-covered ground with clear sky',
        'overcast',
    ],
}


def _non_none(axis: str) -> List[str]:
    """Return the list of non-`none` choices for a given axis."""
    return [v for v in DIVERSITY_OPTIONS[axis] if v != NONE_OPT]


def sample_diversity_combos(n: int, rng: random.Random,
                            full_random: bool = False) -> List[Dict[str, str]]:
    """Return *n* variation dicts. Season is round-robin (so n=12 → 3/season,
    n=4 → 1/season). Other 5 axes are uniformly sampled from non-none choices.
    When ``full_random`` is False (default):
      - ``vegetation_state`` is drawn from a season-conditioned pool
      - ``weather`` for ``winter`` is drawn from a snow-biased pool (other
        seasons use the default weather list)
    """
    seasons = _non_none('season')        # ['spring','summer','autumn','winter']
    n_seasons = len(seasons)
    # Build season list: repeat round-robin so each season appears (n // 4)
    # times, with leftovers (n % 4) drawn at random from the season pool.
    base = (seasons * (n // n_seasons))[:n - (n % n_seasons)]
    extras = rng.sample(seasons, k=n % n_seasons) if (n % n_seasons) else []
    season_seq = base + extras
    rng.shuffle(season_seq)

    other_axes = [k for k in DIVERSITY_KEYS
                  if k not in ('season', 'vegetation_state', 'weather')]
    out: List[Dict[str, str]] = []
    for season in season_seq:
        var: Dict[str, str] = {'season': season}
        # vegetation_state
        veg_pool = (_non_none('vegetation_state')
                    if full_random else _SEASON_VEG_POOL[season])
        var['vegetation_state'] = rng.choice(veg_pool)
        # weather (snow-biased for winter when not --full-random)
        if full_random or _SEASON_WEATHER_POOL.get(season) is None:
            var['weather'] = rng.choice(_non_none('weather'))
        else:
            var['weather'] = rng.choice(_SEASON_WEATHER_POOL[season])
        # remaining axes
        for ax in other_axes:
            var[ax] = rng.choice(_non_none(ax))
        out.append(var)
    return out


# ─── File discovery ─────────────────────────────────────────────────────────

def discover_stems(src_root: Path, split: str) -> List[Tuple[str, Path, Path, Path]]:
    """Scan {src_root}/{split}/{rgb,seg,depth} for matching stems.

    Returns list of (stem, rgb_path, seg_path, depth_path), skipping stems
    that don't have all three files. RGB extension is the source of truth
    (.tif preferred); seg expected as .png; depth as .tif. Issues warnings
    for stems missing one of seg/depth.
    """
    rgb_dir = src_root / split / 'rgb'
    seg_dir = src_root / split / 'seg'
    depth_dir = src_root / split / 'depth'
    if not rgb_dir.is_dir():
        print(f'  [discover] {rgb_dir} missing — skipping split {split!r}')
        return []

    found: List[Tuple[str, Path, Path, Path]] = []
    skipped = 0
    for f in sorted(rgb_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in ('.tif', '.tiff', '.png',
                                                        '.jpg', '.jpeg'):
            continue
        stem = f.stem
        # Try common extensions for seg/depth
        seg_path = next((seg_dir / f'{stem}{e}' for e in ('.png', '.tif', '.tiff')
                         if (seg_dir / f'{stem}{e}').is_file()), None)
        depth_path = next((depth_dir / f'{stem}{e}' for e in ('.tif', '.tiff', '.png')
                           if (depth_dir / f'{stem}{e}').is_file()), None)
        if seg_path is None or depth_path is None:
            skipped += 1
            continue
        found.append((stem, f, seg_path, depth_path))
    if skipped:
        print(f'  [discover] {split}: {len(found)} stems matched, {skipped} skipped (missing seg/depth)')
    else:
        print(f'  [discover] {split}: {len(found)} stems matched')
    return found


# ─── Resume helpers (captions + manifest) ───────────────────────────────────

def _load_json(path: Path, default):
    if path.is_file():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f'  [warn] could not parse {path}: {e}; ignoring')
    return default


def _save_json_atomic(obj, path: Path):
    """Write JSON atomically (write to .tmp then rename)."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ─── Phase: caption ──────────────────────────────────────────────────────────

def run_caption_phase(stems_by_split: Dict[str, List], captions_path: Path,
                      flush_every: int = 25):
    """Caption every source RGB once (or skip if cached). Saves a dict
    ``{split: {stem: caption}}`` to *captions_path* (resumable)."""
    captions = _load_json(captions_path, default={})
    # Decide what's still pending
    pending = []
    for split, items in stems_by_split.items():
        captions.setdefault(split, {})
        for stem, rgb_p, _, _ in items:
            if stem not in captions[split]:
                pending.append((split, stem, rgb_p))
    if not pending:
        print(f'  [caption] all stems already cached at {captions_path}')
        return captions

    print(f'  [caption] loading VLM `{CAPTIONER_MODEL_ID}` for {len(pending)} stems...')
    proc, model, family = _load_captioner(CAPTIONER_MODEL_ID)

    t0 = time.time()
    try:
        for i, (split, stem, rgb_p) in enumerate(pending, 1):
            try:
                img = Image.open(rgb_p).convert('RGB')
                cap = _run_caption(proc, model, family, img)
                cap = ' '.join(cap.split())  # collapse whitespace
            except Exception as e:
                print(f'  [caption:{split}/{stem}] ERROR: {e}')
                cap = ''
            captions[split][stem] = cap
            if i % 10 == 0 or i == len(pending):
                eta = (time.time() - t0) / i * (len(pending) - i)
                print(f'  [caption] {i}/{len(pending)}  eta={eta/60:.1f} min  '
                      f'(last: {split}/{stem} → {cap[:60]!r}...)', flush=True)
            if i % flush_every == 0:
                _save_json_atomic(captions, captions_path)
    finally:
        _save_json_atomic(captions, captions_path)
        _unload_captioner()
    print(f'  [caption] done. Cached at {captions_path}')
    return captions


# ─── Phase: generation ───────────────────────────────────────────────────────

def _build_prompt_obj(base_template: dict, diversity: Dict[str, str],
                      caption: str) -> dict:
    """Inject diversity + reference_caption into a deep-copy of the base
    template. Always sets all 6 axes (uses NONE_OPT for any missing key).

    The base ``dataset/prompt.json`` template hardcodes phrases that fight
    diversity signals at sampling time:
      - ``elements.vegetation``: ``"lush green trees and grass areas"`` —
        directly contradicts ``vegetation_state=leafless bare trees`` and
        ``season=winter``.
      - ``lighting``: ``"bright natural daylight, ..., clear visibility"`` —
        contradicts ``weather=overcast/snow`` and ``time_of_day=dusk``.
      - ``style``: contains ``"natural daylight"`` — same problem.
    Since diversity already covers all three semantics (vegetation_state,
    lighting_mood, weather, time_of_day), strip the redundant hardcoded
    fields so the diversity clause is the sole source of truth.
    """
    obj = json.loads(json.dumps(base_template))  # deep copy via JSON

    # Strip redundant hardcoded fields covered by diversity
    if 'elements' in obj and isinstance(obj['elements'], dict):
        obj['elements'].pop('vegetation', None)
    obj.pop('lighting', None)
    style = obj.get('style', '')
    if isinstance(style, str) and 'natural daylight' in style:
        # remove the "natural daylight" phrase + any surrounding comma/space
        for pat in (', natural daylight', 'natural daylight, ', 'natural daylight'):
            style = style.replace(pat, '')
        obj['style'] = style.strip().rstrip(',').strip()

    div_full = {k: diversity.get(k, NONE_OPT) or NONE_OPT for k in DIVERSITY_KEYS}
    obj['diversity'] = div_full
    cap = (caption or '').strip()
    if cap:
        obj['reference_caption'] = cap
    elif 'reference_caption' in obj:
        obj.pop('reference_caption', None)
    return obj


def _replicate_control(src: Path, dst: Path, mode: str):
    """Copy / symlink / hardlink ``src`` → ``dst``. Idempotent: if ``dst``
    already exists pointing to / equal to ``src``, do nothing."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'symlink':
        # Use absolute target so the link works regardless of cwd.
        os.symlink(src.resolve(), dst)
    elif mode == 'hardlink':
        os.link(src, dst)
    elif mode == 'copy':
        shutil.copy2(src, dst)
    else:
        raise ValueError(f'unknown copy-mode: {mode!r}')


def run_generation_phase(stems_by_split: Dict[str, List],
                          captions: Dict[str, Dict[str, str]],
                          dst_root: Path,
                          base_template: dict,
                          *,
                          n_per_image: int,
                          seed: int,
                          rng_seed: int,
                          num_steps: int,
                          guidance: float,
                          copy_mode: str,
                          full_random: bool,
                          resume: bool,
                          manifest_path: Path,
                          flush_every: int = 5):
    """Main generation loop. Loads checkpoint + Mistral (resident), then for
    each (split, stem) generates n_per_image variations and writes outputs."""
    # Reuse the resident Mistral if STATE already has one (PERSISTENT_TEXT_ENCODER=True);
    # otherwise load it ourselves and unload at the end.
    if getattr(STATE, 'text_encoder', None) is not None:
        text_encoder = STATE.text_encoder
        tokenizer = STATE.tokenizer
        owns_text_encoder = False
        print(f'  [generate] reusing STATE.text_encoder (Mistral resident).')
    else:
        text_encoder_path = STATE.cfg['text_encoder_path']
        print(f'  [generate] loading Mistral text encoder ({text_encoder_path})...')
        text_encoder, tokenizer = load_text_encoder(
            text_encoder_path, device=DEVICE, dtype=TEXT_DTYPE)
        owns_text_encoder = True
    text_seq_len = int(STATE.cfg.get('text_seq_len', 512))
    size = STATE.image_size

    manifest = _load_json(manifest_path, default={})

    total_stems = sum(len(v) for v in stems_by_split.values())
    stem_idx = 0
    t0 = time.time()
    flush_pending = 0

    try:
        for split, items in stems_by_split.items():
            captions_split = captions.get(split, {})
            for stem, rgb_p, seg_p, depth_p in items:
                stem_idx += 1
                cap = captions_split.get(stem, '')

                # Reproducible per-stem RNG so re-runs yield the same variations
                stem_rng = random.Random(rng_seed ^ (hash(stem) & 0xFFFFFFFF))
                variations = sample_diversity_combos(
                    n_per_image, stem_rng, full_random=full_random)

                # Pre-compute output paths and skip whole stem if all done
                rgb_dst_dir = dst_root / split / 'rgb'
                seg_dst_dir = dst_root / split / 'seg'
                depth_dst_dir = dst_root / split / 'depth'
                rgb_dst_dir.mkdir(parents=True, exist_ok=True)
                seg_dst_dir.mkdir(parents=True, exist_ok=True)
                depth_dst_dir.mkdir(parents=True, exist_ok=True)

                # Pre-process seg + depth ONCE, then move to GPU per variation.
                try:
                    seg = preprocess_seg(str(seg_p), size, STATE.num_classes)
                    depth = preprocess_depth(str(depth_p), size)
                except Exception as e:
                    print(f'  [{stem_idx}/{total_stems}:{split}/{stem}] '
                          f'preprocess failed: {e} — skipping stem')
                    continue
                seg_b = seg.unsqueeze(0)              # [1, H, W]
                depth_b = depth.unsqueeze(0)          # [1, 1, H, W]

                for i, var in enumerate(variations):
                    name = f'{stem}__v{i:02d}'
                    rgb_out = rgb_dst_dir / f'{name}.tif'
                    rel_key = f'{split}/{name}.tif'
                    if resume and rgb_out.is_file() and rel_key in manifest:
                        continue

                    try:
                        prompt_obj = _build_prompt_obj(base_template, var, cap)
                        flat = compose_prompt_from_json(prompt_obj)
                        embed = encode_prompts(
                            text_encoder, tokenizer, [flat],
                            max_sequence_length=text_seq_len,
                            device=DEVICE, dtype=DTYPE,
                        )  # [1, L, 15360]

                        with torch.no_grad():
                            rgb = sample_batch(
                                seg_b, depth_b, embed,
                                num_steps=num_steps,
                                guidance_scale=guidance,
                                seeds=[seed],
                                bypass_adapter=False,
                            )  # [1, 3, H, W] CPU
                        img_np = _to_uint8_rgb(rgb[0])
                        Image.fromarray(img_np).save(rgb_out, format='TIFF')

                        # Replicate control inputs (seg, depth) — match each
                        # output extension to the source so HDC2ADataset's
                        # ext-aware loader still resolves siblings correctly.
                        seg_out = seg_dst_dir / f'{name}{seg_p.suffix.lower()}'
                        depth_out = depth_dst_dir / f'{name}{depth_p.suffix.lower()}'
                        _replicate_control(seg_p, seg_out, copy_mode)
                        _replicate_control(depth_p, depth_out, copy_mode)

                        manifest[rel_key] = {
                            'source_stem': stem,
                            'source_split': split,
                            'variation_index': i,
                            'diversity': var,
                            'caption': cap,
                            'seed': seed,
                            'num_steps': num_steps,
                            'guidance': guidance,
                            'seg': str(seg_out.relative_to(dst_root)),
                            'depth': str(depth_out.relative_to(dst_root)),
                            'rgb': str(rgb_out.relative_to(dst_root)),
                        }
                    except Exception as e:
                        tb = traceback.format_exc(limit=4)
                        print(f'  [{stem_idx}/{total_stems}:{split}/{stem} v{i:02d}] '
                              f'FAILED: {type(e).__name__}: {e}\n{tb}')
                        manifest[rel_key] = {
                            'source_stem': stem,
                            'source_split': split,
                            'variation_index': i,
                            'diversity': var,
                            'error': f'{type(e).__name__}: {e}',
                        }

                # Free per-stem GPU tensors before next stem.
                del seg, depth, seg_b, depth_b
                torch.cuda.empty_cache()

                # Periodic flush + progress
                flush_pending += 1
                elapsed = time.time() - t0
                rate = stem_idx / max(elapsed, 1e-6)
                eta = (total_stems - stem_idx) / max(rate, 1e-6)
                print(f'  [generate] stem {stem_idx}/{total_stems} '
                      f'({split}/{stem}) — {n_per_image} variations  '
                      f'elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m', flush=True)
                if flush_pending >= flush_every:
                    _save_json_atomic(manifest, manifest_path)
                    flush_pending = 0
    finally:
        _save_json_atomic(manifest, manifest_path)
        if owns_text_encoder:
            unload_text_encoder(text_encoder, tokenizer)
    print(f'  [generate] done. Manifest at {manifest_path}')
    return manifest


# ─── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src-root', type=Path,
                   default=PROJECT_ROOT / 'dataset',
                   help='source dataset root (with train/, val/ subdirs)')
    p.add_argument('--dst-root', type=Path,
                   default=PROJECT_ROOT / 'dataset_add_diveristy',
                   help='output root (will be created)')
    p.add_argument('--splits', nargs='+', default=['train', 'val'],
                   choices=['train', 'val', 'test'])
    p.add_argument('--n-per-image', type=int, default=12,
                   help='number of diversified RGBs per source stem (default 12)')
    p.add_argument('--seed', type=int, default=0,
                   help='sampling seed (fixed across all stems & variations)')
    p.add_argument('--rng-seed', type=int, default=42,
                   help='RNG seed for diversity-axis sampling (per-stem hashed)')
    p.add_argument('--num-steps', type=int, default=28)
    p.add_argument('--guidance', type=float, default=3.5)
    p.add_argument('--checkpoint', type=Path, default=None,
                   help=f'override checkpoint dir (default: {CKPT_DIR.relative_to(PROJECT_ROOT)})')
    p.add_argument('--captioner-only', action='store_true',
                   help='only run the captioning phase, skip generation')
    p.add_argument('--generate-only', action='store_true',
                   help='only run the generation phase (assumes captions.json exists)')
    p.add_argument('--copy-mode', choices=['copy', 'symlink', 'hardlink'],
                   default='symlink',
                   help='how to replicate seg/depth files (default symlink saves disk)')
    p.add_argument('--limit', type=int, default=0,
                   help='cap stems per split for smoke testing (0 = no limit)')
    p.add_argument('--resume', action='store_true',
                   help='skip rgb outputs that already exist + are in manifest')
    p.add_argument('--full-random', action='store_true',
                   help='disable season↔vegetation_state soft coupling '
                        '(let vegetation be uniformly random)')
    p.add_argument('--prompt-template', type=Path, default=DEFAULT_PROMPT_JSON,
                   help='base prompt JSON template (default: dataset/prompt.json)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.captioner_only and args.generate_only:
        print('ERROR: --captioner-only and --generate-only are mutually exclusive.')
        sys.exit(2)

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # Discover stems
    print(f'\n=== Discover stems in {src_root} ===')
    stems_by_split: Dict[str, List] = {}
    for split in args.splits:
        items = discover_stems(src_root, split)
        if args.limit > 0:
            items = items[:args.limit]
            print(f'  [discover] {split}: --limit {args.limit} → {len(items)} stems')
        stems_by_split[split] = items
    total = sum(len(v) for v in stems_by_split.values())
    if total == 0:
        print('No stems found — nothing to do.')
        sys.exit(1)
    print(f'  [discover] total: {total} stems × {args.n_per_image} variations = '
          f'{total * args.n_per_image} target outputs')

    captions_path = dst_root / 'captions.json'
    manifest_path = dst_root / 'manifest.json'

    # ── Phase 1: caption ────────────────────────────────────────────────
    if not args.generate_only:
        print(f'\n=== Phase 1: VLM caption (cache → {captions_path}) ===')
        captions = run_caption_phase(stems_by_split, captions_path)
    else:
        captions = _load_json(captions_path, default={})
        if not captions:
            print(f'ERROR: --generate-only but no captions found at {captions_path}')
            sys.exit(2)

    if args.captioner_only:
        print('\n--captioner-only: done.')
        return

    # ── Phase 2: generation ────────────────────────────────────────────
    print(f'\n=== Phase 2: Load checkpoint + generate ===')
    ckpt_dir = args.checkpoint.resolve() if args.checkpoint else CKPT_DIR
    STATE.load(ckpt_dir)
    print(f'  [generate] checkpoint loaded from {ckpt_dir}')

    # Read base prompt template
    if not args.prompt_template.is_file():
        print(f'ERROR: prompt template missing: {args.prompt_template}')
        sys.exit(2)
    with open(args.prompt_template) as f:
        base_template = json.load(f)
    # Copy template to dst for downstream training compatibility
    _save_json_atomic(base_template, dst_root / 'prompt.json')

    print(f'\n=== Phase 3: Generate {args.n_per_image} variations × {total} stems ===')
    manifest = run_generation_phase(
        stems_by_split, captions, dst_root, base_template,
        n_per_image=args.n_per_image,
        seed=args.seed,
        rng_seed=args.rng_seed,
        num_steps=args.num_steps,
        guidance=args.guidance,
        copy_mode=args.copy_mode,
        full_random=args.full_random,
        resume=args.resume,
        manifest_path=manifest_path,
    )

    # Final summary
    n_ok = sum(1 for v in manifest.values() if 'error' not in v)
    n_err = len(manifest) - n_ok
    print(f'\n=== Done ===')
    print(f'  manifest entries: {len(manifest)}  ok={n_ok}  errors={n_err}')
    print(f'  outputs at: {dst_root}')


if __name__ == '__main__':
    main()

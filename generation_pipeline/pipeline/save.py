"""Save a run to ``output/<folder_name>/`` — inputs + all generated images +
prompts + metadata. Each image is saved as an independent PNG so the user
can pick them out one by one.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import OUTPUT_DIR


def _safe_dirname(name: str) -> str:
    """Sanitize *name* for use as a directory: keep [A-Za-z0-9._-], replace rest with _."""
    safe = ''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in (name or '').strip())
    return safe or f'run_{int(time.time())}'


def _save_png(arr_or_path, dst: Path):
    """Save *arr_or_path* (numpy HWC uint8 OR an existing path) as a PNG to *dst*."""
    if arr_or_path is None:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr_or_path, (str, Path)):
        src = Path(arr_or_path)
        if not src.exists():
            return False
        # Re-encode through PIL to normalize format
        try:
            Image.open(src).convert('RGB').save(dst, 'PNG')
            return True
        except Exception:
            shutil.copy2(src, dst)
            return True
    arr = np.asarray(arr_or_path)
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr, 'L').save(dst, 'PNG')
    else:
        Image.fromarray(arr[..., :3], 'RGB').save(dst, 'PNG')
    return True


def save_run(folder_name: str, *,
             seg_src_path: str | None,
             depth_src_path: str | None,
             rgb_src_path: str | None,
             seg_preview: np.ndarray | None,
             depth_preview: np.ndarray | None,
             rgb_preview: np.ndarray | None,
             feature_preview: np.ndarray | None,
             ours_tiles: list[np.ndarray] | None,
             baseline_seg_tiles: list[np.ndarray] | None,
             baseline_depth_tiles: list[np.ndarray] | None,
             summary_grid: np.ndarray | None,
             prompt_json_text: str,
             flat_prompt: str,
             seeds: list[int],
             metadata_extra: dict) -> tuple[Path, list[str]]:
    """Persist a run to disk; return (run_dir, list_of_written_files)."""
    safe = _safe_dirname(folder_name)
    run_dir = OUTPUT_DIR / safe
    if run_dir.exists():
        ts = time.strftime('%Y%m%d_%H%M%S')
        run_dir = OUTPUT_DIR / f'{safe}_{ts}'
    run_dir.mkdir(parents=True, exist_ok=False)
    written: list[str] = []

    def _track(ok: bool, path: Path):
        if ok:
            written.append(str(path.relative_to(run_dir)))

    # Inputs — copy originals AND save the colourised previews actually used.
    if seg_src_path:
        _track(_save_png(seg_src_path,   run_dir / 'inputs' / f'seg_original{Path(seg_src_path).suffix.lower()}'),
               run_dir / 'inputs' / f'seg_original{Path(seg_src_path).suffix.lower()}')
    if depth_src_path:
        _track(_save_png(depth_src_path, run_dir / 'inputs' / f'depth_original{Path(depth_src_path).suffix.lower()}'),
               run_dir / 'inputs' / f'depth_original{Path(depth_src_path).suffix.lower()}')
    if rgb_src_path:
        _track(_save_png(rgb_src_path,   run_dir / 'inputs' / f'rgb_reference{Path(rgb_src_path).suffix.lower()}'),
               run_dir / 'inputs' / f'rgb_reference{Path(rgb_src_path).suffix.lower()}')

    _track(_save_png(seg_preview,     run_dir / 'inputs' / 'seg_colorized.png'),     run_dir / 'inputs' / 'seg_colorized.png')
    _track(_save_png(depth_preview,   run_dir / 'inputs' / 'depth_colorized.png'),   run_dir / 'inputs' / 'depth_colorized.png')
    _track(_save_png(rgb_preview,     run_dir / 'inputs' / 'rgb_preview.png'),       run_dir / 'inputs' / 'rgb_preview.png')

    _track(_save_png(feature_preview, run_dir / 'ours' / 'hdc2a_feature.png'), run_dir / 'ours' / 'hdc2a_feature.png')

    for label, tiles in (('ours', ours_tiles),
                         ('baseline_seg_only', baseline_seg_tiles),
                         ('baseline_depth_only', baseline_depth_tiles)):
        if not tiles:
            continue
        for i, (seed, tile) in enumerate(zip(seeds, tiles)):
            p = run_dir / label / f'seed_{seed:03d}.png'
            _track(_save_png(tile, p), p)

    _track(_save_png(summary_grid, run_dir / 'summary_grid.png'), run_dir / 'summary_grid.png')

    # Prompts
    (run_dir / 'prompts').mkdir(parents=True, exist_ok=True)
    (run_dir / 'prompts' / 'prompt.json').write_text(prompt_json_text or '{}')
    (run_dir / 'prompts' / 'flat.txt').write_text(flat_prompt or '')
    written.extend(['prompts/prompt.json', 'prompts/flat.txt'])

    # Metadata
    meta = {
        'folder_name': folder_name,
        'safe_name': safe,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seeds': list(seeds),
        **metadata_extra,
    }
    (run_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    written.append('metadata.json')

    return run_dir, written

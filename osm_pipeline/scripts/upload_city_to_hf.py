#!/usr/bin/env python3
"""Upload one completed city dataset to Hugging Face.

A city is considered complete after both pieces exist locally:
  1. OSM artifacts: osm_pipeline/output/<city>/
  2. Generated RGB artifacts: generation_pipeline/output/osm_batch__<city>__near-nadir-1__depth-png__<ckpt>/

The uploader keeps them separate in the HF dataset repo so partial re-uploads are
safe and easy to inspect:
  <path-prefix>/<city>/osm/
  <path-prefix>/<city>/generation/near-nadir-1_seed64/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_generation_dir(root: Path, city: str, view: str, seed: int, ckpt: str | None = None) -> Path:
    base = root / 'generation_pipeline' / 'output'
    depth = 'depth-png'
    if ckpt:
        p = base / f'osm_batch__{city}__{view}__{depth}__{ckpt}'
        if not p.is_dir():
            raise FileNotFoundError(f'generation output not found: {p}')
        return p
    matches = sorted(base.glob(f'osm_batch__{city}__{view}__{depth}__*'))
    if not matches:
        raise FileNotFoundError(f'no generation output found for city={city}, view={view} under {base}')
    if len(matches) > 1:
        # Prefer the newest modified directory, but print enough context for logs.
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def upload_folder(api, *, repo_id: str, local_dir: Path, path_in_repo: str, repo_type: str, dry_run: bool) -> None:
    print(f'[hf] {local_dir} -> {repo_id}/{path_in_repo}')
    if dry_run:
        return
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description='Upload one SynthUrbanSAT city output to Hugging Face.')
    ap.add_argument('--city', required=True)
    ap.add_argument('--hf-repo', default=os.environ.get('HF_REPO', 'JasonXF/SynthUrbanSAT-5k'))
    ap.add_argument('--repo-type', default='dataset')
    ap.add_argument('--path-prefix', default='', help='Optional prefix inside HF repo, e.g. dataset20')
    ap.add_argument('--seed', type=int, default=64)
    ap.add_argument('--near-nadir', type=int, default=1)
    ap.add_argument('--ckpt', default=None, help='Generation checkpoint suffix. If omitted, newest matching output is used.')
    ap.add_argument('--osm-dir', default=None)
    ap.add_argument('--generation-dir', default=None)
    ap.add_argument('--skip-osm', action='store_true')
    ap.add_argument('--skip-generation', action='store_true')
    ap.add_argument('--create-repo', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    root = repo_root()
    city = args.city
    view = f'near-nadir-{int(args.near_nadir)}'
    prefix = args.path_prefix.strip('/')
    base_path = f'{prefix}/{city}' if prefix else city

    osm_dir = Path(args.osm_dir).expanduser().resolve() if args.osm_dir else root / 'osm_pipeline' / 'output' / city
    generation_dir = Path(args.generation_dir).expanduser().resolve() if args.generation_dir else find_generation_dir(root, city, view, int(args.seed), args.ckpt)

    if not args.skip_osm and not osm_dir.is_dir():
        raise FileNotFoundError(f'OSM output folder not found: {osm_dir}')
    if not args.skip_generation and not generation_dir.is_dir():
        raise FileNotFoundError(f'generation output folder not found: {generation_dir}')

    print(f'[hf] repo={args.hf_repo} repo_type={args.repo_type} city={city}')
    print(f'[hf] base_path={base_path}')
    print(f'[hf] dry_run={args.dry_run}')

    if args.dry_run:
        api = None
    else:
        from huggingface_hub import HfApi
        token = os.environ.get('HF_TOKEN_WRITE') or os.environ.get('HF_TOKEN')
        if not token:
            raise RuntimeError('HF_TOKEN_WRITE or HF_TOKEN must be set for upload')
        api = HfApi(token=token)
        if args.create_repo:
            api.create_repo(repo_id=args.hf_repo, repo_type=args.repo_type, exist_ok=True)

    if not args.skip_osm:
        upload_folder(
            api,
            repo_id=args.hf_repo,
            repo_type=args.repo_type,
            local_dir=osm_dir,
            path_in_repo=f'{base_path}/osm',
            dry_run=args.dry_run,
        )
    if not args.skip_generation:
        upload_folder(
            api,
            repo_id=args.hf_repo,
            repo_type=args.repo_type,
            local_dir=generation_dir,
            path_in_repo=f'{base_path}/generation/{view}_seed{int(args.seed)}',
            dry_run=args.dry_run,
        )
    print('[hf] upload stage complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

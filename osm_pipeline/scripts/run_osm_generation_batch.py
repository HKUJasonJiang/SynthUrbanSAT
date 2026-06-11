#!/usr/bin/env python3
"""Run the 20-city OSM + generation + HF upload batch from a JSON manifest.

The script intentionally runs cities serially on one server. OSM still uses its
own IO/OSM/canopy worker pools, generation can shard across GPUs via ``--gpus``,
and HF upload is a separate final stage. This keeps memory pressure predictable
and makes interruption safe.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if 'machines' not in data:
        raise ValueError(f'manifest missing machines: {path}')
    return data


def city_entries(manifest: dict, machine: str, only: set[str] | None = None) -> list[dict]:
    if machine not in manifest['machines']:
        known = ', '.join(sorted(manifest['machines']))
        raise ValueError(f'unknown machine {machine!r}; known: {known}')
    rows = list(manifest['machines'][machine].get('cities') or [])
    if only:
        rows = [row for row in rows if row['city'] in only]
    return rows


def shellish(cmd: list[str]) -> str:
    return ' '.join(shlex.quote(str(x)) for x in cmd)


def run_logged(cmd: list[str], *, cwd: Path, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(shellish(cmd))
    if dry_run:
        return 0
    started = datetime.now().isoformat(timespec='seconds')
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n===== START {started} =====\n')
        log.write(shellish(cmd) + '\n')
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
            log.write(line)
        rc = proc.wait()
        ended = datetime.now().isoformat(timespec='seconds')
        log.write(f'===== END {ended} rc={rc} =====\n')
        log.flush()
    return rc


def osm_cmd(row: dict, manifest: dict, args) -> list[str]:
    g = manifest.get('global', {})
    workers = g.get('osm_workers', {})
    seed = int(args.seed if args.seed is not None else g.get('seed', 64))
    cmd = [
        sys.executable, 'osm_pipeline/auto_pipeline.py',
        '--city', row['city'],
        '--plan', row['plan'],
        '--height-seed', str(seed),
        '--scatter-seed', str(seed),
        '--tree-height-seed', str(seed),
        '--strategy-tag', str(g.get('osm_strategy_tag', 'dataset20-s64-nn1')),
        '--io-workers', str(args.io_workers or workers.get('io_workers', 8)),
        '--osm-workers', str(args.osm_workers or workers.get('osm_workers', 4)),
        '--canopy-workers', str(args.canopy_workers or workers.get('canopy_workers', 4)),
    ]
    if args.clean_osm:
        cmd.append('--clean')
    return cmd


def generation_cmd(row: dict, manifest: dict, args) -> list[str]:
    g = manifest.get('global', {})
    seed = int(args.seed if args.seed is not None else g.get('seed', 64))
    near_nadir = int(args.near_nadir if args.near_nadir is not None else g.get('near_nadir', 1))
    seed_chunk = int(args.seed_chunk_size if args.seed_chunk_size is not None else g.get('generation_seed_chunk_size', 1))
    cmd = [
        sys.executable, 'generation_pipeline/generation_pipeline.py',
        '--input-dir', f'osm_pipeline/output/{row["city"]}',
        '--near-nadir', str(near_nadir),
        '--seed', str(seed),
        '--skip-existing',
        '--seed-chunk-size', str(seed_chunk),
    ]
    if args.gpus:
        cmd.extend(['--gpus', args.gpus])
    if args.num_steps is not None:
        cmd.extend(['--num-steps', str(args.num_steps)])
    if args.cfg is not None:
        cmd.extend(['--cfg', str(args.cfg)])
    if args.ckpt:
        cmd.extend(['--ckpt', args.ckpt])
    if args.dry_run:
        cmd.append('--dry-run')
    return cmd


def upload_cmd(row: dict, manifest: dict, args) -> list[str]:
    g = manifest.get('global', {})
    seed = int(args.seed if args.seed is not None else g.get('seed', 64))
    near_nadir = int(args.near_nadir if args.near_nadir is not None else g.get('near_nadir', 1))
    hf_repo = args.hf_repo or g.get('hf_repo') or 'JasonXF/SynthUrbanSAT-5k'
    path_prefix = args.hf_path_prefix if args.hf_path_prefix is not None else str(g.get('hf_path_prefix', '') or '')
    cmd = [
        sys.executable, 'osm_pipeline/scripts/upload_city_to_hf.py',
        '--city', row['city'],
        '--hf-repo', hf_repo,
        '--seed', str(seed),
        '--near-nadir', str(near_nadir),
    ]
    if path_prefix:
        cmd.extend(['--path-prefix', path_prefix])
    if args.ckpt:
        cmd.extend(['--ckpt', args.ckpt])
    if args.create_hf_repo:
        cmd.append('--create-repo')
    if args.dry_run:
        cmd.append('--dry-run')
    return cmd


def validate_plans(rows: list[dict], cwd: Path) -> None:
    for row in rows:
        plan = cwd / row['plan']
        if not plan.is_file():
            raise FileNotFoundError(f'missing plan for {row["city"]}: {plan}')
        data = json.loads(plan.read_text(encoding='utf-8'))
        tiles = data.get('tiles') or data.get('tile_plans') or []
        if len(tiles) != int(row['tiles']):
            raise ValueError(f'{row["city"]}: manifest tiles={row["tiles"]}, plan tiles={len(tiles)}')


def main() -> int:
    ap = argparse.ArgumentParser(description='Run dataset20 OSM + generation shards.')
    ap.add_argument('--manifest', default='osm_pipeline/plans/dataset20/manifest.json')
    ap.add_argument('--machine', required=True, choices=['h100', 'h200'])
    ap.add_argument('--stage', choices=['osm', 'generation', 'upload', 'all'], default='all')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only-city', nargs='*', default=[])
    ap.add_argument('--continue-on-error', action='store_true')
    ap.add_argument('--clean-osm', action='store_true')
    ap.add_argument('--gpus', default='', help='Generation GPUs, e.g. 0 or 0,1. Empty = current CUDA device/default.')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--near-nadir', type=int, default=None)
    ap.add_argument('--seed-chunk-size', type=int, default=None)
    ap.add_argument('--num-steps', type=int, default=None)
    ap.add_argument('--cfg', type=float, default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--hf-repo', default=None, help='HF dataset repo for upload stage. Defaults to manifest global.hf_repo or JasonXF/SynthUrbanSAT-5k.')
    ap.add_argument('--hf-path-prefix', default=None, help='Optional path prefix inside HF repo for upload stage.')
    ap.add_argument('--create-hf-repo', action='store_true', help='Create HF repo if needed during upload stage.')
    ap.add_argument('--io-workers', type=int, default=None)
    ap.add_argument('--osm-workers', type=int, default=None)
    ap.add_argument('--canopy-workers', type=int, default=None)
    ap.add_argument('--log-dir', default='logs/dataset20')
    args = ap.parse_args()

    cwd = repo_root()
    manifest_path = (cwd / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    manifest = load_manifest(manifest_path)
    only = set(args.only_city) if args.only_city else None
    rows = city_entries(manifest, args.machine, only)
    if not rows:
        raise SystemExit(f'no cities selected for machine={args.machine}')
    validate_plans(rows, cwd)

    total = sum(int(row['tiles']) for row in rows)
    print(f'Plan: {manifest.get("name", manifest_path.name)}')
    print(f'Machine: {args.machine} | cities={len(rows)} | tiles={total} | stage={args.stage} | dry_run={args.dry_run}')

    log_root = cwd / args.log_dir / args.machine
    failures: list[tuple[str, str, int]] = []
    for row in rows:
        city = row['city']
        print(f'\n### {city} ({row["tiles"]} tiles)')
        city_failed = False
        if args.stage in ('osm', 'all'):
            rc = run_logged(
                osm_cmd(row, manifest, args),
                cwd=cwd,
                log_path=log_root / city / 'osm.log',
                dry_run=args.dry_run,
            )
            if rc != 0:
                failures.append((city, 'osm', rc))
                city_failed = True
                if not args.continue_on_error:
                    break
        if (not city_failed) and args.stage in ('generation', 'all'):
            rc = run_logged(
                generation_cmd(row, manifest, args),
                cwd=cwd,
                log_path=log_root / city / 'generation.log',
                dry_run=args.dry_run,
            )
            if rc != 0:
                failures.append((city, 'generation', rc))
                city_failed = True
                if not args.continue_on_error:
                    break
        if (not city_failed) and args.stage in ('upload', 'all'):
            rc = run_logged(
                upload_cmd(row, manifest, args),
                cwd=cwd,
                log_path=log_root / city / 'upload.log',
                dry_run=args.dry_run,
            )
            if rc != 0:
                failures.append((city, 'upload', rc))
                city_failed = True
                if not args.continue_on_error:
                    break

    if failures:
        print('\nFailures:')
        for city, stage, rc in failures:
            print(f'  {city} {stage} rc={rc}')
        return 1
    print('\nAll requested jobs finished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

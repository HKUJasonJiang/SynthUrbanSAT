#!/usr/bin/env python3
"""Write a traceable shell plan for selected-tile generation.

The plan uses explicit per-GPU queues instead of nested --gpus launcher calls.
Each GPU processes a disjoint tile shard over all requested views.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "write_generation_plan_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_info() -> dict:
    root = _repo_root()
    out: dict[str, object] = {"repo": str(root)}
    try:
        out["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        out["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def _view_arg(view: str) -> list[str]:
    if view == "root":
        return []
    if view.startswith("near-nadir-"):
        return ["--near-nadir", view.rsplit("-", 1)[-1]]
    raise ValueError(f"Unsupported view: {view}")


def _out_dir(out_base: Path, dataset_tag: str, view: str, depth_tag: str, ckpt: str) -> Path:
    return out_base / f"osm_batch__{dataset_tag}__{view}__depth-{depth_tag}__{ckpt}"


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", required=True, help="selected_tiles.json from selection_stats.py")
    ap.add_argument("--input-dir", required=True, help="OSM output folder used by generation_pipeline.py")
    ap.add_argument("--out-dir", default="generation_pipeline/output/selection_omaha984_60tiles")
    ap.add_argument("--dataset-tag", default="omaha-984-selection60")
    ap.add_argument("--generation-output-base", default="generation_pipeline/output")
    ap.add_argument("--python", default="/data/home/jason/miniconda3/envs/flux/bin/python")
    ap.add_argument("--generation-script", default="generation_pipeline/generation_pipeline.py")
    ap.add_argument("--prompt-json", default="osm_pipeline/assets/prompt/prompt.json")
    ap.add_argument("--views", default="root,near-nadir-1,near-nadir-2,near-nadir-3")
    ap.add_argument("--seeds", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--ckpt", default="checkpoint_epoch_0315")
    ap.add_argument("--depth-exr", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args(argv)

    root = _repo_root()
    selection_path = Path(args.selection).expanduser().resolve()
    selection = json.loads(selection_path.read_text())
    tiles = [r["tile"] for r in selection["selected"]]
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not tiles:
        raise SystemExit("Selection contains no tiles")
    if not gpus:
        raise SystemExit("No GPUs provided")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_base = Path(args.generation_output_base).expanduser()
    if not gen_base.is_absolute():
        gen_base = root / gen_base
    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    prompt_json = Path(args.prompt_json).expanduser()
    if not prompt_json.is_absolute():
        prompt_json = root / prompt_json
    gen_script = Path(args.generation_script).expanduser()
    if not gen_script.is_absolute():
        gen_script = root / gen_script

    shards = {gpu: tiles[i::len(gpus)] for i, gpu in enumerate(gpus)}
    depth_tag = "exr" if args.depth_exr else "png"
    plan = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git": _git_info(),
        "selection": str(selection_path),
        "input_dir": str(input_dir),
        "dataset_tag": args.dataset_tag,
        "views": views,
        "seeds": args.seeds,
        "gpus": gpus,
        "shards": shards,
        "outputs": {view: str(_out_dir(gen_base, args.dataset_tag, view, depth_tag, args.ckpt)) for view in views},
    }
    (out_dir / "generation_plan.json").write_text(json.dumps(plan, indent=2))

    log = out_dir / "generation_run.log"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(root))}",
        f"LOG={shlex.quote(str(log))}",
        "mkdir -p \"$(dirname \"$LOG\")\"",
        "{",
        "  echo '==== selection generation run ===='",
        "  date --iso-8601=seconds",
        "  START=$(date +%s)",
        f"  echo script_version={SCRIPT_VERSION}",
        f"  echo selection={shlex.quote(str(selection_path))}",
        f"  echo seeds={shlex.quote(args.seeds)}",
        f"  echo views={shlex.quote(','.join(views))}",
        "} | tee -a \"$LOG\"",
        "",
        "run_gpu_queue() {",
        "  local gpu=\"$1\"",
        "  shift",
        "  echo \"==== GPU $gpu queue start $(date --iso-8601=seconds) ====\" | tee -a \"$LOG\"",
        "  CUDA_VISIBLE_DEVICES=\"$gpu\" \"$@\" 2>&1 | tee -a \"$LOG\"",
        "  echo \"==== GPU $gpu queue end $(date --iso-8601=seconds) ====\" | tee -a \"$LOG\"",
        "}",
        "",
    ]

    gpu_blocks = []
    for gpu in gpus:
        shard_tiles = shards[gpu]
        block_lines = ["("]
        block_lines.append(f"  echo 'GPU {gpu} tiles: {' '.join(shard_tiles)}' | tee -a \"$LOG\"")
        for view in views:
            cmd = [str(args.python), str(gen_script), "--input-dir", str(input_dir), "--tile-names", *shard_tiles, "--seed", args.seeds, "--prompt-json", str(prompt_json), "--ckpt", args.ckpt, "--out", str(_out_dir(gen_base, args.dataset_tag, view, depth_tag, args.ckpt))]
            cmd.extend(_view_arg(view))
            if args.depth_exr:
                cmd.append("--depth-exr")
            if args.skip_existing:
                cmd.append("--skip-existing")
            block_lines.append(f"  echo '==== GENERATE {view} on GPU {gpu} ====' | tee -a \"$LOG\"")
            block_lines.append(f"  run_gpu_queue {shlex.quote(gpu)} {_shell_join(cmd)}")
        block_lines.append(") &")
        gpu_blocks.extend(block_lines)
    lines.extend(gpu_blocks)
    lines.extend([
        "wait",
        "{",
        "  END=$(date +%s)",
        "  echo '==== selection generation complete ===='",
        "  date --iso-8601=seconds",
        "  echo generation_seconds=$((END-START))",
        "  printf 'generation_hms=%02d:%02d:%02d\\n' $(((END-START)/3600)) $((((END-START)%3600)/60)) $(((END-START)%60))",
        "} | tee -a \"$LOG\"",
    ])
    script_path = out_dir / "run_generation_selection.sh"
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)

    print(f"Wrote plan: {out_dir / 'generation_plan.json'}")
    print(f"Wrote shell: {script_path}")
    print(f"Tiles: {len(tiles)} Views: {len(views)} Seeds: {args.seeds} GPUs: {','.join(gpus)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

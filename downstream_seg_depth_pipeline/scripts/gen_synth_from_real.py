"""Bridge generator: turn REAL US3D seg+depth into SYNTHETIC training tiles.

This produces the "S" (synthetic) side of the Phase-1 *Real vs Synthetic*
comparison. For every real tile it:

  1. reads the tile's seg + depth (the labels),
  2. runs the trained HDC2A + FLUX.2-dev ControlNet/LoRA generator
     (reused from ../generation_pipeline) conditioned on that seg + depth,
  3. writes the GENERATED RGB to ``<out>/<split>/rgb/<stem>.png`` and
     COPIES the ORIGINAL seg + depth to ``<out>/<split>/{seg,depth}/`` so the
     labels are byte-identical to the real ones.

=> Only the RGB pixels are synthetic; the layout/labels match real US3D exactly
   (this is the ``us3d_paired`` source S_p in docs/EXPERIMENT_PLAN.md, which
   isolates "is our generated RGB faithful enough to replace real RGB?").

It depends on the generation pipeline (weights + flux_train env). Run AFTER
``generation_pipeline/setup.sh`` has populated the weights.

Example:
    python scripts/gen_synth_from_real.py \
        --real-split ../train_pipeline/dataset/train \
        --out-split  ./dataset_synth_us3d/train \
        --seeds 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PIPE_ROOT = HERE.parent  # downstream_seg_depth_pipeline/
GEN_ROOT = (PIPE_ROOT.parent / "generation_pipeline").resolve()

_RGB_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
_LABEL_EXTS = (".png", ".tif", ".tiff", ".exr", ".jpg", ".jpeg")


def _ensure_gen_on_path():
    if not GEN_ROOT.is_dir():
        raise SystemExit(f"generation_pipeline not found at {GEN_ROOT}")
    if str(GEN_ROOT) not in sys.path:
        sys.path.insert(0, str(GEN_ROOT))


def _find(split_dir: Path, sub: str, stem: str) -> Path | None:
    base = split_dir / sub
    for try_stem in (stem, stem.replace("_RGB_", "_")):
        for ext in _LABEL_EXTS:
            cand = base / f"{try_stem}{ext}"
            if cand.exists():
                return cand
    return None


def _list_stems(split_dir: Path) -> list[str]:
    rgb_dir = split_dir / "rgb"
    if not rgb_dir.is_dir():
        raise SystemExit(f"no rgb/ under {split_dir}")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(rgb_dir)
                  if f.lower().endswith(_RGB_EXTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-split", required=True,
                    help="real split dir with rgb/ seg/ depth/ (e.g. .../dataset/train)")
    ap.add_argument("--out-split", required=True,
                    help="output split dir (e.g. ./dataset_synth_us3d/train)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="one generated RGB per seed (>1 seed multiplies tiles)")
    ap.add_argument("--num-steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint dir name under generation_pipeline/weights/lora")
    ap.add_argument("--prompt-json", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only first N tiles (debug)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    _ensure_gen_on_path()
    # Imports from the generation pipeline (need weights + flux_train env).
    from pipeline import DEFAULT_PROMPT_JSON, list_lora_checkpoints, verify_base_weights
    from pipeline.preprocess import preprocess_depth, preprocess_seg
    from pipeline.state import DEVICE, DTYPE, STATE
    from pipeline.inference import sample_ours
    from batch_eval import compose_prompt_from_json, encode_prompt, to_uint8

    real_split = Path(args.real_split).resolve()
    out_split = Path(args.out_split).resolve()
    for sub in ("rgb", "seg", "depth"):
        (out_split / sub).mkdir(parents=True, exist_ok=True)

    missing = verify_base_weights()
    if missing:
        raise SystemExit(f"MISSING base weights: {missing}\nRun generation_pipeline/setup.sh first.")

    if args.ckpt:
        ckpt_dir = (GEN_ROOT / "weights" / "lora" / args.ckpt).resolve()
    else:
        cks = list_lora_checkpoints()
        if not cks:
            raise SystemExit("No checkpoints in generation_pipeline/weights/lora")
        ckpt_dir = cks[0]
    print(f"[gen] checkpoint: {ckpt_dir.name}")

    STATE.load(ckpt_dir, persistent_text_encoder=True)
    prompt_path = args.prompt_json or str(DEFAULT_PROMPT_JSON)
    prompt_text = compose_prompt_from_json(json.loads(Path(prompt_path).read_text()))
    prompt_embed = encode_prompt(prompt_text)

    size = STATE.image_size
    nc = STATE.num_classes

    stems = _list_stems(real_split)
    if args.limit > 0:
        stems = stems[: args.limit]
    print(f"[gen] {len(stems)} real tiles -> {out_split}  (seeds={args.seeds})")

    done = 0
    for stem in stems:
        seg_path = _find(real_split, "seg", stem)
        depth_path = _find(real_split, "depth", stem)
        if not (seg_path and depth_path):
            print(f"  [skip] {stem}: missing seg/depth")
            continue

        seg = preprocess_seg(str(seg_path), size, nc)
        depth = preprocess_depth(str(depth_path), size)

        seeds = list(args.seeds)
        prompt_B = prompt_embed.to(DEVICE, DTYPE).unsqueeze(0).expand(len(seeds), -1, -1).contiguous()
        STATE.lora_enable(True)
        with torch.no_grad():
            ours_rgb, _ = sample_ours(
                seg.unsqueeze(0), depth.unsqueeze(0), prompt_B,
                num_steps=int(args.num_steps), guidance_scale=float(args.cfg),
                seeds=seeds, progress=None,
            )

        for i, sd in enumerate(seeds):
            out_stem = stem if len(seeds) == 1 else f"{stem}_s{sd}"
            rgb_out = out_split / "rgb" / f"{out_stem}.png"
            if rgb_out.exists() and not args.overwrite:
                continue
            Image.fromarray(to_uint8(ours_rgb[i])).save(rgb_out)
            # Copy the ORIGINAL labels so they stay byte-identical to real.
            shutil.copy2(seg_path, out_split / "seg" / f"{out_stem}{seg_path.suffix}")
            shutil.copy2(depth_path, out_split / "depth" / f"{out_stem}{depth_path.suffix}")
        done += 1
        if done % 25 == 0:
            print(f"  ... {done}/{len(stems)}")

    print(f"[gen] done: wrote synthetic tiles for {done} stems -> {out_split}")


if __name__ == "__main__":
    main()

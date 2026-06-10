from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_probe import load_config, train_one


def _condition_cfg(cfg: dict, condition: str, synth_source: str, synth_count: int) -> dict:
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["eval"] = dict(cfg.get("eval", {}))
    if condition == "R":
        cfg["data"]["real_fraction"] = 1.0
        cfg["data"]["synth_count"] = 0
    elif condition == "S":
        cfg["data"]["real_fraction"] = 0.0
        cfg["data"]["synth_count"] = synth_count
        cfg["data"]["synth_source"] = synth_source
    else:
        raise ValueError(f"condition must be R or S, got {condition!r}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=["R", "S"], required=True)
    parser.add_argument("--tasks", nargs="+", default=["segmentation", "height"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synth-source", default="us3d_paired")
    parser.add_argument("--synth-count", type=int, default=100000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    out_dir = Path(args.out)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for task in args.tasks:
        cfg = _condition_cfg(base_cfg, args.condition, args.synth_source, args.synth_count)
        cfg["task"] = task
        cfg["seed"] = args.seed
        cfg.setdefault("eval", {})["dump_dir"] = None
        cfg["eval"]["dump_n"] = 0
        torch.manual_seed(args.seed)
        model, metrics = train_one(cfg, device)
        checkpoint_path = out_dir / "checkpoints" / f"{args.condition}_{task}.pt"
        torch.save({
            "head": model.head.state_dict(),
            "metrics": metrics,
            "cfg": cfg,
            "condition": args.condition,
            "task": task,
        }, checkpoint_path)
        results.append({
            "task": task,
            "condition": args.condition,
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
        })
        print(f"[worker {args.condition}] saved {checkpoint_path}")

    with open(out_dir / "logs" / f"{args.condition}_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
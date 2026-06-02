"""Phase 1: Real vs Synthetic (NO scaling, NO low-data curve).

For each task (segmentation, height) it trains two probes that differ ONLY in
the training data, then evaluates both on the SAME real US3D test set:

    R  : real US3D only            (real_fraction=1.0, synth_count=0)
    S  : synthetic only (TSTR)     (real_fraction=0.0, synth_count=big,
                                     synth_source=us3d_paired)

Outputs (under --out):
    results.json / results.csv          numbers (R vs S, per task)
    figures/<task>_R_vs_S_bars.png      bar chart with the headline result
    figures/qual_<task>_<R|S>/*.png     RGB | GT | pred panels per condition

This is the first thing to run on the server. Look at the bars + panels, then
decide whether to launch the full scaling sweep (experiments/run_scaling.py).
"""

import argparse
import copy
import csv
import json
import os
import statistics

import torch

from scripts.train_probe import load_config, train_one


def _primary(task, m):
    return m["mIoU"] if task == "segmentation" else m["RMSE_m"]


def _run(cfg, task, cond, seed, device, out_dir, synth_big, synth_source):
    c = copy.deepcopy(cfg)
    c["task"] = task
    c["seed"] = seed
    if cond == "R":
        c["data"]["real_fraction"] = 1.0
        c["data"]["synth_count"] = 0
    else:  # S
        c["data"]["real_fraction"] = 0.0
        c["data"]["synth_count"] = synth_big
        c["data"]["synth_source"] = synth_source
    # Dump a few qualitative panels for the FIRST seed only.
    c.setdefault("eval", {})
    if seed == cfg["seed"]:
        c["eval"]["dump_dir"] = os.path.join(out_dir, "figures", f"qual_{task}_{cond}")
        c["eval"]["dump_n"] = 6
    else:
        c["eval"]["dump_dir"] = None
        c["eval"]["dump_n"] = 0
    torch.manual_seed(seed)
    _, metrics = train_one(c, device)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), os.pardir,
                                                     "configs", "default.yaml"))
    ap.add_argument("--tasks", nargs="+", default=["segmentation", "height"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--synth-source", default="us3d_paired",
                    help="which synthetic source feeds S (default: us3d_paired)")
    ap.add_argument("--synth-count", type=int, default=100000,
                    help="cap on synthetic tiles for S (large = use all available)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), os.pardir,
                                                  "output", "phase1"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    rows = []
    for task in args.tasks:
        for cond in ("R", "S"):
            ms = [_run(cfg, task, cond, s, device, args.out, args.synth_count, args.synth_source)
                  for s in args.seeds]
            prim = [_primary(task, m) for m in ms]
            rows.append({
                "task": task, "condition": cond,
                "synth_source": args.synth_source if cond == "S" else "",
                "primary_mean": statistics.mean(prim),
                "primary_std": statistics.pstdev(prim) if len(prim) > 1 else 0.0,
                "raw": ms,
            })
            print(f"[phase1] {task} {cond}: {prim}")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "condition", "synth_source", "primary_mean", "primary_std"])
        for r in rows:
            w.writerow([r["task"], r["condition"], r["synth_source"],
                        f"{r['primary_mean']:.4f}", f"{r['primary_std']:.4f}"])

    # Headline bar chart per task (R vs S).
    try:
        from scripts.visualize import plot_condition_bars

        for task in args.tasks:
            plot_condition_bars(
                rows, task,
                os.path.join(args.out, "figures", f"{task}_R_vs_S_bars.png"),
                title=f"{task}: Real vs Synthetic")
        print(f"[phase1] figures -> {os.path.join(args.out, 'figures')}")
    except Exception as e:  # pragma: no cover
        print(f"[phase1] figure step skipped ({e})")

    print(f"[phase1] done -> {args.out}")


if __name__ == "__main__":
    main()

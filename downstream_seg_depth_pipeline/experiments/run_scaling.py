"""Orchestrate the core ICLR experiments and write a results table.

Conditions per task (segmentation, height), each over multiple seeds:
  * R    : real-only            (real_fraction varies, synth_count=0)
  * S    : synthetic-only TSTR  (real_fraction=0, synth_count=N)
  * R+S  : real + synthetic     (real_fraction varies, synth_count=N)

Two curves:
  * low-data:  real_fraction in {0.1,0.25,0.5,1.0}, +/- fixed synthetic
  * synth-scale: synth_count in {0,1k,5k,10k,25k,50k}, fixed real

Results are aggregated (mean/std across seeds) into a JSON + CSV under output/.
This is a thin driver around scripts.train_probe.train_one.
"""

import argparse
import copy
import csv
import json
import os
import statistics

import torch

from scripts.train_probe import load_config, train_one


def _run(cfg, task, real_fraction, synth_count, seed, device, synth_source=None):
    c = copy.deepcopy(cfg)
    c["task"] = task
    c["seed"] = seed
    c["data"]["real_fraction"] = real_fraction
    c["data"]["synth_count"] = synth_count
    if synth_source is not None:
        c["data"]["synth_source"] = synth_source
    torch.manual_seed(seed)
    _, metrics = train_one(c, device)
    return metrics


def _primary(task, metrics):
    return metrics["mIoU"] if task == "segmentation" else metrics["RMSE_m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), os.pardir,
                                                     "configs", "default.yaml"))
    ap.add_argument("--tasks", nargs="+", default=["segmentation", "height"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--real-fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--synth-counts", nargs="+", type=int, default=[0, 1000, 5000, 10000, 25000, 50000])
    ap.add_argument("--synth-source", default="osm",
                    help="named synthetic source for the curves: 'us3d_paired' or 'osm'")
    ap.add_argument("--fixed-synth", type=int, default=10000,
                    help="synth_count used in the low-data (real-fraction) curve")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), os.pardir,
                                                  "output", "scaling_results"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    rows = []

    def record(task, cond, rf, sc, metrics_list):
        prim = [_primary(task, m) for m in metrics_list]
        rows.append({
            "task": task, "condition": cond, "real_fraction": rf, "synth_count": sc,
            "synth_source": args.synth_source,
            "primary_mean": statistics.mean(prim),
            "primary_std": statistics.pstdev(prim) if len(prim) > 1 else 0.0,
            "raw": metrics_list,
        })

    for task in args.tasks:
        # --- low-data curve: real fraction sweep, +/- fixed synthetic ---
        for rf in args.real_fractions:
            for sc, cond in ((0, "R"), (args.fixed_synth, "R+S")):
                ms = [_run(cfg, task, rf, sc, s, device, args.synth_source) for s in args.seeds]
                record(task, cond, rf, sc, ms)
        # --- synthetic-only TSTR at full synthetic budget ---
        ms = [_run(cfg, task, 0.0, max(args.synth_counts), s, device, args.synth_source)
              for s in args.seeds]
        record(task, "S", 0.0, max(args.synth_counts), ms)
        # --- synth-scale curve: fixed full real, growing synthetic ---
        for sc in args.synth_counts:
            ms = [_run(cfg, task, 1.0, sc, s, device, args.synth_source) for s in args.seeds]
            record(task, "scale", 1.0, sc, ms)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "condition", "real_fraction", "synth_count", "synth_source",
                    "primary_mean", "primary_std"])
        for r in rows:
            w.writerow([r["task"], r["condition"], r["real_fraction"], r["synth_count"],
                        r.get("synth_source", ""),
                        f"{r['primary_mean']:.4f}", f"{r['primary_std']:.4f}"])
    print(f"[done] wrote results to {args.out}")

    # Emit the human-readable KR figures (bars + curves) next to the tables.
    try:
        from scripts.visualize import figures_from_results

        fig_dir = os.path.join(args.out, "figures")
        made = figures_from_results(os.path.join(args.out, "results.json"), fig_dir)
        print(f"[figures] wrote {len(made)} PNGs to {fig_dir}")
    except Exception as e:  # pragma: no cover - plotting is best-effort
        print(f"[figures] skipped ({e})")


if __name__ == "__main__":
    main()

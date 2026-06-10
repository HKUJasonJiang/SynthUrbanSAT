#!/usr/bin/env python3
"""Score generated RGB candidates with traceable artifact metrics.

This is a bad-image filter, not a final scientific quality oracle. It compares
synthetic image statistics against real RGB statistics from the selected OSM
manifest and writes per-candidate scores plus aggregate failure tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_VERSION = "artifact_score_v1"
SEED_RE = re.compile(r"rgb_seed_(\d+)\.png$")
VIEW_RE = re.compile(r"osm_batch__.+?__(root|near-nadir-\d+)__depth-(png|exr)__")


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


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _read_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "selected" not in data:
        raise ValueError(f"Selection manifest missing 'selected': {path}")
    return data


def _load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size and img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _metrics(path: Path, *, size: int | None = 512) -> dict[str, float]:
    arr = _load_rgb(path, size=size).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    # Fast Laplacian variance without OpenCV.
    gray = luma
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )

    hist, _ = np.histogram((luma * 255).astype(np.uint8), bins=256, range=(0, 255), density=False)
    probs = hist.astype(np.float64)
    probs = probs / max(probs.sum(), 1.0)
    entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum())

    return {
        "mean_luma": float(luma.mean()),
        "std_luma": float(luma.std()),
        "p01_luma": float(np.quantile(luma, 0.01)),
        "p99_luma": float(np.quantile(luma, 0.99)),
        "black_ratio": float((luma < 0.02).mean()),
        "white_ratio": float((luma > 0.98).mean()),
        "saturation_mean": float(sat.mean()),
        "saturation_p95": float(np.quantile(sat, 0.95)),
        "laplacian_var": float(lap.var()),
        "entropy": entropy,
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    xs = np.asarray(values, dtype=np.float64)
    return {
        "p01": float(np.quantile(xs, 0.01)),
        "p05": float(np.quantile(xs, 0.05)),
        "p50": float(np.quantile(xs, 0.50)),
        "p95": float(np.quantile(xs, 0.95)),
        "p99": float(np.quantile(xs, 0.99)),
        "mean": float(xs.mean()),
        "std": float(xs.std()),
    }


def build_thresholds(real_metric_rows: list[dict[str, float]], args: argparse.Namespace) -> dict[str, dict[str, float]]:
    by_metric = {k: [r[k] for r in real_metric_rows] for k in real_metric_rows[0]}
    stats = {k: _percentiles(v) for k, v in by_metric.items()}
    thresholds = {
        "mean_luma": {"min": max(0.0, stats["mean_luma"]["p01"] - args.luma_margin), "max": min(1.0, stats["mean_luma"]["p99"] + args.luma_margin)},
        "std_luma": {"min": stats["std_luma"]["p01"] * args.std_low_factor},
        "black_ratio": {"max": max(stats["black_ratio"]["p99"] * args.clip_factor, args.black_ratio_floor)},
        "white_ratio": {"max": max(stats["white_ratio"]["p99"] * args.clip_factor, args.white_ratio_floor)},
        "saturation_mean": {"min": max(0.0, stats["saturation_mean"]["p01"] - args.saturation_margin), "max": min(1.0, stats["saturation_mean"]["p99"] + args.saturation_margin)},
        "saturation_p95": {"max": min(1.0, stats["saturation_p95"]["p99"] + args.saturation_margin)},
        "laplacian_var": {"min": stats["laplacian_var"]["p01"] * args.lap_low_factor},
        "entropy": {"min": stats["entropy"]["p01"] * args.entropy_low_factor},
    }
    return {"real_stats": stats, "thresholds": thresholds}


def _apply_thresholds(metrics: dict[str, float], thresholds: dict[str, dict[str, float]]) -> tuple[list[str], float]:
    flags: list[str] = []
    severity = 0.0
    for metric, rule in thresholds.items():
        val = metrics[metric]
        if "min" in rule and val < rule["min"]:
            flags.append(f"{metric}_low")
            severity += (rule["min"] - val) / (abs(rule["min"]) + 1e-6)
        if "max" in rule and val > rule["max"]:
            flags.append(f"{metric}_high")
            severity += (val - rule["max"]) / (abs(rule["max"]) + 1e-6)
    return flags, float(severity)


def _view_from_root(root: Path) -> tuple[str, str]:
    m = VIEW_RE.search(root.name)
    if m:
        return m.group(1), m.group(2)
    if "near-nadir" in root.name:
        m2 = re.search(r"near-nadir-\d+", root.name)
        return (m2.group(0) if m2 else "unknown", "png")
    if "root" in root.name:
        return "root", "png"
    return "unknown", "png"


def scan_generated(generated_roots: list[Path], selected_tiles: set[str]) -> list[dict]:
    rows: list[dict] = []
    for root in generated_roots:
        view, depth_tag = _view_from_root(root)
        for p in sorted(root.glob("tile_*/**/rgb_seed_*.png")):
            tile = p.parts[p.parts.index(root.name) + 1] if root.name in p.parts else p.parents[2].name
            if tile not in selected_tiles:
                continue
            m = SEED_RE.search(p.name)
            seed = int(m.group(1)) if m else -1
            parts = p.relative_to(root).parts
            view_name = parts[1] if len(parts) > 2 else view
            depth_name = parts[2] if len(parts) > 3 else f"depth_{depth_tag}"
            rows.append({
                "tile": tile,
                "view": view_name,
                "depth_tag": depth_name.replace("depth_", ""),
                "seed": seed,
                "path": str(p),
                "generated_root": str(root),
            })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for vals, items in sorted(groups.items(), key=lambda kv: kv[0]):
        n = len(items)
        failed = sum(str(i.get("failed", "0")) in ("1", "True", "true") for i in items)
        flags = sum(1 for i in items if i.get("flags"))
        rec = {k: v for k, v in zip(keys, vals)}
        rec.update({"count": n, "failed": failed, "flagged": flags, "failure_rate": failed / n if n else 0.0})
        out.append(rec)
    return out


def _make_review_grid(rows: list[dict], out_path: Path, *, title: str, limit: int, thumb: int = 192) -> None:
    rows = rows[:limit]
    if not rows:
        return
    cols = min(5, len(rows))
    label_h = 48
    rows_n = int(math.ceil(len(rows) / cols))
    img = Image.new("RGB", (cols * thumb, rows_n * (thumb + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = _font(13)
    for i, rec in enumerate(rows):
        x = (i % cols) * thumb
        y = (i // cols) * (thumb + label_h)
        tile = Image.open(rec["path"]).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        img.paste(tile, (x, y + label_h))
        draw.rectangle([x, y, x + thumb, y + label_h], fill=(35, 35, 38))
        label = f"{rec['tile']} {rec['view']} s={rec['seed']}"
        label2 = f"score={float(rec['artifact_score']):.2f} {rec.get('flags','')[:30]}"
        draw.text((x + 4, y + 5), label, fill=(240, 240, 240), font=font)
        draw.text((x + 4, y + 25), label2, fill=(220, 220, 220), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", required=True, help="selected_tiles.json from selection_stats.py")
    ap.add_argument("--generated-root", nargs="+", required=True, help="One or more osm_batch output folders")
    ap.add_argument("--out", default="generation_pipeline/output/selection_omaha984_60tiles/artifact_eval")
    ap.add_argument("--metric-size", type=int, default=512)
    ap.add_argument("--luma-margin", type=float, default=0.08)
    ap.add_argument("--saturation-margin", type=float, default=0.15)
    ap.add_argument("--std-low-factor", type=float, default=0.70)
    ap.add_argument("--lap-low-factor", type=float, default=0.50)
    ap.add_argument("--entropy-low-factor", type=float, default=0.85)
    ap.add_argument("--clip-factor", type=float, default=2.0)
    ap.add_argument("--black-ratio-floor", type=float, default=0.05)
    ap.add_argument("--white-ratio-floor", type=float, default=0.05)
    ap.add_argument("--fail-flag-count", type=int, default=2, help="fail if candidate has at least this many hard flags")
    args = ap.parse_args(argv)

    selection_path = Path(args.selection).expanduser().resolve()
    generated_roots = [Path(x).expanduser().resolve() for x in args.generated_root]
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = _read_manifest(selection_path)
    selected_rows = selection["selected"]
    selected_tiles = {r["tile"] for r in selected_rows}
    by_tile = {r["tile"]: r for r in selected_rows}

    real_metric_rows = []
    for rec in selected_rows:
        rgb = rec.get("rgb_path")
        if rgb and Path(rgb).exists():
            m = _metrics(Path(rgb), size=args.metric_size)
            m.update({"tile": rec["tile"], "selection_group": rec["selection_group"], "path": rgb})
            real_metric_rows.append(m)
    if not real_metric_rows:
        raise SystemExit("No real RGB references found in selection manifest")

    metric_only = [{k: v for k, v in r.items() if isinstance(v, float)} for r in real_metric_rows]
    threshold_pack = build_thresholds(metric_only, args)
    thresholds = threshold_pack["thresholds"]

    candidates = scan_generated(generated_roots, selected_tiles)
    scored: list[dict] = []
    for cand in candidates:
        p = Path(cand["path"])
        metric = _metrics(p, size=args.metric_size)
        flags, severity = _apply_thresholds(metric, thresholds)
        sel = by_tile.get(cand["tile"], {})
        rec = dict(cand)
        rec.update({
            "selection_group": sel.get("selection_group", ""),
            "building_ratio": sel.get("building_ratio", ""),
            "foliage_ratio": sel.get("foliage_ratio", ""),
            "water_ratio": sel.get("water_ratio", ""),
        })
        rec.update(metric)
        rec["flags"] = ";".join(flags)
        rec["flag_count"] = len(flags)
        rec["artifact_score"] = severity
        rec["failed"] = int(len(flags) >= args.fail_flag_count)
        scored.append(rec)

    scored = sorted(scored, key=lambda r: (int(r["failed"]), float(r["artifact_score"]), str(r["tile"])), reverse=True)
    _write_csv(out_dir / "real_rgb_metrics.csv", real_metric_rows)
    _write_csv(out_dir / "artifact_scores.csv", scored)
    _write_csv(out_dir / "artifact_summary_by_view.csv", _aggregate(scored, ["view"]))
    _write_csv(out_dir / "artifact_summary_by_seed.csv", _aggregate(scored, ["seed"]))
    _write_csv(out_dir / "artifact_summary_by_group.csv", _aggregate(scored, ["selection_group"]))
    _write_csv(out_dir / "artifact_summary_by_view_seed.csv", _aggregate(scored, ["view", "seed"]))
    _write_csv(out_dir / "artifact_summary_by_view_group.csv", _aggregate(scored, ["view", "selection_group"]))

    failed = [r for r in scored if int(r["failed"])]
    accepted = [r for r in scored if not int(r["failed"])]
    borderline = [r for r in accepted if int(r["flag_count"]) > 0]
    (out_dir / "failed_candidates.json").write_text(json.dumps(failed, indent=2))
    (out_dir / "borderline_candidates.json").write_text(json.dumps(borderline, indent=2))
    (out_dir / "accepted_candidates.json").write_text(json.dumps(accepted, indent=2))

    review_dir = out_dir / "review_grids"
    _make_review_grid(failed, review_dir / "worst_failed.png", title="worst failed", limit=25)
    _make_review_grid(scored, review_dir / "worst_overall.png", title="worst overall", limit=25)
    rng = np.random.default_rng(0)
    if accepted:
        sample_idx = rng.choice(len(accepted), size=min(25, len(accepted)), replace=False)
        _make_review_grid([accepted[i] for i in sample_idx], review_dir / "random_accepted.png", title="random accepted", limit=25)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "git": _git_info(),
        "selection": str(selection_path),
        "generated_roots": [str(p) for p in generated_roots],
        "threshold_pack": threshold_pack,
        "counts": {"real_refs": len(real_metric_rows), "candidates": len(scored), "failed": len(failed), "borderline": len(borderline), "accepted": len(accepted)},
    }
    (out_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))

    summary_lines = [
        "# Artifact Evaluation Summary",
        "",
        f"Generated: {manifest['created_utc']}",
        f"Selection: `{selection_path}`",
        f"Candidates: {len(scored)}",
        f"Failed: {len(failed)}",
        f"Borderline: {len(borderline)}",
        f"Accepted: {len(accepted)}",
        "",
        "## Thresholds",
        "",
        "```json",
        json.dumps(thresholds, indent=2),
        "```",
    ]
    (out_dir / "artifact_summary.md").write_text("\n".join(summary_lines) + "\n")

    print(f"Candidates: {len(scored)} failed={len(failed)} borderline={len(borderline)} accepted={len(accepted)}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

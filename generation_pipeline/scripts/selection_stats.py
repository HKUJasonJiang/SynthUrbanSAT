#!/usr/bin/env python3
"""Select representative OSM tiles by segmentation distribution.

Outputs are intentionally traceable: all tile metrics, selected groups, rules,
command line, and summary statistics are written next to preview grids.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_VERSION = "selection_stats_v1"
DEFAULT_CLASSES = {
    "road": (0, 0, 255),
    "water": (0, 225, 255),
    "foliage": (0, 255, 0),
    "building": (255, 0, 0),
    "grass": (128, 0, 128),
    "ground": (0, 0, 0),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_color_map() -> Path:
    return _repo_root() / "train_pipeline" / "configs" / "color_map.json"


def _git_info() -> dict:
    root = _repo_root()
    out: dict[str, object] = {"repo": str(root)}
    try:
        out["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        out["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def _tile_sort_key(path: Path):
    import re

    m = re.search(r"(\d+)$", path.name)
    return (int(m.group(1)) if m else 10**12, path.name)


def _load_class_map(path: Path) -> dict[str, tuple[int, int, int]]:
    if not path.exists():
        return DEFAULT_CLASSES.copy()
    data = json.loads(path.read_text())
    out: dict[str, tuple[int, int, int]] = {}
    if "classes" in data:
        data = data["classes"]
    for _, spec in sorted(data.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])):
        out[str(spec["name"])] = tuple(int(x) for x in spec["rgb"])
    return out


def _decode_rgb_seg(seg_path: Path, classes: dict[str, tuple[int, int, int]]) -> tuple[np.ndarray, dict[str, int], int]:
    arr = np.array(Image.open(seg_path).convert("RGB"), dtype=np.uint8)
    flat = arr.reshape(-1, 3)
    encoded = flat[:, 0].astype(np.uint32) << 16 | flat[:, 1].astype(np.uint32) << 8 | flat[:, 2].astype(np.uint32)
    vals, counts = np.unique(encoded, return_counts=True)
    by_color = {int(v): int(c) for v, c in zip(vals, counts)}
    class_counts: dict[str, int] = {}
    for name, rgb in classes.items():
        code = int(rgb[0]) << 16 | int(rgb[1]) << 8 | int(rgb[2])
        class_counts[name] = by_color.get(code, 0)
    known = sum(class_counts.values())
    class_counts["unknown"] = int(flat.shape[0] - known)
    return arr, class_counts, int(flat.shape[0])


def _entropy(ratios: Iterable[float]) -> float:
    xs = np.asarray([x for x in ratios if x > 0], dtype=np.float64)
    if xs.size == 0:
        return 0.0
    return float(-(xs * np.log2(xs)).sum())


def _sha256_short(path: Path, n_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def scan_tiles(input_dir: Path, classes: dict[str, tuple[int, int, int]], seg_name: str, rgb_name: str) -> list[dict]:
    rows: list[dict] = []
    for tile_dir in sorted([p for p in input_dir.glob("tile_*") if p.is_dir()], key=_tile_sort_key):
        seg_path = tile_dir / seg_name
        if not seg_path.exists():
            continue
        _, counts, total = _decode_rgb_seg(seg_path, classes)
        rec: dict[str, object] = {
            "tile": tile_dir.name,
            "tile_dir": str(tile_dir),
            "seg_path": str(seg_path),
            "rgb_path": str(tile_dir / rgb_name) if (tile_dir / rgb_name).exists() else "",
            "seg_sha256_1mb": _sha256_short(seg_path),
            "total_pixels": total,
        }
        for name in classes:
            rec[f"{name}_pixels"] = counts.get(name, 0)
            rec[f"{name}_ratio"] = counts.get(name, 0) / total
        rec["unknown_pixels"] = counts.get("unknown", 0)
        rec["unknown_ratio"] = counts.get("unknown", 0) / total
        rec["green_total_ratio"] = float(rec.get("foliage_ratio", 0.0)) + float(rec.get("grass_ratio", 0.0))
        class_ratios = [float(rec[f"{name}_ratio"]) for name in classes]
        rec["class_entropy"] = _entropy(class_ratios)
        rec["dominant_class"] = max(classes, key=lambda name: float(rec[f"{name}_ratio"]))
        rows.append(rec)
    return rows


def _quantiles(rows: list[dict], key: str) -> dict[str, float]:
    xs = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return {
        "mean": float(xs.mean()) if xs.size else 0.0,
        "p50": float(np.quantile(xs, 0.50)) if xs.size else 0.0,
        "p75": float(np.quantile(xs, 0.75)) if xs.size else 0.0,
        "p90": float(np.quantile(xs, 0.90)) if xs.size else 0.0,
        "p95": float(np.quantile(xs, 0.95)) if xs.size else 0.0,
        "p99": float(np.quantile(xs, 0.99)) if xs.size else 0.0,
        "max": float(xs.max()) if xs.size else 0.0,
        "count_gt_0": int((xs > 0).sum()),
        "count_ge_10pct": int((xs >= 0.10).sum()),
    }


def summarize(rows: list[dict], classes: dict[str, tuple[int, int, int]]) -> dict:
    keys = [f"{name}_ratio" for name in classes] + ["green_total_ratio", "class_entropy"]
    return {key: _quantiles(rows, key) for key in keys}


def _select_ranked(rows: list[dict], *, metric: str, count: int, used: set[str], threshold: float | None = None) -> tuple[list[dict], list[dict]]:
    candidates = [r for r in rows if r["tile"] not in used]
    if threshold is not None:
        preferred = [r for r in candidates if float(r[metric]) >= threshold]
        fallback = [r for r in candidates if float(r[metric]) < threshold]
        ordered = sorted(preferred, key=lambda r: (float(r[metric]), str(r["tile"])), reverse=True)
        selected = ordered[:count]
        if len(selected) < count:
            selected.extend(sorted(fallback, key=lambda r: (float(r[metric]), str(r["tile"])), reverse=True)[: count - len(selected)])
    else:
        selected = sorted(candidates, key=lambda r: (float(r[metric]), str(r["tile"])), reverse=True)[:count]
    return selected, candidates


def select_groups(rows: list[dict], group_size: int, water_threshold: float) -> tuple[dict[str, list[dict]], dict]:
    used: set[str] = set()
    groups: dict[str, list[dict]] = {}

    water, _ = _select_ranked(rows, metric="water_ratio", count=group_size, used=used, threshold=water_threshold)
    groups["water"] = water
    used.update(str(r["tile"]) for r in water)

    building, _ = _select_ranked(rows, metric="building_ratio", count=group_size, used=used)
    groups["building"] = building
    used.update(str(r["tile"]) for r in building)

    tree, _ = _select_ranked(rows, metric="foliage_ratio", count=group_size, used=used)
    groups["tree"] = tree
    used.update(str(r["tile"]) for r in tree)

    info = {
        "policy": "disjoint groups; priority water -> building -> tree",
        "group_size": group_size,
        "water_threshold": water_threshold,
        "water_candidates_ge_threshold": int(sum(float(r["water_ratio"]) >= water_threshold for r in rows)),
        "water_fallback_used": int(sum(float(r["water_ratio"]) < water_threshold for r in water)),
    }
    return groups, info


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_preview(group: str, rows: list[dict], out_path: Path, *, thumb: int = 160, cols: int = 5) -> None:
    if not rows:
        return
    cell_w = thumb * 2
    label_h = 44
    cell_h = thumb + label_h
    rows_n = int(math.ceil(len(rows) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows_n * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    small = _font(12)
    for idx, rec in enumerate(rows):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        rgb_path = Path(str(rec.get("rgb_path") or ""))
        seg_path = Path(str(rec["seg_path"]))
        rgb = Image.open(rgb_path).convert("RGB") if rgb_path.exists() else Image.new("RGB", (thumb, thumb), (90, 90, 90))
        seg = Image.open(seg_path).convert("RGB")
        rgb.thumbnail((thumb, thumb), Image.LANCZOS)
        seg.thumbnail((thumb, thumb), Image.NEAREST)
        canvas.paste(rgb.resize((thumb, thumb)), (x, y + label_h))
        canvas.paste(seg.resize((thumb, thumb), Image.NEAREST), (x + thumb, y + label_h))
        draw.rectangle([x, y, x + cell_w, y + label_h], fill=(35, 35, 38))
        text = f"{rec['tile']} {group}"
        nums = f"b={float(rec['building_ratio']):.2f} f={float(rec['foliage_ratio']):.2f} w={float(rec['water_ratio']):.2f}"
        draw.text((x + 4, y + 4), text, fill=(240, 240, 240), font=font)
        draw.text((x + 4, y + 23), nums, fill=(220, 220, 220), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_summary_md(path: Path, *, input_dir: Path, rows: list[dict], groups: dict[str, list[dict]], summary: dict, select_info: dict, args: argparse.Namespace) -> None:
    lines = [
        "# Omaha Selection Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Input: `{input_dir}`",
        f"Tiles scanned: {len(rows)}",
        f"Group size: {args.group_size}",
        f"Water threshold: {args.water_threshold}",
        f"Water fallback used: {select_info['water_fallback_used']}",
        "",
        "## Distribution Summary",
        "",
        "| Metric | Mean | P50 | P75 | P90 | P95 | P99 | Max | Count >= 10% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, val in summary.items():
        lines.append(
            f"| {key} | {val['mean']:.4f} | {val['p50']:.4f} | {val['p75']:.4f} | {val['p90']:.4f} | {val['p95']:.4f} | {val['p99']:.4f} | {val['max']:.4f} | {val['count_ge_10pct']} |"
        )
    lines.extend(["", "## Selected Groups", ""])
    for group, selected in groups.items():
        metric = "water_ratio" if group == "water" else "building_ratio" if group == "building" else "foliage_ratio"
        lines.extend([f"### {group}", "", "| Tile | Building | Foliage | Grass | Water | Road | Entropy |", "|---|---:|---:|---:|---:|---:|---:|"])
        for r in selected:
            lines.append(
                f"| {r['tile']} | {float(r['building_ratio']):.4f} | {float(r['foliage_ratio']):.4f} | {float(r['grass_ratio']):.4f} | {float(r['water_ratio']):.4f} | {float(r['road_ratio']):.4f} | {float(r['class_entropy']):.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, help="OSM output folder, e.g. osm_pipeline/output/omaha-984")
    ap.add_argument("--out", default="generation_pipeline/output/selection_omaha984_60tiles")
    ap.add_argument("--color-map", default=str(_default_color_map()))
    ap.add_argument("--seg-name", default="4_seg.png")
    ap.add_argument("--rgb-name", default="2_rgb.png")
    ap.add_argument("--group-size", type=int, default=20)
    ap.add_argument("--water-threshold", type=float, default=0.10)
    ap.add_argument("--preview-thumb", type=int, default=160)
    args = ap.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = _load_class_map(Path(args.color_map).expanduser().resolve())

    rows = scan_tiles(input_dir, classes, args.seg_name, args.rgb_name)
    if not rows:
        raise SystemExit(f"No tiles with {args.seg_name} found under {input_dir}")
    summary = summarize(rows, classes)
    groups, select_info = select_groups(rows, args.group_size, args.water_threshold)

    write_csv(out_dir / "seg_distribution_all_tiles.csv", rows)
    selected_rows = []
    for group, selected in groups.items():
        for rank, rec in enumerate(selected, start=1):
            row = dict(rec)
            row["selection_group"] = group
            row["selection_rank"] = rank
            selected_rows.append(row)
    write_csv(out_dir / "selected_tiles.csv", selected_rows)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "git": _git_info(),
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "class_map": {k: list(v) for k, v in classes.items()},
        "summary": summary,
        "selection_info": select_info,
        "groups": {g: [r["tile"] for r in rs] for g, rs in groups.items()},
        "selected": selected_rows,
    }
    (out_dir / "selected_tiles.json").write_text(json.dumps(manifest, indent=2))

    for group, selected in groups.items():
        make_preview(group, selected, out_dir / f"preview_grid_{group}.png", thumb=args.preview_thumb)
    write_summary_md(out_dir / "selection_summary.md", input_dir=input_dir, rows=rows, groups=groups, summary=summary, select_info=select_info, args=args)

    print(f"Scanned {len(rows)} tiles")
    for group, selected in groups.items():
        print(f"{group}: {len(selected)} tiles -> {', '.join(str(r['tile']) for r in selected[:5])}{' ...' if len(selected) > 5 else ''}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

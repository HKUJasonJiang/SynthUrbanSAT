from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import TileDataset
from scripts.labels import LabelSpace
from scripts.model import ProbeModel
from scripts.train_probe import load_config
from scripts.visualize import colorize_seg, denorm_rgb


def _load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = ProbeModel(
        task=ckpt["task"], num_classes=cfg["num_classes"],
        backbone_name=cfg["backbone"]["name"], out_indices=tuple(cfg["backbone"]["out_indices"]),
        head_type=cfg["head"]["type"], hidden_dim=cfg["head"]["hidden_dim"],
        freeze_backbone=cfg["backbone"]["freeze"], ndsm_max_m=cfg["ndsm_max_m"],
    ).to(device)
    model.head.load_state_dict(ckpt["head"])
    model.eval()
    return model, ckpt


def _height_rgb(height: np.ndarray, vmax: float) -> np.ndarray:
    x = np.asarray(height, dtype=np.float32)
    x = np.where(np.isfinite(x), x, np.nan)
    x = np.clip(x / max(vmax, 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap("viridis")
    rgb = (cmap(np.nan_to_num(x, nan=0.0))[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(height)] = np.array([80, 80, 80], dtype=np.uint8)
    return rgb


def _err_rgb(err: np.ndarray, vmax: float) -> np.ndarray:
    x = np.clip(np.asarray(err, dtype=np.float32) / max(vmax, 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap("magma")
    return (cmap(np.nan_to_num(x, nan=0.0))[..., :3] * 255).astype(np.uint8)


def _read_synth_rgb(root: Path, stem: str, size: int) -> np.ndarray:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = root / "test" / "rgb" / f"{stem}{ext}"
        if path.exists():
            return np.array(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))
    raise FileNotFoundError(stem)


def _seconds_label(seconds: float) -> str:
    minutes = seconds / 60.0
    return f"{minutes:.1f} min" if minutes < 120 else f"{minutes / 60.0:.2f} h"


def _plot_curves(checkpoints: dict, out_dir: Path) -> None:
    train_times = {k: v["metrics"].get("train_seconds", 0.0) for k, v in checkpoints.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for task, ax in zip(("segmentation", "height"), axes):
        for cond, color in (("R", "#2f6fbb"), ("S", "#c25b35")):
            hist = checkpoints[(cond, task)]["metrics"].get("history", {})
            loss = hist.get("train_loss", [])
            if loss:
                ax.plot(range(1, len(loss) + 1), loss, label=f"{cond} ({_seconds_label(train_times[(cond, task)])})", color=color)
        ax.set_title(f"{task} train loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fair R vs S loss curves; train uses train+val, test uses real test")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for cond, color in (("R", "#2f6fbb"), ("S", "#c25b35")):
        hist = checkpoints[(cond, "height")]["metrics"].get("history", {})
        mse = hist.get("train_mse_m2", [])
        if mse:
            ax.plot(range(1, len(mse) + 1), mse, label=f"{cond} ({_seconds_label(train_times[(cond, 'height')])})", color=color)
    ax.set_title("height train MSE curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (m^2)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "height_mse_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_results(checkpoints: dict, out_dir: Path) -> None:
    rows = []
    for (cond, task), ckpt in checkpoints.items():
        metrics = ckpt["metrics"]
        primary = metrics.get("mIoU") if task == "segmentation" else metrics.get("RMSE_m")
        rows.append({"condition": cond, "task": task, "primary": primary, "metrics": metrics})
    with open(out_dir / "results.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "task", "primary", "train_seconds", "train_size"])
        for row in rows:
            m = row["metrics"]
            writer.writerow([row["condition"], row["task"], row["primary"], m.get("train_seconds"), m.get("train_size")])


def _make_qualitative(cfg: dict, checkpoints: dict, out_dir: Path, sample_count: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = LabelSpace()
    size = int(cfg["image_size"])
    real_root = Path(cfg["data"]["real_root"])
    synth_root = Path(cfg["data"]["synth_sources"]["us3d_paired"])
    ds = TileDataset(str(real_root / "test"), labels, size, "segmentation")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    models = {}
    for cond in ("R", "S"):
        for task in ("segmentation", "height"):
            models[(cond, task)], _ = _load_model(out_dir / "checkpoints" / f"{cond}_{task}.pt", device)

    rows = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= sample_count:
                break
            stem = batch["stem"][0]
            rgb_t = batch["rgb"].to(device)
            rgb_real = denorm_rgb(batch["rgb"][0].numpy())
            rgb_synth = _read_synth_rgb(synth_root, stem, size)
            gt_seg = colorize_seg(batch["seg"][0].numpy(), labels.rgb, cfg["ignore_index"])
            gt_h = batch["height"][0, 0].numpy()
            vmax = float(np.nanpercentile(gt_h[np.isfinite(gt_h)], 99)) if np.isfinite(gt_h).any() else cfg["ndsm_max_m"]
            vmax = max(vmax, 1.0)
            panels = [("real RGB", rgb_real), ("synth RGB", rgb_synth), ("GT seg", gt_seg), ("GT depth", _height_rgb(gt_h, vmax))]
            for cond in ("R", "S"):
                seg_pred = models[(cond, "segmentation")](rgb_t).argmax(1)[0].cpu().numpy()
                h_pred = models[(cond, "height")](rgb_t)[0, 0].cpu().numpy()
                panels.append((f"{cond} seg", colorize_seg(seg_pred, labels.rgb, cfg["ignore_index"])))
                panels.append((f"{cond} depth", _height_rgb(h_pred, vmax)))
                panels.append((f"{cond} |err|", _err_rgb(np.abs(h_pred - gt_h), vmax)))
            rows.append((stem, panels))

    cols = max(len(p) for _, p in rows)
    thumb = 170
    label_h = 28
    stem_w = 135
    fig_w = stem_w + cols * thumb
    fig_h = max(1, len(rows)) * (thumb + label_h)
    canvas = Image.new("RGB", (fig_w, fig_h), "white")
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(canvas)
    for r, (stem, panels) in enumerate(rows):
        y = r * (thumb + label_h)
        draw.text((4, y + 8), stem, fill=(0, 0, 0))
        for c, (title, img) in enumerate(panels):
            x = stem_w + c * thumb
            draw.rectangle([x, y, x + thumb, y + label_h], fill=(35, 35, 35))
            draw.text((x + 5, y + 7), title, fill=(245, 245, 245))
            tile = Image.fromarray(img).resize((thumb, thumb), Image.BILINEAR)
            canvas.paste(tile, (x, y + label_h))
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    canvas.save(figures / "qualitative_comparison_grid.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out)
    checkpoints = {}
    for cond in ("R", "S"):
        for task in ("segmentation", "height"):
            ckpt = torch.load(out_dir / "checkpoints" / f"{cond}_{task}.pt", map_location="cpu", weights_only=False)
            checkpoints[(cond, task)] = ckpt
    cfg = load_config(args.config) if Path(args.config).exists() else checkpoints[("R", "segmentation")]["cfg"]
    _write_results(checkpoints, out_dir)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_curves(checkpoints, figures)
    _make_qualitative(cfg, checkpoints, out_dir, args.samples)
    print(f"[report] figures -> {figures}")


if __name__ == "__main__":
    main()
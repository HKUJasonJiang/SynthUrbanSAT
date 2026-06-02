"""Human-readable result figures (the KR deliverables).

Every Key Result in docs/EXPERIMENT_PLAN.md maps to a PNG produced here so a
reader can *see* the conclusion, not just read a number:

  * qualitative panels   -> save_seg_triptych / save_height_panel
  * condition bars        -> plot_condition_bars   (R / S / R+S)
  * low-data curve        -> plot_lowdata_curve     (R vs R+S over real fraction)
  * synthetic scaling     -> plot_scale_curve       (metric vs synth count)

All functions take plain numpy arrays / parsed result rows and write a PNG;
matplotlib uses the non-interactive 'Agg' backend so this runs headless on a
server. Only matplotlib + numpy are required.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def denorm_rgb(rgb_chw: np.ndarray) -> np.ndarray:
    """ImageNet-normalised CHW float -> HWC uint8 for display."""
    x = np.asarray(rgb_chw, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)


def colorize_seg(idx_map: np.ndarray, palette_rgb, ignore_index: int = 255) -> np.ndarray:
    """Class-index map -> RGB uint8 using the label palette."""
    idx = np.asarray(idx_map)
    h, w = idx.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(palette_rgb):
        out[idx == i] = np.array(color, dtype=np.uint8)
    out[idx == ignore_index] = np.array([128, 128, 128], dtype=np.uint8)
    return out


def _ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# qualitative panels
# --------------------------------------------------------------------------- #
def save_seg_triptych(rgb_chw, gt_idx, pred_idx, palette_rgb, path,
                      title="segmentation", ignore_index=255):
    """RGB | ground-truth | prediction, side by side."""
    _ensure_dir(path)
    rgb = denorm_rgb(rgb_chw)
    gt = colorize_seg(gt_idx, palette_rgb, ignore_index)
    pr = colorize_seg(pred_idx, palette_rgb, ignore_index)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    for a, im, t in zip(ax, (rgb, gt, pr), ("RGB", "GT", "Pred")):
        a.imshow(im)
        a.set_title(t, fontsize=10)
        a.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_height_panel(rgb_chw, gt_m, pred_m, path, title="AGL height (m)", vmax=None):
    """RGB | GT height | Pred height | abs-error, with shared colour scale."""
    _ensure_dir(path)
    rgb = denorm_rgb(rgb_chw)
    gt = np.asarray(gt_m, dtype=np.float32)
    pr = np.asarray(pred_m, dtype=np.float32)
    finite = np.isfinite(gt)
    if vmax is None:
        vmax = float(np.nanpercentile(gt[finite], 99)) if finite.any() else 1.0
        vmax = max(vmax, 1e-3)
    err = np.where(finite, np.abs(pr - gt), np.nan)

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    ax[0].imshow(rgb)
    ax[0].set_title("RGB", fontsize=10)
    ax[0].axis("off")
    for a, im, t in ((ax[1], gt, "GT height"), (ax[2], pr, "Pred height")):
        m = a.imshow(im, cmap="viridis", vmin=0, vmax=vmax)
        a.set_title(t, fontsize=10)
        a.axis("off")
        fig.colorbar(m, ax=a, fraction=0.046, pad=0.04)
    me = ax[3].imshow(err, cmap="magma", vmin=0, vmax=vmax)
    ax[3].set_title("|error| (m)", fontsize=10)
    ax[3].axis("off")
    fig.colorbar(me, ax=ax[3], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# quantitative plots (consume rows written by experiments/run_scaling.py)
# --------------------------------------------------------------------------- #
def _metric_label(task):
    return "mIoU" if task == "segmentation" else "RMSE (m)"


def plot_condition_bars(rows, task, path, title=None):
    """Bar chart of primary metric across conditions (R / S / R+S) with std bars."""
    _ensure_dir(path)
    conds, means, stds = [], [], []
    for r in rows:
        if r["task"] != task:
            continue
        if r["condition"] in ("R", "S", "R+S") and r.get("real_fraction", 1.0) in (1.0, 0.0):
            conds.append(r["condition"])
            means.append(r["primary_mean"])
            stds.append(r["primary_std"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    x = np.arange(len(conds))
    ax.bar(x, means, yerr=stds, capsize=4, color=["#888", "#e377c2", "#2ca02c"][: len(conds)])
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel(_metric_label(task))
    ax.set_title(title or f"{task}: condition comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lowdata_curve(rows, task, path, title=None):
    """Primary metric vs real fraction, one line for R and one for R+S."""
    _ensure_dir(path)
    series = {"R": {}, "R+S": {}}
    for r in rows:
        if r["task"] != task or r["condition"] not in series:
            continue
        series[r["condition"]][r["real_fraction"]] = (r["primary_mean"], r["primary_std"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for cond, style in (("R", "o-"), ("R+S", "s-")):
        pts = sorted(series[cond].items())
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1][0] for p in pts]
        es = [p[1][1] for p in pts]
        ax.errorbar(xs, ys, yerr=es, fmt=style, capsize=3, label=cond)
    ax.set_xlabel("real-data fraction")
    ax.set_ylabel(_metric_label(task))
    ax.set_title(title or f"{task}: low-data regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scale_curve(rows, task, path, title=None):
    """Primary metric vs synthetic count (condition == 'scale')."""
    _ensure_dir(path)
    pts = sorted((r["synth_count"], r["primary_mean"], r["primary_std"])
                 for r in rows if r["task"] == task and r["condition"] == "scale")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3, color="#1f77b4")
    ax.set_xlabel("# synthetic tiles")
    ax.set_ylabel(_metric_label(task))
    ax.set_title(title or f"{task}: synthetic scaling")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def figures_from_results(results_json, out_dir):
    """Convenience: read results.json and emit all curve/bar PNGs for both tasks."""
    import json

    with open(results_json) as f:
        rows = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    made = []
    tasks = sorted({r["task"] for r in rows})
    prefix = {"segmentation": "kr1", "height": "kr2"}
    for task in tasks:
        p = prefix.get(task, task)
        made.append(plot_condition_bars(rows, task, os.path.join(out_dir, f"{p}_condition_bars.png")))
        made.append(plot_lowdata_curve(rows, task, os.path.join(out_dir, f"{p}_lowdata_curve.png")))
        made.append(plot_scale_curve(rows, task, os.path.join(out_dir, f"{p}_scaling_curve.png")))
    return made

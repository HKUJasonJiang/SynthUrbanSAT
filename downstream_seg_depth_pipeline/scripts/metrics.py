"""Evaluation metrics for the downstream tasks.

Segmentation : per-class IoU, mIoU, overall accuracy (OA), mean F1.
Height (AGL) : RMSE, MAE, threshold accuracy (delta < 1.25), all in metres.
Combined     : DFC2019 mIoU-3 (a pixel is a true positive only if the semantic
               label is correct AND the height error is below a threshold, 1 m).

All accumulators work on integer/float numpy or torch tensors and ignore
``ignore_index`` (segmentation) or non-finite values (height).
"""

import numpy as np


def _to_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


class SegMeter:
    """Streaming confusion-matrix accumulator for semantic segmentation."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        pred = _to_numpy(pred).reshape(-1)
        target = _to_numpy(target).reshape(-1)
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        # also drop any out-of-range predictions defensively
        k = (target >= 0) & (target < self.num_classes)
        pred, target = pred[k], target[k]
        idx = self.num_classes * target.astype(np.int64) + pred.astype(np.int64)
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.conf += binc.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        conf = self.conf.astype(np.float64)
        tp = np.diag(conf)
        fp = conf.sum(axis=0) - tp
        fn = conf.sum(axis=1) - tp
        denom_iou = tp + fp + fn
        iou = np.where(denom_iou > 0, tp / np.maximum(denom_iou, 1e-9), np.nan)
        denom_f1 = 2 * tp + fp + fn
        f1 = np.where(denom_f1 > 0, 2 * tp / np.maximum(denom_f1, 1e-9), np.nan)
        oa = tp.sum() / max(conf.sum(), 1e-9)
        return {
            "mIoU": float(np.nanmean(iou)),
            "per_class_IoU": iou.tolist(),
            "mF1": float(np.nanmean(f1)),
            "OA": float(oa),
        }


class HeightMeter:
    """Streaming accumulator for AGL height regression metrics (metres)."""

    def __init__(self, delta_thresh: float = 1.25):
        self.delta_thresh = delta_thresh
        self.sse = 0.0      # sum of squared errors
        self.sae = 0.0      # sum of absolute errors
        self.n = 0
        self.delta_hits = 0

    def update(self, pred, target):
        pred = _to_numpy(pred).reshape(-1).astype(np.float64)
        target = _to_numpy(target).reshape(-1).astype(np.float64)
        valid = np.isfinite(target) & np.isfinite(pred)
        pred, target = pred[valid], target[valid]
        if pred.size == 0:
            return
        err = pred - target
        self.sse += float(np.sum(err ** 2))
        self.sae += float(np.sum(np.abs(err)))
        self.n += pred.size
        # threshold accuracy: max(p/t, t/p) < delta, on the positive-height subset
        pos = (target > 1e-3) & (pred > 1e-3)
        if np.any(pos):
            ratio = np.maximum(pred[pos] / target[pos], target[pos] / pred[pos])
            self.delta_hits += int(np.sum(ratio < self.delta_thresh))
            self._pos_n = getattr(self, "_pos_n", 0) + int(np.sum(pos))

    def compute(self) -> dict:
        n = max(self.n, 1)
        pos_n = max(getattr(self, "_pos_n", 0), 1)
        return {
            "RMSE_m": float(np.sqrt(self.sse / n)),
            "MAE_m": float(self.sae / n),
            f"delta<{self.delta_thresh}": float(self.delta_hits / pos_n),
        }


class MIoU3Meter:
    """DFC2019 mIoU-3: TP requires correct class AND |height error| < thresh (m)."""

    def __init__(self, num_classes: int, ignore_index: int = 255, height_thresh_m: float = 1.0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.height_thresh_m = height_thresh_m
        # Columns 0..K-1 are real predictions; column K is a "void" bucket for
        # height-failed pixels (counted as FN for the true class, never as FP).
        self.conf = np.zeros((num_classes, num_classes + 1), dtype=np.int64)

    def update(self, seg_pred, seg_target, height_pred, height_target):
        seg_pred = _to_numpy(seg_pred).reshape(-1)
        seg_target = _to_numpy(seg_target).reshape(-1)
        hp = _to_numpy(height_pred).reshape(-1).astype(np.float64)
        ht = _to_numpy(height_target).reshape(-1).astype(np.float64)

        valid = (seg_target != self.ignore_index) & np.isfinite(ht) & np.isfinite(hp)
        valid &= (seg_target >= 0) & (seg_target < self.num_classes)
        seg_pred, seg_target = seg_pred[valid], seg_target[valid]
        hp, ht = hp[valid], ht[valid]

        # Height failure -> route the pixel to the void column K so it can never
        # be a TP and never becomes a false positive of any real class.
        height_ok = np.abs(hp - ht) < self.height_thresh_m
        eff_pred = np.where(height_ok, seg_pred, self.num_classes).astype(np.int64)

        cols = self.num_classes + 1
        idx = cols * seg_target.astype(np.int64) + eff_pred
        binc = np.bincount(idx, minlength=self.num_classes * cols)
        self.conf += binc.reshape(self.num_classes, cols)

    def compute(self) -> dict:
        conf = self.conf.astype(np.float64)
        real = conf[:, : self.num_classes]      # K x K real-prediction block
        tp = np.diag(real)
        fp = real.sum(axis=0) - tp              # FP only over real predictions
        fn = conf.sum(axis=1) - tp              # FN includes the void column
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / np.maximum(denom, 1e-9), np.nan)
        return {"mIoU3": float(np.nanmean(iou)), "per_class_IoU3": iou.tolist()}

"""Offline smoke test: metrics, label decoding, data reader, and DPT head.

Runs without GPU, network, or real datasets. The DINOv2 backbone (which needs
torch.hub download) is exercised separately; here we validate the head with
random feature maps so the tensor plumbing is checked end-to-end.

Run:  python -m tests.test_smoke   (from the downstream_pipeline directory)
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_seg_metrics_perfect():
    from scripts.metrics import SegMeter

    m = SegMeter(num_classes=3, ignore_index=255)
    t = np.array([[0, 1, 2], [2, 1, 0]])
    m.update(t.copy(), t.copy())
    out = m.compute()
    assert abs(out["mIoU"] - 1.0) < 1e-6, out
    assert abs(out["OA"] - 1.0) < 1e-6, out
    print("ok seg metrics (perfect):", out)


def test_height_metrics():
    from scripts.metrics import HeightMeter

    m = HeightMeter()
    pred = np.array([1.0, 2.0, 3.0])
    tgt = np.array([1.0, 2.0, 3.0])
    m.update(pred, tgt)
    out = m.compute()
    assert out["RMSE_m"] < 1e-6 and out["MAE_m"] < 1e-6, out
    print("ok height metrics:", out)


def test_miou3_height_gate():
    from scripts.metrics import MIoU3Meter

    m = MIoU3Meter(num_classes=3, height_thresh_m=1.0)
    seg = np.array([0, 1, 2])
    # class correct everywhere, but pixel 1 has a 5 m height error -> not a TP
    m.update(seg_pred=seg, seg_target=seg,
             height_pred=np.array([0.0, 5.0, 0.0]),
             height_target=np.array([0.0, 0.0, 0.0]))
    out = m.compute()
    assert out["mIoU3"] < 1.0, out  # gated below perfect
    print("ok mIoU-3 height gate:", out)


def test_label_decode_palette():
    from PIL import Image

    from scripts.labels import LabelSpace

    ls = LabelSpace()
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)  # palette indices
    img = Image.fromarray(arr, mode="P")
    dec = ls.decode_seg(img)
    assert dec.shape == (2, 3)
    assert dec.min() >= 0 and dec.max() < ls.num_classes
    print("ok label decode (palette):", dec.tolist())


def test_data_reader_roundtrip():
    from PIL import Image

    from scripts.data import TileDataset
    from scripts.labels import LabelSpace

    ls = LabelSpace()
    with tempfile.TemporaryDirectory() as d:
        for sub in ("rgb", "seg", "depth"):
            os.makedirs(os.path.join(d, sub))
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(os.path.join(d, "rgb", "t0.png"))
        Image.fromarray(np.ones((16, 16), np.uint8), mode="P").save(os.path.join(d, "seg", "t0.png"))
        # height as float TIFF (metres)
        Image.fromarray(np.full((16, 16), 9.0, np.float32)).save(os.path.join(d, "depth", "t0.tif"))
        ds = TileDataset(d, ls, image_size=28, task="height")
        assert len(ds) == 1
        s = ds[0]
        assert s["rgb"].shape == (3, 28, 28)
        assert s["seg"].shape == (28, 28)
        assert s["height"].shape == (1, 28, 28)
        print("ok data reader roundtrip:", {k: tuple(v.shape) for k, v in s.items() if hasattr(v, "shape")})


def test_dpt_head_shapes():
    import torch

    from scripts.heads import DPTHead

    dims = [32, 32, 32, 32]
    head = DPTHead(dims, out_ch=6, hidden_dim=16)
    feats = [torch.randn(2, 32, 8, 8) for _ in dims]
    out = head(feats, out_hw=(56, 56))
    assert out.shape == (2, 6, 56, 56), out.shape
    print("ok DPT head shapes:", tuple(out.shape))


def test_resolve_synth_source():
    from scripts.data import resolve_synth_root

    cfg = {"data": {"synth_sources": {"us3d_paired": "A", "osm": "B"},
                    "synth_source": "osm", "synth_root": "fallback"}}
    assert resolve_synth_root(cfg) == "B"
    cfg["data"]["synth_source"] = "us3d_paired"
    assert resolve_synth_root(cfg) == "A"
    # legacy fallback when no sources defined
    assert resolve_synth_root({"data": {"synth_root": "C"}}) == "C"
    print("ok resolve synth source")


def test_visualize_outputs():
    from scripts.visualize import (plot_lowdata_curve, plot_scale_curve,
                                   save_height_panel, save_seg_triptych)

    palette = [(0, 0, 255), (0, 225, 255), (0, 255, 0),
               (255, 0, 0), (128, 0, 128), (0, 0, 0)]
    with tempfile.TemporaryDirectory() as d:
        rgb = np.random.rand(3, 32, 32).astype(np.float32)
        gt = np.random.randint(0, 6, (32, 32))
        pred = np.random.randint(0, 6, (32, 32))
        p1 = save_seg_triptych(rgb, gt, pred, palette, os.path.join(d, "seg.png"))
        h_gt = np.random.rand(32, 32).astype(np.float32) * 20
        h_pr = h_gt + np.random.randn(32, 32).astype(np.float32)
        p2 = save_height_panel(rgb, h_gt, h_pr, os.path.join(d, "h.png"), vmax=25.0)
        rows = [
            {"task": "segmentation", "condition": "R", "real_fraction": 0.1,
             "synth_count": 0, "primary_mean": 0.30, "primary_std": 0.01},
            {"task": "segmentation", "condition": "R+S", "real_fraction": 0.1,
             "synth_count": 10000, "primary_mean": 0.42, "primary_std": 0.02},
            {"task": "segmentation", "condition": "R", "real_fraction": 1.0,
             "synth_count": 0, "primary_mean": 0.55, "primary_std": 0.01},
            {"task": "segmentation", "condition": "R+S", "real_fraction": 1.0,
             "synth_count": 10000, "primary_mean": 0.58, "primary_std": 0.01},
            {"task": "segmentation", "condition": "scale", "real_fraction": 1.0,
             "synth_count": 0, "primary_mean": 0.55, "primary_std": 0.01},
            {"task": "segmentation", "condition": "scale", "real_fraction": 1.0,
             "synth_count": 50000, "primary_mean": 0.60, "primary_std": 0.01},
        ]
        p3 = plot_lowdata_curve(rows, "segmentation", os.path.join(d, "low.png"))
        p4 = plot_scale_curve(rows, "segmentation", os.path.join(d, "scale.png"))
        for p in (p1, p2, p3, p4):
            assert os.path.exists(p) and os.path.getsize(p) > 0, p
    print("ok visualize outputs (seg/height panels + curves)")


if __name__ == "__main__":
    test_seg_metrics_perfect()
    test_height_metrics()
    test_miou3_height_gate()
    test_label_decode_palette()
    test_resolve_synth_source()
    test_data_reader_roundtrip()
    test_dpt_head_shapes()
    test_visualize_outputs()
    print("\nALL SMOKE TESTS PASSED")

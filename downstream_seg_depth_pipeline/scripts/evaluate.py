"""Evaluation on the (always real) US3D test set."""

import os

import torch

from scripts.metrics import HeightMeter, MIoU3Meter, SegMeter


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index=255,
             height_thresh_m=1.0, joint=False, dump_dir=None, dump_n=0,
             palette_rgb=None, ndsm_max_m=25.0):
    """Run the model over ``loader`` and return a metrics dict.

    If ``joint`` is True the model must output a (seg_logits, height) tuple and
    both task metrics plus the DFC mIoU-3 are reported. Otherwise the single
    active task (model.task) is evaluated.

    If ``dump_dir`` is set, the first ``dump_n`` tiles are saved as qualitative
    panels (seg triptych / height panel) — the KR1.x / KR2.x deliverables.
    """
    model.eval()
    task = getattr(model, "task", "segmentation")

    seg_meter = SegMeter(num_classes, ignore_index) if (joint or task == "segmentation") else None
    h_meter = HeightMeter() if (joint or task == "height") else None
    m3 = MIoU3Meter(num_classes, ignore_index, height_thresh_m) if joint else None

    dumped = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        out = model(rgb)

        if joint:
            seg_logits, height_pred = out
        elif task == "segmentation":
            seg_logits, height_pred = out, None
        else:
            seg_logits, height_pred = None, out

        if seg_meter is not None:
            pred = seg_logits.argmax(1)
            seg_meter.update(pred, batch["seg"])
        if h_meter is not None:
            h_meter.update(height_pred.squeeze(1), batch["height"].squeeze(1))
        if m3 is not None:
            m3.update(seg_logits.argmax(1), batch["seg"],
                      height_pred.squeeze(1), batch["height"].squeeze(1))

        if dump_dir and dumped < dump_n:
            dumped = _dump_qualitative(
                batch, seg_logits, height_pred, dump_dir, dumped, dump_n,
                palette_rgb, ndsm_max_m)

    result = {}
    if seg_meter is not None:
        result.update(seg_meter.compute())
    if h_meter is not None:
        result.update(h_meter.compute())
    if m3 is not None:
        result.update(m3.compute())
    return result


def _dump_qualitative(batch, seg_logits, height_pred, dump_dir, dumped, dump_n,
                      palette_rgb, ndsm_max_m):
    """Save per-tile qualitative panels until ``dump_n`` are written."""
    from scripts.visualize import save_height_panel, save_seg_triptych

    os.makedirs(dump_dir, exist_ok=True)
    rgb = batch["rgb"].cpu().numpy()
    bsz = rgb.shape[0]
    for i in range(bsz):
        if dumped >= dump_n:
            break
        stem = batch.get("stem", [f"tile{dumped}"] * bsz)[i]
        if seg_logits is not None:
            save_seg_triptych(
                rgb[i], batch["seg"][i].cpu().numpy(),
                seg_logits[i].argmax(0).cpu().numpy(),
                palette_rgb or [], os.path.join(dump_dir, f"seg_{stem}.png"),
                title=f"seg: {stem}")
        if height_pred is not None:
            save_height_panel(
                rgb[i], batch["height"][i].squeeze(0).cpu().numpy(),
                height_pred[i].squeeze(0).cpu().numpy(),
                os.path.join(dump_dir, f"height_{stem}.png"),
                title=f"height: {stem}", vmax=ndsm_max_m)
        dumped += 1
    return dumped

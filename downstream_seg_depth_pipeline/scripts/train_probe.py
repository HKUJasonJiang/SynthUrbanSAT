"""Frozen-probe training loop for a single task / data-mixture run.

Trains only the head (backbone frozen). Loss:
  * segmentation -> cross-entropy (ignore_index aware)
  * height       -> masked L1 on finite-AGL pixels (metres)

CLI runs one experiment condition; the data mixture is the experimental
variable (see configs/default.yaml and scripts/data.py).
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from scripts.data import TileDataset, build_mixture, resolve_synth_root
from scripts.evaluate import evaluate
from scripts.labels import LabelSpace
from scripts.model import ProbeModel


def _seg_loss(logits, target, ignore_index):
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def _height_loss(pred, target, scale_m=25.0, positive_weight=4.0, positive_threshold_m=1.0):
    target = target.to(pred.dtype)
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    scale = max(float(scale_m), 1e-6)
    err = torch.abs(pred[mask] / scale - target[mask] / scale)
    weights = torch.ones_like(err)
    weights = weights + float(positive_weight) * (target[mask] > float(positive_threshold_m)).to(err.dtype)
    return (err * weights).sum() / weights.sum().clamp_min(1.0)


def _height_mse(pred, target):
    target = target.to(pred.dtype)
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return None
    return torch.mean((pred[mask] - target[mask]) ** 2).detach().float().item()


def _training_split_names(cfg):
    splits = cfg.get("data", {}).get("train_splits", ["train"])
    if isinstance(splits, str):
        splits = [splits]
    return splits or ["train"]


def _build_train_dataset(cfg, real_root, synth_root, label_space, image_size, task):
    parts = []
    for split in _training_split_names(cfg):
        part = build_mixture(
            real_split_dir=os.path.join(real_root, split),
            synth_split_dir=os.path.join(synth_root, split),
            label_space=label_space, image_size=image_size, task=task,
            real_fraction=cfg["data"]["real_fraction"],
            synth_count=cfg["data"]["synth_count"],
            seed=cfg["seed"],
        )
        parts.append(part)
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def train_one(cfg, device):
    ls = LabelSpace()
    task = cfg["task"]
    s = cfg["image_size"]

    real_root = cfg["data"]["real_root"]
    synth_root = resolve_synth_root(cfg)
    train_ds = _build_train_dataset(cfg, real_root, synth_root, ls, s, task)
    test_ds = TileDataset(os.path.join(real_root, "test"), ls, s, task)

    tcfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                              num_workers=tcfg["num_workers"], pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             num_workers=tcfg["num_workers"], pin_memory=True)

    model = ProbeModel(
        task=task, num_classes=cfg["num_classes"],
        backbone_name=cfg["backbone"]["name"], out_indices=tuple(cfg["backbone"]["out_indices"]),
        head_type=cfg["head"]["type"], hidden_dim=cfg["head"]["hidden_dim"],
        freeze_backbone=cfg["backbone"]["freeze"], ndsm_max_m=cfg["ndsm_max_m"],
    ).to(device)

    lr = tcfg.get("height_lr", tcfg["lr"]) if task == "height" else tcfg["lr"]
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr,
                            weight_decay=tcfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["amp"] and device.type == "cuda")

    history = {"train_loss": [], "train_mse_m2": []}
    start_time = time.time()
    for epoch in range(tcfg["epochs"]):
        model.train()
        running = 0.0
        mse_running = 0.0
        mse_count = 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(rgb)
                if task == "segmentation":
                    loss = _seg_loss(out, batch["seg"].to(device), cfg["ignore_index"])
                else:
                    loss = _height_loss(
                        out.squeeze(1), batch["height"].to(device).squeeze(1), cfg["ndsm_max_m"],
                        tcfg.get("height_positive_weight", 4.0),
                        tcfg.get("height_positive_threshold_m", 1.0),
                    )
                    mse = _height_mse(out.squeeze(1), batch["height"].to(device).squeeze(1))
                    if mse is not None:
                        mse_running += mse
                        mse_count += 1
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        epoch_loss = running / max(len(train_loader), 1)
        history["train_loss"].append(epoch_loss)
        if task == "height":
            history["train_mse_m2"].append(mse_running / max(mse_count, 1))
        print(f"[epoch {epoch+1}/{tcfg['epochs']}] loss={epoch_loss:.4f}")

    metrics = evaluate(model, test_loader, device, cfg["num_classes"],
                       cfg["ignore_index"], cfg["eval"]["height_threshold_m"],
                       dump_dir=cfg["eval"].get("dump_dir"),
                       dump_n=cfg["eval"].get("dump_n", 0),
                       palette_rgb=ls.rgb, ndsm_max_m=cfg["ndsm_max_m"])
    metrics["history"] = history
    metrics["train_seconds"] = time.time() - start_time
    metrics["train_size"] = len(train_ds)
    metrics["train_splits"] = _training_split_names(cfg)
    print(f"[test] {metrics}")
    return model, metrics


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), os.pardir,
                                                     "configs", "default.yaml"))
    ap.add_argument("--task", choices=["segmentation", "height"])
    ap.add_argument("--real-fraction", type=float)
    ap.add_argument("--synth-count", type=int)
    ap.add_argument("--synth-source", default=None,
                    help="named synthetic source: 'us3d_paired' or 'osm'")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save", default=None, help="optional path to save head weights")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.task is not None:
        cfg["task"] = args.task
    if args.real_fraction is not None:
        cfg["data"]["real_fraction"] = args.real_fraction
    if args.synth_count is not None:
        cfg["data"]["synth_count"] = args.synth_count
    if args.synth_source is not None:
        cfg["data"]["synth_source"] = args.synth_source
    if args.seed is not None:
        cfg["seed"] = args.seed

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metrics = train_one(cfg, device)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save({"head": model.head.state_dict(), "metrics": metrics, "cfg": cfg}, args.save)
        print(f"[save] {args.save}")


if __name__ == "__main__":
    main()

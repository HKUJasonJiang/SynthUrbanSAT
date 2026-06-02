"""Frozen-probe training loop for a single task / data-mixture run.

Trains only the head (backbone frozen). Loss:
  * segmentation -> cross-entropy (ignore_index aware)
  * height       -> masked L1 on finite-AGL pixels (metres)

CLI runs one experiment condition; the data mixture is the experimental
variable (see configs/default.yaml and scripts/data.py).
"""

import argparse
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from scripts.data import TileDataset, build_mixture, resolve_synth_root
from scripts.evaluate import evaluate
from scripts.labels import LabelSpace
from scripts.model import ProbeModel


def _seg_loss(logits, target, ignore_index):
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def _height_loss(pred, target):
    target = target.to(pred.dtype)
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred[mask], target[mask])


def train_one(cfg, device):
    ls = LabelSpace()
    task = cfg["task"]
    s = cfg["image_size"]

    real_root = cfg["data"]["real_root"]
    synth_root = resolve_synth_root(cfg)
    train_ds = build_mixture(
        real_split_dir=os.path.join(real_root, "train"),
        synth_split_dir=os.path.join(synth_root, "train"),
        label_space=ls, image_size=s, task=task,
        real_fraction=cfg["data"]["real_fraction"],
        synth_count=cfg["data"]["synth_count"],
        seed=cfg["seed"],
    )
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

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["amp"] and device.type == "cuda")

    for epoch in range(tcfg["epochs"]):
        model.train()
        running = 0.0
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(rgb)
                if task == "segmentation":
                    loss = _seg_loss(out, batch["seg"].to(device), cfg["ignore_index"])
                else:
                    loss = _height_loss(out.squeeze(1), batch["height"].to(device).squeeze(1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        print(f"[epoch {epoch+1}/{tcfg['epochs']}] loss={running/max(len(train_loader),1):.4f}")

    metrics = evaluate(model, test_loader, device, cfg["num_classes"],
                       cfg["ignore_index"], cfg["eval"]["height_threshold_m"],
                       dump_dir=cfg["eval"].get("dump_dir"),
                       dump_n=cfg["eval"].get("dump_n", 0),
                       palette_rgb=ls.rgb, ndsm_max_m=cfg["ndsm_max_m"])
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

"""Build leakage-free train/val/test split directories from a flat data root.

Input root must contain flat ``rgb/ seg/ depth/`` folders. Tiles are assigned
to *test* by city/tile prefix (geographic holdout, so test tiles never share a
city with training tiles), the rest are split into train/val by fraction.

Outputs symlinked split dirs (``--copy`` to hard-copy on filesystems without
symlink permission, e.g. some Windows setups):

    <out>/{train,val,test}/{rgb,seg,depth}/...

The split is deterministic given ``--seed``.
"""

import argparse
import os
import random
import shutil

_RGB_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _list_stems(rgb_dir):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(rgb_dir)
                  if f.lower().endswith(_RGB_EXTS))


def _find(root, sub, stem):
    base = os.path.join(root, sub)
    exts = (".png", ".tif", ".tiff", ".exr", ".jpg", ".jpeg")
    for try_stem in (stem, stem.replace("_RGB_", "_")):
        for ext in exts:
            cand = os.path.join(base, try_stem + ext)
            if os.path.exists(cand):
                return cand
    return None


def _place(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="flat root with rgb/ seg/ depth/")
    ap.add_argument("--out", required=True, help="output root for split dirs")
    ap.add_argument("--test-prefixes", nargs="*", default=[],
                    help="tile-name prefixes assigned to the test set (geographic holdout)")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rgb_dir = os.path.join(args.src, "rgb")
    stems = _list_stems(rgb_dir)
    if not stems:
        raise SystemExit(f"No RGB tiles under {rgb_dir}")

    test = [s for s in stems if any(s.startswith(p) for p in args.test_prefixes)]
    pool = [s for s in stems if s not in set(test)]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    n_val = int(round(len(pool) * args.val_fraction))
    val, train = pool[:n_val], pool[n_val:]

    counts = {"train": train, "val": val, "test": test}
    rgb_ext = {os.path.splitext(f)[0]: os.path.splitext(f)[1]
               for f in os.listdir(rgb_dir) if f.lower().endswith(_RGB_EXTS)}
    for split, group in counts.items():
        for stem in group:
            rgb_src = os.path.join(rgb_dir, stem + rgb_ext[stem])
            seg_src = _find(args.src, "seg", stem)
            d_src = _find(args.src, "depth", stem)
            if not (seg_src and d_src):
                continue
            _place(rgb_src, os.path.join(args.out, split, "rgb", os.path.basename(rgb_src)), args.copy)
            _place(seg_src, os.path.join(args.out, split, "seg", os.path.basename(seg_src)), args.copy)
            _place(d_src, os.path.join(args.out, split, "depth", os.path.basename(d_src)), args.copy)
        print(f"[split] {split}: {len(group)} tiles")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PIPE_ROOT = HERE.parent
GEN_ROOT = (PIPE_ROOT.parent / "generation_pipeline").resolve()
TRAIN_ROOT = (PIPE_ROOT.parent / "train_pipeline").resolve()

for path in (GEN_ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from batch_eval import compose_prompt_from_json
from pipeline import DEFAULT_PROMPT_JSON, LORA_DIR, TEXT_ENCODER_PATH, TOKENIZER_DIR, list_lora_checkpoints
from pipeline.state import DEVICE, DTYPE
from scripts.text_encoder import encode_prompts, load_text_encoder, unload_text_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint dir name under generation_pipeline/weights/lora")
    parser.add_argument("--prompt-json", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.ckpt:
        ckpt_dir = (LORA_DIR / args.ckpt).resolve()
    else:
        checkpoints = list_lora_checkpoints()
        if not checkpoints:
            raise SystemExit("No checkpoints in generation_pipeline/weights/lora")
        ckpt_dir = checkpoints[0]

    meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu", weights_only=False)
    config = meta.get("config", {})
    max_sequence_length = int(config.get("text_seq_len", 512))

    prompt_path = Path(args.prompt_json or DEFAULT_PROMPT_JSON)
    prompt_text = compose_prompt_from_json(json.loads(prompt_path.read_text()))
    print(f"[prompt] checkpoint={ckpt_dir.name} seq_len={max_sequence_length}")

    text_encoder, tokenizer = load_text_encoder(str(TEXT_ENCODER_PATH), device=DEVICE, dtype=DTYPE)
    try:
        embed = encode_prompts(
            text_encoder, tokenizer, [prompt_text],
            max_sequence_length=max_sequence_length,
            device=DEVICE, dtype=DTYPE,
        )[0].detach().cpu()
    finally:
        unload_text_encoder(text_encoder, tokenizer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embed, out)
    print(f"[prompt] saved {tuple(embed.shape)} -> {out}")


if __name__ == "__main__":
    main()
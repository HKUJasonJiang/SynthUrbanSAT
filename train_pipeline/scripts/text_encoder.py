"""Mistral-3.1 Text Encoder for Flux2.

Supports two model sources:
  1. Single FP8 safetensors file (e.g. mistral_3_small_flux2_fp8.safetensors)
     - Weights are dequantized from float8_e4m3fn → bfloat16
     - Tokenizer is extracted from the embedded 'tekken_model' key
  2. HuggingFace directory with text_encoder/ and tokenizer/ subfolders

Flux2 extracts hidden states from layers 10, 20, 30 of the 30-layer
pruned Mistral-3.1-Small and concatenates them (3 × 5120 = 15360).

External deps: transformers, safetensors.
"""

import base64
import gc
import json
import os
from typing import List, Optional, Union

import numpy as np
import torch
from safetensors.torch import load_file


# ── Chat template (matches ComfyUI / Flux2 convention) ─────────────────────
LLAMA_TEMPLATE = (
    '[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. '
    'You give structured responses focusing on object relationships, object\n'
    'attribution and actions without speculation.[/SYSTEM_PROMPT]'
    '[INST]{}[/INST]'
)

DEFAULT_SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. "
    "You give structured responses focusing on object relationships, "
    "object attribution and actions without speculation."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer from embedded tekken_model
# ═══════════════════════════════════════════════════════════════════════════════

def _build_tokenizer_from_tekken(tekken_data):
    """Build a LlamaTokenizerFast from the tekken_model bytes in the safetensors.

    Args:
        tekken_data: uint8 tensor or raw bytes of the tekken JSON vocab.

    Returns:
        LlamaTokenizerFast tokenizer.
    """
    from transformers import LlamaTokenizerFast
    try:
        from transformers.integrations.mistral import MistralConverter
    except ModuleNotFoundError:
        from transformers.models.pixtral.convert_pixtral_weights_to_hf import MistralConverter

    if torch.is_tensor(tekken_data):
        tekken_data = tekken_data.numpy().tobytes()

    mistral_vocab = json.loads(tekken_data)

    special_tokens = {}
    vocab = {}

    max_vocab = mistral_vocab["config"]["default_vocab_size"]
    max_vocab -= len(mistral_vocab["special_tokens"])

    for w in mistral_vocab["vocab"]:
        r = w["rank"]
        if r >= max_vocab:
            continue
        vocab[base64.b64decode(w["token_bytes"])] = r

    for w in mistral_vocab["special_tokens"]:
        if "token_bytes" in w:
            special_tokens[base64.b64decode(w["token_bytes"])] = w["rank"]
        else:
            special_tokens[w["token_str"]] = w["rank"]

    all_special = list(special_tokens.keys())
    special_tokens.update(vocab)
    vocab = special_tokens

    tokenizer_obj = MistralConverter(
        vocab=vocab, additional_special_tokens=all_special
    ).converted()
    tokenizer = LlamaTokenizerFast(
        tokenizer_object=tokenizer_obj, legacy=False
    )
    # Mistral tekken: pad_token_id=11 (matches ComfyUI convention)
    tokenizer.pad_token_id = 11
    tokenizer.padding_side = 'left'
    return tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_text_encoder(model_path, device='cuda', dtype=torch.bfloat16,
                      low_cpu_mem_usage=True):
    """Load Mistral-3.1 text encoder and tokenizer.

    Accepts either:
      - A .safetensors file path (FP8 quantized, with embedded tekken tokenizer)
      - A HuggingFace model directory with text_encoder/ and tokenizer/ subfolders

    Returns:
        (text_encoder, tokenizer)
    """
    if model_path.endswith('.safetensors'):
        return _load_from_safetensors(model_path, device, dtype)
    else:
        return _load_from_hf_dir(model_path, device, dtype, low_cpu_mem_usage)


def _load_from_safetensors(safetensors_path, device, dtype):
    """Load text encoder from a single FP8 safetensors file."""
    from transformers import MistralConfig, MistralModel
    from safetensors import safe_open

    print(f'  Loading safetensors: {safetensors_path}')
    f = safe_open(safetensors_path, framework='pt')
    keys = list(f.keys())

    # ── Extract tokenizer ───────────────────────────────────────────────────
    print('  Building tokenizer from embedded tekken_model ...')
    tekken_data = f.get_tensor('tekken_model')
    tokenizer = _build_tokenizer_from_tekken(tekken_data)

    # ── Detect model shape ──────────────────────────────────────────────────
    layer_nums = sorted({
        int(k.split('.')[2])
        for k in keys if k.startswith('model.layers.')
    })
    num_layers = len(layer_nums)
    embed_weight = f.get_tensor('model.embed_tokens.weight')
    vocab_size, hidden_size = embed_weight.shape

    # Infer head counts from projection shapes
    q_shape = f.get_tensor('model.layers.0.self_attn.q_proj.weight').shape
    k_shape = f.get_tensor('model.layers.0.self_attn.k_proj.weight').shape
    o_shape = f.get_tensor('model.layers.0.self_attn.o_proj.weight').shape
    mlp_shape = f.get_tensor('model.layers.0.mlp.gate_proj.weight').shape

    # Mistral-Small: head_dim=128, q_proj=[num_heads*128, hidden], k_proj=[num_kv_heads*128, hidden]
    # o_proj=[hidden, num_heads*128], verify consistency: o_in = q_out
    head_dim = 128
    num_attention_heads = q_shape[0] // head_dim   # 4096/128 = 32
    num_kv_heads = k_shape[0] // head_dim           # 1024/128 = 8
    intermediate_size = mlp_shape[0]                 # 32768

    has_norm = 'model.norm.weight' in keys

    print(f'  Detected: {num_layers} layers, hidden={hidden_size}, '
          f'heads={num_attention_heads}, kv_heads={num_kv_heads}, '
          f'ffn={intermediate_size}, vocab={vocab_size}')

    # ── Build config & model ────────────────────────────────────────────────
    config = MistralConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=8192,
        rope_theta=1_000_000_000.0,
        hidden_act="silu",
    )

    print(f'  Initialising MistralModel ({num_layers} layers)...')
    text_encoder = MistralModel(config)

    # ── Dequantize FP8 weights → bf16 and load ─────────────────────────────
    print('  Dequantizing FP8 weights → bf16 ...')
    state_dict = {}
    skip_keys = {'scaled_fp8', 'tekken_model'}

    for k in keys:
        if k in skip_keys:
            continue
        if k.endswith('.input_scale') or k.endswith('.weight_scale'):
            continue

        tensor = f.get_tensor(k)

        # Dequantize FP8 linear weights
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = k.replace('.weight', '.weight_scale')
            if scale_key in keys:
                scale = f.get_tensor(scale_key).float()
                tensor = tensor.to(torch.float32) * scale
            tensor = tensor.to(dtype)

        # Strip 'model.' prefix for MistralModel
        new_key = k[len('model.'):] if k.startswith('model.') else k
        state_dict[new_key] = tensor.to(dtype)

    missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
    if missing:
        # norm.weight is expected to be missing for pruned Flux2 models
        non_norm_missing = [m for m in missing if 'norm.weight' not in m]
        if non_norm_missing:
            print(f'  WARNING: missing keys: {non_norm_missing}')
    if unexpected:
        print(f'  WARNING: unexpected keys: {unexpected}')

    text_encoder = text_encoder.to(device=device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)

    params_b = sum(p.numel() for p in text_encoder.parameters()) / 1e9
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f'  Text encoder: {params_b:.1f}B params, VRAM: {vram_gb:.2f} GB')

    return text_encoder, tokenizer


def _load_from_hf_dir(model_path, device, dtype, low_cpu_mem_usage):
    """Load text encoder from a HuggingFace model directory."""
    from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

    print(f'  Loading tokenizer from {model_path}/tokenizer ...')
    tokenizer = PixtralProcessor.from_pretrained(
        model_path, subfolder="tokenizer"
    )

    print(f'  Loading text encoder from {model_path}/text_encoder ...')
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        model_path, subfolder="text_encoder",
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    text_encoder = text_encoder.to(device).eval()
    text_encoder.requires_grad_(False)

    params_b = sum(p.numel() for p in text_encoder.parameters()) / 1e9
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f'  Text encoder: {params_b:.1f}B params, VRAM: {vram_gb:.1f} GB')

    return text_encoder, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding
# ═══════════════════════════════════════════════════════════════════════════════

def encode_prompts(text_encoder, tokenizer, prompts,
                   max_sequence_length=512,
                   system_message=DEFAULT_SYSTEM_MESSAGE,
                   hidden_states_layers=(10, 20, 30),
                   device='cuda', dtype=torch.bfloat16):
    """Encode text prompts → [B, max_seq_len, 15360] embeddings.

    Extracts hidden states from layers 10, 20, 30 of Mistral-3.1-Small
    and concatenates them (3 × 5120 = 15360).

    Supports both LlamaTokenizerFast (from safetensors) and PixtralProcessor.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    from transformers import LlamaTokenizerFast

    if isinstance(tokenizer, LlamaTokenizerFast):
        # Safetensors path: apply chat template manually
        templated = [LLAMA_TEMPLATE.format(p) for p in prompts]
        inputs = tokenizer(
            templated,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
    else:
        # HuggingFace dir path: use PixtralProcessor apply_chat_template
        messages_batch = [
            [
                {"role": "system",
                 "content": [{"type": "text", "text": system_message}]},
                {"role": "user",
                 "content": [{"type": "text", "text": p}]},
            ]
            for p in prompts
        ]
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Stack hidden states from specified layers
    out = torch.stack(
        [output.hidden_states[k] for k in hidden_states_layers], dim=1
    )
    out = out.to(dtype=dtype, device=device)

    # Reshape: [B, num_layers, seq_len, hidden_dim] → [B, seq_len, num_layers * hidden_dim]
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )

    return prompt_embeds


def unload_text_encoder(text_encoder, tokenizer):
    """Unload text encoder from GPU to free VRAM."""
    del text_encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'  Text encoder unloaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')


def precompute_and_save_embeddings(model_path, dataset_dir, output_path,
                                   prompts_source='captions.json',
                                   max_sequence_length=512,
                                   batch_size=4,
                                   device='cuda', dtype=torch.bfloat16):
    """Precompute text embeddings for all samples and save to disk.

    This loads the text encoder, processes all prompts, saves embeddings,
    then unloads — so it can run before loading the transformer.

    Args:
        model_path: path to FLUX.2-dev model (or HF model ID)
        dataset_dir: dataset root (containing train/, val/)
        output_path: where to save the .pt file with all embeddings
        prompts_source: filename of JSON with prompts (in dataset_dir/split/)
                        JSON format: {"filename.png": "caption text", ...}
        max_sequence_length: max token length
        batch_size: encoding batch size
        device: device
        dtype: dtype

    Saves:
        {output_path}: dict with keys like "train/filename.png" → tensor [512, 15360]
    """
    print('\n=== Precomputing Text Embeddings ===')
    text_encoder, tokenizer = load_text_encoder(model_path, device, dtype)

    all_embeddings = {}

    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_dir, split)
        prompts_path = os.path.join(split_dir, prompts_source)

        if not os.path.exists(prompts_path):
            print(f'  [{split}] No {prompts_source} found at {prompts_path}, skipping.')
            continue

        with open(prompts_path) as f:
            prompts_dict = json.load(f)

        filenames = list(prompts_dict.keys())
        captions = [prompts_dict[fn] for fn in filenames]
        print(f'  [{split}] Encoding {len(captions)} prompts...')

        # Encode in batches
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i + batch_size]
            batch_filenames = filenames[i:i + batch_size]

            embeds = encode_prompts(
                text_encoder, tokenizer, batch_captions,
                max_sequence_length=max_sequence_length,
                device=device, dtype=dtype,
            )

            for j, fn in enumerate(batch_filenames):
                key = f'{split}/{fn}'
                all_embeddings[key] = embeds[j].cpu()

            if (i // batch_size + 1) % 10 == 0:
                print(f'    Encoded {min(i + batch_size, len(captions))}/{len(captions)}')

        print(f'  [{split}] Done: {len([k for k in all_embeddings if k.startswith(split)])} embeddings')

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(all_embeddings, output_path)
    print(f'  Saved embeddings to {output_path} ({len(all_embeddings)} total)')

    # Unload
    unload_text_encoder(text_encoder, tokenizer)
    print('=== Text Embedding Precomputation Done ===\n')

    return output_path


def load_precomputed_embeddings(embeddings_path):
    """Load precomputed embeddings from disk.

    Returns:
        dict: {"split/filename.png": tensor [512, 15360]}
              OR {"global": tensor [512, 15360]} for single-prompt mode
    """
    embeddings = torch.load(embeddings_path, map_location='cpu', weights_only=True)
    if 'global' in embeddings:
        print(f'  Loaded global text embedding from {embeddings_path} '
              f'(shape: {embeddings["global"].shape})')
    else:
        print(f'  Loaded {len(embeddings)} precomputed text embeddings from {embeddings_path}')
    return embeddings


def precompute_single_prompt_embeddings(model_path, prompt_text, output_path,
                                        max_sequence_length=512,
                                        device='cuda', dtype=torch.bfloat16):
    """Encode a single global prompt once and save with key 'global'.

    Much more efficient than encoding the same text per-sample when all
    samples share one prompt.

    Args:
        model_path: path to text encoder weights (.safetensors or HF dir)
        prompt_text: the single text prompt string to encode
        output_path: where to save the .pt file
        max_sequence_length: max token length (default 512)
        device: 'cuda' or 'cpu'
        dtype: encoding dtype (bfloat16)

    Saves:
        {output_path}: {'global': tensor [seq_len, 15360]}
    """
    print('\n=== Precomputing Global Text Embedding ===')
    print(f'  Prompt: {prompt_text[:120]}{"..." if len(prompt_text) > 120 else ""}')
    text_encoder, tokenizer = load_text_encoder(model_path, device, dtype)

    embeds = encode_prompts(
        text_encoder, tokenizer, [prompt_text],
        max_sequence_length=max_sequence_length,
        device=device, dtype=dtype,
    )
    embed_tensor = embeds[0].cpu()   # [seq_len, 15360]
    print(f'  Embedding shape: {embed_tensor.shape}')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save({'global': embed_tensor}, output_path)
    print(f'  Saved global embedding → {output_path}')

    unload_text_encoder(text_encoder, tokenizer)
    print('=== Global Text Embedding Done ===\n')
    return output_path

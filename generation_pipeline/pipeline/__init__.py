"""Resolve paths to model weights for the self-contained pipeline.

Layout under ``generation_pipeline/weights``::

    base/
        flux2_dev_fp8mixed.safetensors
        flux2-vae.safetensors
        mistral_3_small_flux2_fp8.safetensors
        FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors
    tokenizer/
        ... (PixtralProcessor files)
    lora/
        checkpoint_epoch_0315/
            meta.pt  hdc2a.pt  control_params.pt  ...

Each file may be a regular file OR a symlink populated by ``setup.sh``.
"""

from __future__ import annotations

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PIPELINE_ROOT / 'weights'
OUTPUT_DIR = PIPELINE_ROOT / 'output'
TRAIN_PIPELINE_ROOT = PIPELINE_ROOT.parent / 'train_pipeline'
VIDEOX_FUN_ROOT = TRAIN_PIPELINE_ROOT / 'models'

BASE_DIR = WEIGHTS_DIR / 'base'
LORA_DIR = WEIGHTS_DIR / 'lora'
TOKENIZER_DIR = WEIGHTS_DIR / 'tokenizer'

TRANSFORMER_PATH = BASE_DIR / 'flux2_dev_fp8mixed.safetensors'
VAE_PATH = BASE_DIR / 'flux2-vae.safetensors'
TEXT_ENCODER_PATH = BASE_DIR / 'mistral_3_small_flux2_fp8.safetensors'
CONTROLNET_PATH = BASE_DIR / 'FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors'

COLOR_MAP_PATH = TRAIN_PIPELINE_ROOT / 'configs' / 'color_map.json'
DEFAULT_PROMPT_JSON = TRAIN_PIPELINE_ROOT / 'dataset' / 'prompt.json'


def list_lora_checkpoints() -> list[Path]:
    """Return all checkpoint dirs inside ``weights/lora`` that contain meta.pt."""
    if not LORA_DIR.is_dir():
        return []
    return sorted(
        [p for p in LORA_DIR.iterdir() if p.is_dir() and (p / 'meta.pt').is_file()],
        reverse=True,
    )


def verify_base_weights() -> list[str]:
    """Return human-readable list of missing required weight files."""
    missing = []
    for p in (TRANSFORMER_PATH, VAE_PATH, TEXT_ENCODER_PATH, CONTROLNET_PATH):
        if not p.exists():
            missing.append(str(p.relative_to(PIPELINE_ROOT)))
    if not TOKENIZER_DIR.exists() or not any(TOKENIZER_DIR.iterdir()):
        missing.append(str(TOKENIZER_DIR.relative_to(PIPELINE_ROOT)))
    return missing

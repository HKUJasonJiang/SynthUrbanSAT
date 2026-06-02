#!/usr/bin/env python
"""HDC²A + Flux2 ControlNet Training Script  —  Main Entry Point

Workflow (executed sequentially):
  1. Text embedding precompute  (if text_encoder_path is set; frees encoder before step 2)
  2. Load VAE  (AutoencoderKLFlux2)
  3. Load Transformer  (FP8 dequant -> diffusers key conversion -> controlnet merge -> FP8)
  4. Create HDC²A adapter  (trainable)
  5. Build optimizer  (AdamW over HDC²A + ControlNet trainable params)
  6. Optional resume from checkpoint
  7. Quick forward sanity check
  8. Train loop with validation + WandB logging + periodic checkpointing

Configuration:
  Edit the CONFIG dict below. All paths are relative to this file's directory.
  See README.md for full configuration reference.

Usage:
    python train_script.py
"""

import argparse
import os
import sys
import time

# Force line-buffered stdout so output appears in real time when piped/redirected
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Prevent rich/wandb service subprocess from sending terminal probe sequences
# (OSC 11 background-color query, DA1 device-attributes query) which cause
# escape-sequence garbage in piped output and at the shell prompt after exit.
# Must be set before any rich/wandb import so subprocesses inherit it.
os.environ.setdefault('NO_COLOR', '1')

import psutil
import torch
import wandb

# ── Make project root the working directory & importable ────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from scripts.models import HDC2AAdapter
from scripts.dataprep import create_dataloaders
from scripts.utility import (
    check_memory, clear_cache, load_vae, load_transformer,
)
from scripts.train import (
    train_one_epoch, validate, save_checkpoint, load_checkpoint,
    test_forward_pass,
)
from scripts.text_encoder import (
    precompute_single_prompt_embeddings, load_precomputed_embeddings,
)
from scripts.colors import (
    bold_cyan, bold_green, bold_red, bold_yellow, bold_magenta, bold_blue,
    cyan, green, yellow, magenta, gray, bold,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Log tee: strip ANSI codes when writing to file, keep colors on terminal
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re
_ANSI_RE = _re.compile(r'\x1B\[[0-9;]*m')

class _TeeLogger:
    """Forward writes to stream (with colors) and file (ANSI stripped)."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file
    def write(self, data):
        self._stream.write(data)
        self._log.write(_ANSI_RE.sub('', data))
        self._log.flush()
    def flush(self):
        self._stream.flush()
        self._log.flush()
    def fileno(self):
        return self._stream.fileno()
    def isatty(self):
        return self._stream.isatty() if hasattr(self._stream, 'isatty') else False


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Paths (all relative to PROJECT_ROOT) ──
    'transformer_path': 'weights/flux2_dev_fp8mixed.safetensors',
    'vae_path': 'weights/flux2-vae.safetensors',
    'controlnet_path': 'weights/FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors',
    'dataset_dir': 'dataset',
    'color_map_path': 'configs/color_map.json',
    'output_dir': 'output/run_hdc2a_only',  # subfolder per run

    # ── Model ──
    'image_size': 1024,
    'num_classes': 6,
    'control_in_dim': 3072,
    'fusion_dim': 768,
    'num_fusion_blocks': 3,
    'num_heads': 12,
    'num_fourier_bands': 32,
    'boundary_threshold': 0.1,

    # ── Training ──
    'num_epochs': 100,
    'batch_size': 1,
    'learning_rate': 3e-4,      # Adapter-only run: larger LR safe with only 52M params
    'adapter_lr': 3e-4,         # HDC²A adapter learning rate
    'backbone_lr': 0.0,         # ControlNet backbone LR (0 = completely frozen)
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'grad_accum_steps': 4,      # effective batch size = batch_size × grad_accum_steps
    'guidance_scale': 3.5,
    'num_workers': 0,

    # ── Training mode ──
    # True  = only train 52M HDC²A adapter (recommended for ≤2000 samples)
    # False = also fine-tune 4B ControlNet backbone (set backbone_lr > 0)
    'freeze_controlnet_backbone': False,

    # ── LR scheduler ──
    'warmup_steps': 400,        # ~1 epoch warmup (400 samples / grad_accum=4 = 100 opt steps/epoch)
    'lr_scheduler': 'cosine',   # 'cosine' | 'constant'

    # ── Loss ──
    'logit_normal_timestep': True,   # bias timestep sampling toward t≈0.5
    'minsnr_loss_weight': True,      # weight loss by 1/(t*(1-t)) for better signal

    # ── Checkpointing ──
    'keep_last_n_checkpoints': 3,    # delete older checkpoints to save disk

    # ── Text encoder ──
    'text_encoder_path': 'weights/mistral_3_small_flux2_fp8.safetensors',
    'precomputed_embeddings': 'output/text_embeddings_global.pt',
    'text_seq_len': 512,
    'text_dim': 15360,

    # ── Hardware ──
    'device': 'cuda',
    'dtype': torch.bfloat16,

    # ── Logging ──
    'log_interval': 10,             # log every N steps within each epoch
    'save_every_n_epochs': 5,
    'val_every_n_epochs': 1,

    # ── WandB ──
    'wandb_entity': 'hku-xgboost',
    'wandb_project': '<plz rewrite project name> via --name',        # can modify per run via --name

    # ── Resume ──
    'resume_from': None,            # e.g. 'output/checkpoint_epoch_0027'
}


def parse_args():
    p = argparse.ArgumentParser(description='HDC²A + Flux2 ControlNet Training')

    # ── Mode ──
    p.add_argument('--test', action='store_true',
                   help='Smoke test: load models, forward+backward 1 step, exit.')
    p.add_argument('--test-data', action='store_true',
                   help='Like --test but also loads dataset and runs 1 train step.')

    # ── Paths ──
    p.add_argument('--transformer-path', type=str, default=None)
    p.add_argument('--vae-path', type=str, default=None)
    p.add_argument('--controlnet-path', type=str, default=None)
    p.add_argument('--dataset-dir', type=str, default=None)
    p.add_argument('--color-map-path', type=str, default=None)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--text-encoder-path', type=str, default=None)
    p.add_argument('--precomputed-embeddings', type=str, default=None)
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from checkpoint dir.')

    # ── Model ──
    p.add_argument('--image-size', type=int, default=None)
    p.add_argument('--num-classes', type=int, default=None)
    p.add_argument('--control-in-dim', type=int, default=None)
    p.add_argument('--fusion-dim', type=int, default=None)
    p.add_argument('--num-fusion-blocks', type=int, default=None)
    p.add_argument('--num-heads', type=int, default=None)
    p.add_argument('--num-fourier-bands', type=int, default=None)
    p.add_argument('--boundary-threshold', type=float, default=None)

    # ── Training ──
    p.add_argument('--num-epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', '--learning-rate', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=None)
    p.add_argument('--max-grad-norm', type=float, default=None)
    p.add_argument('--grad-accum-steps', type=int, default=None)
    p.add_argument('--guidance-scale', type=float, default=None)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)

    # ── Text ──
    p.add_argument('--text-seq-len', type=int, default=None)
    p.add_argument('--text-dim', type=int, default=None)

    # ── Logging ──
    p.add_argument('--log-interval', type=int, default=None)
    p.add_argument('--save-every-n-epochs', type=int, default=None)
    p.add_argument('--val-every-n-epochs', type=int, default=None)

    # ── WandB ──
    p.add_argument('--wandb-entity', type=str, default=None)
    p.add_argument('--wandb-project', type=str, default=None)
    p.add_argument('--no-wandb', action='store_true',
                   help='Disable WandB logging (WANDB_MODE=disabled).')

    # ── ControlNet (4B) Backbone freeze ──
    _freeze_grp = p.add_mutually_exclusive_group()
    _freeze_grp.add_argument('--freeze-backbone', dest='freeze_backbone', action='store_true',
                             default=None,
                             help='Freeze ControlNet backbone, train only HDC\u00b2A (52M). Default in CONFIG.')
    _freeze_grp.add_argument('--unfreeze-backbone', dest='freeze_backbone', action='store_false',
                             help='Unfreeze ControlNet backbone (set --backbone-lr too).')
    p.add_argument('--adapter-lr', type=float, default=None,
                   help='Learning rate for HDC\u00b2A adapter (default: 3e-4).')
    p.add_argument('--backbone-lr', type=float, default=None,
                   help='Learning rate for ControlNet backbone when unfrozen (e.g. 1e-5).')

    # ── Run name, corresponding to output dir and wandb ──
    p.add_argument('--name', type=str, default=None,
                   help=('Run name. Creates output/<name>_<timestamp>/ automatically '
                         'and writes train.log inside. '
                         'E.g. python train_script.py --name hdc2a_only'))

    return p.parse_args()


def phase(msg, config=None):
    """Print a phase banner."""
    banner = f'\n{cyan("="*60)}\n  {bold_cyan(msg)}\n{cyan("="*60)}'
    print(banner)


def compose_prompt_from_json(prompt_json_path):
    """Build a single flat text string from prompt.json.

    Expected format::

        {
            "scene":    "...",
            "style":    "...",
            "elements": {"key": "val", ...},
            "lighting": "...",
            "quality":  "..."
        }

    Returns a single comma-joined string suitable for the Mistral text encoder.
    """
    import json
    with open(prompt_json_path) as f:
        d = json.load(f)
    parts = []
    if 'scene' in d:
        parts.append(d['scene'])
    if 'style' in d:
        parts.append(d['style'])
    if 'elements' in d:
        for k, v in d['elements'].items():
            parts.append(f'{k}: {v}')
    if 'lighting' in d:
        parts.append(d['lighting'])
    if 'quality' in d:
        parts.append(d['quality'])
    return ', '.join(parts)


def run_preflight_checks(config):
    """Run all pre-flight checks before loading any heavy models.

    Exits with a clear error message if any check fails.
    Checks:
      1. Required Python packages
      2. GPU availability + VRAM
      3. Weight files
      4. Dataset structure + non-empty split directories
      5. prompt.json
    """
    import importlib
    errors = []

    print(f'\n{bold_cyan("Pre-flight checks...")}')

    # ── 1. Packages ─────────────────────────────────────────────────────────
    required_packages = [
        ('torch', 'torch'),
        ('diffusers', 'diffusers'),
        ('safetensors', 'safetensors'),
        ('PIL', 'Pillow'),
        ('tifffile', 'tifffile'),
        ('wandb', 'wandb'),
        ('transformers', 'transformers'),
        ('psutil', 'psutil'),
    ]
    for mod_name, pkg_name in required_packages:
        try:
            importlib.import_module(mod_name)
            print(f'  {green("✓")} {pkg_name}')
        except ImportError:
            errors.append(f'Missing package: {pkg_name}  (install: pip install {pkg_name})')
            print(f'  {bold_red("✗")} {pkg_name}  — NOT FOUND')

    # ── 2. GPU ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        print(f'  {green("✓")} GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)')
        if vram_gb < 20:
            errors.append(f'GPU VRAM too low: {vram_gb:.1f} GB < 20 GB required')
    else:
        errors.append('No CUDA GPU available')
        print(f'  {bold_red("✗")} No CUDA GPU')

    # ── 3. Weight files ──────────────────────────────────────────────────────
    weight_keys = ['transformer_path', 'vae_path', 'controlnet_path', 'text_encoder_path']
    for key in weight_keys:
        path = config.get(key)
        if path and os.path.exists(path):
            size_gb = os.path.getsize(path) / 1e9
            print(f'  {green("✓")} {key}: {path} ({size_gb:.1f} GB)')
        elif path:
            errors.append(f'Weight file missing: {key} = {path}')
            print(f'  {bold_red("✗")} {key}: {path}  — NOT FOUND')

    # ── 4. Dataset structure ─────────────────────────────────────────────────
    dataset_dir = config['dataset_dir']
    for split in ('train', 'val'):
        for subdir in ('rgb', 'seg', 'depth'):
            d = os.path.join(dataset_dir, split, subdir)
            if os.path.isdir(d):
                n = len([f for f in os.listdir(d)
                         if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg'))])
                if n == 0:
                    errors.append(f'Empty directory: {d}')
                    print(f'  {bold_red("✗")} {split}/{subdir}: empty')
                else:
                    print(f'  {green("✓")} {split}/{subdir}: {n} files')
            else:
                errors.append(f'Missing dataset directory: {d}')
                print(f'  {bold_red("✗")} {split}/{subdir}: directory missing')

    # ── 5. Prompt JSON ───────────────────────────────────────────────────────
    prompt_json = os.path.join(dataset_dir, 'prompt.json')
    if os.path.exists(prompt_json):
        print(f'  {green("✓")} prompt.json found')
    else:
        errors.append(f'Missing prompt.json at {prompt_json}')
        print(f'  {bold_red("✗")} prompt.json missing at {prompt_json}')

    # ── Summary ──────────────────────────────────────────────────────────────
    if errors:
        print(f'\n{bold_red("Pre-flight FAILED:")}')
        for e in errors:
            print(f'  {bold_red("✗")} {e}')
        sys.exit(1)
    else:
        print(f'\n{bold_green("All pre-flight checks passed.")}')


def main():
    args = parse_args()

    config = CONFIG.copy()

    # ── CLI overrides (arg name → config key) ──────────────────────────────
    _ARG_MAP = {
        'transformer_path': 'transformer_path',
        'vae_path': 'vae_path',
        'controlnet_path': 'controlnet_path',
        'dataset_dir': 'dataset_dir',
        'color_map_path': 'color_map_path',
        'output_dir': 'output_dir',
        'text_encoder_path': 'text_encoder_path',
        'precomputed_embeddings': 'precomputed_embeddings',
        'resume': 'resume_from',
        'image_size': 'image_size',
        'num_classes': 'num_classes',
        'control_in_dim': 'control_in_dim',
        'fusion_dim': 'fusion_dim',
        'num_fusion_blocks': 'num_fusion_blocks',
        'num_heads': 'num_heads',
        'num_fourier_bands': 'num_fourier_bands',
        'boundary_threshold': 'boundary_threshold',
        'num_epochs': 'num_epochs',
        'batch_size': 'batch_size',
        'lr': 'learning_rate',
        'weight_decay': 'weight_decay',
        'max_grad_norm': 'max_grad_norm',
        'grad_accum_steps': 'grad_accum_steps',
        'guidance_scale': 'guidance_scale',
        'num_workers': 'num_workers',
        'text_seq_len': 'text_seq_len',
        'text_dim': 'text_dim',
        'log_interval': 'log_interval',
        'save_every_n_epochs': 'save_every_n_epochs',
        'val_every_n_epochs': 'val_every_n_epochs',
        'wandb_entity': 'wandb_entity',
        'wandb_project': 'wandb_project',
        'adapter_lr': 'adapter_lr',
        'backbone_lr': 'backbone_lr',
    }
    args_dict = vars(args)
    for arg_name, config_key in _ARG_MAP.items():
        val = args_dict.get(arg_name)
        if val is not None:
            config[config_key] = val

    # freeze_backbone uses store_true/store_false so None means "not given"
    if args_dict.get('freeze_backbone') is not None:
        config['freeze_controlnet_backbone'] = args_dict['freeze_backbone']

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    # ── --name: fixed output dir + internal log tee ───────────────────
    # output/<name>/  is created automatically.
    # train.log is written there (ANSI-stripped) — no shell redirect needed.
    if args.name is not None and args_dict.get('output_dir') is None:
        config['output_dir'] = os.path.join('output', args.name)

    os.makedirs(config['output_dir'], exist_ok=True)

    log_path = os.path.join(config['output_dir'], 'train.log')
    _log_file = open(log_path, 'a', buffering=1, encoding='utf-8')
    sys.stdout = _TeeLogger(sys.__stdout__, _log_file)
    sys.stderr = _TeeLogger(sys.__stderr__, _log_file)
    print(f'Run dir : {config["output_dir"]}')
    print(f'Log file: {log_path}')

    # In test mode, force small epoch count and high log frequency
    if args.test or args.test_data:
        config['num_epochs'] = 1
        config['log_interval'] = 1

    # ── WandB init ──────────────────────────────────────────────────────────
    mode = 'test' if (args.test or args.test_data) else 'train'
    _run_label = args.name if args.name else 'hdc2a'
    run = wandb.init(
        entity=config['wandb_entity'],
        project=config['wandb_project'],
        name=f'{_run_label}-{mode}-{time.strftime("%Y%m%d-%H%M%S")}',
        config={k: str(v) for k, v in config.items()},
        tags=[mode],
        settings=wandb.Settings(
            console='off',           # 'wrap' probes terminal with OSC11/DA1 queries → garbage in piped output
            x_disable_stats=True,    # disable built-in 21 system metrics
        ),
    )
    config['_wandb_active'] = True

    device = config['device']
    dtype = config['dtype']

    gpu_name = torch.cuda.get_device_name()
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {bold(gpu_name)} | VRAM: {cyan(f"{vram_total:.1f} GB")} | PyTorch: {torch.__version__}')

    # ── Print final resolved config ────────────────────────────────────────
    print(f'\n{bold_cyan("Final Configuration:")}')
    _SECTIONS = [
        ('Paths', [
            'transformer_path', 'vae_path', 'controlnet_path',
            'dataset_dir', 'color_map_path', 'output_dir',
            'text_encoder_path', 'precomputed_embeddings',
        ]),
        ('Model', [
            'image_size', 'num_classes', 'control_in_dim', 'fusion_dim',
            'num_fusion_blocks', 'num_heads', 'num_fourier_bands', 'boundary_threshold',
        ]),
        ('Training', [
            'num_epochs', 'batch_size', 'learning_rate', 'weight_decay',
            'max_grad_norm', 'grad_accum_steps', 'guidance_scale', 'num_workers',
        ]),
        ('Text Encoder', [
            'text_seq_len', 'text_dim',
        ]),
        ('Logging', [
            'log_interval', 'save_every_n_epochs', 'val_every_n_epochs',
        ]),
        ('WandB', [
            'wandb_entity', 'wandb_project',
        ]),
        ('Resume', [
            'resume_from',
        ]),
    ]
    for section_name, keys in _SECTIONS:
        print(f'  {bold(section_name)}:')
        for k in keys:
            v = config.get(k)
            v_str = str(v) if v is not None else gray('(not set)')
            print(f'    {k:<28s} {v_str}')

    check_memory('pre-flight')

    # ── Pre-flight checks (skip for --test mode which uses random data) ─────
    if not (args.test or args.test_data):
        run_preflight_checks(config)

    # ── Text Embeddings (precompute BEFORE loading transformer to save VRAM) ──
    phase('[1/8] Text Embeddings', config)
    embeddings_dict = None
    precomp_path = config.get('precomputed_embeddings')
    if precomp_path and os.path.exists(precomp_path):
        # Reuse cached embedding
        print(f'  Loading cached embedding from {precomp_path}')
        embeddings_dict = load_precomputed_embeddings(precomp_path)
    elif config.get('text_encoder_path') and os.path.exists(config['text_encoder_path']):
        # Build prompt string from prompt.json and encode once before loading transformer
        prompt_json_path = os.path.join(config['dataset_dir'], 'prompt.json')
        if not os.path.exists(prompt_json_path):
            print(f'{bold_red("ERROR:")} prompt.json not found at {prompt_json_path}')
            sys.exit(1)
        prompt_text = compose_prompt_from_json(prompt_json_path)
        print(f'  Composed prompt ({len(prompt_text)} chars):\n    {prompt_text[:160]}...')
        save_path = precomp_path or os.path.join(config['output_dir'], 'text_embeddings_global.pt')
        os.makedirs(config['output_dir'], exist_ok=True)
        precompute_single_prompt_embeddings(
            model_path=config['text_encoder_path'],
            prompt_text=prompt_text,
            output_path=save_path,
            max_sequence_length=config['text_seq_len'],
            device=device,
            dtype=dtype,
        )
        embeddings_dict = load_precomputed_embeddings(save_path)
        config['precomputed_embeddings'] = save_path
    else:
        # No text encoder and no precomputed embeddings — fail for real training
        if args.test or args.test_data:
            print('[NOTE] No text embeddings — using random placeholders for test mode.')
        else:
            print(f'{bold_red("ERROR:")} No text_encoder_path and no precomputed_embeddings.')
            print('       Set text_encoder_path in CONFIG or provide precomputed_embeddings.')
            sys.exit(1)

    # ── Load VAE ────────────────────────────────────────────────────────────
    phase('[2/8] Loading VAE', config)
    t0 = time.time()
    vae, bn_mean, bn_std = load_vae(config['vae_path'], device, dtype)
    print(f'  Done ({time.time()-t0:.1f}s), VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    check_memory('after VAE')

    # ── Load Transformer ────────────────────────────────────────────────────
    phase('[3/8] Loading Transformer', config)
    t0 = time.time()
    transformer = load_transformer(
        config['transformer_path'],
        config['controlnet_path'],
        config['control_in_dim'],
        device, dtype,
    )
    print(f'  Done ({time.time()-t0:.1f}s), VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    transformer.gradient_checkpointing = True
    print('  Gradient checkpointing: enabled')

    # When freezing controlnet backbone, disable requires_grad for ALL transformer params.
    # Gradients still flow through frozen ops back to hdc2a (control_context has grad).
    if config.get('freeze_controlnet_backbone', False):
        for p in transformer.parameters():
            p.requires_grad_(False)
        print('  Backbone FROZEN: all transformer params set requires_grad=False')
        print('  Gradients will still propagate to HDC²A via control_context autograd')
    check_memory('after Transformer')

    # ── Create HDC²A Adapter ────────────────────────────────────────────────
    phase('[4/8] Creating HDC²A Adapter', config)
    hdc2a = HDC2AAdapter(
        num_classes=config['num_classes'],
        fusion_dim=config['fusion_dim'],
        output_dim=config['control_in_dim'],
        num_heads=config['num_heads'],
        num_fusion_blocks=config['num_fusion_blocks'],
        num_fourier_bands=config['num_fourier_bands'],
        boundary_threshold=config['boundary_threshold'],
        image_size=config['image_size'],
    ).to(device, dtype)

    hdc2a_params = sum(p.numel() for p in hdc2a.parameters())
    ctrl_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_trainable = hdc2a_params + ctrl_params
    print(f'HDC²A: {hdc2a_params/1e6:.1f}M params')
    print(f'Control: {ctrl_params/1e6:.1f}M params')
    print(f'Total trainable: {total_trainable/1e6:.1f}M params')

    # ── Optimizer ───────────────────────────────────────────────────────────
    phase('[5/8] Building Optimizer', config)

    # Separate param groups: HDC2A adapter (trainable) vs controlnet backbone
    adapter_lr = config.get('adapter_lr', config['learning_rate'])
    backbone_lr = config.get('backbone_lr', 0.0)
    param_groups = [{'params': list(hdc2a.parameters()), 'lr': adapter_lr, 'name': 'adapter'}]
    ctrl_trainable = [p for p in transformer.parameters() if p.requires_grad]
    if backbone_lr > 0 and ctrl_trainable:
        param_groups.append({'params': ctrl_trainable, 'lr': backbone_lr, 'name': 'backbone'})
    elif not ctrl_trainable:
        print('  Note: transformer has no requires_grad params (backbone fully frozen)')

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    print(f'  AdamW: adapter_lr={adapter_lr:.2e}, backbone_lr={backbone_lr:.2e}')
    for pg in optimizer.param_groups:
        print(f"    param_group '{pg['name']}': {len(pg['params'])} tensors, lr={pg['lr']:.2e}")

    # LR scheduler: linear warmup → cosine decay
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    # Optimizer steps per epoch (training steps / grad_accum), rough estimate before data loads
    steps_per_epoch_est = (400 // config['batch_size']) // max(config['grad_accum_steps'], 1)
    total_optimizer_steps = max(steps_per_epoch_est * config['num_epochs'], 1)
    warmup_steps = config.get('warmup_steps', 200)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                            total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=max(total_optimizer_steps - warmup_steps, 1),
                                     eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                             milestones=[warmup_steps])
    print(f'  Scheduler: {warmup_steps} warmup steps → cosine over ~{total_optimizer_steps} steps')

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    if config['resume_from'] is not None:
        phase('[6/8] Resuming from Checkpoint', config)
        print(f'  Path: {config["resume_from"]}')
        start_epoch = load_checkpoint(
            config['resume_from'], hdc2a, transformer, optimizer, device) + 1
        print(f'  Resuming from epoch {start_epoch}')
    else:
        print('  [6/8] Resume: skipped (no checkpoint specified)')

    # ── Quick forward test ──────────────────────────────────────────────────
    phase('[7/8] Forward Sanity Check', config)
    test_forward_pass(hdc2a, transformer, vae, bn_mean, bn_std, config)
    check_memory('after test')

    # ── --test mode: exit after forward test ────────────────────────────────
    if args.test:
        print(f'\n{bold_green("*** --test passed: all models loaded, forward test OK. Exiting. ***")}')
        wandb.finish()
        return

    # ── Data ────────────────────────────────────────────────────────────────
    phase('[8/8] Loading Data', config)
    train_loader, val_loader = create_dataloaders(
        config['dataset_dir'],
        config['color_map_path'],
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        num_classes=config['num_classes'],
        embeddings_dict=embeddings_dict,
    )

    if train_loader is None:
        print('\n[WARNING] No training data found. Pipeline verified successfully.')
        print('Place your data in dataset/train/ (rgb/, seg/, depth/, captions.json) and re-run.')
        print('Then set text_encoder_path to precompute text embeddings, or provide precomputed_embeddings.')
        wandb.finish()
        return

    # ── --test-data mode: run 1 train step then exit ────────────────────────
    if args.test_data:
        if train_loader is None:
            print('\n*** --test-data FAILED: no training data found. ***')
            wandb.finish()
            return
        print('\n*** --test-data: running 1 training step... ***')
        train_loss = train_one_epoch(
            0, hdc2a, transformer, vae, bn_mean, bn_std,
            train_loader, optimizer, None, config, scheduler=scheduler,
        )
        print(f'*** --test-data passed: 1 epoch OK, loss={train_loss:.6f} ***')
        wandb.finish()
        return

    # ── Training Loop ───────────────────────────────────────────────────────
    total_epochs = config['num_epochs']
    total_steps = total_epochs * len(train_loader)
    print(f'\n{bold("="*70)}')
    print(bold(f'Starting training: {total_epochs} epochs × {len(train_loader)} steps = {total_steps} total steps'))
    print(f'  batch_size={config["batch_size"]}, grad_accum={config["grad_accum_steps"]}, '
          f'effective_bs={config["batch_size"] * config["grad_accum_steps"]}')
    print(f'  adapter_lr={config.get("adapter_lr", config["learning_rate"]):.2e}, '
          f'weight_decay={config["weight_decay"]}')
    print(f'{bold("="*70)}')

    best_val_loss = float('inf')
    best_ckpt_path = None
    keep_last_n = config.get('keep_last_n_checkpoints', 3)
    train_start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        epoch_progress = (epoch - start_epoch) / max(total_epochs - start_epoch, 1) * 100

        # Train
        print(f'\n--- {bold_magenta(f"Epoch {epoch}/{total_epochs-1}")}  ({bold_cyan(f"{epoch_progress:.0f}%")} done) ---')
        train_loss = train_one_epoch(
            epoch, hdc2a, transformer, vae, bn_mean, bn_std,
            train_loader, optimizer, None, config, scheduler=scheduler,
        )
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start_time
        epochs_done = epoch - start_epoch + 1
        eta = elapsed / epochs_done * (total_epochs - start_epoch - epochs_done)
        print(f'  Train loss: {yellow(f"{train_loss:.6f}")} ({epoch_time:.1f}s)  '
              f'ETA: {cyan(f"{eta/60:.0f}min")}')

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "adapter_lr": optimizer.param_groups[0]['lr'],
            "gpu_mem_gb": torch.cuda.memory_allocated() / 1e9,
            "cpu_pct": psutil.cpu_percent(),
        })

        # Validate
        val_loss = None
        if val_loader is not None and (epoch + 1) % config['val_every_n_epochs'] == 0:
            val_loss = validate(
                epoch, hdc2a, transformer, vae, bn_mean, bn_std,
                val_loader, config,
            )
            print(f'  Val loss: {val_loss:.6f}')
            wandb.log({"val_loss": val_loss})

            # save_checkpoint handles best tracking + rotation internally
            _, _, best_val_loss, best_ckpt_path = save_checkpoint(
                epoch, hdc2a, transformer, optimizer,
                val_loss, config['output_dir'], config,
                keep_last_n=keep_last_n,
                best_loss=best_val_loss,
                best_ckpt_path=best_ckpt_path,
            )

        # Periodic checkpoint (even when no val this epoch)
        elif (epoch + 1) % config['save_every_n_epochs'] == 0:
            _, _, best_val_loss, best_ckpt_path = save_checkpoint(
                epoch, hdc2a, transformer, optimizer,
                train_loss, config['output_dir'], config,
                keep_last_n=keep_last_n,
                best_loss=best_val_loss,
                best_ckpt_path=best_ckpt_path,
            )

        check_memory(f'epoch {epoch} end')

    # ── Cleanup ─────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('Training complete.')
    print(f'{"="*70}')

    wandb.summary.update({
        "final_train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "total_epochs": config['num_epochs'],
    })
    wandb.finish()


if __name__ == '__main__':
    main()

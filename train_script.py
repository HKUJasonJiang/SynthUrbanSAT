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
# === OVERFIT TEST === 减少 CUDA 显存碎片化，有助于 backward 阶段分配激活梯度缓冲区
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# === END OVERFIT TEST ===

import psutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
# === OVERFIT TEST === 过拟合辅助模块（正常训练时无需修改，--overfit 时自动激活）
from scripts.overfit import (
    apply_lora_to_control_blocks,
    print_param_stats,
    gradient_check,
    generate_overfit_samples,
    save_overfit_grid,
    save_step_vis_single,
    save_milestone_big_grid,
    save_lora_checkpoint,
    load_lora_checkpoint,
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
# Distributed training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def setup_distributed():
    """Initialize distributed training if launched via torchrun.

    Returns:
        (rank, world_size, local_rank)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')                 # init distributed backend, A100
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main_process():
    """Return True if this is rank 0 (or single-GPU mode)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_distributed():
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


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
    'batch_size': 4,            # lora=4, controlnet=1
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
    'freeze_controlnet_backbone': True,

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
    'milestone_pct': 10,             # milestone visualization interval (%)

    # ── WandB ──
    'wandb_entity': '',              # empty = personal namespace of logged-in user
    'wandb_project': '<plz rewrite project name> via --name',        # can modify per run via --name

    # ── Resume ──
    'resume_from': None,            # e.g. 'output/checkpoint_epoch_0027'
}


# === OVERFIT TEST ===
# 过拟合 sanity check 的超参数默认值。
# 使用 --overfit 时自动应用（仍可被其他 CLI 参数覆盖）。
# 快速切换：--overfit 开启，不加 --overfit 为正常训练。
OVERFIT_DEFAULTS = {
    # ── 小数据集（5 张图）──
    'dataset_dir':             'dataset_small',
    # ── 图像尺寸：512 代替 1024，latent tokens 从 4096 降到 1024（4x 少），大幅节省显存
    # 过拟合 sanity check 不需要完整分辨率
    'image_size':              512,
    # ── 训练轮数：足够让 loss 趋近 0 ──
    'num_epochs':              2000,
    'batch_size':              5,           # 全部 5 张图一次过（显存够的话）
    # ── 学习率：过拟合不需要小 lr，但也别太大 ──
    'adapter_lr':              1e-4,        # HDC²A adapter LR
    'backbone_lr':             1e-4,        # LoRA LR（backbone_lr>0 → optimizer 会加 LoRA group）
    'weight_decay':            0.0,         # 过拟合测试不需要正则化
    # ── LR scheduler：固定 LR，不用 cosine/warmup ──
    'lr_scheduler':            'constant',
    'warmup_steps':            0,
    # ── 梯度累积：过拟合用小 batch，不需要累积 ──
    'grad_accum_steps':        1,
    # ── 冻结策略：backbone 完全冻结，LoRA + adapter 可训练 ──
    'freeze_controlnet_backbone': True,
    # ── 日志 / 保存频率 ──
    'log_interval':            1,
    'save_every_n_epochs':     100,
    'val_every_n_epochs':      99999,       # 过拟合不做 validation
    'num_workers':             0,
    # ── 输出目录 ──
    'output_dir':              'output/overfit_test',
}
# === END OVERFIT TEST ===


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

    # ── Data augmentation ──
    _aug_grp = p.add_mutually_exclusive_group()
    _aug_grp.add_argument('--augment', dest='use_augment', action='store_true', default=None,
                          help='Enable data augmentation on the training split '
                               '(flips, rotations, scale-crop, colour jitter, blur, noise, cutout). '
                               'Applied to RGB+seg+depth jointly where spatial; RGB-only otherwise.')
    _aug_grp.add_argument('--no-augment', dest='use_augment', action='store_false',
                          help='Disable data augmentation (default).')

    # ── Run name, corresponding to output dir and wandb ──
    p.add_argument('--name', type=str, required=True,
                   help=('Run name (required). Output is saved to output/<name>/. '
                         'E.g. python train_script.py --name hdc2a_only'))

    # ── Ablation controls ──
    p.add_argument('--disable-depth', action='store_true',
                   help='Zero out depth input to HDC²A (seg-only ablation).')
    p.add_argument('--no-minsnr', action='store_true',
                   help='Disable min-SNR loss weighting; use uniform weight w=1.')

    # ── LoRA (enabled by default, disable with --no-lora) ──
    p.add_argument('--no-lora', action='store_true',
                   help='Disable LoRA injection (by default LoRA is ON with rank=32).')
    p.add_argument('--lora-rank', type=int, default=32,
                   help='LoRA rank (default: 32).')
    p.add_argument('--lora-alpha', type=float, default=None,
                   help='LoRA alpha (default: same as --lora-rank).')

    # ── Auto-upload to HuggingFace ──
    p.add_argument('--hf-repo', type=str, default=None,
                   help='HuggingFace repo ID for auto-upload at each milestone '
                        '(e.g. JasonXF/SynthUrbanSAT-Output). Disabled if not set.')

    # === OVERFIT TEST ===
    p.add_argument('--overfit', action='store_true',
                   help=('过拟合 sanity check 模式。自动使用 dataset_small，'
                         '给 ControlNet 加 LoRA，固定 LR，2000 epochs。'
                         '改为 False（不传此参数）即恢复正常训练。'))
    p.add_argument('--vis-every', type=int, default=50,
                   help='每隔 N 个 epoch 保存可视化图像（仅 --overfit 模式，默认 50）。')
    # === END OVERFIT TEST ===

    # ── Milestone visualization ──
    p.add_argument('--milestone-pct', type=int, default=None,
                   help='Milestone visualization interval as integer percentage '
                        '(e.g. 10 = every 10%%, 5 = every 5%%). Default: 10.')

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
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f'  {green("✓")} GPU: {gpu_name} ({vram_gb:.1f} GiB VRAM)')
        if vram_gb < 20:
            errors.append(f'GPU VRAM too low: {vram_gb:.1f} GiB < 20 GiB required')
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
    for split in ('train', 'val', 'test'):
        for subdir in ('rgb', 'seg', 'depth'):
            d = os.path.join(dataset_dir, split, subdir)
            if os.path.isdir(d):
                n = len([f for f in os.listdir(d)
                         if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg'))])
                if n == 0:
                    # test split is optional — warn instead of error
                    if split == 'test':
                        print(f'  {yellow("○")} {split}/{subdir}: empty (optional)')
                    else:
                        errors.append(f'Empty directory: {d}')
                        print(f'  {bold_red("✗")} {split}/{subdir}: empty')
                else:
                    print(f'  {green("✓")} {split}/{subdir}: {n} files')
            else:
                if split == 'test':
                    print(f'  {yellow("○")} {split}/{subdir}: not found (optional)')
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

    # ── Distributed training setup ──────────────────────────────────────────
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    # === OVERFIT TEST ===
    # 先把 OVERFIT_DEFAULTS 合并进来，顺序：CONFIG → OVERFIT_DEFAULTS → CLI args
    # 这样 CLI 参数依然可以覆盖 overfit 默认值（保持灵活性）
    if args.overfit:
        config.update(OVERFIT_DEFAULTS)
        config['_overfit_mode'] = True
        if is_main:
            print(f'\n{bold_red("="*60)}\n  {bold_red("OVERFIT TEST MODE 过拟合测试模式")} — 仅用于 sanity check\n{bold_red("="*60)}')
    else:
        config['_overfit_mode'] = False
    lora_modules = {}  # 仅 overfit 模式下填充
    # === END OVERFIT TEST ===

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
        'milestone_pct': 'milestone_pct',
    }
    args_dict = vars(args)
    for arg_name, config_key in _ARG_MAP.items():
        val = args_dict.get(arg_name)
        if val is not None:
            config[config_key] = val

    # freeze_backbone uses store_true/store_false so None means "not given"
    if args_dict.get('freeze_backbone') is not None:
        config['freeze_controlnet_backbone'] = args_dict['freeze_backbone']

    # wandb project defaults to run name (so output dir and WandB project are always aligned)
    if args_dict.get('wandb_project') is None:
        config['wandb_project'] = args.name

    if args.seed is not None:
        torch.manual_seed(args.seed + rank)  # different seed per rank for data variety
        torch.cuda.manual_seed_all(args.seed + rank)

    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    # ── Ablation flags → config ─────────────────────────────────────────────
    if args.disable_depth:
        config['disable_depth'] = True
    if args.no_minsnr:
        config['minsnr_loss_weight'] = False

    # ── --name: fixed output dir + internal log tee ───────────────────
    # output/<name>/  is created automatically.
    # train.log is written there (ANSI-stripped) — no shell redirect needed.
    config['output_dir'] = os.path.join('output', args.name)

    os.makedirs(config['output_dir'], exist_ok=True)

    # Only rank 0 tees stdout/stderr to the log file
    if is_main:
        log_path = os.path.join(config['output_dir'], 'train.log')
        _log_file = open(log_path, 'a', buffering=1, encoding='utf-8')
        sys.stdout = _TeeLogger(sys.__stdout__, _log_file)
        sys.stderr = _TeeLogger(sys.__stderr__, _log_file)
        print(f'Run dir : {config["output_dir"]}')
        print(f'Log file: {log_path}')
        if world_size > 1:
            print(f'Distributed: {world_size} GPUs (rank {rank}, local_rank {local_rank})')

    # In test mode, force small epoch count and high log frequency
    if args.test or args.test_data:
        config['num_epochs'] = 1
        config['log_interval'] = 1

    device = config['device']
    dtype = config['dtype']

    # In multi-GPU mode, each rank uses its own GPU via local_rank
    if world_size > 1:
        device = f'cuda:{local_rank}'
        config['device'] = device

    # Fail fast on broken CUDA runtime before starting WandB/retry loops.
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            print(
                f"[FATAL] CUDA requested (device={device}) but torch.cuda.is_available() is False. "
                "Please fix NVIDIA driver/runtime first, then resume from checkpoint."
            )
            sys.exit(2)
        gpu_name = torch.cuda.get_device_name(local_rank)
        vram_total = torch.cuda.get_device_properties(local_rank).total_memory / (1024 ** 3)
        if is_main:
            print(f'GPU: {bold(gpu_name)} | VRAM: {cyan(f"{vram_total:.1f} GiB")} | PyTorch: {torch.__version__}')
    else:
        if is_main:
            print(f'GPU: {gray("CPU mode")} | PyTorch: {torch.__version__}')

    # ── WandB init (rank 0 only) ──────────────────────────────────────────────
    mode = 'test' if (args.test or args.test_data) else 'train'
    _run_label = args.name if args.name else 'hdc2a'
    if not is_main:
        os.environ['WANDB_MODE'] = 'disabled'
        wandb.init(mode='disabled')
        config['_wandb_active'] = False
    else:
        try:
            _wandb_kwargs = dict(
                project=config['wandb_project'],
                name=f'{_run_label}-{mode}-{time.strftime("%Y%m%d-%H%M%S")}',
                config={k: str(v) for k, v in config.items()},
                tags=[mode],
                settings=wandb.Settings(
                    console='off',
                    x_disable_stats=True,
                ),
            )
            # Only set entity when explicitly configured — empty means personal namespace
            _wandb_entity = config.get('wandb_entity', '')
            if _wandb_entity:
                _wandb_kwargs['entity'] = _wandb_entity
            run = wandb.init(**_wandb_kwargs)
            config['_wandb_active'] = True
        except Exception as _wandb_err:
            print(f'\n{bold_yellow("WARNING: WandB init failed — training will continue without logging.")}')
            print(f'  Error: {_wandb_err}')
            print(f'  Fix: check WANDB_API_KEY permissions, or run with --no-wandb to silence this.\n')
            os.environ['WANDB_MODE'] = 'disabled'
            wandb.init(mode='disabled')
            config['_wandb_active'] = False

    # ── Print final resolved config ────────────────────────────────────────
    if is_main:
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
    if not (args.test or args.test_data) and is_main:
        run_preflight_checks(config)
    # Barrier: wait for rank 0 preflight before all ranks proceed
    if world_size > 1:
        dist.barrier()

    # ── Text Embeddings (precompute BEFORE loading transformer to save VRAM) ──
    phase('[1/8] Text Embeddings', config)
    embeddings_dict = None
    precomp_path = config.get('precomputed_embeddings')
    if precomp_path and os.path.exists(precomp_path):
        # Reuse cached embedding
        if is_main:
            print(f'  Loading cached embedding from {precomp_path}')
        embeddings_dict = load_precomputed_embeddings(precomp_path)
    elif config.get('text_encoder_path') and os.path.exists(config['text_encoder_path']):
        # Rank 0 encodes; other ranks wait, then all load from file
        if is_main:
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
            config['precomputed_embeddings'] = save_path
        if world_size > 1:
            dist.barrier()
        # All ranks load from the saved file
        _emb_path = config.get('precomputed_embeddings') or os.path.join(config['output_dir'], 'text_embeddings_global.pt')
        embeddings_dict = load_precomputed_embeddings(_emb_path)
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
    print(f'  Done ({time.time()-t0:.1f}s), VRAM: {torch.cuda.memory_allocated()/(1024**3):.2f} GiB')
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
    print(f'  Done ({time.time()-t0:.1f}s), VRAM: {torch.cuda.memory_allocated()/(1024**3):.2f} GiB')
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

    # === LoRA injection (--use-lora or --overfit) ===
    # 给 ControlNet control_transformer_blocks 的 attention 层注入 LoRA。
    # 原始权重完全冻结，只训 rank 很小的 A/B 矩阵，大幅减少可训练参数量。
    _apply_lora = not getattr(args, 'no_lora', False) or config.get('_overfit_mode', False)
    if _apply_lora:
        phase('[4.5/8] Applying LoRA to ControlNet Control Blocks', config)
        _lora_rank = getattr(args, 'lora_rank', 32)
        _lora_alpha = getattr(args, 'lora_alpha', None) or float(_lora_rank)
        print(f'  LoRA rank={_lora_rank}, alpha={_lora_alpha}, dropout=0')
        lora_modules = apply_lora_to_control_blocks(
            transformer, rank=_lora_rank, alpha=_lora_alpha,
        )
        print_param_stats(hdc2a, transformer, lora_modules)
    # === END LoRA ===

    # ── DDP wrap HDC²A before optimizer ─────────────────────────────────────
    # Transformer is NOT wrapped (mostly frozen; trainable params get manual gradient sync).
    hdc2a_raw = hdc2a  # keep reference for checkpoint save/load
    if world_size > 1:
        hdc2a = DDP(hdc2a, device_ids=[local_rank])
        print(f'  HDC²A wrapped in DDP (device_ids=[{local_rank}])')

    # ── Optimizer ───────────────────────────────────────────────────────────
    phase('[5/8] Building Optimizer', config)

    # Separate param groups: HDC2A adapter (trainable) vs controlnet backbone
    adapter_lr = config.get('adapter_lr', config['learning_rate'])
    backbone_lr = config.get('backbone_lr', 0.0)
    param_groups = [{'params': list(hdc2a.parameters()), 'lr': adapter_lr, 'name': 'adapter'}]
    ctrl_trainable = [p for p in transformer.parameters() if p.requires_grad]
    if backbone_lr > 0 and ctrl_trainable:
        # === OVERFIT TEST === 过拟合模式下，ctrl_trainable 实际上是 LoRA A/B 参数
        _pg_name = 'lora' if config.get('_overfit_mode', False) else 'backbone'
        param_groups.append({'params': ctrl_trainable, 'lr': backbone_lr, 'name': _pg_name})
        # === END OVERFIT TEST ===
    elif not ctrl_trainable:
        print('  Note: transformer has no requires_grad params (backbone fully frozen)')

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    print(f'  AdamW: adapter_lr={adapter_lr:.2e}, backbone_lr={backbone_lr:.2e}')
    for pg in optimizer.param_groups:
        print(f"    param_group '{pg['name']}': {len(pg['params'])} tensors, lr={pg['lr']:.2e}")

    # LR scheduler: linear warmup → cosine decay
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
    # Optimizer steps per epoch (training steps / grad_accum), rough estimate before data loads
    steps_per_epoch_est = (400 // config['batch_size']) // max(config['grad_accum_steps'], 1)
    total_optimizer_steps = max(steps_per_epoch_est * config['num_epochs'], 1)
    warmup_steps = config.get('warmup_steps', 200)

    # === OVERFIT TEST === 过拟合模式不用 warmup/cosine，直接固定 LR
    if config.get('_overfit_mode', False) or config.get('lr_scheduler', 'cosine') == 'constant':
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=total_optimizer_steps)
        print(f'  Scheduler: constant LR (overfit/constant mode)')
    else:
        # === END OVERFIT TEST ===
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
    _resume_global_step = 0
    _resume_best_val_loss = float('inf')
    if config['resume_from'] is not None:
        phase('[6/8] Resuming from Checkpoint', config)
        print(f'  Path: {config["resume_from"]}')
        _ckpt_info = load_checkpoint(
            config['resume_from'], hdc2a_raw, transformer, optimizer, device,
            scheduler=scheduler)
        start_epoch = _ckpt_info['epoch'] + 1
        _resume_global_step = _ckpt_info.get('global_step', 0)
        _resume_best_val_loss = _ckpt_info.get('best_val_loss', float('inf'))
        print(f'  Resuming from epoch {start_epoch}, '
              f'global_step={_resume_global_step}, '
              f'best_val_loss={_resume_best_val_loss:.6f}')
    else:
        print('  [6/8] Resume: skipped (no checkpoint specified)')

    # ── Quick forward test ──────────────────────────────────────────────────
    phase('[7/8] Forward Sanity Check', config)
    test_forward_pass(hdc2a, transformer, vae, bn_mean, bn_std, config)
    check_memory('after test')

    # === OVERFIT TEST === 过拟合模式下，在正式训练前做一次梯度检查
    if config.get('_overfit_mode', False) and lora_modules:
        _grad_ok = gradient_check(
            hdc2a, transformer, vae, bn_mean, bn_std, lora_modules, config)
        if not _grad_ok:
            print(f'\n{bold_red("⚠ 梯度检查失败！部分参数无梯度，请查看上方警告后再启动正式训练。")}')
            # 不强制退出，允许用户继续观察，但会打印明显警告
    # === END OVERFIT TEST ===

    # ── --test mode: exit after forward test ────────────────────────────────
    if args.test:
        if is_main:
            print(f'\n{bold_green("*** --test passed: all models loaded, forward test OK. Exiting. ***")}')
        wandb.finish()
        cleanup_distributed()
        return

    # ── Data ────────────────────────────────────────────────────────────────
    phase('[8/8] Loading Data', config)
    # === OVERFIT TEST === 过拟合模式 shuffle=False，确保每 epoch 顺序相同，便于 debug
    _shuffle_train = not config.get('_overfit_mode', False)
    # === END OVERFIT TEST ===
    # Augmentation: disabled in overfit mode; respect --augment / --no-augment otherwise
    _use_augment = False
    if not config.get('_overfit_mode', False):
        _use_augment = bool(args_dict.get('use_augment'))
    if is_main:
        if _use_augment:
            print(f'[Data] {bold_cyan("Data augmentation: ENABLED")} '
                  f'(flips, rotations, scale-crop, colour jitter, blur, noise, cutout)')
        else:
            print(f'[Data] Data augmentation: disabled')

    # Always load test split for milestone visualization
    _overfit_mode = config.get('_overfit_mode', False)
    _aug = False if _overfit_mode else _use_augment
    _distributed = (world_size > 1)
    train_loader, val_loader, test_loader = create_dataloaders(
        config['dataset_dir'],
        config['color_map_path'],
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        num_classes=config['num_classes'],
        embeddings_dict=embeddings_dict,
        shuffle_train=_shuffle_train,
        use_augment=_aug,
        include_test=True,
        distributed=_distributed,
    )

    if train_loader is None:
        if is_main:
            print('\n[WARNING] No training data found. Pipeline verified successfully.')
            print('Place your data in dataset/train/ (rgb/, seg/, depth/, captions.json) and re-run.')
            print('Then set text_encoder_path to precompute text embeddings, or provide precomputed_embeddings.')
        wandb.finish()
        cleanup_distributed()
        return

    # ── --test-data mode: run 1 train step then exit ────────────────────────
    if args.test_data:
        if train_loader is None:
            if is_main:
                print('\n*** --test-data FAILED: no training data found. ***')
            wandb.finish()
            cleanup_distributed()
            return
        if is_main:
            print('\n*** --test-data: running 1 training step... ***')
        train_loss, _ = train_one_epoch(
            0, hdc2a, transformer, vae, bn_mean, bn_std,
            train_loader, optimizer, None, config, scheduler=scheduler,
        )
        if is_main:
            print(f'*** --test-data passed: 1 epoch OK, loss={train_loss:.6f} ***')
        wandb.finish()
        cleanup_distributed()
        return

    # ── Training Loop ───────────────────────────────────────────────────────
    total_epochs = config['num_epochs']
    total_steps = total_epochs * len(train_loader)
    if is_main:
        _eff_bs = config['batch_size'] * config['grad_accum_steps'] * world_size
        print(f'\n{bold("="*70)}')
        print(bold(f'Starting training: {total_epochs} epochs × {len(train_loader)} steps = {total_steps} total steps'))
        print(f'  batch_size={config["batch_size"]}, grad_accum={config["grad_accum_steps"]}, '
              f'world_size={world_size}, effective_bs={_eff_bs}')
        print(f'  adapter_lr={config.get("adapter_lr", config["learning_rate"]):.2e}, '
              f'weight_decay={config["weight_decay"]}')
        print(f'{bold("="*70)}')

    best_val_loss = _resume_best_val_loss
    best_ckpt_path = None
    keep_last_n = config.get('keep_last_n_checkpoints', 3)
    train_start_time = time.time()
    _train_losses = []   # for loss-curve PNG
    _cumulative_opt_steps = _resume_global_step  # tracks optimizer steps across epochs

    # === OVERFIT TEST ===
    _overfit_mode = config.get('_overfit_mode', False)

    # ── Milestone visualization setup ──────────────────────────────────────
    # Vis at 5 equidistant steps: 0%, 20%, 40%, 60%, 100% of total_steps.
    # At each milestone the current model is used to generate samples for
    # 3 fixed batches (train / val / test), and per-step images are saved.
    # After all milestones are done a 5-row × 8-col big grid is saved per split.
    _step_vis_dir = os.path.join(config['output_dir'], 'step_vis')
    _n_classes = config.get('num_classes', 6)

    # Compute milestone global steps (configurable via --milestone-pct, default 10%)
    # total_steps = total_epochs × steps_per_epoch
    _ms_interval = config.get('milestone_pct', 10) / 100  # e.g. 10 → 0.1
    _milestone_pcts = [i * _ms_interval for i in range(int(1 / _ms_interval) + 1)]
    _milestone_steps = sorted(set(
        int(round(p * max(total_steps - 1, 1))) for p in _milestone_pcts
    ))
    # Labels shown in the big grid header (e.g. "step 0", "step 200")
    _milestone_labels = [f'step {s}' for s in _milestone_steps]
    print(f'  {bold_cyan("[Milestone Vis]")} steps: {_milestone_steps}')

    # Fixed vis batches — first N samples from each split
    # train/val: 10 samples, test: all samples
    def _fixed_batch(loader, n=10):
        if loader is None:
            return None
        # collect samples one by one to pick first n without re-ordering
        samples = []
        for b in loader:
            for i in range(b['rgb'].shape[0]):
                samples.append({k: v[i:i+1] for k, v in b.items()
                                if isinstance(v, torch.Tensor)})
                if len(samples) >= n:
                    break
            if len(samples) >= n:
                break
        if not samples:
            return None
        # Stack into a single batch dict
        out = {}
        for k in samples[0]:
            out[k] = torch.cat([s[k] for s in samples], dim=0)
        return out

    _vis_train_batch = _fixed_batch(train_loader, n=10)
    _vis_val_batch   = _fixed_batch(val_loader,   n=10)
    _vis_test_batch  = _fixed_batch(test_loader,  n=999)

    # Split each batch into chunks of 5 for readable grids.
    # _vis_chunks: list of (grid_label, sub_batch) — e.g. "train_0", "test_2"
    _GRID_CHUNK = 5

    def _chunk_batch(name, batch):
        """Split a batch dict into chunks of _GRID_CHUNK samples."""
        if batch is None:
            return []
        B = batch['rgb'].shape[0]
        chunks = []
        for ci, start in enumerate(range(0, B, _GRID_CHUNK)):
            end = min(start + _GRID_CHUNK, B)
            sub = {k: v[start:end] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            label = f'{name}_{ci}' if B > _GRID_CHUNK else name
            chunks.append((label, sub))
        return chunks

    _vis_chunks = (
        _chunk_batch('train', _vis_train_batch) +
        _chunk_batch('val',   _vis_val_batch) +
        _chunk_batch('test',  _vis_test_batch)
    )
    _chunk_desc = ", ".join(f"{lbl}({b['rgb'].shape[0]})" for lbl, b in _vis_chunks)
    print(f'  {bold_cyan("[Milestone Vis]")} {len(_vis_chunks)} grids: {_chunk_desc}')

    # Accumulate generated images per chunk per milestone:
    # {grid_label: {milestone_label: tensor [B,3,H,W]}}
    _milestone_images: dict[str, dict] = {lbl: {} for lbl, _ in _vis_chunks}
    _milestone_set = set(_milestone_steps)

    def _run_milestone_vis(global_step):
        """Generate and save per-step vis for all chunks at this milestone.
        Also immediately saves an updated big grid so each checkpoint
        produces a self-contained comparison PNG.
        """
        label = f'step {global_step}'
        for grid_label, vis_batch in _vis_chunks:
            try:
                gen_rgb = generate_overfit_samples(
                    hdc2a, transformer, vae, bn_mean, bn_std,
                    vis_batch, config, num_steps=20,
                )
                _milestone_images[grid_label][label] = gen_rgb.cpu()
                # Save per-step individual grid to step_vis/
                save_step_vis_single(
                    step=global_step, batch=vis_batch, generated_rgb=gen_rgb,
                    output_dir=_step_vis_dir,
                    num_classes=_n_classes,
                    tag=f'{grid_label}_step_{global_step:06d}',
                )
                hdc2a.train()
                print(f'  {bold_cyan("[MilestoneVis]")} {grid_label} step {global_step} ✓')

                # Save an updated big grid right now (columns = milestones seen so far)
                ordered = {lbl: _milestone_images[grid_label][lbl]
                           for lbl in _milestone_labels
                           if lbl in _milestone_images[grid_label]}
                _grid_dir = os.path.join(config['output_dir'], 'milestone_vis')
                _gp = save_milestone_big_grid(
                    split_name=grid_label,
                    milestone_images=ordered,
                    batch=vis_batch,
                    output_dir=_grid_dir,
                    num_classes=_n_classes,
                    thumb_size=256,
                )
                print(f'  {bold_green("[MilestoneGrid]")} {grid_label} → {_gp}')
                # Log to WandB
                try:
                    wandb.log({
                        f'vis/{grid_label}': wandb.Image(_gp, caption=f'{grid_label} step {global_step}'),
                    })
                except Exception:
                    pass
            except Exception as _e:
                print(f'  {bold_yellow("[MilestoneVis WARN]")} {grid_label} step {global_step}: {_e}')

        # ── Auto-upload to HuggingFace (overwrites previous upload) ────────
        _hf_repo = args.hf_repo
        if _hf_repo:
            try:
                from ControlNet_training.HDC2A_training.scripts.version_file.upload import upload_to_hf
                _hf_path = f'output/{args.name}'
                # Exclude checkpoint dirs and .pt files (large); upload vis + logs only
                _ignore = ['checkpoint_*/**', '*.pt', 'wandb/**']
                print(f'  {bold_cyan("[HF Upload]")} syncing to {_hf_repo}/{_hf_path} ...')
                _hf_url = upload_to_hf(
                    local_dir=config['output_dir'],
                    repo=_hf_repo,
                    path_in_repo=_hf_path,
                    ignore_patterns=_ignore,
                )
                print(f'  {bold_green("[HF Upload]")} ✓ {_hf_url}')
            except Exception as _hf_err:
                print(f'  {bold_yellow("[HF Upload WARN]")} upload failed: {_hf_err}')

    # Callback for the train loop: fires only at milestone steps
    def _milestone_step_callback(_global_step, _batch):
        if _global_step in _milestone_set:
            _run_milestone_vis(_global_step)

    _active_step_vis_callback = _milestone_step_callback

    # === END OVERFIT TEST ===

    for epoch in range(start_epoch, total_epochs):
        # Distributed: set epoch on sampler for proper shuffling
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if val_loader is not None and hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)

        epoch_start = time.time()
        epoch_progress = (epoch - start_epoch) / max(total_epochs - start_epoch, 1) * 100

        # Train
        if is_main:
            print(f'\n--- {bold_magenta(f"Epoch {epoch}/{total_epochs-1}")}  ({bold_cyan(f"{epoch_progress:.0f}%")} done) ---')
        train_loss, _epoch_opt_steps = train_one_epoch(
            epoch, hdc2a, transformer, vae, bn_mean, bn_std,
            train_loader, optimizer, None, config, scheduler=scheduler,
            step_vis_callback=_active_step_vis_callback if is_main else None,
            step_vis_interval=1,   # callback checks internally; pass every step
        )
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start_time
        epochs_done = epoch - start_epoch + 1
        eta = elapsed / epochs_done * (total_epochs - start_epoch - epochs_done)
        _cumulative_opt_steps += _epoch_opt_steps
        if is_main:
            print(f'  Train loss: {yellow(f"{train_loss:.6f}")} ({epoch_time:.1f}s)  '
                  f'ETA: {cyan(f"{eta/60:.0f}min")}')
        _train_losses.append((epoch, train_loss))

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "adapter_lr": optimizer.param_groups[0]['lr'],
            "gpu_mem_gib": torch.cuda.memory_allocated() / (1024 ** 3),
            "cpu_pct": psutil.cpu_percent(),
        })

        # Validate
        val_loss = None
        if val_loader is not None and (epoch + 1) % config['val_every_n_epochs'] == 0:
            val_loss, _val_t_bins = validate(
                epoch, hdc2a, transformer, vae, bn_mean, bn_std,
                val_loader, config,
            )
            # Print overall + per-bin losses
            if is_main:
                _bin_str = '  '.join(f'{k}={v:.4f}' for k, v in _val_t_bins.items())
                print(f'  Val loss: {val_loss:.6f}  [{_bin_str}]')
            _wandb_val = {"val_loss": val_loss}
            _wandb_val.update({f'val/{k}': v for k, v in _val_t_bins.items()})
            wandb.log(_wandb_val)

            # save_checkpoint handles best tracking + rotation internally (rank 0 only)
            if is_main:
                _, _, best_val_loss, best_ckpt_path = save_checkpoint(
                    epoch, hdc2a_raw, transformer, optimizer,
                    val_loss, config['output_dir'], config,
                    keep_last_n=keep_last_n,
                    best_loss=best_val_loss,
                    best_ckpt_path=best_ckpt_path,
                    scheduler=scheduler,
                    global_step=_cumulative_opt_steps,
                )
            if world_size > 1:
                dist.barrier()

        # Periodic checkpoint (even when no val this epoch)
        elif (epoch + 1) % config['save_every_n_epochs'] == 0:
            # === OVERFIT TEST === 过拟合模式用轻量 LoRA checkpoint（只保存 A/B + adapter）
            if _overfit_mode and lora_modules:
                if is_main:
                    save_lora_checkpoint(
                        epoch, hdc2a_raw, lora_modules, config['output_dir'], loss=train_loss,
                    )
            else:
                # === END OVERFIT TEST ===
                if is_main:
                    _, _, best_val_loss, best_ckpt_path = save_checkpoint(
                        epoch, hdc2a_raw, transformer, optimizer,
                        train_loss, config['output_dir'], config,
                        keep_last_n=keep_last_n,
                        best_loss=best_val_loss,
                        best_ckpt_path=best_ckpt_path,
                        scheduler=scheduler,
                        global_step=_cumulative_opt_steps,
                    )
            if world_size > 1:
                dist.barrier()

        check_memory(f'epoch {epoch} end')

    # ── Post-training: loss curve ────────────────────────────────────────────
    # (milestone big grids are already saved at each checkpoint above)
    # Loss curve PNG (always saved, rank 0 only)
    if _train_losses and is_main:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            epochs_x   = [e for e, _ in _train_losses]
            losses_y   = [l for _, l in _train_losses]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(epochs_x, losses_y, linewidth=1.0, color='steelblue', label='train loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Loss — {args.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            curve_path = os.path.join(config['output_dir'], 'loss_curve.png')
            fig.savefig(curve_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'  {bold_green("[LossCurve]")} saved → {curve_path}')
            try:
                wandb.log({'loss_curve': wandb.Image(curve_path)})
            except Exception:
                pass
        except Exception as _e:
            print(f'  {bold_yellow("[LossCurve WARN]")} could not save loss curve: {_e}')

    # ── Final HF upload (after loss curve is saved) ───────────────────────
    if is_main and args.hf_repo:
        try:
            from ControlNet_training.HDC2A_training.scripts.version_file.upload import upload_to_hf
            _hf_path = f'output/{args.name}'
            _ignore = ['checkpoint_*/**', '*.pt', 'wandb/**']
            print(f'  {bold_cyan("[HF Upload]")} final sync to {args.hf_repo}/{_hf_path} ...')
            _hf_url = upload_to_hf(
                local_dir=config['output_dir'],
                repo=args.hf_repo,
                path_in_repo=_hf_path,
                ignore_patterns=_ignore,
            )
            print(f'  {bold_green("[HF Upload]")} ✓ {_hf_url}')
        except Exception as _hf_err:
            print(f'  {bold_yellow("[HF Upload WARN]")} final upload failed: {_hf_err}')

    # ── Cleanup ─────────────────────────────────────────────────────────────
    if is_main:
        print(f'\n{"="*70}')
        print('Training complete.')
        print(f'{"="*70}')

    wandb.summary.update({
        "final_train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "total_epochs": config['num_epochs'],
    })
    wandb.finish()
    cleanup_distributed()


if __name__ == '__main__':
    main()

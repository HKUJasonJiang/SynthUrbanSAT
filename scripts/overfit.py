"""# === OVERFIT TEST ===
过拟合 Sanity Check 辅助模块 — HDC²A + Flux2 ControlNet

提供以下功能:
  LoRALinear                 — 轻量 LoRA 包装，无需 peft 库
  apply_lora_to_control_blocks(transformer, rank, alpha)
                             — 给 control_transformer_blocks 的 attention 层加 LoRA，返回 lora_modules dict
  print_param_stats(hdc2a, transformer, lora_modules)
                             — 打印 Adapter / ControlNet / Flux2 参数统计
  gradient_check(...)        — 运行 1 step forward+backward，检查所有可训练参数是否有梯度
  generate_overfit_samples(...) — 多步 Euler 采样，返回 [B, 3, H, W] 生成图像 (float [0,1])
  save_overfit_grid(epoch, batch, generated_rgb, output_dir)
                             — 保存 [Depth | Seg | Generated | GT] 四列对比图
  save_lora_checkpoint(epoch, hdc2a, lora_modules, output_dir)
                             — 只保存 HDC²A + LoRA A/B 权重（节省磁盘）
  load_lora_checkpoint(ckpt_dir, hdc2a, lora_modules, device)
                             — 从轻量 checkpoint 恢复（load_state_dict strict=False）

本模块的所有代码均可在正式训练时完全移除，不影响主干逻辑。
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# 颜色工具（若某环境无该模块则 fallback 到无颜色版本）
try:
    from scripts.colors import (
        bold_cyan, bold_green, bold_red, bold_yellow,
        green, yellow, cyan, gray,
    )
except ImportError:
    def bold_cyan(s): return s
    def bold_green(s): return s
    def bold_red(s): return s
    def bold_yellow(s): return s
    def green(s): return s
    def yellow(s): return s
    def cyan(s): return s
    def gray(s): return s


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA 实现（手写，无 peft 依赖）
# ═══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """用 LoRA 包装一个已有的 nn.Linear（原始权重保持冻结）。

    forward(x) = frozen_linear(x) + (x @ lora_A.T @ lora_B.T) * (alpha / rank)

    lora_A [rank, in_features]  : Kaiming 初始化
    lora_B [out_features, rank] : 零初始化 → 初始时 LoRA 增量为 0，保留预训练模型输出

    之所以零初始化 lora_B：确保训练开始时模型行为与原模型完全一致，
    避免引入随机噪声破坏预训练特征。
    """

    def __init__(self, linear: nn.Linear, rank: int = 32, alpha: float = 32.0):
        super().__init__()

        if not isinstance(linear, nn.Linear):
            raise TypeError(f'LoRALinear expects nn.Linear, got {type(linear)}')

        # 保存原始 frozen linear
        self.linear = linear

        self.rank = rank
        # LoRA 缩放系数：按 alpha/rank 缩放（LoRA 论文标准做法）
        self.scale = alpha / rank

        d_in = linear.in_features
        d_out = linear.out_features
        # 使用与原 linear 相同的 dtype 和 device
        # 注意：apply_lora 在 transformer.to(device) 之后调用，所以 weight 已在 CUDA 上
        dtype = linear.weight.dtype
        device = linear.weight.device

        # === OVERFIT TEST === LoRA A/B 参数（唯一可训练参数）
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype, device=device))

        # Kaiming uniform 初始化 A（与 nn.Linear 默认 init 一致）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 零初始化（不改变初始输出）

        # 冻结原始权重（LoRA 训练不改动原始参数）
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 冻结分支（梯度不更新 linear.weight，但仍可传播到 x）
        base_out = self.linear(x)
        # LoRA 增量分支（梯度流向 lora_A 和 lora_B）
        # 注意：不能用 @ 链式写法 x @ lora_A.T @ lora_B.T 在半精度下有风险
        # 用 F.linear 保证正确性
        lora_delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_delta

    # 让外部代码可以通过 .weight / .bias 访问原始线性层的属性（兼容性）
    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return getattr(self.linear, 'bias', None)

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features


# ═══════════════════════════════════════════════════════════════════════════════
# 把 LoRA 注入到 control_transformer_blocks 的 attention 层
# ═══════════════════════════════════════════════════════════════════════════════

def apply_lora_to_control_blocks(
    transformer,
    rank: int = 32,
    alpha: float = 32.0,
) -> dict:
    """给 transformer.control_transformer_blocks 的 attention Q/K/V/O 层加 LoRA。

    Flux2TransformerBlock 的 attention 结构（attn = Flux2Attention）：
      attn.to_q, attn.to_k, attn.to_v        — 图像流 Q/K/V 投影
      attn.add_q_proj, attn.add_k_proj, attn.add_v_proj — 文本流 Q/K/V 投影
      attn.to_out[0]                          — 输出投影（to_out 是 ModuleList[Linear, Dropout]）

    为什么加 LoRA：控制块是训练的核心，通过低秩矩阵微调可以在冻结预训练权重的前提下
    用极少参数（~几MB）进行有效的过拟合测试，验证梯度通路和学习能力。

    Returns:
        lora_modules: dict {module_path_str: LoRALinear}，用于参数追踪和 checkpoint 保存
    """
    # 目标 attention 属性名（图像流 QKV + 文本流 QKV + 输出）
    TARGET_ATTRS = ['to_q', 'to_k', 'to_v', 'add_q_proj', 'add_k_proj', 'add_v_proj']
    lora_modules: dict[str, LoRALinear] = {}

    if not hasattr(transformer, 'control_transformer_blocks'):
        raise AttributeError('transformer has no control_transformer_blocks attribute. '
                             'Is it a Flux2ControlTransformer2DModel?')

    num_blocks = len(transformer.control_transformer_blocks)

    for block_idx, block in enumerate(transformer.control_transformer_blocks):
        is_last_block = (block_idx == num_blocks - 1)
        attn = getattr(block, 'attn', None)
        if attn is None:
            print(f'  [LoRA] block {block_idx}: no attn attribute, skipping')
            continue

        # 处理 to_q, to_k, to_v, add_*_proj
        for attr_name in TARGET_ATTRS:
            # 跳过最后一个 control block 的 add_*_proj：
            # forward_control 中最后一个 block 的 encoder_hidden_states 输出被丢弃，
            # add_*_proj 的输出无法传递到 loss，其 lora_B 梯度永远为零，无法训练。
            if is_last_block and attr_name.startswith('add_'):
                continue

            linear = getattr(attn, attr_name, None)
            if linear is None:
                continue  # 部分 block 可能没有 add_*_proj
            if not isinstance(linear, nn.Linear):
                continue  # 已被替换或类型不匹配，跳过

            lora = LoRALinear(linear, rank=rank, alpha=alpha)
            setattr(attn, attr_name, lora)
            key = f'control_transformer_blocks.{block_idx}.attn.{attr_name}'
            lora_modules[key] = lora
            print(f'  {green("LoRA")} {key} '
                  f'[{linear.in_features}→{linear.out_features}]')

        # 处理 to_out[0]（输出投影；to_out 是 ModuleList([Linear, Dropout])）
        to_out = getattr(attn, 'to_out', None)
        if to_out is not None and len(to_out) > 0:
            if isinstance(to_out[0], nn.Linear):
                linear = to_out[0]
                lora = LoRALinear(linear, rank=rank, alpha=alpha)
                to_out[0] = lora
                key = f'control_transformer_blocks.{block_idx}.attn.to_out.0'
                lora_modules[key] = lora
                print(f'  {green("LoRA")} {key} '
                      f'[{linear.in_features}→{linear.out_features}]')

    total_lora_params = sum(
        m.lora_A.numel() + m.lora_B.numel() for m in lora_modules.values()
    )
    print(f'\n  LoRA modules injected: {bold_cyan(str(len(lora_modules)))}')
    print(f'  LoRA trainable params: {bold_cyan(f"{total_lora_params / 1e6:.2f}M")}')
    return lora_modules


# ═══════════════════════════════════════════════════════════════════════════════
# 参数统计
# ═══════════════════════════════════════════════════════════════════════════════

def print_param_stats(hdc2a, transformer, lora_modules: dict = None):
    """打印三个模块的参数统计表：HDC²A Adapter / ControlNet / Flux2 backbone。

    预期结果：
      - HDC²A: total = trainable (所有参数可训练)
      - ControlNet: total >> LoRA trainable（冻结主干 + 少量 LoRA）
      - Flux2: trainable = 0（完全冻结）
    """
    print(f'\n{bold_cyan("Parameter Statistics:")}')

    # HDC²A Adapter
    adapter_total = sum(p.numel() for p in hdc2a.parameters())
    adapter_train = sum(p.numel() for p in hdc2a.parameters() if p.requires_grad)
    print(f'  {"HDC²A Adapter:":<22} total={adapter_total/1e6:.1f}M  '
          f'trainable={bold_green(f"{adapter_train/1e6:.1f}M")}')

    # ControlNet blocks (control_transformer_blocks + control_img_in 等)
    def _ctrl_params():
        for n, p in transformer.named_parameters():
            if 'control' in n:
                yield n, p
    ctrl_total = sum(p.numel() for _, p in _ctrl_params())
    if lora_modules is not None:
        # LoRA 模式：只有 A/B 是可训练的
        lora_trainable = sum(
            m.lora_A.numel() + m.lora_B.numel() for m in lora_modules.values()
        )
        print(f'  {"ControlNet (frozen):":<22} total={ctrl_total/1e6:.1f}M  '
              f'LoRA trainable={bold_green(f"{lora_trainable/1e6:.2f}M")}')
    else:
        ctrl_train = sum(p.numel() for _, p in _ctrl_params() if p.requires_grad)
        print(f'  {"ControlNet:":<22} total={ctrl_total/1e6:.1f}M  '
              f'trainable={bold_green(f"{ctrl_train/1e6:.1f}M")}')

    # Flux2 backbone（非 control 部分）
    def _backbone_params():
        for n, p in transformer.named_parameters():
            if 'control' not in n:
                yield p
    flux_total = sum(p.numel() for p in _backbone_params())
    flux_train = sum(p.numel() for p in _backbone_params() if p.requires_grad)
    flux_train_str = bold_red(f'{flux_train/1e6:.1f}M (应为0!)') if flux_train > 0 \
        else bold_green(f'{flux_train/1e6:.1f}M ✓')
    print(f'  {"Flux2 backbone:":<22} total={flux_total/1e6:.1f}M  '
          f'trainable={flux_train_str}')

    # 汇总：HDC²A + LoRA（或 ControlNet trainable）
    if lora_modules is not None:
        lora_trainable = sum(
            m.lora_A.numel() + m.lora_B.numel() for m in lora_modules.values()
        )
        total_trainable = adapter_train + lora_trainable
        print(f'  {"─"*50}')
        print(f'  {"Total trainable:":<22} HDC²A {adapter_train/1e6:.1f}M '
              f'+ LoRA {lora_trainable/1e6:.2f}M '
              f'= {bold_cyan(f"{total_trainable/1e6:.2f}M")}')
    else:
        ctrl_train = sum(p.numel() for _, p in _ctrl_params() if p.requires_grad)
        total_trainable = adapter_train + ctrl_train + flux_train
        print(f'  {"─"*50}')
        print(f'  {"Total trainable:":<22} {bold_cyan(f"{total_trainable/1e6:.2f}M")}')


# ═══════════════════════════════════════════════════════════════════════════════
# 梯度检查
# ═══════════════════════════════════════════════════════════════════════════════

def gradient_check(hdc2a, transformer, vae, bn_mean, bn_std,
                   lora_modules: dict, config) -> bool:
    """运行 1 step forward+backward（随机数据），检查所有可训练参数是否有梯度。

    检查范围：
      - HDC²A 的所有 requires_grad=True 参数
      - LoRA A/B 参数（通过 lora_modules dict 定位）

    如果某参数没有梯度，打印警告和排查建议。
    检查完毕后清空所有梯度（避免污染第一个正式 step）。

    Returns:
        True  — 所有可训练参数都有非零梯度
        False — 存在无梯度参数（需要排查）
    """
    from scripts.utility import encode_rgb_to_latent, prepare_latent_ids, prepare_text_ids
    from scripts.train import flow_matching_forward

    device = config['device']
    dtype = config['dtype']
    img_size = config['image_size']

    print(f'\n{bold_cyan("="*60)}\n  {bold_cyan("[Grad Check] Overfit mode gradient verification")}\n{bold_cyan("="*60)}')
    print('  使用随机数据跑 1 step forward+backward，验证梯度通路...')

    # 构造随机输入（不需要真实数据）
    rgb = torch.rand(1, 3, img_size, img_size, device=device, dtype=dtype)
    seg = torch.randint(0, config.get('num_classes', 5), (1, img_size, img_size), device=device)
    depth = torch.rand(1, 1, img_size, img_size, device=device, dtype=dtype)

    hdc2a.train()
    # transformer 保持 eval（backbone 冻结，LoRA 虽在 eval 内但梯度仍流动）
    transformer.eval()

    # 清空历史梯度（确保干净起点）
    hdc2a.zero_grad()
    for m in lora_modules.values():
        m.lora_A.grad = None
        m.lora_B.grad = None

    # Forward
    target_packed, patchified = encode_rgb_to_latent(vae, rgb, bn_mean, bn_std, dtype)
    latent_ids = prepare_latent_ids(patchified, device)

    t = torch.tensor([0.5], device=device)
    noisy_latent, noise, flow_target = flow_matching_forward(target_packed, t)

    seq_len = config.get('text_seq_len', 512)
    text_dim = config.get('text_dim', 15360)
    prompt_embeds = torch.randn(1, seq_len, text_dim, device=device, dtype=dtype)
    text_ids = prepare_text_ids(prompt_embeds, device)
    guidance = torch.tensor([config.get('guidance_scale', 3.5)], device=device, dtype=dtype)

    with torch.amp.autocast('cuda', dtype=dtype):
        control_context = hdc2a(seg, depth)
        output = transformer(
            hidden_states=noisy_latent,
            encoder_hidden_states=prompt_embeds,
            timestep=t.to(dtype),
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=guidance,
            control_context=control_context.to(transformer.dtype),
            return_dict=False,
        )
        loss = F.mse_loss(output[0], flow_target)

    print(f'  Forward done. loss={yellow(f"{loss.item():.6f}")}')

    # Backward
    loss.backward()
    print('  Backward done.')

    # ── 检查 HDC²A 参数 ──
    hdc2a_total = sum(1 for p in hdc2a.parameters() if p.requires_grad)
    hdc2a_with_grad = sum(
        1 for p in hdc2a.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0
    )
    no_grad_adapter = [
        n for n, p in hdc2a.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)
    ]

    # ── 检查 LoRA 参数 ──
    # 注意：lora_A 的梯度依赖 lora_B，lora_B 零初始化时 d_loss/d_lora_A = lora_B.T @ grad = 0
    # 这是预期行为（不是梯度通路断裂），从第2步起 lora_B 变为非零后 lora_A 才会有非零梯度
    # → "no gradient" 的真正含义是 grad is None（autograd 图断裂），而非 abs().max()==0
    lora_total = len(lora_modules) * 2  # A + B per module
    lora_with_any_grad = sum(
        1 for m in lora_modules.values()
        for p in [m.lora_A, m.lora_B]
        if p.grad is not None
    )
    lora_with_nonzero_grad = sum(
        1 for m in lora_modules.values()
        for p in [m.lora_A, m.lora_B]
        if p.grad is not None and p.grad.abs().max() > 0
    )
    # 真正的断裂：grad tensor 不存在（autograd 图被截断）
    truly_disconnected_lora = [
        f'{path}.{pname}'
        for path, m in lora_modules.items()
        for pname, p in [('lora_A', m.lora_A), ('lora_B', m.lora_B)]
        if p.grad is None
    ]
    # 零梯度（grad exists but ==0）：lora_A 在第 0 步因 lora_B=0 而正常为零
    zero_grad_lora_A = [
        f'{path}.lora_A' for path, m in lora_modules.items()
        if m.lora_A.grad is not None and m.lora_A.grad.abs().max() == 0
    ]
    zero_grad_lora_B = [
        f'{path}.lora_B' for path, m in lora_modules.items()
        if m.lora_B.grad is not None and m.lora_B.grad.abs().max() == 0
    ]

    # ── 打印结果 ──
    print(f'\n  HDC²A  params with grad: {green(f"{hdc2a_with_grad}/{hdc2a_total}")}')
    print(f'  LoRA   params with grad tensor: {green(f"{lora_with_any_grad}/{lora_total}")}  '
          f'(non-zero: {lora_with_nonzero_grad}/{lora_total})')
    if zero_grad_lora_A:
        print(f'  {cyan("[NOTE]")} {len(zero_grad_lora_A)} lora_A 梯度为零（正常！因为 lora_B 零初始化，'
              f'd_L/d_lora_A = lora_B.T @ grad = 0，第2步后会有非零梯度）')
    if zero_grad_lora_B:
        print(f'  {bold_yellow("[WARN]")} {len(zero_grad_lora_B)} lora_B 梯度为零（排查 control 块梯度通路）')

    all_ok = True

    if no_grad_adapter:
        all_ok = False
        print(f'\n  {bold_red("⚠ HDC²A 无梯度参数")} ({len(no_grad_adapter)} 个):')
        for n in no_grad_adapter[:5]:
            print(f'    {bold_red("✗")} {n}')
        if len(no_grad_adapter) > 5:
            print(f'    ... 还有 {len(no_grad_adapter) - 5} 个')
        print(f'  {bold_red("排查方向:")} 检查 HDC²A 构建时参数是否正确 requires_grad=True.')

    if truly_disconnected_lora:
        all_ok = False
        print(f'\n  {bold_red("⚠ LoRA 梯度完全断裂")} (grad is None, {len(truly_disconnected_lora)} 个):')
        for n in truly_disconnected_lora[:5]:
            print(f'    {bold_red("✗")} {n}')
        if len(truly_disconnected_lora) > 5:
            print(f'    ... 还有 {len(truly_disconnected_lora) - 5} 个')
        print(f'  {bold_red("排查方向:")}')
        print('    1. 确认 apply_lora_to_control_blocks 已在 optimizer 创建前调用')
        print('    2. 确认 LoRALinear.linear.weight.requires_grad = False')
        print('    3. 确认 LoRALinear.lora_A / lora_B.requires_grad = True')
        print('    4. 检查 transformer.control_transformer_blocks 的梯度通路是否被截断')

    if all_ok:
        # lora_A 零梯度是预期的（lora_B=0 导致），不算失败
        # 只要 lora_B 有非零梯度、grad tensor 存在即可确认通路正常
        lora_B_ok = all(
            m.lora_B.grad is not None and m.lora_B.grad.abs().max() > 0
            for m in lora_modules.values()
        )
        if lora_B_ok:
            print(f'\n  {bold_green("✓ 梯度通路验证通过！")}')
            print(f'    lora_B 全部有非零梯度（✓），lora_A 因 lora_B=0 初始化梯度为零（正常，第2步自动激活）')
        else:
            all_ok = False
            print(f'\n  {bold_red("⚠ lora_B 存在零梯度，梯度通路可能有问题！")}')
            for path, m in lora_modules.items():
                if m.lora_B.grad is None or m.lora_B.grad.abs().max() == 0:
                    print(f'    {bold_red("✗")} {path}.lora_B  grad={m.lora_B.grad}')

    # 清空梯度（避免污染第一个正式训练 step）
    hdc2a.zero_grad()
    for m in lora_modules.values():
        m.lora_A.grad = None
        m.lora_B.grad = None

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Latent 解码辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _unpatchify_latents(patched: torch.Tensor) -> torch.Tensor:
    """patchify 的逆操作: [B, C*4, H/2, W/2] → [B, C, H, W]

    patchify 操作：
      [B, C, H, W] → view(B, C, H//2, 2, W//2, 2) → permute(0,1,3,5,2,4)
                   → reshape(B, C*4, H//2, W//2)
    逆操作（此函数）：
      [B, C*4, H2, W2] → reshape(B, C, 2, 2, H2, W2) → permute(0,1,4,2,5,3)
                       → reshape(B, C, H2*2, W2*2)
    """
    B, C4, H2, W2 = patched.shape
    C = C4 // 4
    x = patched.reshape(B, C, 2, 2, H2, W2)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.reshape(B, C, H2 * 2, W2 * 2)
    return x


def _decode_packed_latent(
    packed: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_std: torch.Tensor,
    vae,
) -> torch.Tensor:
    """将 packed latent [B, N, 128] 解码为 RGB 图像 [B, 3, H, W] 在 [0, 1]。

    解码链：
      packed [B, N, 128]
        → unpack  → [B, 128, H2, W2]  （N = H2 * W2）
        → unnorm  → [B, 128, H2, W2]  （× bn_std + bn_mean）
        → unpatch → [B, 32, H, W]     （反 patchify，H = H2*2）
        → VAE decode → [B, 3, H*8, W*8]  （VAE 8× 放大）
        → clip & rescale → [0, 1]
    """
    B, N, C = packed.shape
    H2 = W2 = int(math.sqrt(N))
    assert H2 * W2 == N, f'N={N} is not a perfect square'

    # 1. unpack: [B, N, 128] → [B, 128, H2, W2]
    x = packed.permute(0, 2, 1).reshape(B, C, H2, W2)

    # 2. unnormalize: reverse of (patchified - bn_mean) / bn_std
    #    bn_mean/bn_std shape: [1, 128, 1, 1]
    x = x * bn_std + bn_mean

    # 3. unpatchify: [B, 128, H2, W2] → [B, 32, H2*2, W2*2]
    x = _unpatchify_latents(x)

    # 4. VAE decode: [B, 32, H_lat, W_lat] → [B, 3, H_img, W_img]
    # VAE weights are bfloat16; cast x to match (intermediate ops can stay float32)
    with torch.no_grad():
        vae_dtype = next(iter(vae.parameters())).dtype
        decoded = vae.decode(x.to(vae_dtype)).sample  # output [-1, 1]

    # 5. 映射到 [0, 1] 并 clamp
    rgb = (decoded.clamp(-1, 1) + 1.0) / 2.0
    return rgb.to(packed.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# 图像生成（Euler 采样）
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_overfit_samples(
    hdc2a,
    transformer,
    vae,
    bn_mean: torch.Tensor,
    bn_std: torch.Tensor,
    batch: dict,
    config: dict,
    num_steps: int = 20,
) -> torch.Tensor:
    """从 seg + depth 条件生成 RGB 图像（Euler flow matching 采样）。

    理论：flow matching 的 Euler 积分：
      x_{t-dt} = x_t + dt * v_theta(x_t, t)
    从 t=1（纯噪声）积分到 t=0（生成图像）。

    为什么能用于过拟合可视化：
      如果模型已经过拟合，v_theta 精确近似真实速度场，
      Euler 采样应能还原出与 GT 几乎一模一样的图像。

    Args:
        num_steps: Euler 步数（默认 20，step 数越多质量越好）

    Returns:
        generated_rgb: [B, 3, H, W] float tensor in [0, 1]
    """
    from scripts.utility import prepare_latent_ids, prepare_text_ids

    device = config['device']
    dtype = config['dtype']
    img_size = config['image_size']

    seg = batch['seg'].to(device)
    depth = batch['depth'].to(device, dtype=dtype)
    B = seg.shape[0]

    # 文本 embedding（使用 batch 中预计算的，或随机 placeholder）
    seq_len = config.get('text_seq_len', 512)
    text_dim = config.get('text_dim', 15360)
    if 'prompt_embeds' in batch:
        prompt_embeds = batch['prompt_embeds'].to(device, dtype=dtype)
    else:
        prompt_embeds = torch.randn(B, seq_len, text_dim, device=device, dtype=dtype)
    text_ids = prepare_text_ids(prompt_embeds, device)

    # 从 img_size 推断 patchified latent 的 spatial 尺寸
    # VAE 压缩倍数 = 8（由 tile_latent_min_size = sample_size / 2^(n_blocks-1) = 1024/8 推得）
    # patchify 再折叠一半：H2 = img_size // 8 // 2 = img_size // 16
    H2 = W2 = img_size // 16
    N = H2 * W2  # 1024x1024 → N=4096
    C = 128      # patchified channels: 32 latent channels * 4 (2×2 patch)

    # position IDs（用 dummy patchified 获取正确 shape）
    dummy_patchified = torch.zeros(B, C, H2, W2, device=device)
    latent_ids = prepare_latent_ids(dummy_patchified, device)

    guidance = torch.full((B,), config.get('guidance_scale', 3.5), device=device, dtype=dtype)

    # 从 t=1 的纯噪声出发
    x = torch.randn(B, N, C, device=device, dtype=dtype)

    # Euler steps: t: 1.0 → 0.0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    hdc2a.eval()
    transformer.eval()

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr  # 负数（t 递减）

        t_batch = t_curr.expand(B).to(dtype)

        # control context（每步重新计算，seg/depth 固定）
        control_context = hdc2a(seg, depth)

        with torch.amp.autocast('cuda', dtype=dtype):
            output = transformer(
                hidden_states=x,
                encoder_hidden_states=prompt_embeds,
                timestep=t_batch,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=guidance,
                control_context=control_context.to(transformer.dtype),
                return_dict=False,
            )
        v_pred = output[0].to(dtype)  # 预测速度场 [B, N, C]

        # Euler 步进（dt 为负，x 从噪声向生成图像移动）
        x = x + dt * v_pred

    # 解码最终 latent 为 RGB 图像
    generated_rgb = _decode_packed_latent(x.float(), bn_mean.float(), bn_std.float(), vae)
    return generated_rgb  # [B, 3, H_img, W_img] in [0, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# 可视化 Grid 保存
# ═══════════════════════════════════════════════════════════════════════════════

# 分割图着色 palette（6 个类，含背景）
_SEG_PALETTE = [
    [200, 200, 200],  # class 0: background (gray)
    [255,  85,  85],  # class 1: building (red)
    [ 85, 170, 255],  # class 2: water (blue)
    [ 85, 255,  85],  # class 3: vegetation (green)
    [255, 170,  85],  # class 4: road (orange)
    [170,  85, 255],  # class 5: other (purple)
]

_LABEL_BG = (20, 20, 20)
_LABEL_FG = (200, 200, 200)
_HEADER_H  = 28   # pixels tall for column/row header bars


def _seg_to_rgb(seg_tensor: torch.Tensor, num_classes: int = 6) -> np.ndarray:
    """将 [H, W] long 分割图转为 [H, W, 3] uint8 RGB。"""
    seg_np = seg_tensor.cpu().numpy().astype(np.int64)
    H, W = seg_np.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in enumerate(_SEG_PALETTE[:num_classes]):
        mask = (seg_np == cls_id)
        rgb[mask] = color
    return rgb


def _draw_label_bar(width: int, height: int, text: str) -> Image.Image:
    """Return a PIL image of size (width, height) with centered text."""
    bar = Image.new('RGB', (width, height), color=_LABEL_BG)
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(bar)
        # estimate x so it's roughly centered
        x = max(4, width // 2 - len(text) * 3)
        draw.text((x, 4), text, fill=_LABEL_FG)
    except Exception:
        pass
    return bar


def _tensor_to_pil(t: torch.Tensor, thumb: int, resample=Image.LANCZOS) -> Image.Image:
    """Convert a [3,H,W] or [1,H,W] float [0,1] tensor to a resized PIL RGB image."""
    arr = t.cpu().float()
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    np_arr = (arr.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_arr).resize((thumb, thumb), resample)


def save_step_vis_single(
    step: int,
    batch: dict,
    generated_rgb: torch.Tensor,
    output_dir: str,
    thumb_size: int = 256,
    num_classes: int = 6,
    tag: str = None,
) -> str:
    """保存单个 step 的对比图（4 列：Seg | Depth | Generated | GT）到 step_vis/ 目录。
    同时单独保存每个 sample 的生成图。

    Returns: path to the saved grid image.
    """
    os.makedirs(output_dir, exist_ok=True)

    gt_rgb  = batch['rgb']    # [B, 3, H, W]
    seg     = batch['seg']    # [B, H, W] long
    depth   = batch['depth']  # [B, 1, H, W]
    B = gt_rgb.shape[0]
    T = thumb_size

    col_labels = ['Seg', 'Depth', f'Step {step}', 'GT']
    n_cols = len(col_labels)

    # Header row
    header = Image.new('RGB', (T * n_cols, _HEADER_H), color=_LABEL_BG)
    header.paste(_draw_label_bar(T * n_cols, _HEADER_H,
                                 '  |  '.join(col_labels)), (0, 0))

    rows = []
    for i in range(B):
        seg_img   = Image.fromarray(_seg_to_rgb(seg[i], num_classes)).resize((T, T), Image.NEAREST)
        depth_img = _tensor_to_pil(depth[i], T)
        gen_img   = _tensor_to_pil(generated_rgb[i], T)
        gt_img    = _tensor_to_pil(gt_rgb[i], T)

        row = Image.new('RGB', (T * n_cols, T))
        for col_i, img in enumerate([seg_img, depth_img, gen_img, gt_img]):
            row.paste(img, (col_i * T, 0))
        rows.append(row)

    total_h = _HEADER_H + T * B
    grid = Image.new('RGB', (T * n_cols, total_h))
    grid.paste(header, (0, 0))
    for i, row in enumerate(rows):
        grid.paste(row, (0, _HEADER_H + i * T))

    prefix = tag if tag is not None else f'step_{step:06d}'
    save_path = os.path.join(output_dir, f'{prefix}.png')
    grid.save(save_path)

    # Individual generated images
    for i in range(B):
        np_arr = (generated_rgb[i].cpu().float().permute(1, 2, 0).numpy() * 255
                  ).clip(0, 255).astype(np.uint8)
        Image.fromarray(np_arr).save(
            os.path.join(output_dir, f'{prefix}_sample{i:02d}.png'))

    return save_path


# keep old name as alias so existing call-sites still work
def save_overfit_grid(
    epoch: int,
    batch: dict,
    generated_rgb: torch.Tensor,
    output_dir: str,
    thumb_size: int = 256,
    num_classes: int = 6,
    tag: str = None,
) -> str:
    """Thin wrapper — delegates to save_step_vis_single (backward-compat)."""
    return save_step_vis_single(
        step=epoch, batch=batch, generated_rgb=generated_rgb,
        output_dir=output_dir, thumb_size=thumb_size,
        num_classes=num_classes, tag=tag,
    )


def build_milestone_big_grid(
    milestone_images: dict,
    batch: dict,
    num_classes: int = 6,
    thumb_size: int = 256,
) -> Image.Image:
    """Build the big 5×8 milestone grid for one split (train / val / test).

    Layout (cols): Seg | Depth | step_0 | step_1 | ... | step_N | GT
    Layout (rows): one row per sample in batch (up to 5 samples shown)

    Args:
        milestone_images: OrderedDict  {step_label_str: tensor [B,3,H,W] [0,1]}
                          e.g. {'step 0': tensor, 'step 200': tensor, ...}
        batch:            dict with keys 'rgb' [B,3,H,W], 'seg' [B,H,W], 'depth' [B,1,H,W]
        num_classes:      number of segmentation classes
        thumb_size:       side length of each thumbnail cell (px)

    Returns:
        PIL Image of the full grid
    """
    from PIL import ImageDraw

    T = thumb_size
    gt_rgb = batch['rgb']
    seg    = batch['seg']
    depth  = batch['depth']
    B = min(gt_rgb.shape[0], 5)  # at most 5 rows

    step_labels = list(milestone_images.keys())   # e.g. ['step 0', 'step 200', ...]
    col_labels = ['Seg', 'Depth'] + step_labels + ['GT']
    n_cols = len(col_labels)

    row_label_w = 80  # pixels for left row-label column
    total_w = row_label_w + T * n_cols
    total_h = _HEADER_H + T * B

    canvas = Image.new('RGB', (total_w, total_h), color=(10, 10, 10))

    # ── Column header ────────────────────────────────────────────────────────
    header = Image.new('RGB', (total_w, _HEADER_H), color=_LABEL_BG)
    draw = ImageDraw.Draw(header)
    # Row-label column header (empty)
    for col_i, label in enumerate(col_labels):
        x = row_label_w + col_i * T + max(2, T // 2 - len(label) * 3)
        draw.text((x, 6), label, fill=_LABEL_FG)
    canvas.paste(header, (0, 0))

    # ── Rows ─────────────────────────────────────────────────────────────────
    for row_i in range(B):
        y0 = _HEADER_H + row_i * T

        # Row label
        row_label_bar = Image.new('RGB', (row_label_w, T), color=(30, 30, 30))
        dl = ImageDraw.Draw(row_label_bar)
        dl.text((4, T // 2 - 6), f'#{row_i}', fill=_LABEL_FG)
        canvas.paste(row_label_bar, (0, y0))

        cells = []
        cells.append(Image.fromarray(_seg_to_rgb(seg[row_i], num_classes))
                     .resize((T, T), Image.NEAREST))               # Seg
        cells.append(_tensor_to_pil(depth[row_i], T))              # Depth
        for label in step_labels:
            gen = milestone_images[label]
            cells.append(_tensor_to_pil(gen[row_i], T))            # step_X
        cells.append(_tensor_to_pil(gt_rgb[row_i], T))             # GT

        for col_i, cell in enumerate(cells):
            canvas.paste(cell, (row_label_w + col_i * T, y0))

    return canvas


def save_milestone_big_grid(
    split_name: str,
    milestone_images: dict,
    batch: dict,
    output_dir: str,
    num_classes: int = 6,
    thumb_size: int = 256,
) -> str:
    """Save the big milestone grid for one split and return its path.

    File name: milestone_grid_{split_name}.png   (saved in output_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    grid = build_milestone_big_grid(
        milestone_images, batch, num_classes=num_classes, thumb_size=thumb_size)
    save_path = os.path.join(output_dir, f'milestone_grid_{split_name}.png')
    grid.save(save_path)
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA Checkpoint 保存 / 加载
# ═══════════════════════════════════════════════════════════════════════════════

def save_lora_checkpoint(
    epoch: int,
    hdc2a,
    lora_modules: dict,
    output_dir: str,
    loss: float = float('nan'),
) -> str:
    """保存轻量 checkpoint：HDC²A 权重 + LoRA A/B 权重。

    只保存可训练参数（不保存冻结的 4B ControlNet 主干权重），节省磁盘空间。

    Returns:
        ckpt_dir: 保存目录路径
    """
    ckpt_dir = os.path.join(output_dir, f'overfit_ckpt_epoch_{epoch:04d}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # HDC²A 完整权重
    torch.save(hdc2a.state_dict(), os.path.join(ckpt_dir, 'hdc2a.pt'))

    # LoRA A/B 权重（仅可训练参数，极小）
    lora_sd = {}
    for key, lora_mod in lora_modules.items():
        lora_sd[key + '.lora_A'] = lora_mod.lora_A.data.cpu()
        lora_sd[key + '.lora_B'] = lora_mod.lora_B.data.cpu()
    torch.save(lora_sd, os.path.join(ckpt_dir, 'lora_params.pt'))

    # 元数据
    meta = {
        'epoch': epoch,
        'loss': loss,
        'lora_keys': list(lora_modules.keys()),
    }
    torch.save(meta, os.path.join(ckpt_dir, 'meta.pt'))

    lora_mb = sum(v.numel() * 2 for v in lora_sd.values()) / 1e6
    print(f'  {bold_green("Overfit ckpt saved:")} {ckpt_dir}  '
          f'(LoRA: {lora_mb:.1f} MB)')
    return ckpt_dir


def load_lora_checkpoint(
    ckpt_dir: str,
    hdc2a,
    lora_modules: dict,
    device: str = 'cuda',
):
    """从 save_lora_checkpoint 保存的 ckpt 恢复 HDC²A + LoRA。

    Returns:
        epoch: 恢复的 epoch 编号
    """
    # HDC²A
    hdc2_path = os.path.join(ckpt_dir, 'hdc2a.pt')
    if os.path.exists(hdc2_path):
        hdc2a.load_state_dict(torch.load(hdc2_path, map_location=device))
        print(f'  HDC²A loaded from {hdc2_path}')
    else:
        print(f'  hdc2a.pt not found in {ckpt_dir}')

    # LoRA
    lora_path = os.path.join(ckpt_dir, 'lora_params.pt')
    if os.path.exists(lora_path):
        lora_sd = torch.load(lora_path, map_location=device)
        for key, lora_mod in lora_modules.items():
            key_A = key + '.lora_A'
            key_B = key + '.lora_B'
            if key_A in lora_sd:
                lora_mod.lora_A.data.copy_(lora_sd[key_A])
            if key_B in lora_sd:
                lora_mod.lora_B.data.copy_(lora_sd[key_B])
        print(f'  LoRA params ({len(lora_sd)//2} modules) loaded from {lora_path}')
    else:
        print(f'  lora_params.pt not found in {ckpt_dir}')

    # 元数据
    meta_path = os.path.join(ckpt_dir, 'meta.pt')
    epoch = 0
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location='cpu')
        epoch = meta.get('epoch', 0)
        print(f'  Resumed from epoch {epoch}, loss={meta.get("loss", float("nan")):.6f}')
    return epoch

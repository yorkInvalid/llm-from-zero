"""现代语言模型组件：RMSNorm、SwiGLU 和按优化步骤计算的学习率。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 半精度输入使用 float32 统计，避免平方溢出；不减均值、不加 bias。
        stable = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = stable * torch.rsqrt(stable.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 三个投影，隐层取约 8/3*d，参数量接近 4*d 隐层的两投影 FFN。
        self.hidden_dim = math.ceil((8 * d_model / 3) / 8) * 8
        self.gate = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.up = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.down = nn.Linear(self.hidden_dim, d_model, bias=False)

    def forward(self, x):
        tokens = x.transpose(-2, -1)
        return self.down(F.silu(self.gate(tokens)) * self.up(tokens)).transpose(-2, -1)


def learning_rate_at_step(step, total_steps, warmup_steps, peak_lr, min_ratio=.1):
    """step 从 1 开始：线性 warmup 至 peak，再 cosine 衰减至 peak*min_ratio。"""
    if not 1 <= step <= total_steps or not 0 <= warmup_steps < total_steps:
        raise ValueError('学习率步骤范围无效')
    if peak_lr <= 0 or not 0 <= min_ratio <= 1:
        raise ValueError('学习率须为正，最小比例须在 [0,1]')
    if warmup_steps and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    first = max(1, warmup_steps)
    progress = (step - first) / max(1, total_steps - first)
    return peak_lr * (min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * progress)))

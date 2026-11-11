"""多头注意力：输入、输出都按列排列 token；导入本文件不会启动训练。"""

import torch
from torch import nn

from position_encoding import apply_rope


class MultiHeadAttention(nn.Module):
    """保留逐头计算，便于与讲义中的 Q、K、V、A、A′、B 对照。"""

    def __init__(self, d_in=10, num_heads=2, d_k=5, d_v=5, use_rope=False):
        super().__init__()
        if use_rope and d_k < 2:
            raise ValueError("RoPE 每头至少需要 2 维")
        self.num_heads = num_heads
        self.d_k = d_k
        self.use_rope = use_rope

        # ParameterList 会把每个头的参数注册到模块中。
        # 保留原来的初始化顺序和尺度，便于对照整理前后的训练结果。
        self.W_q = nn.ParameterList([
            nn.Parameter(torch.randn(d_k, d_in) * 0.1) for _ in range(num_heads)
        ])
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn(d_k, d_in) * 0.1) for _ in range(num_heads)
        ])
        self.W_v = nn.ParameterList([
            nn.Parameter(torch.randn(d_v, d_in) * 0.1) for _ in range(num_heads)
        ])
        self.W_o = nn.Parameter(torch.randn(d_in, num_heads * d_v) * 0.1)

    def forward(self, X):
        """X: (d_in, n)；返回 B: (d_in, n)、权重: (num_heads, n, n)。"""
        positions = torch.arange(X.shape[1], dtype=X.dtype, device=X.device)
        head_outputs = []
        head_weights = []

        for h in range(self.num_heads):
            Q = self.W_q[h] @ X
            K = self.W_k[h] @ X
            V = self.W_v[h] @ X
            # RoPE 作用于各头的 Q、K；V 不旋转。
            if self.use_rope:
                Q = apply_rope(Q, positions)
                K = apply_rope(K, positions)
            A = K.T @ Q  # 行是 key，列是 query
            A = A / (self.d_k ** 0.5)
            A_prime = torch.softmax(A, dim=0)  # 每列权重之和为 1
            B_head = V @ A_prime
            head_outputs.append(B_head)
            head_weights.append(A_prime)

        B_concat = torch.cat(head_outputs, dim=0)
        B = self.W_o @ B_concat
        return B, torch.stack(head_weights)

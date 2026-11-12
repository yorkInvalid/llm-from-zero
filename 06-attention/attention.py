"""多头注意力：输入、输出都按列排列 token；导入本文件不会启动训练。"""

import torch
from torch import nn

from position_encoding import apply_rope


class MultiHeadAttention(nn.Module):
    """保留逐头计算，便于与讲义中的 Q、K、V、A、A′、B 对照。"""

    def __init__(self, d_in=10, num_heads=2, d_k=5, d_v=5, use_rope=False,
                 dropout=0.0):
        super().__init__()
        if use_rope and d_k < 2:
            raise ValueError("RoPE 每头至少需要 2 维")
        self.num_heads = num_heads
        self.d_in = d_in
        self.d_k = d_k
        self.use_rope = use_rope
        self.attention_dropout = nn.Dropout(dropout)

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

    def forward(self, X, padding_mask=None, causal=False, memory=None):
        """X: (...,d_in,target_len)；memory 可为 (...,d_in,source_len)。

        memory=None 为自注意力；否则 Q 来自 X，K/V 来自 memory。
        输出与 X 同形状，权重为 (...,head,key_len,query_len)。padding_mask
        始终对应 K/V 序列，True 表示 PAD。Cross-Attention 不用 causal/RoPE。
        拒绝全 PAD 样本；个别 query 无可读 key 时，其权重和注意力输出为零。
        """
        if X.ndim not in (2, 3) or X.shape[-1] == 0:
            raise ValueError("X 必须为非空的 (d_in,n) 或 (batch,d_in,n)")
        if X.shape[-2] != self.d_in:
            raise ValueError("X 的特征维度必须等于 d_in")
        key_value_input = X
        if memory is not None:
            if causal:
                raise ValueError("Cross-Attention 不使用 causal mask")
            if self.use_rope:
                raise ValueError("当前 Cross-Attention 不启用 RoPE，请使用 use_rope=False 的独立实例")
            if memory.ndim != X.ndim or memory.shape[:-2] != X.shape[:-2]:
                raise ValueError("memory 与 X 的 batch 维度必须一致，不允许隐式广播")
            if memory.shape[-2] != self.d_in or memory.shape[-1] == 0:
                raise ValueError("memory 特征维度必须等于 d_in，源序列不能为空")
            if memory.device != X.device or memory.dtype != X.dtype:
                raise ValueError("memory 与 X 必须具有相同的设备和 dtype")
            key_value_input = memory
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                raise TypeError("padding_mask 必须为 bool，True 表示 PAD")
            if padding_mask.shape != key_value_input.shape[:-2] + key_value_input.shape[-1:]:
                raise ValueError("padding_mask 形状必须与 K/V 的 batch 和序列长度一致")
            if padding_mask.device != X.device:
                raise ValueError("padding_mask 和 X 必须在同一设备")
            if padding_mask.all(dim=-1).any():
                raise ValueError("每条序列必须至少包含一个非 PAD token")
        positions = torch.arange(X.shape[-1], dtype=X.dtype, device=X.device)
        # 两种 mask 语义独立，最后取并集；True 始终表示禁止读取。
        padding_blocked = padding_mask.unsqueeze(-1) if padding_mask is not None else None
        future_blocked = None
        if causal:
            token_positions = torch.arange(X.shape[-1], device=X.device)
            # 行 key、列 query：key > query 为未来位置，屏蔽严格下三角。
            future_blocked = token_positions[:, None] > token_positions[None, :]
        blocked = padding_blocked
        if future_blocked is not None:
            blocked = future_blocked if blocked is None else blocked | future_blocked
        no_readable_key = blocked.all(dim=-2, keepdim=True) if blocked is not None else None
        head_outputs = []
        head_weights = []

        for h in range(self.num_heads):
            Q = self.W_q[h] @ X
            K = self.W_k[h] @ key_value_input
            V = self.W_v[h] @ key_value_input
            # RoPE 作用于各头的 Q、K；V 不旋转。
            if self.use_rope:
                Q = apply_rope(Q, positions)
                K = apply_rope(K, positions)
            A = K.transpose(-2, -1) @ Q  # 行是 key，列是 query
            A = A / (self.d_k ** 0.5)
            if blocked is not None:
                A = A.masked_fill(blocked, float("-inf"))
                # 左侧 PAD 的早期 query 可能全被屏蔽；先避免 softmax(-inf,...)
                # 产生 NaN，再将这些 query 的全部权重置零。
                A = A.masked_fill(no_readable_key, 0.0)
            A_prime = torch.softmax(A, dim=-2)  # 沿 key 归一化
            if blocked is not None:
                A_prime = A_prime.masked_fill(blocked, 0.0)
            # 返回的仍是 dropout 前的概率；实际聚合使用 dropout 后的权重。
            B_head = V @ self.attention_dropout(A_prime)
            head_outputs.append(B_head)
            head_weights.append(A_prime)

        B_concat = torch.cat(head_outputs, dim=-2)
        B = self.W_o @ B_concat
        return B, torch.stack(head_weights, dim=-3)

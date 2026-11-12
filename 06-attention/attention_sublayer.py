"""Encoder 的注意力子层：多头注意力 → 残差连接 → LayerNorm（Post-LN）。"""

from torch import nn

from attention import MultiHeadAttention


def forward_with_residual(attention, X, padding_mask=None):
    """只执行注意力和残差；attention 应为 MultiHeadAttention。"""
    B, weights = attention(X, padding_mask=padding_mask)
    Y = X + B
    return Y, weights


class AttentionSublayer(nn.Module):
    """尚不包含 FFN；输入输出均为 (..., d_in, token 数)。"""

    def __init__(self, d_in=10, num_heads=2, d_k=5, d_v=5, use_rope=False,
                 dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(d_in, num_heads, d_k, d_v, use_rope,
                                            dropout=dropout)
        self.output_dropout = nn.Dropout(dropout)
        # 每个特征一组 gamma/beta，在各 token 间共享；注册到优化器中。
        self.norm = nn.LayerNorm(d_in)

    def forward(self, X, padding_mask=None):
        B, weights = self.attention(X, padding_mask=padding_mask)
        Y = X + self.output_dropout(B)  # 只丢弃分支，不丢弃残差直通路径
        # token 按列排列，转置后才是沿每个 token 的特征维度归一化。
        Z = self.norm(Y.transpose(-2, -1)).transpose(-2, -1)
        return Z, weights

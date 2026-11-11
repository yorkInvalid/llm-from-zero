"""Post-LN Encoder Layer：注意力和 FFN 后各有一次残差与 LayerNorm。"""

from torch import nn

from attention_sublayer import AttentionSublayer
from feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """主体流程：Attention → Add & Norm → FFN → Add & Norm。"""

    def __init__(self, d_model=10, num_heads=2, d_k=5, d_v=5,
                 d_ff=40, use_rope=False):
        super().__init__()
        self.attention_sublayer = AttentionSublayer(
            d_in=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v,
            use_rope=use_rope,
        )
        self.ffn = FeedForward(d_model, d_ff)
        # 与注意力子层的 LayerNorm 独立，不共享 gamma/beta。
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X):
        """输入输出均为 (d_model, n)，同时返回 (head, key, query) 权重。"""
        Z, weights = self.attention_sublayer(X)
        F = self.ffn(Z)
        Y = Z + F  # 加回 FFN 的输入 Z，而不是整个 Block 的输入 X。
        output = self.norm2(Y.T).T
        return output, weights

"""Post-LN Encoder Layer：注意力和 FFN 后各有一次残差与 LayerNorm。"""

from torch import nn

from attention_sublayer import AttentionSublayer
from feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """主体流程：Attention → Add & Norm → FFN → Add & Norm。"""

    def __init__(self, d_model=10, num_heads=2, d_k=5, d_v=5,
                 d_ff=40, use_rope=False, dropout=0.0):
        super().__init__()
        self.attention_sublayer = AttentionSublayer(
            d_in=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v,
            use_rope=use_rope, dropout=dropout,
        )
        self.ffn = FeedForward(d_model, d_ff)
        self.ffn_dropout = nn.Dropout(dropout)
        # 与注意力子层的 LayerNorm 独立，不共享 gamma/beta。
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X, padding_mask=None):
        """输入输出 (...,d_model,n)，返回 (...,head,key,query) 权重。"""
        Z, weights = self.attention_sublayer(X, padding_mask=padding_mask)
        F = self.ffn(Z)
        Y = Z + self.ffn_dropout(F)  # 只对 FFN 分支应用 dropout
        output = self.norm2(Y.transpose(-2, -1)).transpose(-2, -1)
        return output, weights

"""经典 Encoder–Decoder 的一个 Post-LN Decoder Block。"""

from torch import nn

from attention import MultiHeadAttention
from feed_forward import FeedForward


class DecoderLayer(nn.Module):
    """因果自注意力 → Add & Norm → 交叉注意力 → Add & Norm → FFN → Add & Norm。"""

    def __init__(self, d_model=10, num_heads=2, d_k=5, d_v=5,
                 d_ff=40, use_rope=False, dropout=0.0):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model, num_heads, d_k, d_v, use_rope=use_rope, dropout=dropout)
        # 独立参数；交叉注意力不使用 causal 或自注意力的 RoPE。
        self.cross_attention = MultiHeadAttention(
            d_model, num_heads, d_k, d_v, use_rope=False, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff)
        self.self_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, X, memory, target_padding_mask=None, source_padding_mask=None):
        """X: (...,d_model,T)，memory: (...,d_model,S)，允许 S != T。

        两个 bool mask 分别对应目标和源序列，True 表示 PAD。
        返回 output 和 {"self": 自注意力概率, "cross": 交叉注意力概率}。
        """
        if memory is None:
            raise ValueError("DecoderLayer 需要 Encoder 输出 memory")

        B, self_weights = self.self_attention(
            X, padding_mask=target_padding_mask, causal=True)
        Y1 = X + self.self_dropout(B)
        decoder_hidden = self.norm1(Y1.transpose(-2, -1)).transpose(-2, -1)

        C, cross_weights = self.cross_attention(
            decoder_hidden, memory=memory, padding_mask=source_padding_mask)
        Y2 = decoder_hidden + self.cross_dropout(C)
        H = self.norm2(Y2.transpose(-2, -1)).transpose(-2, -1)

        F = self.ffn(H)
        Y3 = H + self.ffn_dropout(F)
        output = self.norm3(Y3.transpose(-2, -1)).transpose(-2, -1)
        return output, {"self": self_weights, "cross": cross_weights}

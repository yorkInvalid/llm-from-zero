"""多个独立 DecoderLayer 堆叠；每层查询同一份 Encoder 最终输出。"""

from torch import nn

from decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, num_layers=3, d_model=10, num_heads=2,
                 d_k=5, d_v=5, d_ff=40, use_rope=False, dropout=0.0):
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers 必须是正整数")
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_k=d_k,
                         d_v=d_v, d_ff=d_ff, use_rope=use_rope, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X, memory, target_padding_mask=None, source_padding_mask=None):
        """返回 (...,d_model,T) 输出和逐层 {self, cross} 注意力字典列表。

        memory 为同一份 (...,d_model,S) Encoder 最终输出，不按 Decoder 层拆分。
        两种 mask 均为 (...,length) bool 张量，True 表示 PAD。
        """
        H = X
        attention_weights = []
        for layer in self.layers:
            H, weights = layer(H, memory,
                               target_padding_mask=target_padding_mask,
                               source_padding_mask=source_padding_mask)
            attention_weights.append(weights)
        return H, attention_weights

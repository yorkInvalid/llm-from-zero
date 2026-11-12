"""由独立的 Post-LN EncoderLayer 堆叠而成的 Encoder。"""

from torch import nn

from encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, num_layers=3, d_model=10, num_heads=2,
                 d_k=5, d_v=5, d_ff=40, use_rope=False, dropout=0.0):
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers 必须是正整数")
        # 每次创建新对象，参数独立；ModuleList 将所有层注册到模型中。
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, d_k=d_k,
                         d_v=d_v, d_ff=d_ff, use_rope=use_rope, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X, padding_mask=None):
        """返回 (...,d_model,n) 输出和逐层 (...,head,key,query) 权重。"""
        H = X
        attention_weights = []
        for layer in self.layers:
            H, weights = layer(H, padding_mask=padding_mask)
            attention_weights.append(weights)
        return H, attention_weights

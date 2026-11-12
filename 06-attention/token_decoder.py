"""目标 Token Embedding → 位置编码 → Decoder → 可选词表预测头。"""

import torch
from torch import nn

from decoder import Decoder
from position_encoding import sinusoidal_encoding


class TokenDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=10, num_layers=3, num_heads=2,
                 d_k=5, d_v=5, d_ff=40, position_encoding="sinusoidal",
                 padding_idx=None, dropout=0.0):
        super().__init__()
        if position_encoding not in {"none", "sinusoidal", "rope"}:
            raise ValueError("未知的位置编码模式")
        self.position_encoding = position_encoding
        # 与源 TokenEncoder 的 Embedding 独立，允许源/目标词表不同。
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.input_dropout = nn.Dropout(dropout)
        self.decoder = Decoder(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads,
            d_k=d_k, d_v=d_v, d_ff=d_ff,
            use_rope=position_encoding == "rope", dropout=dropout)

    def forward(self, token_ids, memory, source_padding_mask=None):
        """目标 ID (...,T) → hidden (...,d_model,T) 和逐层注意力。

        memory 是 Encoder 输出 (...,d_model,S)，源 PAD 必须单独传入。
        本层不做目标序列右移；训练时由调用方准备右移后的输入。
        """
        if token_ids.ndim not in (1, 2) or token_ids.numel() == 0:
            raise ValueError("token_ids 必须是非空的一维或二维序列")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids 必须是整数索引")
        if (token_ids < 0).any() or (token_ids >= self.embedding.num_embeddings).any():
            raise ValueError("token ID 超出词表范围")
        target_padding_mask = (token_ids == self.embedding.padding_idx
                               if self.embedding.padding_idx is not None else None)
        X = self.embedding(token_ids).transpose(-2, -1)
        # 与 TokenEncoder 一致：当前不额外乘 sqrt(d_model)。
        if self.position_encoding == "sinusoidal":
            X = X + sinusoidal_encoding(X)
        X = self.input_dropout(X)
        return self.decoder(X, memory, target_padding_mask=target_padding_mask,
                            source_padding_mask=source_padding_mask)


class TokenDecoderLM(nn.Module):
    """逐位置预测目标词表：返回原始 logits，可直接用于交叉熵。"""

    def __init__(self, vocab_size, d_model=10, num_layers=3, num_heads=2,
                 d_k=5, d_v=5, d_ff=40, position_encoding="sinusoidal",
                 padding_idx=None, dropout=0.0):
        super().__init__()
        self.token_decoder = TokenDecoder(
            vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            d_k=d_k, d_v=d_v, d_ff=d_ff, position_encoding=position_encoding,
            padding_idx=padding_idx, dropout=dropout)
        # 第一版不共享 Embedding 与输出投影的权重，便于观察两者更新。
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, memory, source_padding_mask=None):
        hidden, weights = self.token_decoder(token_ids, memory, source_padding_mask)
        logits = self.lm_head(hidden.transpose(-2, -1))
        return logits, weights  # (...,T,vocab_size)，不做池化或 softmax。

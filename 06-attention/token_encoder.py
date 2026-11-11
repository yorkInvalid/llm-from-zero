"""可学习的 token 查表 → 位置编码 → 多层 Encoder，以及教学分类头。"""

import torch
from torch import nn

from encoder import Encoder
from position_encoding import sinusoidal_encoding


class TokenEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=10, num_layers=3, num_heads=2,
                 d_k=5, d_v=5, d_ff=40, position_encoding="sinusoidal"):
        super().__init__()
        if position_encoding not in {"none", "sinusoidal", "rope"}:
            raise ValueError("未知的位置编码模式")
        self.position_encoding = position_encoding
        # 随机初始化一次；weight 是模型参数，之后由优化器更新。
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                               use_rope=position_encoding == "rope")

    def forward(self, token_ids):
        """一维整数 ID (n,) → 输出 (d_model, n) 和逐层注意力列表。"""
        if token_ids.ndim != 1 or token_ids.numel() == 0:
            raise ValueError("token_ids 必须是非空的一维序列，暂不支持 batch")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids 必须是整数索引")
        if (token_ids < 0).any() or (token_ids >= self.embedding.num_embeddings).any():
            raise ValueError("token ID 超出词表范围")
        X = self.embedding(token_ids).T  # (n, d_model) → (d_model, n)
        # 每次 forward 重新查当前的表，不能在训练循环外缓存或 detach。
        if self.position_encoding == "sinusoidal":
            X = X + sinusoidal_encoding(X)
        return self.encoder(X)


class TokenClassifier(nn.Module):
    """逐 token 分类：输出未经 softmax 的 logits，供交叉熵使用。"""

    def __init__(self, vocab_size, num_classes=3, d_model=10,
                 num_layers=3, position_encoding="sinusoidal"):
        super().__init__()
        self.token_encoder = TokenEncoder(vocab_size, d_model=d_model,
                                          num_layers=num_layers,
                                          position_encoding=position_encoding)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids):
        encoded, weights = self.token_encoder(token_ids)
        return self.classifier(encoded.T), weights  # (n, num_classes)

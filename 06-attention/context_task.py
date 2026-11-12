"""重复 token 识别：数据、无上下文基线及忽略 PAD 的评估。"""

from collections import Counter
import random

import torch
from torch import nn

from feed_forward import FeedForward


VOCAB_SIZE = 8
PAD_ID = VOCAB_SIZE
IGNORE_INDEX = -100


def repetition_labels(sequence):
    """某 token 在当前序列出现至少两次时标为 1，否则为 0。"""
    counts = Counter(sequence)
    return [int(counts[token] > 1) for token in sequence]


def make_splits(train_size=1024, validation_size=256, seed=42):
    """长度 3–8；按 token 多重集合去重，验证集也不会只是训练序列的重排。"""
    if train_size < 1 or validation_size < 1:
        raise ValueError("训练集和验证集大小必须为正")
    # 组合总数有限，教学实验限制大小，防止无限拒绝采样。
    if train_size + validation_size > 5000:
        raise ValueError("教学实验最多生成 5000 个不同 token 多重集合")
    rng = random.Random(seed)
    examples, seen = [], set()
    while len(examples) < train_size + validation_size:
        sequence = tuple(rng.randrange(VOCAB_SIZE) for _ in range(rng.randint(3, 8)))
        key = tuple(sorted(sequence))
        if key not in seen:
            seen.add(key)
            examples.append(sequence)
    rng.shuffle(examples)
    return examples[:train_size], examples[train_size:]


def collate_sequences(sequences):
    """右侧补 PAD；返回 (batch,length) ID 和标签。"""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("batch 和每条序列都不能为空")
    if any(token < 0 or token >= VOCAB_SIZE for seq in sequences for token in seq):
        raise ValueError("输入序列只能含有效 token，PAD 由 collate 添加")
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), PAD_ID, dtype=torch.long)
    labels = torch.full_like(ids, IGNORE_INDEX)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence)
        labels[index, :len(sequence)] = torch.tensor(repetition_labels(sequence))
    return ids, labels


class TokenwiseBaseline(nn.Module):
    """Embedding + 多层逐 token FFN，无位置编码、Attention 或跨 token 聚合。"""

    def __init__(self, d_model=32, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE + 1, d_model, padding_idx=PAD_ID)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([FeedForward(d_model, 40) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, ids):
        H = self.input_dropout(self.embedding(ids)).transpose(-2, -1)
        for layer, norm in zip(self.layers, self.norms):
            H = norm((H + self.dropout(layer(H))).transpose(-2, -1)).transpose(-2, -1)
        return self.classifier(H.transpose(-2, -1)), []


def evaluate(model, sequences, batch_size=64):
    """按有效 token 汇总，避免最后一个小 batch 或 PAD 改变统计权重。"""
    if not sequences or batch_size < 1:
        raise ValueError("评估集不能为空且 batch_size 必须为正")
    was_training = model.training
    model.eval()
    total_loss = total = correct = sequence_correct = 0
    class_total = [0, 0]
    class_correct = [0, 0]
    try:
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                ids, labels = collate_sequences(sequences[start:start + batch_size])
                logits, _ = model(ids)
                valid = labels != IGNORE_INDEX
                prediction = logits.argmax(dim=-1)
                total_loss += nn.functional.cross_entropy(
                    logits.reshape(-1, 2), labels.reshape(-1),
                    ignore_index=IGNORE_INDEX, reduction="sum",
                ).item()
                total += valid.sum().item()
                correct += ((prediction == labels) & valid).sum().item()
                sequence_correct += ((prediction == labels) | ~valid).all(dim=-1).sum().item()
                for category in (0, 1):
                    selected = labels == category
                    class_total[category] += selected.sum().item()
                    class_correct[category] += ((prediction == labels) & selected).sum().item()
    finally:
        model.train(was_training)
    recalls = [class_correct[i] / class_total[i] if class_total[i] else None for i in (0, 1)]
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "sequence_accuracy": sequence_correct / len(sequences),
        "recall_unique": recalls[0], "recall_repeated": recalls[1],
        "balanced_accuracy": sum(recalls) / 2 if None not in recalls else None,
        "valid_tokens": total,
        "positive_fraction": class_total[1] / total,
    }

"""倒序任务：互不重复的数据划分、右移目标和右侧 PAD。"""

import random

import torch
from torch.nn.utils.rnn import pad_sequence

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
VOCAB_SIZE = 13


def make_splits(train_size=1024, validation_size=128, seed=42,
                min_length=2, max_length=6):
    if train_size < 1 or validation_size < 1 or not 1 <= min_length <= max_length:
        raise ValueError("数据量须为正，长度范围须满足 1 <= min <= max")
    capacity = sum((VOCAB_SIZE - 3) ** n for n in range(min_length, max_length + 1))
    if train_size + validation_size > capacity // 2:
        raise ValueError("采样量不能超过序列空间的一半，请减小数据量或增加长度")
    rng = random.Random(seed)
    seen, samples = set(), []
    while len(samples) < train_size + validation_size:
        sample = tuple(rng.randrange(3, VOCAB_SIZE)
                       for _ in range(rng.randint(min_length, max_length)))
        if sample not in seen:
            seen.add(sample)
            samples.append(sample)
    rng.shuffle(samples)
    return samples[:train_size], samples[train_size:]


def collate_sequences(samples, device=None):
    if not samples or any(not s or any(t < 3 or t >= VOCAB_SIZE for t in s) for s in samples):
        raise ValueError("样本必须是非空的普通 token ID 序列")
    sources = [torch.tensor(s, dtype=torch.long) for s in samples]
    # 分别构造输入和标签，再补 PAD：EOS 只在标签中作为停止目标。
    inputs = [torch.tensor([BOS_ID, *reversed(s)]) for s in samples]
    labels = [torch.tensor([*reversed(s), EOS_ID]) for s in samples]
    return tuple(pad_sequence(group, batch_first=True, padding_value=PAD_ID).to(device)
                 for group in (sources, inputs, labels))

"""小矩阵 GQA 实验；随机权重，仅验证计算，不代表 Qwen 的生成能力。"""

import torch
from torch import nn


class TinyGQA(nn.Module):
    def __init__(self, d_model=16, query_heads=4, kv_heads=2, head_dim=8):
        super().__init__()
        if min(d_model, query_heads, kv_heads, head_dim) < 1 or query_heads % kv_heads:
            raise ValueError('维度须为正，Q 头数须能被 KV 头数整除')
        self.hq, self.hkv, self.dh = query_heads, kv_heads, head_dim
        self.q_proj = nn.Linear(d_model, query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(query_heads * head_dim, d_model, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.hq, self.dh).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.hkv, self.dh).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.hkv, self.dh).transpose(1, 2)
        # 为便于理解显式展开：真实缓存保存展开前的 K/V，不应存重复副本。
        keys = k.repeat_interleave(self.hq // self.hkv, dim=1)
        values = v.repeat_interleave(self.hq // self.hkv, dim=1)
        scores = q @ keys.transpose(-2, -1) / self.dh**.5
        future = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        weights = scores.masked_fill(future, -torch.inf).softmax(-1)
        mixed = (weights @ values).transpose(1, 2).reshape(b, t, self.hq * self.dh)
        return self.o_proj(mixed), dict(q=q, k=k, v=v, weights=weights)


def run_demo():
    torch.manual_seed(42)
    model = TinyGQA()
    x = torch.randn(1, 5, 16)
    output, detail = model(x)
    print('随机教学模型，不加载 Qwen 权重；省略 QK Norm/RoPE/FFN，仅观察 GQA。')
    print('输入:', tuple(x.shape), '输出:', tuple(output.shape))
    for key in ('q', 'k', 'v', 'weights'):
        print(key, tuple(detail[key].shape))
    print('共享关系: Q0,Q1 → KV0；Q2,Q3 → KV1')
    print('不同 Q 仍有自己的权重:', not torch.allclose(detail['weights'][:, 0], detail['weights'][:, 1]))
    return model, x, output, detail


if __name__ == '__main__':
    run_demo()

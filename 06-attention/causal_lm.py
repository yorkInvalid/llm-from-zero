"""小型 Decoder-only 语言模型：只有因果自注意力，没有 Encoder/memory。"""

import torch
from torch import nn

from attention import MultiHeadAttention
from feed_forward import FeedForward
from position_encoding import sinusoidal_encoding
from lm_components import RMSNorm, SwiGLU


class CausalBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout, norm_style='pre', norm_type='rms',
                 position_encoding='rope', ffn_type='swiglu'):
        super().__init__()
        self.norm_style = norm_style
        self.attention = MultiHeadAttention(d_model, num_heads, d_model // num_heads,
                                            d_model // num_heads, use_rope=position_encoding == 'rope', dropout=dropout)
        self.ffn = SwiGLU(d_model) if ffn_type == 'swiglu' else FeedForward(d_model, 4 * d_model)
        norm = RMSNorm if norm_type == 'rms' else nn.LayerNorm
        self.norm1, self.norm2 = norm(d_model), norm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask, *, use_cache=False, kv_cache=None):
        def attention(inputs):
            return self.attention(inputs, padding_mask=padding_mask, causal=True,
                                  use_cache=use_cache, kv_cache=kv_cache)
        if self.norm_style == 'pre':
            normalized = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
            result = attention(normalized)
            branch = result[0]
            x = x + self.dropout(branch)
            normalized = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
            output = x + self.dropout(self.ffn(normalized))
        else:
            result = attention(x)
            x = self.norm1((x + self.dropout(result[0])).transpose(-2, -1)).transpose(-2, -1)
            output = self.norm2((x + self.dropout(self.ffn(x))).transpose(-2, -1)).transpose(-2, -1)
        return (output, result[2]) if use_cache else output


class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_layers=2, num_heads=4,
                 dropout=.1, context_length=64, pad_id=0, norm_style='pre', norm_type='rms',
                 position_encoding='rope', ffn_type='swiglu'):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads or num_layers < 1 or context_length < 1:
            raise ValueError('模型维度/层数/上下文长度无效')
        if norm_style not in ('pre', 'post') or norm_type not in ('rms', 'layer'):
            raise ValueError('未知的归一化配置')
        if position_encoding not in ('rope', 'sinusoidal') or ffn_type not in ('swiglu', 'relu'):
            raise ValueError('未知的位置编码或 FFN 配置')
        self.config = dict(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
                           num_heads=num_heads, dropout=dropout, context_length=context_length, pad_id=pad_id,
                           norm_style=norm_style, norm_type=norm_type,
                           position_encoding=position_encoding, ffn_type=ffn_type)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([CausalBlock(d_model, num_heads, dropout, norm_style, norm_type,
                                               position_encoding, ffn_type) for _ in range(num_layers)])
        norm = RMSNorm if norm_type == 'rms' else nn.LayerNorm
        self.final_norm = norm(d_model) if norm_style == 'pre' else nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids, *, use_cache=False, kv_cache=None):
        """常规返回 logits；缓存推理返回 (新位置 logits, 逐层缓存)。

        缓存仅用于同一模型参数、同一序列的连续推理，不保存进 state_dict。
        传入缓存时 ids 仅包含新增 token；调用方须使用 eval + no_grad。
        """
        if ids.ndim not in (1, 2) or ids.numel() == 0 or ids.shape[-1] > self.config['context_length']:
            raise ValueError('ID 须为非空的一维/二维张量，长度不能超过 context_length')
        if kv_cache is not None and (not use_cache or len(kv_cache) != len(self.layers)):
            raise ValueError('缓存必须启用 use_cache 且与模型层数一致')
        past_length = kv_cache[0]['padding_mask'].shape[-1] if kv_cache is not None else 0
        if kv_cache is not None and any(c['padding_mask'].shape[-1] != past_length for c in kv_cache):
            raise ValueError('每层缓存长度必须相同')
        if past_length + ids.shape[-1] > self.config['context_length']:
            raise ValueError('缓存加新输入超过上下文长度，请使用当前窗口重新 prefill')
        x = self.embedding(ids).transpose(-2, -1)
        if self.config['position_encoding'] == 'sinusoidal':
            x = x + sinusoidal_encoding(x, position_offset=past_length)
        x = self.input_dropout(x)
        mask = ids == self.config['pad_id']
        caches = []
        for index, layer in enumerate(self.layers):
            if use_cache:
                x, cache = layer(x, mask, use_cache=True, kv_cache=kv_cache[index] if kv_cache is not None else None)
                caches.append(cache)
            else:
                x = layer(x, mask)
        logits = self.lm_head(self.final_norm(x.transpose(-2, -1)))
        return (logits, caches) if use_cache else logits

    @torch.no_grad()
    def generate(self, ids, bos_id=1, eos_id=2, max_new_tokens=64, use_cache=True):
        if ids.ndim != 1 or not ids.numel() or max_new_tokens < 0:
            raise ValueError('生成需要非空一维前缀及非负长度')
        modes = {m: m.training for m in self.modules()}
        result = ids.clone()
        cache = None
        try:
            self.eval()
            for _ in range(max_new_tokens):
                if use_cache:
                    if cache is None or cache[0]['padding_mask'].shape[-1] >= self.config['context_length']:
                        # 旧生成语义会截断前缀并重置位置；必须重建每层表示，不能只丢最旧 K/V。
                        logits, cache = self(result[-self.config['context_length']:], use_cache=True)
                    else:
                        logits, cache = self(result[-1:], use_cache=True, kv_cache=cache)
                else:
                    logits = self(result[-self.config['context_length']:])
                scores = logits[-1].clone()
                scores[[self.config['pad_id'], bos_id]] = -torch.inf
                token = scores.argmax().reshape(1)
                result = torch.cat((result, token))
                if token.item() == eos_id:
                    break
        finally:
            for m, mode in modes.items():
                m.training = mode
        return result

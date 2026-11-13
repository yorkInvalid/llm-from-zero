"""离线读取固定 Qwen3 配置；只做整数计算，不分配真实模型权重。"""

import argparse
import hashlib
import json
from pathlib import Path

ASSETS = Path(__file__).parent / 'assets'


def load_config():
    raw = (ASSETS / 'qwen3-0.6b-config.json').read_bytes()
    source = json.loads((ASSETS / 'source.json').read_text())
    if hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('配置校验失败，请检查 assets 文件')
    return json.loads(raw)


def describe(config, batch=1, tokens=1024, bytes_per_element=2):
    if min(batch, tokens, bytes_per_element) < 1:
        raise ValueError('batch、tokens 和每元素字节数须为正')
    d, hq, hkv, dh = (config[k] for k in ('hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim'))
    if hq % hkv:
        raise ValueError('Q 头数必须能被 KV 头数整除')
    layers, hidden, vocab = (config[k] for k in ('num_hidden_layers', 'intermediate_size', 'vocab_size'))
    embedding = vocab * d
    attention = d * (hq * dh + 2 * hkv * dh + hq * dh)
    ffn = 3 * d * hidden
    norms = 2 * d + 2 * dh  # 每层的两个 RMSNorm，以及按 head_dim 共享的 Q/K Norm。
    output = 0 if config['tie_word_embeddings'] else vocab * d
    total = embedding + layers * (attention + ffn + norms) + d + output
    kv_bytes = 2 * batch * tokens * layers * hkv * dh * bytes_per_element
    mha_bytes = 2 * batch * tokens * layers * hq * dh * bytes_per_element
    return dict(model='Qwen3-0.6B (fixed config)', layers=layers, hidden_size=d,
                query_heads=hq, kv_heads=hkv, head_dim=dh, queries_per_kv=hq // hkv,
                shapes=dict(input=[batch, tokens, d], q_projection=[batch, tokens, hq * dh],
                            q_heads=[batch, hq, tokens, dh], k_heads=[batch, hkv, tokens, dh],
                            v_heads=[batch, hkv, tokens, dh], concatenated=[batch, tokens, hq * dh],
                            attention_output=[batch, tokens, d], logits=[batch, tokens, vocab]),
                parameters=dict(embedding=embedding, attention_per_layer=attention,
                                swiglu_per_layer=ffn, norms_per_layer=norms,
                                final_norm=d, extra_output_head=output, total=total),
                storage=dict(bytes_per_element=bytes_per_element,
                             weights_gib=total * bytes_per_element / 2**30,
                             kv_cache_mib=kv_bytes / 2**20,
                             hypothetical_mha_cache_mib=mha_bytes / 2**20,
                             kv_ratio_to_mha=kv_bytes / mha_bytes),
                limits='Dense Qwen3 formula, no projection biases, tied head. Weight/cache storage only; not peak training or inference memory.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=1024)
    parser.add_argument('--bytes', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(describe(load_config(), args.batch, args.tokens, args.bytes), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

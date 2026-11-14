"""课件的计时和逐 token 生成工具；只支持单条、无 padding 的输入。"""
import statistics
import time
import torch

MODEL_ID = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def eos_ids(model):
    value = model.generation_config.eos_token_id
    return set(value if isinstance(value, list) else [value]) if value is not None else set()


def choose_token(logits, *, sample=False, temperature=0.7, top_k=20, generator=None):
    """logits 为 [1,V]。抽样在 CPU 上进行，方便跨设备使用固定随机种子。"""
    if not sample:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature <= 0 or top_k < 1:
        raise ValueError('采样要求 temperature > 0，top_k >= 1')
    values, indices = torch.topk(logits.float().cpu() / temperature, min(top_k, logits.shape[-1]))
    probabilities = values.softmax(dim=-1)
    selected = torch.multinomial(probabilities, 1, generator=generator)
    return indices.gather(-1, selected).to(logits.device)


@torch.inference_mode()
def manual_generate(model, input_ids, *, max_new_tokens=32, use_cache=True,
                    sample=False, temperature=0.7, top_k=20, seed=42,
                    stop_at_eos=True, verbose=False, tokenizer=None):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('本教学循环只接受 [1,T] 且 T > 0，无 padding')
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens 必须 >= 1')
    ids = input_ids.clone()
    prompt_length = ids.shape[1]
    cache = None
    trace = []
    generator = torch.Generator(device='cpu').manual_seed(seed)
    stop_reason = 'max_new_tokens'
    synchronize(ids.device)
    start = time.perf_counter()
    for step in range(max_new_tokens):
        # prefill 输入全部 prompt；已有缓存后，只输入尚未处理的最后一个 token。
        current = ids[:, -1:] if use_cache and cache is not None else ids
        output = model(input_ids=current, attention_mask=torch.ones_like(ids),
                       past_key_values=cache, use_cache=use_cache, logits_to_keep=1)
        next_id = choose_token(output.logits[:, -1, :], sample=sample,
                               temperature=temperature, top_k=top_k, generator=generator)
        cache = output.past_key_values if use_cache else None
        trace.append({'step': step + 1, 'input_tokens': current.shape[1],
                      'cache_tokens': cache.get_seq_length() if cache is not None else 0,
                      'token_id': next_id.item()})
        ids = torch.cat([ids, next_id], dim=1)
        if verbose and tokenizer is not None:
            print(trace[-1], '累计文本:', repr(tokenizer.decode(ids[0, prompt_length:], skip_special_tokens=True)))
        if stop_at_eos and next_id.item() in eos_ids(model):
            stop_reason = 'eos'
            break
    synchronize(ids.device)
    elapsed = time.perf_counter() - start
    return {'ids': ids, 'new_ids': ids[:, prompt_length:], 'trace': trace,
            'stop_reason': stop_reason, 'seconds': elapsed,
            'tokens_per_second': len(trace) / elapsed}


@torch.inference_mode()
def check_cache(model, input_ids, steps=3):
    """两条路径喂同样的 token；比较每一步 logits，避免不同生成轨迹混淆结论。"""
    ids = input_ids.clone()
    cached = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
    cache = cached.past_key_values
    errors = []
    for _ in range(steps):
        next_id = cached.logits[:, -1:].argmax(dim=-1)
        ids = torch.cat([ids, next_id], dim=1)
        cached = model(input_ids=next_id, attention_mask=torch.ones_like(ids),
                       past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = cached.past_key_values
        full = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
        error = (cached.logits - full.logits).abs().max().item()
        # 本课使用 float32；跨设备浮点运算允许少量数值差异。
        torch.testing.assert_close(cached.logits, full.logits, rtol=1e-3, atol=2e-3)
        assert cache.get_seq_length() == ids.shape[1]
        errors.append(error)
    return errors


def benchmark(model, input_ids, tokens=12, repeats=3):
    # 同样的 prompt、固定步数、贪心生成；包括 prefill，不含分词/下载/打印。
    for cached in (False, True):
        manual_generate(model, input_ids, max_new_tokens=2, use_cache=cached, stop_at_eos=False)
    samples = {False: [], True: []}
    for repeat in range(repeats):
        # 交替顺序，减少始终先跑一种方式的偏差。
        for cached in ((False, True) if repeat % 2 == 0 else (True, False)):
            result = manual_generate(model, input_ids, max_new_tokens=tokens,
                                     use_cache=cached, stop_at_eos=False)
            samples[cached].append(result['seconds'])
    return [{'use_cache': cached, 'new_tokens': tokens, 'repeats': repeats,
             'seconds': times, 'median_seconds': statistics.median(times),
             'tokens_per_second': tokens / statistics.median(times)}
            for cached, times in samples.items()]

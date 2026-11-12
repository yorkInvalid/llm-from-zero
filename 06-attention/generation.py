"""单条源序列的贪心自回归生成：Encoder 一次，Decoder 逐步扩展。"""

import torch


@torch.no_grad()
def greedy_generate(encoder, decoder, source_ids, bos_id, eos_id, max_new_tokens=50):
    """返回一维目标 ID，包含 BOS，以及选中时的 EOS。

    source_ids 为一维整数张量；模型和输入须在同一设备。
    每步选择除 BOS/PAD 外分数最高的 token，达到 EOS 或长度上限停止。
    max_new_tokens 不包含初始 BOS，允许为 0。暂不支持 batch 或 KV cache。
    推理期间关闭 Dropout/梯度，结束后恢复所有模块原先的训练状态。
    """
    if source_ids.ndim != 1 or source_ids.numel() == 0:
        raise ValueError("生成目前只支持非空的一维 source_ids")
    if source_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("source_ids 必须是整数索引")
    if (source_ids < 0).any() or (source_ids >= encoder.embedding.num_embeddings).any():
        raise ValueError("源 token ID 超出词表范围")
    source_pad = encoder.embedding.padding_idx
    source_mask = source_ids == source_pad if source_pad is not None else None
    if source_mask is not None and source_mask.all():
        raise ValueError("源序列不能全部是 PAD")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens 必须是非负整数")
    vocab_size = decoder.token_decoder.embedding.num_embeddings
    target_pad = decoder.token_decoder.embedding.padding_idx
    for name, token_id in (("bos_id", bos_id), ("eos_id", eos_id)):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
            raise ValueError(f"{name} 必须是目标词表内的整数 ID")
    if bos_id == eos_id or target_pad in (bos_id, eos_id):
        raise ValueError("BOS、EOS 和目标 PAD 必须使用不同 ID")

    generated = torch.tensor([bos_id], dtype=torch.long, device=source_ids.device)
    if max_new_tokens == 0:
        return generated
    modes = {module: module.training for model in (encoder, decoder) for module in model.modules()}
    try:
        encoder.eval()
        decoder.eval()
        memory, _ = encoder(source_ids)
        for _ in range(max_new_tokens):
            logits, _ = decoder(generated, memory, source_padding_mask=source_mask)
            scores = logits[-1].clone()  # 最后一个位置预测下一个 token。
            scores[bos_id] = -torch.inf
            if target_pad is not None:
                scores[target_pad] = -torch.inf
            next_token = scores.argmax(dim=-1).reshape(1)
            generated = torch.cat((generated, next_token))
            if next_token.item() == eos_id:
                break
        return generated
    finally:
        for module, training in modes.items():
            module.training = training

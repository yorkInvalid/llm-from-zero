"""小型 Qwen 结构的预训练实验。真实反向传播；没有加载 Qwen3-0.6B 权重。"""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import Qwen3Config, Qwen3ForCausalLM

HERE = Path(__file__).resolve().parent
PAD, BOS, EOS = 0, 1, 2


def load_assets():
    corpus = json.loads((HERE/'assets/corpus.json').read_text())
    tokenizer = Tokenizer.from_file(str(HERE/'assets/tokenizer.json'))
    return corpus, tokenizer


def new_model(vocab_size, seed=42):
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=vocab_size, hidden_size=64, intermediate_size=176,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, max_position_embeddings=256,
                         attention_dropout=0., tie_word_embeddings=True,
                         pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS,
                         use_cache=False)
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config)


def windows(paragraphs, tokenizer, length=64):
    """每块带一个额外目标 token，相邻块仅共享边界 token，预测目标不重复。"""
    if length < 1:
        raise ValueError('length 必须为正')
    result = []
    for paragraph in paragraphs:
        ids = [BOS, *tokenizer.encode(paragraph).ids, EOS]
        for start in range(0, len(ids)-1, length):
            result.append(torch.tensor(ids[start:start+length+1], dtype=torch.long))
    return result


def collate(samples):
    ids = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=PAD)
    mask = ids.ne(PAD).long()
    labels = ids.clone()  # 不提前移位！HF 的 CausalLM loss 内部会处理。
    labels[mask == 0] = -100
    return dict(input_ids=ids, attention_mask=mask, labels=labels)


def batch_for(samples, generator, batch_size=8, device='cpu'):
    indices = torch.randint(len(samples), (batch_size,), generator=generator).tolist()
    return {k: v.to(device) for k,v in collate([samples[i] for i in indices]).items()}


def manual_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def evaluate(model, samples, batch_size=16):
    was_training = model.training
    model.eval()
    total, count = 0., 0
    device = next(model.parameters()).device
    try:
        for start in range(0, len(samples), batch_size):
            batch = {k: v.to(device) for k,v in collate(samples[start:start+batch_size]).items()}
            out = model(**batch, use_cache=False)
            n = batch['labels'][:, 1:].ne(-100).sum().item()
            total += out.loss.item()*n
            count += n
    finally:
        model.train(was_training)
    loss = total/count
    return dict(loss=loss, perplexity=math.exp(loss), target_tokens=count)


def train_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch, use_cache=False)
    output.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return output.loss.item(), float(grad_norm)


def train_steps(model, optimizer, samples, validation, generator, steps, start=0, every=25):
    history = []
    device = next(model.parameters()).device
    for step in range(start+1, start+steps+1):
        loss, grad = train_step(model, optimizer, batch_for(samples, generator, device=device))
        if step % every == 0 or step == start+steps:
            val = evaluate(model, validation)
            row = dict(step=step, batch_loss=loss, validation_loss=val['loss'], gradient_norm=grad)
            history.append(row)
            print(f"step={step:3d}  当前batch loss={loss:.4f}  固定验证集loss={val['loss']:.4f}", flush=True)
    return history


def tokenizer_hash():
    return hashlib.sha256((HERE/'assets/tokenizer.json').read_bytes()).hexdigest()


def save_resume(path, model, optimizer, generator, step):
    """精确恢复实验限定 CPU、单进程、dropout=0、固定语料与常数学习率。"""
    torch.save(dict(config=model.config.to_dict(), model=model.state_dict(),
                    optimizer=optimizer.state_dict(), generator=generator.get_state(),
                    torch_rng=torch.get_rng_state(), step=step,
                    tokenizer_sha256=tokenizer_hash()), path)


def load_resume(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['tokenizer_sha256'] != tokenizer_hash():
        raise ValueError('分词器不一致，不能直接复用 Embedding/LM head')
    config = Qwen3Config.from_dict(saved['config'])
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config)
    model.load_state_dict(saved['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(saved['optimizer'])
    generator = torch.Generator().set_state(saved['generator'])
    torch.set_rng_state(saved['torch_rng'])
    return model, optimizer, generator, saved['step']


@torch.no_grad()
def completion(model, tokenizer, prompt='Alice was', tokens=32):
    was_training = model.training
    model.eval()
    ids = torch.tensor([[BOS, *tokenizer.encode(prompt).ids]], device=next(model.parameters()).device)
    try:
        output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=tokens,
                                do_sample=False, use_cache=True, repetition_penalty=1.0)
        return tokenizer.decode(output[0].tolist())
    finally:
        model.train(was_training)

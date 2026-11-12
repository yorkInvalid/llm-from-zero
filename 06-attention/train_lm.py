"""可观察的书籍预训练：prepare / inspect / train / test / generate。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from causal_lm import CausalLM
from text_pretraining import prepare, load_data, collate, BOS
from lm_report import write_observation_report
from lm_components import learning_rate_at_step

ROOT = Path(__file__).parent / 'artifacts/book-lm'
OBSERVATION_PROMPTS = ('Alice was', 'The rabbit', 'She said')


@torch.no_grad()
def evaluate(model, samples, batch_size=16):
    modes = {m: m.training for m in model.modules()}
    device = next(model.parameters()).device
    loss_sum, correct, count = 0., 0, 0
    try:
        model.eval()
        for x, y in DataLoader(samples, batch_size=batch_size, collate_fn=collate):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                    y.reshape(-1), ignore_index=-100, reduction='sum').item()
            valid = y != -100
            count += valid.sum().item()
            correct += ((logits.argmax(-1) == y) & valid).sum().item()
    finally:
        for m, mode in modes.items():
            m.training = mode
    loss = loss_sum / count
    return dict(loss=loss, perplexity=math.exp(min(loss, 700)), token_accuracy=correct / count, tokens=count)


def inspect_batch(tokenizer, samples, model=None):
    x, y = collate(samples[:2])
    report = dict(input_shape=list(x.shape), label_shape=list(y.shape),
                  input_ids=x.tolist(), labels=y.tolist(),
                  input_text=tokenizer.decode(x[0].tolist()), rows=[])
    logits = None
    if model is not None:
        with torch.no_grad():
            logits = model(x.to(next(model.parameters()).device)).cpu()
        report['logits_shape'] = list(logits.shape)
    for position, (input_id, label_id) in enumerate(zip(x[0].tolist(), y[0].tolist())):
        row = dict(position=position, input_id=input_id, input_token=tokenizer.id_to_token(input_id),
                   label_id=label_id, label_token=tokenizer.id_to_token(label_id) if label_id >= 0 else 'IGNORED')
        if logits is not None:
            distribution = logits[0, position].softmax(-1)
            probabilities, indices = distribution.topk(min(3, distribution.numel()))
            row['top3'] = [dict(id=i, token=tokenizer.id_to_token(i), probability=p)
                           for i, p in zip(indices.tolist(), probabilities.tolist())]
            row['label_probability'] = distribution[label_id].item() if label_id >= 0 else None
            row['label_in_top3'] = label_id in indices.tolist() if label_id >= 0 else None
        report['rows'].append(row)
    return report


@torch.no_grad()
def observe_epoch(model, tokenizer, samples, epoch, max_new_tokens=32):
    """固定验证窗口和提示词，关闭 Dropout；不更新参数或消耗训练随机数。"""
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        predictions = inspect_batch(tokenizer, samples, model)
        generations = []
        for prompt in OBSERVATION_PROMPTS:
            prefix = torch.tensor([BOS, *tokenizer.encode(prompt).ids],
                                  device=next(model.parameters()).device)
            output = model.generate(prefix, max_new_tokens=max_new_tokens).tolist()
            generations.append(dict(prompt=prompt, output_ids=output,
                                    text=tokenizer.decode(output)))
        return dict(epoch=epoch, predictions=predictions, generations=generations)
    finally:
        for module, mode in modes.items():
            module.training = mode


def restore(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = dict(norm_style='post', norm_type='layer', position_encoding='sinusoidal', ffn_type='relu')
    config.update(saved['config'])  # 旧 checkpoint 没有这些字段时按原结构重建。
    model = CausalLM(**config).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    return model, Tokenizer.from_str(saved['tokenizer']), saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'inspect', 'train', 'test', 'generate'])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'run')
    parser.add_argument('--text-file', type=Path)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--context-length', type=int, default=64)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--prompt', default='Alice was')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--norm-style', choices=['pre', 'post'], default='pre')
    parser.add_argument('--norm-type', choices=['rms', 'layer'], default='rms')
    parser.add_argument('--position-encoding', choices=['rope', 'sinusoidal'], default='rope')
    parser.add_argument('--ffn-type', choices=['swiglu', 'relu'], default='swiglu')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--warmup-ratio', type=float, default=.1)
    parser.add_argument('--min-lr-ratio', type=float, default=.1)
    parser.add_argument('--lr-schedule', choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--kv-cache', action=argparse.BooleanOptionalAction, default=True,
                        help='生成时使用 KV cache；--no-kv-cache 用于与完整重算对照')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.context_length < 1 or args.max_new_tokens < 0:
        parser.error('epochs/batch/context 必须为正，生成长度不能为负')
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1 or args.log_every < 1:
        parser.error('请检查学习率、warmup/min-lr 比例和日志间隔')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if args.command == 'prepare':
        print(json.dumps(prepare(args.data_dir, args.text_file), ensure_ascii=False, indent=2))
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.command in ('test', 'generate'):
        model, tokenizer, saved = restore(args.run_dir / 'best.pt', args.device)
        if args.command == 'generate':
            prefix = torch.tensor([BOS, *tokenizer.encode(args.prompt).ids], device=args.device)
            result = model.generate(prefix, max_new_tokens=args.max_new_tokens, use_cache=args.kv_cache).tolist()
            print(json.dumps(dict(prompt=args.prompt, prompt_ids=prefix.tolist(), output_ids=result,
                                  text=tokenizer.decode(result)), ensure_ascii=False, indent=2))
        else:
            current_hash = hashlib.sha256((args.data_dir / 'tokenizer.json').read_bytes()).hexdigest()
            if current_hash != saved['tokenizer_sha256']:
                raise ValueError('测试数据分词器与 checkpoint 不一致')
            _, splits = load_data(args.data_dir, model.config['context_length'], ('test',))
            metrics = evaluate(model, splits['test'], args.batch_size)
            (args.run_dir / 'test.json').write_text(json.dumps(metrics, indent=2))
            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter(str(args.run_dir / 'tensorboard')) as writer:
                for name, value in metrics.items():
                    writer.add_scalar(f'test/{name}', value, saved.get('global_step', saved['epoch']))
            print(json.dumps(metrics, indent=2))
        return
    tokenizer, splits = load_data(args.data_dir, args.context_length)
    if args.command == 'inspect':
        report = inspect_batch(tokenizer, splits['train'])
        report['first_original_paragraph'] = json.loads((args.data_dir / 'train.json').read_text())[0]
        print(json.dumps(report, ensure_ascii=False, indent=2))
        (args.run_dir / 'inspect.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return
    model = CausalLM(tokenizer.get_vocab_size(), context_length=args.context_length,
                     norm_style=args.norm_style, norm_type=args.norm_type,
                     position_encoding=args.position_encoding, ffn_type=args.ffn_type).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    total_steps = args.epochs * len(loader)
    warmup_steps = min(total_steps - 1, int(total_steps * args.warmup_ratio))
    training_config = dict(peak_lr=args.learning_rate, warmup_steps=warmup_steps, total_steps=total_steps,
                           min_lr_ratio=args.min_lr_ratio, schedule=args.lr_schedule,
                           batch_size=args.batch_size, seed=42, device=args.device)
    from training_monitor import TrainingMonitor
    with TrainingMonitor(args.run_dir / 'tensorboard', dict(model=model.config, training=training_config)) as monitor:
        history = [dict(epoch=0, train=evaluate(model, splits['train'], args.batch_size),
                        validation=evaluate(model, splits['validation'], args.batch_size))]
        print(json.dumps(history[0]), flush=True)
        observations = [observe_epoch(model, tokenizer, splits['validation'], 0)]
        monitor.epoch(0, history[0], observations[0])
        global_step = 0
        window_start, window_tokens = time.monotonic(), 0
        best = float('inf')
        tokenizer_json = (args.data_dir / 'tokenizer.json').read_bytes()
        start = time.monotonic()
        for epoch in range(1, args.epochs + 1):
            model.train()
            for step, (x, y) in enumerate(loader):
                global_step += 1
                lr = (learning_rate_at_step(global_step, total_steps, warmup_steps, args.learning_rate, args.min_lr_ratio)
                      if args.lr_schedule == 'cosine' else args.learning_rate)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
                before = model.embedding.weight.detach().clone() if epoch == 1 and step == 0 else None
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
                gradient = model.embedding.weight.grad.norm().item() if before is not None else None
                optimizer.step()
                window_tokens += (y != -100).sum().item()
                if global_step == 1 or global_step % args.log_every == 0 or step == len(loader) - 1:
                    monitor.step(global_step, loss.item(), lr, gradient_norm.item(),
                                 window_tokens / max(time.monotonic() - window_start, 1e-9))
                    window_start, window_tokens = time.monotonic(), 0
                if before is not None:
                    print('FIRST STEP:', json.dumps(dict(input_shape=list(x.shape), logits_shape=list(logits.shape),
                          loss=loss.item(), embedding_gradient_norm=gradient,
                          embedding_update_norm=(model.embedding.weight.detach() - before).norm().item())), flush=True)
            metrics = dict(epoch=epoch, train=evaluate(model, splits['train'], args.batch_size),
                           validation=evaluate(model, splits['validation'], args.batch_size))
            history.append(metrics)
            observations.append(observe_epoch(model, tokenizer, splits['validation'], epoch))
            monitor.epoch(global_step, metrics, observations[-1])
            (args.run_dir / 'observations.json').write_text(
                json.dumps(dict(history=history, observations=observations), ensure_ascii=False, indent=2))
            print(json.dumps(metrics), flush=True)
            if metrics['validation']['loss'] < best:
                best = metrics['validation']['loss']
                torch.save(dict(config=model.config, model=model.state_dict(), tokenizer=tokenizer_json.decode(),
                                tokenizer_sha256=hashlib.sha256(tokenizer_json).hexdigest(), epoch=epoch,
                                global_step=global_step, training_config=training_config),
                           args.run_dir / 'best.pt')
            window_start, window_tokens = time.monotonic(), 0  # 吞吐量不计入每轮验证时间。
        model, _, saved = restore(args.run_dir / 'best.pt', args.device)
        report = dict(history=history, best_epoch=saved['epoch'], seconds=time.monotonic() - start,
                      parameters=sum(p.numel() for p in model.parameters()), device=args.device,
                      model_config=model.config, training_config=training_config)
        (args.run_dir / 'report.json').write_text(json.dumps(report, indent=2))
        inspection = inspect_batch(tokenizer, splits['validation'], model)
        (args.run_dir / 'predictions.json').write_text(json.dumps(inspection, ensure_ascii=False, indent=2))
        write_observation_report(args.run_dir, history, observations, saved['epoch'])
        print(f'最佳模型与报告已保存至 {args.run_dir}；test 命令单独评估未参与选模型的测试集。')


if __name__ == '__main__':
    main()

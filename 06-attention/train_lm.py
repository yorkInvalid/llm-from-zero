"""可观察的书籍预训练：prepare / inspect / train / test / generate。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import platform
from datetime import datetime, timezone

import torch
from torch import nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from causal_lm import CausalLM
from text_pretraining import prepare, load_data, collate, BOS
from lm_report import write_observation_report
from lm_components import learning_rate_at_step
from training_progress import TrainingProgress

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
def observe_epoch(model, tokenizer, samples, epoch, max_new_tokens=32, prompts=OBSERVATION_PROMPTS):
    """固定验证窗口和提示词，关闭 Dropout；不更新参数或消耗训练随机数。"""
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        predictions = inspect_batch(tokenizer, samples, model)
        generations = []
        for prompt in prompts:
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
    config = dict(norm_style='post', norm_type='layer', position_encoding='sinusoidal', ffn_type='relu', attention_init='legacy')
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
    parser.add_argument('--run-label', default='', help='TensorBoard 中显示的实验名称')
    parser.add_argument('--run-note', default='', help='本次训练目的与变化说明')
    parser.add_argument('--init-checkpoint', type=Path, help='从权重继续训练；重新初始化优化器和学习率计划')
    parser.add_argument('--resume-checkpoint', type=Path, help='完整续训；保持原训练计划和数据')
    parser.add_argument('--keep-optimizer-state', action='store_true',
                        help='init-checkpoint 使用完整断点，保留 Adam/GradScaler，开始新的学习率计划')
    parser.add_argument('--full-train-eval', action=argparse.BooleanOptionalAction, default=True,
                        help='默认全训练集评估；关闭后只评估固定训练探针，标签明确为 train_probe')
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--train-probe-windows', type=int, default=512)
    parser.add_argument('--stop-after-step', type=int, default=0, help='在指定全局步数保存并暂停，0 表示完整运行')
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=None,
                        help='新模型默认0.1；init-checkpoint可覆盖；完整resume保持断点值')
    parser.add_argument('--attention-init', choices=['legacy', 'fan_in'], default='fan_in',
                        help='新训练默认 fan_in；legacy 用于复现历史实验；加载 checkpoint 时沿用保存的结构和权重')
    parser.add_argument('--observation-prompts', nargs='+', default=list(OBSERVATION_PROMPTS))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--context-length', type=int, default=64)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--precision', choices=['float32', 'fp16'], default='float32')
    parser.add_argument('--prompt', default='Alice was')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--norm-style', choices=['pre', 'post'], default='pre')
    parser.add_argument('--norm-type', choices=['rms', 'layer'], default='rms')
    parser.add_argument('--position-encoding', choices=['rope', 'sinusoidal'], default='rope')
    parser.add_argument('--ffn-type', choices=['swiglu', 'relu'], default='swiglu')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--warmup-ratio', type=float, default=.1)
    parser.add_argument('--min-lr-ratio', type=float, default=.1)
    parser.add_argument('--lr-schedule', choices=['cosine', 'constant', 'wsd'], default='cosine')
    parser.add_argument('--cooldown-ratio', type=float, default=.2,
                        help='WSD最后多少比例的更新从peak余弦衰减至min-lr-ratio')
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default=None,
                        help='新优化器默认adam；复用状态时默认继承断点配方')
    parser.add_argument('--beta1', type=float, default=None)
    parser.add_argument('--beta2', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--kv-cache', action=argparse.BooleanOptionalAction, default=True,
                        help='生成时使用 KV cache；--no-kv-cache 用于与完整重算对照')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.context_length < 1 or args.max_new_tokens < 0:
        parser.error('epochs/batch/context 必须为正，生成长度不能为负')
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1 or args.log_every < 1:
        parser.error('请检查学习率、warmup/min-lr 比例和日志间隔')
    if args.dropout is not None and not 0 <= args.dropout < 1:
        parser.error('dropout须在[0,1)内')
    if any(value is not None and not 0 <= value < 1 for value in (args.beta1, args.beta2)):
        parser.error('Adam betas须在[0,1)内')
    if args.weight_decay is not None and (not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error('weight-decay须为有限非负值')
    if args.lr_schedule == 'wsd' and not 0 < args.cooldown_ratio <= 1:
        parser.error('WSD cooldown-ratio须在(0,1]内')
    if min(args.d_model, args.num_layers, args.num_heads) < 1 or args.d_model % args.num_heads:
        parser.error('模型维度、层数和头数必须为正，模型维度须整除头数')
    if args.precision == 'fp16' and args.device != 'cuda':
        parser.error('fp16 训练仅支持 CUDA，其他设备请使用 float32')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA 不可用；不会退回 CPU')
    if args.command == 'train' and args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('run-dir 非空，请给每次训练使用新的目录')
    if args.init_checkpoint and args.resume_checkpoint:
        parser.error('--init-checkpoint 和 --resume-checkpoint 不能同时使用')
    if args.keep_optimizer_state and not args.init_checkpoint:
        parser.error('--keep-optimizer-state 需要 --init-checkpoint 指向完整断点')
    if args.eval_every < 1 or args.train_probe_windows < 1 or args.stop_after_step < 0:
        parser.error('检查验证间隔、训练探针数和暂停步数')
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
        if (args.data_dir / 'train.json').exists():
            report['first_original_paragraph'] = json.loads((args.data_dir / 'train.json').read_text())[0]
        else:
            import gzip
            with gzip.open(args.data_dir / 'training_documents.jsonl.gz', 'rt') as stream:
                report['first_original_paragraph'] = json.loads(next(stream))['text']
        print(json.dumps(report, ensure_ascii=False, indent=2))
        (args.run_dir / 'inspect.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return
    model = CausalLM(tokenizer.get_vocab_size(), context_length=args.context_length,
                     d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, attention_init=args.attention_init,
                     dropout=.1 if args.dropout is None else args.dropout,
                     norm_style=args.norm_style, norm_type=args.norm_type,
                     position_encoding=args.position_encoding, ffn_type=args.ffn_type).to(args.device)
    resumed = None
    starting_checkpoint = args.resume_checkpoint or args.init_checkpoint
    if starting_checkpoint:
        model, _, initial = restore(starting_checkpoint, args.device)
        if args.resume_checkpoint: resumed = initial
        current_hash = hashlib.sha256((args.data_dir / 'tokenizer.json').read_bytes()).hexdigest()
        if current_hash != initial['tokenizer_sha256'] or args.context_length != model.config['context_length']:
            raise ValueError('继续训练要求相同 tokenizer 和上下文长度')
        if args.resume_checkpoint and args.dropout is not None and args.dropout != model.config['dropout']:
            raise ValueError('完整resume要求相同dropout；修改配方请使用init-checkpoint')
    if args.dropout is not None:
        model.config['dropout'] = args.dropout
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = args.dropout
            if hasattr(module, 'dropout_p'):
                module.dropout_p = args.dropout
    from training_state import checkpoint_optimizer_config
    preserving_optimizer = args.keep_optimizer_state or args.resume_checkpoint is not None
    optimizer_config = (checkpoint_optimizer_config(initial) if preserving_optimizer else
                        dict(optimizer='adam', betas=[.9, .999], weight_decay=0., optimizer_eps=1e-8))
    requested_recipe = dict(optimizer=args.optimizer,
                            betas=[optimizer_config['betas'][0] if args.beta1 is None else args.beta1,
                                   optimizer_config['betas'][1] if args.beta2 is None else args.beta2],
                            weight_decay=args.weight_decay)
    for key, value in requested_recipe.items():
        if value is None:
            continue
        if preserving_optimizer and value != optimizer_config[key]:
            raise ValueError(f'复用优化器状态要求相同{key}；修改配方请重置优化器')
        optimizer_config[key] = value
    optimizer_type = torch.optim.AdamW if optimizer_config['optimizer'] == 'adamw' else torch.optim.Adam
    optimizer = optimizer_type(model.parameters(), lr=args.learning_rate, betas=tuple(optimizer_config['betas']),
                               weight_decay=optimizer_config['weight_decay'], eps=optimizer_config['optimizer_eps'])
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16', init_scale=1024.)
    if args.keep_optimizer_state:
        from training_state import reuse_optimizer_state
        reuse_optimizer_state(optimizer, scaler, initial, args.precision)
    loader = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    total_steps = args.epochs * len(loader)
    warmup_steps = min(total_steps - 1, int(total_steps * args.warmup_ratio))
    cooldown_steps = max(1, int(total_steps * args.cooldown_ratio)) if args.lr_schedule == 'wsd' else 0
    if cooldown_steps > total_steps - warmup_steps:
        raise ValueError('WSD warmup与cooldown更新区间重叠')
    from indexed_tokens import dataset_counts, split_fingerprint
    counts = {name: dataset_counts(rows) for name, rows in splits.items()}
    training_config = dict(peak_lr=args.learning_rate, warmup_steps=warmup_steps, total_steps=total_steps,
                           min_lr_ratio=args.min_lr_ratio, schedule=args.lr_schedule, cooldown_steps=cooldown_steps,
                           dropout=model.config['dropout'], **optimizer_config,
                           batch_size=args.batch_size, seed=42, device=args.device, precision=args.precision, evaluation_precision='float32',
                           observation_prompts=args.observation_prompts,
                           eval_every=args.eval_every, train_probe_windows=args.train_probe_windows,
                           full_train_eval=args.full_train_eval,
                           train_token_budget=counts['train']['tokens'] * args.epochs,
                           keep_optimizer_state=args.keep_optimizer_state,
                           init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None)
    source_path = args.data_dir / 'source.json'
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    source.pop('document_hashes', None)
    split_hashes = {name:split_fingerprint(args.data_dir, name) for name in ('train','validation')}
    checkpoint_hash = (hashlib.sha256(starting_checkpoint.read_bytes()).hexdigest()
                       if starting_checkpoint else None)
    metadata = dict(label=args.run_label or args.run_dir.name, note=args.run_note,
                    started_at_utc=datetime.now(timezone.utc).isoformat(),
                    training_start=('full_resume' if args.resume_checkpoint else
                                    'continue_weights_and_optimizer_new_schedule' if args.keep_optimizer_state else
                                    'continue_weights_reset_optimizer' if args.init_checkpoint else 'random_initialization'),
                    checkpoint_sha256=checkpoint_hash, model=model.config, training=training_config,
                    data=source, split_sha256=split_hashes,
                    tokenizer_sha256=hashlib.sha256((args.data_dir/'tokenizer.json').read_bytes()).hexdigest(),
                    counts=counts,
                    hardware=dict(device=args.device, gpu=torch.cuda.get_device_name() if args.device=='cuda' else platform.machine(),
                                  torch=str(torch.__version__), cuda=torch.version.cuda, precision=args.precision, evaluation_precision='float32'),
                    code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    from training_loop import train
    train(args,model,tokenizer,splits,optimizer,scaler,training_config,metadata,resumed)


if __name__ == '__main__':
    main()

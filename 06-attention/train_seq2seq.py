"""从 ID 到自回归生成的倒序任务；可直接在 CPU 运行。"""

import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn

from sequence_task import PAD_ID, BOS_ID, EOS_ID, VOCAB_SIZE, make_splits, collate_sequences
from transformer import Transformer


def build_model(d_model=32, num_layers=2, dropout=0.0):
    if d_model < 2 or d_model % 2:
        raise ValueError("d_model 必须是正偶数")
    return Transformer(VOCAB_SIZE, VOCAB_SIZE, d_model=d_model,
                       num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                       num_heads=2, d_k=d_model // 2, d_v=d_model // 2, d_ff=d_model * 4,
                       source_padding_idx=PAD_ID, target_padding_idx=PAD_ID, dropout=dropout)


def load_model(path, device='cpu'):
    """加载统一 checkpoint，也兼容此前分开保存的 Encoder/Decoder 权重。"""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if 'model' in checkpoint:
        model = Transformer(**checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['model'])
    else:
        model = build_model(**checkpoint['model_config']).to(device)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, samples, batch_size=64, generate=False, max_new_tokens=8):
    if not samples or batch_size < 1:
        raise ValueError("验证数据不能为空，batch_size 须为正")
    modes = {m: m.training for m in model.modules()}
    device = next(model.parameters()).device
    total_loss, correct, count = 0., 0, 0
    examples, exact, ended = [], 0, 0
    try:
        model.eval()
        for start in range(0, len(samples), batch_size):
            source, target, labels = collate_sequences(samples[start:start + batch_size], device)
            logits = model(source, target)
            total_loss += nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1),
                ignore_index=PAD_ID, reduction='sum').item()
            valid = labels != PAD_ID
            count += valid.sum().item()
            correct += ((logits.argmax(-1) == labels) & valid).sum().item()
        if generate:
            for sample in samples:
                predicted = model.generate(torch.tensor(sample, device=device),
                                            BOS_ID, EOS_ID, max_new_tokens).tolist()[1:]
                expected = [*reversed(sample), EOS_ID]
                exact += predicted == expected
                ended += predicted[-1:] == [EOS_ID]
                if len(examples) < 8:
                    examples.append(dict(source=list(sample), expected=expected, predicted=predicted))
    finally:
        for module, training in modes.items():
            module.training = training
    metrics = dict(loss=total_loss / count, token_accuracy=correct / count)
    if generate:
        metrics.update(exact_match=exact / len(samples), eos_rate=ended / len(samples), examples=examples)
    return metrics


def train_model(model, train_data, validation_data, epochs=40,
                batch_size=64, learning_rate=.001, seed=42):
    if epochs < 0 or batch_size < 1 or learning_rate <= 0 or not train_data:
        raise ValueError("请检查训练轮数、batch 大小、学习率和数据")
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    device = next(model.parameters()).device
    rng, history = random.Random(seed), []
    for epoch in range(epochs + 1):
        if epoch:
            order = list(train_data)
            rng.shuffle(order)
            model.train()
            for start in range(0, len(order), batch_size):
                source, target, labels = collate_sequences(order[start:start + batch_size], device)
                optimizer.zero_grad()
                logits = model(source, target)
                loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                                   labels.reshape(-1), ignore_index=PAD_ID)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
        metrics = dict(epoch=epoch, train=evaluate(model, train_data, batch_size),
                       validation=evaluate(model, validation_data, batch_size))
        history.append(metrics)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"epoch={epoch:3d} train_loss={metrics['train']['loss']:.4f} "
                  f"val_loss={metrics['validation']['loss']:.4f} "
                  f"val_token_acc={metrics['validation']['token_accuracy']:.1%}", flush=True)
    model.eval()
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--train-size', type=int, default=1024)
    parser.add_argument('--validation-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'artifacts/seq2seq')
    args = parser.parse_args()
    if args.epochs < 0 or args.batch_size < 1:
        parser.error('epochs 须非负，batch-size 须为正')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train, validation = make_splits(args.train_size, args.validation_size, args.seed)
    model_config = dict(d_model=32, num_layers=2, dropout=0.0)
    model = build_model(**model_config)
    history = train_model(model, train, validation, args.epochs, args.batch_size, seed=args.seed)
    generated = evaluate(model, validation, args.batch_size, generate=True)
    report = dict(config={**vars(args), 'output_dir': str(args.output_dir),
                          'model': model_config, 'learning_rate': .001,
                          'min_length': 2, 'max_length': 6, 'max_new_tokens': 8},
                  split_rule='unique source sequences; no overlap between train and validation',
                  history=history, generation=generated)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    torch.save(dict(model=model.state_dict(), model_config=model.config,
                    vocab_size=VOCAB_SIZE, pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID),
               args.output_dir / 'model.pt')
    print(f"生成整句准确率={generated['exact_match']:.1%} EOS率={generated['eos_rate']:.1%}")
    for example in generated['examples']:
        print(example)
    print(f"报告和模型已保存至 {args.output_dir}")


if __name__ == '__main__':
    main()

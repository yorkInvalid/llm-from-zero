"""训练 Encoder 与无 Attention 基线，比较未见过序列上的重复识别能力。"""

import argparse
import json
from pathlib import Path
import random

import torch

from context_task import (VOCAB_SIZE, PAD_ID, TokenwiseBaseline,
                          collate_sequences, evaluate, make_splits)
from token_encoder import TokenClassifier
from train_tokens import token_classification_loss


def train_model(model, train_data, validation_data, epochs=30, batch_size=64,
                learning_rate=0.003, seed=42):
    if epochs < 0 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs 不能为负，batch_size 和 learning_rate 必须为正")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    history = []
    for epoch in range(epochs + 1):
        if epoch:
            order = list(range(len(train_data)))
            rng.shuffle(order)
            model.train()
            for start in range(0, len(order), batch_size):
                ids, labels = collate_sequences([train_data[i] for i in order[start:start + batch_size]])
                optimizer.zero_grad()
                logits, _ = model(ids)
                loss = token_classification_loss(logits, labels)
                loss.backward()
                optimizer.step()
        metrics = {"epoch": epoch,
                   "train": evaluate(model, train_data, batch_size),
                   "validation": evaluate(model, validation_data, batch_size)}
        history.append(metrics)
        if epoch % 5 == 0 or epoch == epochs:
            train, validation = metrics['train'], metrics['validation']
            print(f"epoch={epoch:2d} train_loss={train['loss']:.4f} "
                  f"val_loss={validation['loss']:.4f} "
                  f"train_acc={train['accuracy']:.1%} val_acc={validation['accuracy']:.1%}")
    model.eval()
    return history


def run_experiment(epochs=30, train_size=1024, validation_size=256,
                   batch_size=64, seed=42, dropout=0.1):
    train_data, validation_data = make_splits(train_size, validation_size, seed)
    report = {"config": {"epochs": epochs, "train_size": train_size,
                         "validation_size": validation_size, "batch_size": batch_size,
                         "seed": seed, "dropout": dropout, "position_encoding": "none",
                         "d_model": 32, "num_layers": 3, "learning_rate": .003},
              "split_rule": "disjoint token multisets; length 3..8; token IDs 0..7",
              "models": {}}
    for name in ('encoder', 'tokenwise_baseline'):
        torch.manual_seed(seed)
        if name == 'encoder':
            model = TokenClassifier(VOCAB_SIZE + 1, num_classes=2, d_model=32,
                                    num_layers=3, padding_idx=PAD_ID,
                                    position_encoding='none', dropout=dropout)
        else:
            model = TokenwiseBaseline(dropout=dropout)
        print(f"\n{name}（不使用位置编码；同一数据划分、训练步数和优化器设置）")
        history = train_model(model, train_data, validation_data, epochs, batch_size, seed=seed)
        report['models'][name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "history": history,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="依赖上下文的重复 token 识别实验")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train-size', type=int, default=1024)
    parser.add_argument('--validation-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'artifacts/context-task-report.json')
    args = parser.parse_args()
    if (args.epochs < 0 or args.train_size < 1 or args.validation_size < 1
            or args.train_size + args.validation_size > 5000 or args.batch_size < 1
            or not 0 <= args.dropout <= 1):
        parser.error('请检查轮数、数据量（总量不超过 5000）、batch 大小和 dropout 概率')
    # 小矩阵实验减少线程调度开销；仅 CLI 设置，不影响导入方。
    torch.set_num_threads(1)
    report = run_experiment(args.epochs, args.train_size, args.validation_size,
                            args.batch_size, args.seed, args.dropout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"\n完整训练/验证记录：{args.output}")


if __name__ == '__main__':
    main()

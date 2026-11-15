"""Evaluate a training-only add-one unigram baseline on validation, never test."""
import argparse
import json
import math
from pathlib import Path
import torch
from text_pretraining import load_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    args = p.parse_args()
    tokenizer, splits = load_data(args.data_dir, 256)
    counts = torch.ones(tokenizer.get_vocab_size(), dtype=torch.float64)
    for _, y in splits['train']:
        counts += torch.bincount(y, minlength=len(counts))
    log_prob = (counts / counts.sum()).log()
    report = {'smoothing': 'add-one', 'fit_split': 'train', 'uniform_loss': math.log(len(counts))}
    for name, samples in splits.items():
        n = sum(y.numel() for _, y in samples)
        loss = sum(-log_prob[y].sum().item() for _, y in samples) / n
        report[name] = dict(loss=loss, perplexity=math.exp(loss), tokens=n,
                            token_accuracy=sum((y == counts.argmax()).sum().item() for _, y in samples)/n)
    (args.data_dir / 'unigram-baseline.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

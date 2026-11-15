"""Compare starting and selected weights on the frozen legacy validation set."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from text_pretraining import load_data
from train_lm import evaluate, observe_epoch, restore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','mps','cuda'),default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(1)
    report=json.loads((args.run_dir/'report.json').read_text())
    context=report['model_config']['context_length']
    _,splits=load_data(args.data_dir,context,('validation_legacy',))
    tokenizer_hash=hashlib.sha256((args.data_dir/'tokenizer.json').read_bytes()).hexdigest()
    results={};observations={}
    for label,path in [('before',args.before),('after',args.run_dir/'best.pt')]:
        model,tokenizer,saved=restore(path,args.device)
        if saved['tokenizer_sha256']!=tokenizer_hash or model.config['context_length']!=context:
            raise ValueError('对照模型的 tokenizer 或上下文长度不一致')
        results[label]=evaluate(model,splits['validation_legacy'],32)
        observations[label]=observe_epoch(model,tokenizer,splits['validation_legacy'],saved['epoch'],
            prompts=report['training_config']['observation_prompts'])['generations']
        del model,saved
        if args.device=='cuda':torch.cuda.empty_cache()
    (args.run_dir/'legacy-validation-comparison.json').write_text(json.dumps(results,indent=2))
    (args.run_dir/'generation-comparison.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2))
    best_step=report['best_step']
    with SummaryWriter(str(args.run_dir/'tensorboard')) as writer:
        for name,result in results.items():
            for metric,value in result.items():
                writer.add_scalar(f'legacy_validation/{metric}',value,0 if name=='before' else best_step)
        writer.add_text('run/legacy_comparison',json.dumps(results,indent=2),best_step)
        writer.add_text('run/generation_comparison',json.dumps(observations,ensure_ascii=False,indent=2),best_step)
    print(json.dumps(dict(legacy_validation=results,generations=observations),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

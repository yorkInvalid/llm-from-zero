"""Compare completed historical and fan-in runs by token exposure, not optimizer steps.

The historical run reset Adam/schedule after epoch 1 and changed batch size.
This is an effectiveness comparison, not a single-variable initialization ablation.
"""
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

ROOT=Path(__file__).parent/'artifacts/book-lm'


def main():
    old=json.loads((ROOT/'industry-33m-run/report.json').read_text())
    continued=json.loads((ROOT/'industry-33m-continue/report.json').read_text())
    new=json.loads((ROOT/'industry-33m-fanin-run/report.json').read_text())
    histories={
        'legacy':old['history'] + [dict(row,epoch=row['epoch']+1) for row in continued['history'][1:]],
        'fan_in':new['history'],
    }
    summary={'caveat':__doc__, 'histories':histories}
    for name,rows in histories.items():
        with SummaryWriter(str(ROOT/'industry-comparison'/name)) as writer:
            for row in rows:
                seen=row['epoch']*row['train']['tokens']
                for split in ('train','validation'):
                    for metric in ('loss','perplexity','token_accuracy'):
                        writer.add_scalar(f'{split}/{metric}_by_tokens_seen',row[split][metric],seen)
            writer.add_text('comparison_method',__doc__,0)
        best=min(rows,key=lambda row:row['validation']['loss'])
        summary[name]=dict(best_epoch=best['epoch'],validation=best['validation'])
    (ROOT/'industry-comparison/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({name:summary[name] for name in histories},indent=2))


if __name__=='__main__':main()

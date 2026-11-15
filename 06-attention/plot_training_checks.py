"""Plot recorded validation/probe points without TensorBoard smoothing."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot(directory):
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text())
    checks=report['checks']
    token_axis=all('tokens_seen' in c for c in checks[1:])
    def x(check):
        return check.get('tokens_seen',0)/1e6 if token_axis else check['global_step']
    fig,ax=plt.subplots(figsize=(11,5.4),constrained_layout=True)
    values=[]
    for key,label,color in [('validation','Full validation','#087f9c'),
                            ('train_probe','Fixed training probe','#d88117')]:
        losses=[c[key]['loss'] for c in checks]
        values.extend(losses)
        ax.plot([x(c) for c in checks],losses,'-o',color=color,label=label,
                linewidth=2,markersize=4)
    if report.get('train_metric_scope','full_training_set')=='full_training_set':
        by_step={c['global_step']:x(c) for c in checks}
        history=report['history']
        losses=[c['train']['loss'] for c in history]
        values.extend(losses)
        ax.plot([by_step[c['global_step']] for c in history],losses,'--s',color='#707986',
                label='Full training set (epoch endpoints)',markersize=5)
    best=next(c for c in checks if c['global_step']==report['best_step'])
    ax.annotate(f"Best validation: {best['validation']['loss']:.3f}",
                xy=(x(best),best['validation']['loss']),
                xytext=((-175 if x(best)>x(checks[-1])*.3 else 40),65),textcoords='offset points',
                arrowprops=dict(arrowstyle='->',color='#087f9c'),color='#087f9c')
    span=max(max(values)-min(values),.1)
    ax.set_ylim(min(values)-.12*span,max(values)+.22*span)
    ax.set(title=f"Own {report['parameters']/1e6:.1f}M LM: recorded training progress",
           xlabel='Training tokens (millions)' if token_axis else 'Optimizer updates',
           ylabel='Cross-entropy loss')
    ax.text(.02,.97,f'{len(checks)} checkpoints; no smoothing',transform=ax.transAxes,
            va='top',color='#586270',fontsize=10)
    ax.grid(alpha=.18)
    ax.legend(loc='upper right',bbox_to_anchor=(1,.88),frameon=False)
    path=directory/'validation_progress.png'
    fig.savefig(path,dpi=150)
    plt.close(fig)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    print(plot(parser.parse_args().run_dir))

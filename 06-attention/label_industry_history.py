"""Add explicit retrospective descriptions without changing historical scalar curves."""
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
ROOT=Path(__file__).parent/'artifacts/book-lm'
LABELS={
 'industry-33m-run':('01 · Mac · 旧初始化 · 从零 1 遍','320 篇 AI 文档；batch=2；legacy 初始化。'),
 'industry-33m-continue':('02 · Mac · 旧权重追加 3 遍','同一小数据；batch=8；重置优化器；不是断点恢复。'),
 'industry-33m-fanin-run':('03 · Mac · 修正初始化 · 从零 4 遍','同一小数据；batch=8；fan_in 初始化；最佳验证 loss=6.6021。'),
 'industry-unigram':('基线 · 只按词频预测','不训练神经网络；只在训练集拟合频率。'),
 'progress-smoke-run':('工程检查 · 进度条短跑','极小数据和小模型；不可作为模型质量对照。'),
}

def main():
 for name,(label,note) in LABELS.items():
  p=ROOT/name
  if not p.exists():continue
  report=json.loads((p/'report.json').read_text()) if (p/'report.json').exists() else {}
  info=dict(label=label,note=note,annotation='retrospective; historical scalars unchanged',
            model=report.get('model_config'),training=report.get('training_config'),
            status='completed' if report else 'reference')
  (p/'run-description.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
  with SummaryWriter(str(p/'tensorboard')) as writer:
   writer.add_text('run/description',label+'\n\n'+note,0)
   writer.add_text('run/annotation',json.dumps(info,ensure_ascii=False,indent=2),0)
 print('Historical runs labeled')

if __name__=='__main__':main()

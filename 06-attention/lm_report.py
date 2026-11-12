"""离线训练观察报告：曲线、固定提示续写、逐位置真实标签与候选。"""

from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def write_observation_report(directory, history, observations, best_epoch):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    epochs = [item['epoch'] for item in history]
    for ax, metric, title in zip(axes, ('loss', 'perplexity', 'token_accuracy'),
                                ('Cross-entropy loss', 'Perplexity (log scale)', 'Token accuracy')):
        for split in ('train', 'validation'):
            ax.plot(epochs, [item[split][metric] for item in history], marker='o', label=split)
        ax.axvline(best_epoch, color='gray', linestyle=':', label='best checkpoint')
        if metric == 'perplexity':
            ax.set_yscale('log')
        if metric == 'token_accuracy':
            ax.set_ylim(0, 1)
        ax.set(title=title, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(directory / 'curves.png', dpi=150)
    plt.close(fig)
    sections = []
    for observation in observations:
        epoch = observation['epoch']
        metric = next(item for item in history if item['epoch'] == epoch)
        generations = ''.join(
            f'<tr><td>{escape(g["prompt"])}</td><td>{escape(g["text"])}</td></tr>'
            for g in observation['generations'])
        rows = []
        valid = [r for r in observation['predictions']['rows'] if r['label_id'] >= 0]
        hits = sum(r['label_in_top3'] for r in valid)
        for row in observation['predictions']['rows']:
            candidates = ' | '.join(f'{escape(c["token"])} ({c["probability"]:.1%})'
                                    for c in row['top3'])
            probability = '忽略' if row['label_probability'] is None else f'{row["label_probability"]:.1%}'
            mark = '✓' if row['label_in_top3'] else ('—' if row['label_id'] < 0 else '×')
            rows.append(f'<tr><td>{row["position"]}</td><td>{escape(row["input_token"])}</td>'
                        f'<td>{escape(row["label_token"])}</td><td>{probability}</td>'
                        f'<td>{candidates}</td><td>{mark}</td></tr>')
        sections.append(f'''<details {'open' if epoch in (0, best_epoch) else ''}>
<summary>Epoch {epoch} {'（最佳验证 checkpoint）' if epoch == best_epoch else ''}
 · 验证 loss {metric['validation']['loss']:.4f}
 · 固定窗口 top-3 命中 {hits}/{len(valid)}</summary>
<h3>自由续写：后续输入来自模型自己的预测</h3>
<table><tr><th>固定提示词</th><th>完整输出（包含提示词）</th></tr>{generations}</table>
<h3>固定验证窗口：每个位置都提供真实前缀</h3>
<pre>{escape(observation['predictions']['input_text'])}</pre>
<p>Ġ 表示空格。✓ 表示真实标签进入前三候选；这里的命中率只针对这个窗口。</p>
<div class="scroll"><table><tr><th>位置</th><th>输入 token</th><th>真实下一 token</th>
<th>真实标签概率</th><th>Top-3 候选（概率）</th><th>命中</th></tr>{''.join(rows)}</table></div>
</details>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>语言模型训练观察</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#182638;background:#f6f8fb}
img{width:100%;background:white}details{background:white;border:1px solid #ccd6e0;border-radius:8px;padding:16px;margin:18px 0}
summary{cursor:pointer;font-weight:600}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{border:1px solid #dce3ea;padding:8px;text-align:left;vertical-align:top}th{background:#edf3f9}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f9;padding:12px}.scroll{overflow-x:auto}
</style><h1>语言模型训练观察</h1>
<p>Epoch 0 是随机初始化。所有轮次使用相同验证窗口、相同提示词、贪心解码和最多 32 个新 token；观察时关闭梯度与 Dropout。</p>
<h2>怎么看曲线</h2><p>训练和验证 loss 一起下降：对未训练段落的预测正在改善。
训练 loss 下降但验证 loss 上升：可能过拟合。困惑度等于 exp(loss)，不是准确率。
验证采用真实前缀，续写采用模型自己的前缀，两种表现不能直接等同。</p>
<img src="curves.png" alt="训练与验证损失、困惑度和 token 准确率曲线">
<p>模型选择只依据全验证集 loss。测试集不用于逐轮观察；下方文字展示的是各轮模型，未统一替换成最佳模型。</p>
<h2>逐轮对照（点击展开）</h2>''' + ''.join(sections) + '</html>'
    (directory / 'training_observation.html').write_text(document, encoding='utf-8')

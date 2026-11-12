"""本地 TensorBoard：日志按 run 目录隔离，周期刷新供网页实时读取。"""

import json
from torch.utils.tensorboard import SummaryWriter


class TrainingMonitor:
    def __init__(self, directory, config):
        self.writer = SummaryWriter(str(directory), flush_secs=5)
        self.writer.add_text('configuration', '```json\n' + json.dumps(config, indent=2) + '\n```', 0)

    def step(self, step, loss, learning_rate, gradient_norm, tokens_per_second):
        for name, value in dict(loss=loss, learning_rate=learning_rate,
                                gradient_norm=gradient_norm, tokens_per_second=tokens_per_second).items():
            self.writer.add_scalar(f'optimization/{name}', value, step)
        self.writer.flush()

    def epoch(self, step, metrics, observation):
        for split in ('train', 'validation'):
            for name in ('loss', 'perplexity', 'token_accuracy'):
                self.writer.add_scalar(f'{split}/{name}', metrics[split][name], step)
        for index, generated in enumerate(observation['generations']):
            self.writer.add_text(f'generation/prompt_{index}',
                                 f"Epoch {metrics['epoch']} — {generated['prompt']}\n\n"
                                 f"```text\n{generated['text']}\n```", step)
        rows = observation['predictions']['rows']
        table = '|位置|输入|真实下一 token|真实概率|Top 3|\n|---|---|---|---|---|\n'
        for row in rows:
            if row['label_id'] < 0:
                continue
            clean = lambda s: s.replace('|', '\\|').replace('\n', ' ')
            candidates = ', '.join(f"{clean(c['token'])}: {c['probability']:.1%}" for c in row['top3'])
            table += (f"|{row['position']}|{clean(row['input_token'])}|{clean(row['label_token'])}|"
                      f"{row['label_probability']:.1%}|{candidates}|\n")
        self.writer.add_text('validation/fixed_predictions', table, step)
        self.writer.flush()

    def close(self):
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""本地 TensorBoard：日志按 run 目录隔离，周期刷新供网页实时读取。"""

import json
from pathlib import Path
from datetime import datetime, timezone
from torch.utils.tensorboard import SummaryWriter


class TrainingMonitor:
    def __init__(self, directory, config):
        self.metadata_path = Path(directory).parent / 'run.json'
        self.metadata = dict(config, status='running')
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2))
        self.last_step = 0
        self.writer = SummaryWriter(str(directory), flush_secs=5)
        self.writer.add_text('run/description', config.get('label','') + '\n\n' + config.get('note',''), 0)
        self.writer.add_text('run/status', 'running', 0)
        self.writer.add_text('configuration', '```json\n' + json.dumps(config, indent=2) + '\n```', 0)

    def step(self, step, loss, learning_rate, gradient_norm, tokens_per_second):
        self.last_step = step
        for name, value in dict(loss=loss, learning_rate=learning_rate,
                                gradient_norm=gradient_norm, tokens_per_second=tokens_per_second).items():
            self.writer.add_scalar(f'optimization/{name}', value, step)
        self.writer.flush()

    def epoch(self, step, metrics, observation, train_tag='train', log_train=True, log_validation=True):
        for split in ('train', 'validation'):
            if (split == 'train' and not log_train) or (split == 'validation' and not log_validation):
                continue
            tag = train_tag if split == 'train' else split
            for name in ('loss', 'perplexity', 'token_accuracy'):
                self.writer.add_scalar(f'{tag}/{name}', metrics[split][name], step)
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

    def __exit__(self, exc_type, exc_value, traceback):
        status = 'failed' if exc_type else self.metadata.get('final_status','completed')
        self.metadata.update(status=status, finished_at_utc=datetime.now(timezone.utc).isoformat())
        if exc_type:
            self.metadata['error_type'] = exc_type.__name__
        self.metadata_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2))
        self.writer.add_text('run/status', status, self.last_step)
        self.close()

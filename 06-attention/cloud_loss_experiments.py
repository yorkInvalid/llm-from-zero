"""Run a controlled learning-rate comparison, then continue on prepared fresh data."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path('/root/autodl-tmp/ownlm-industry-v2')
BASELINE = ROOT / 'runs/06-v100-v3-100m-1ep-fp16-b128'
STATE = ROOT / 'loss-experiments-state.json'


def save_state(state):
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temporary.replace(STATE)


def run(name, checkpoint, data, learning_rate, note, short=False):
    directory = ROOT / 'runs' / name
    command = [sys.executable, '-u', 'train_lm.py', 'train',
               '--data-dir', str(data), '--run-dir', str(directory),
               '--init-checkpoint', str(checkpoint), '--keep-optimizer-state',
               '--context-length', '256', '--device', 'cuda', '--precision', 'fp16',
               '--batch-size', '128', '--epochs', '1',
               '--learning-rate', str(learning_rate), '--warmup-ratio', '0.03',
               '--eval-every', '200' if short else '500',
               '--train-probe-windows', '1024', '--no-full-train-eval',
               '--run-label', name, '--run-note', note,
               '--observation-prompts', '人工智能', '机器学习的主要方法', '数据', '科学研究的意义']
    if short:
        command += ['--lr-schedule', 'constant', '--stop-after-step', '400']
    with (ROOT / f'{name}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    snapshot = directory / 'code'
    snapshot.mkdir(exist_ok=True)
    for source in ROOT.glob('*.py'):
        shutil.copy2(source, snapshot / source.name)
    report = json.loads((directory / 'report.json').read_text())
    return dict(run=name, learning_rate=learning_rate,
                final_validation_loss=report['checks'][-1]['validation']['loss'],
                best_validation_loss=report['best_validation']['loss'],
                tokens_seen=report['tokens_seen'], seconds=report['seconds'],
                checkpoint=str(directory / 'last.pt'))


def main():
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if STATE.exists():
        raise RuntimeError('An experiment state already exists. Inspect it before starting another queue.')
    state = dict(status='running', purpose='学习率对照后新增2亿token，持续检查验证loss', comparisons=[])
    baseline_loss = json.loads((BASELINE / 'report.json').read_text())['checks'][-1]['validation']['loss']
    selected = dict(checkpoint=str(BASELINE / 'last.pt'), learning_rate=1e-4,
                    final_validation_loss=baseline_loss, run=BASELINE.name)
    try:
        for number, learning_rate in enumerate((5e-5, 1e-4, 2e-4), 7):
            name = f'{number:02d}-v100-lr-{learning_rate:.0e}-probe400'
            state['active_run'] = name
            save_state(state)
            result = run(name, BASELINE / 'last.pt', ROOT / 'data-v3-100m', learning_rate,
                         '学习率对照。从同一06完整断点独立开始，保留Adam/GradScaler；同样seed42、同样400批次，'
                         '重复v3语料约1210万目标token。恒定学习率，按第400步完整验证loss选择。'
                         'paused表示按计划结束短测。初始及每200步验证，未评估test。', short=True)
            state['comparisons'].append(result)
            if result['final_validation_loss'] < selected['final_validation_loss']:
                selected = result
            save_state(state)
        state['selected'] = selected
        ready = json.loads((ROOT / 'data-v4-200m/READY.json').read_text())
        if ready['target_tokens'] != 200_000_000 or not ready['verified']:
            raise RuntimeError('Fresh data verification is incomplete')
        name = '10-v100-fresh200m-selected-lr'
        state['active_run'] = name
        save_state(state)
        result = run(name, Path(selected['checkpoint']), ROOT / 'data-v4-200m', selected['learning_rate'],
                     '新增2亿训练目标token，AI/教育/科技/新闻各5000万。排除v2、v3训练文档精确重复和留出集长片段重叠。'
                     f"按07/08/09同起点400步对照的末步验证选择{selected['run']}及学习率{selected['learning_rate']}。"
                     '保留所选完整断点权重、Adam和GradScaler，重新设置1遍cosine计划、3%warmup。'
                     '固定8192词表、验证集，context256，batch128，FP16；每500步完整验证，固定1024窗口train_probe。')
        state.update(status='completed', final=result)
        save_state(state)
        for filename in ('best.pt', 'last.pt'):
            path = ROOT / 'runs' / name / filename
            with path.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            with (path.parent / 'checksums.sha256').open('a') as checksums:
                checksums.write(f'{checksum}  {filename}\n')
    except Exception as error:
        state.update(status='failed', error=str(error))
        save_state(state)
        raise


if __name__ == '__main__':
    main()

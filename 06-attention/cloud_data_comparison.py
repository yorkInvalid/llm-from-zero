"""Compare replayed and newly sampled text from the same full checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path('/root/autodl-tmp/ownlm-industry-v2')
ORIGIN = ROOT / 'runs/10-v100-fresh200m-selected-lr/last.pt'
STATE = ROOT / 'data-comparison-state.json'


def record(state):
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temporary.replace(STATE)


def main():
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if STATE.exists():
        raise RuntimeError('Inspect the existing comparison state before starting a new queue')
    state = dict(status='running', comparisons=[], purpose='降低固定验证loss，比较本次新采样语料与已学语料')
    try:
        for number, arm in [(13, 'replay'), (14, 'fresh')]:
            directory = ROOT / 'runs' / f'{number:02d}-v100-data-{arm}'
            data = ROOT / 'data-comparison-v1' / arm
            state.update(active_run=directory.name, status='waiting_for_verified_data')
            record(state)
            while not (data / 'READY.json').exists():
                time.sleep(10)
            ready = json.loads((data / 'READY.json').read_text())
            target_tokens = ready['target_tokens']
            manifest = json.loads((data / 'train.index.json').read_text())
            if (not ready['verified'] or target_tokens != manifest['windows'] * 256
                    or manifest['windows'] % 128):
                raise RuntimeError(f'Invalid data verification for {arm}')
            if 'target_tokens_per_arm' in state and state['target_tokens_per_arm'] != target_tokens:
                raise RuntimeError('The comparison arms have different token budgets')
            state['target_tokens_per_arm'] = target_tokens
            total_steps = manifest['windows'] // 128
            command = [sys.executable, '-u', 'train_lm.py', 'train',
                       '--data-dir', str(data), '--run-dir', str(directory),
                       '--init-checkpoint', str(ORIGIN), '--keep-optimizer-state',
                       '--context-length', '256', '--device', 'cuda', '--precision', 'fp16',
                       '--batch-size', '128', '--epochs', '1',
                       '--learning-rate', '0.00005', '--warmup-ratio', '0.03', '--min-lr-ratio', '0.4',
                       '--eval-every', '384', '--train-probe-windows', '1024', '--no-full-train-eval',
                       '--run-label', f'{number} · {arm} · {target_tokens/1e6:.2f}M token · 自有3360万模型',
                       '--run-note',
                       f'{arm}语料对照。从10同一完整断点独立开始，保留Adam和GradScaler。'
                       f'四领域等窗口配额，完整256目标窗口，共{manifest["windows"]}窗口、{total_steps}次batch128更新、{target_tokens:,}目标token。'
                       '两臂同词表、验证集、seed42、FP16及cosine学习率计划：峰值5e-5，warmup3%，结束2e-5。'
                       '完整窗口筛选减少文档尾窗和EOS。新采样语料可能来自不同shard，结论限于本次语料策略。'
                       '主比较为相同固定完整验证集的末步loss。各臂train_probe取自各自训练集，仅用于自身走势。',
                       '--observation-prompts', '人工智能', '机器学习的主要方法', '数据', '科学研究的意义']
            state.update(status='running')
            record(state)
            with (ROOT / f'{directory.name}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            code = directory / 'code'
            code.mkdir()
            for source in ROOT.glob('*.py'):
                shutil.copy2(source, code / source.name)
            report = json.loads((directory / 'report.json').read_text())
            assert report['tokens_seen'] == target_tokens and report['global_step'] == total_steps
            state['comparisons'].append(dict(arm=arm, run=directory.name,
                                             initial_validation=report['checks'][0]['validation'],
                                             final_validation=report['checks'][-1]['validation'],
                                             best_validation=report['best_validation'],
                                             best_step=report['best_step'], seconds=report['seconds'],
                                             tokens_seen=report['tokens_seen']))
            with (directory / 'checksums.sha256').open('w') as output:
                for filename in ('best.pt', 'last.pt'):
                    with (directory / filename).open('rb') as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    output.write(f'{digest}  {filename}\n')
            record(state)
        state['status'] = 'training_completed'
        record(state)
        with (ROOT / 'data-comparison-evaluation.log').open('w') as log:
            subprocess.run([sys.executable, '-u', 'evaluate_data_comparison.py'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['status'] = 'completed'
        record(state)
    except Exception as error:
        state.update(status='failed', error=str(error))
        record(state)
        raise


if __name__ == '__main__':
    main()

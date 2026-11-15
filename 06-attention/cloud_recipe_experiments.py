"""Run matched recipe pilots, then continue the selected model to one billion tokens."""

import fcntl
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
CODE = Path(__file__).resolve().parent
DATA = ROOT / 'data-longrun-v1'
STATE = ROOT / 'recipe-experiment-state.json'
ORIGIN = ROOT / 'runs/14-v100-data-fresh/last.pt'
ORIGIN_TOKENS = 387_173_513
TOTAL_TOKENS = 1_000_000_000
PILOT_TOKENS = 25_000_000
PROMPTS = ['人工智能', '机器学习的主要方法', '数据', '科学研究的意义']


def read_json(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(state):
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temporary.replace(STATE)


def wait_for_data(directory):
    while not (directory / 'READY.json').exists():
        time.sleep(10)
    ready = read_json(directory / 'READY.json')
    if not ready['verified']:
        raise ValueError(f'Data verification failed for {directory}')
    from indexed_tokens import split_fingerprint
    if ready['train_sha256'] != split_fingerprint(directory, 'train'):
        raise ValueError(f'Data changed after verification for {directory}')
    return read_json(directory / 'train.index.json')


def run_training(state, name, parent, data, dropout, *, reset_optimizer=False,
                 optimizer='adam', beta2=.999, weight_decay=0., long_run=False):
    for filename, digest in state['code_sha256'].items():
        if checksum(CODE / filename) != digest:
            raise ValueError(f'Frozen experiment code changed: {filename}')
    directory = ROOT / 'runs' / name
    manifest = wait_for_data(data)
    targets = manifest['target_tokens']
    if directory.exists():
        raise FileExistsError(f'Inspect existing run before retrying: {directory}')
    config = dict(dropout=dropout, optimizer=optimizer, beta1=.9,
                  beta2=beta2, weight_decay=weight_decay,
                  weight_decay_scope='all parameters in one group',
                  optimizer_state='reset' if reset_optimizer else 'preserved')
    note = (f'自有3360万参数模型配方实验。起点为{Path(parent["checkpoint"]).parent.name}，'
            f'起点累计{parent["lineage_tokens"]:,}个有效目标token。'
            f'本段{targets:,}目标token，训练后累计{parent["lineage_tokens"] + targets:,}。'
            f'配置{json.dumps(config, ensure_ascii=False)}。'
            '保留原始文档边界、短文和尾窗，使用原8192词表及固定200篇验证集。')
    if long_run:
        note += '本段采用3%warmup、稳定学习率、末20%cosine冷却至5e-6，按完整预算统一调度。'
    else:
        note += '配方对照共享训练数据、seed42和5e-5至2e-5的cosine计划。选择完整末步断点。'
    command = [sys.executable, '-u', str(CODE / 'train_lm.py'), 'train',
               '--data-dir', str(data), '--run-dir', str(directory),
               '--init-checkpoint', str(parent['checkpoint']),
               '--context-length', '256', '--device', 'cuda', '--precision', 'fp16',
               '--batch-size', '128', '--epochs', '1', '--dropout', str(dropout),
               '--learning-rate', '0.00005', '--warmup-ratio', '0.03',
               '--min-lr-ratio', '0.1' if long_run else '0.4',
               '--lr-schedule', 'wsd' if long_run else 'cosine',
               '--eval-every', '1000' if long_run else '256',
               '--train-probe-windows', '1024', '--no-full-train-eval',
               '--run-label', f'{name} · {targets / 1e6:.2f}M token',
               '--run-note', note, '--observation-prompts', *PROMPTS]
    if reset_optimizer:
        command.extend(['--optimizer', optimizer, '--beta1', '0.9',
                        '--beta2', str(beta2), '--weight-decay', str(weight_decay)])
    else:
        command.append('--keep-optimizer-state')
    if long_run:
        command.extend(['--cooldown-ratio', '0.2'])
    state.update(status='training', active_run=name, active_token_budget=targets,
                 active_lineage_tokens_start=parent['lineage_tokens'])
    record(state)
    with (ROOT / f'{name}.log').open('w') as log:
        subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read_json(directory / 'report.json')
    if report['paused'] or report['tokens_seen'] != targets:
        raise ValueError(f'Incomplete training budget for {name}')
    if abs(report['checks'][0]['validation']['loss'] - parent['validation_loss']) > 1e-5:
        raise ValueError(f'Starting validation changed for {name}')
    snapshot = directory / 'code'
    snapshot.mkdir()
    for path in CODE.glob('*.py'):
        shutil.copy2(path, snapshot / path.name)
    metadata = read_json(directory / 'run.json')
    for filename, digest in metadata['code_sha256'].items():
        if checksum(snapshot / filename) != digest:
            raise ValueError(f'Source changed during training: {filename}')
    with (directory / 'checksums.sha256').open('w') as stream:
        for filename in ('best.pt', 'last.pt'):
            stream.write(f'{checksum(directory / filename)}  {filename}\n')
    recipe = report['training_config']
    result = dict(run=name, checkpoint=str(directory / 'last.pt'),
                  lineage_tokens=parent['lineage_tokens'] + targets,
                  parent_checkpoint=str(parent['checkpoint']),
                  parent_checkpoint_sha256=metadata['checkpoint_sha256'],
                  validation_loss=report['checks'][-1]['validation']['loss'],
                  best_validation=report['best_validation'], best_step=report['best_step'],
                  global_step=report['global_step'], dropout=report['model_config']['dropout'],
                  optimizer=recipe['optimizer'], betas=recipe['betas'],
                  weight_decay=recipe['weight_decay'], tokens_seen=targets,
                  seconds=report['seconds'])
    (directory / 'lineage.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    state['runs'].append(result)
    record(state)
    return result


def write_pilot_conclusion(state):
    from torch.utils.tensorboard import SummaryWriter
    text = json.dumps(dict(dropout_selection=state['dropout_selection'],
                           long_training_origin=state['selected_origin'],
                           pilot_runs=state['runs'],
                           selection='lowest terminal fixed-validation loss among complete checkpoints and the starting checkpoint',
                           scope='Single training seed and reused validation set. Adam vs AdamW comparison includes beta2 and weight decay.'),
                      ensure_ascii=False, indent=2)
    for item in state['runs']:
        with SummaryWriter(str(ROOT / 'runs' / item['run'] / 'tensorboard'),
                           filename_suffix='.recipe') as writer:
            writer.add_text('run/recipe_comparison', text, item['global_step'])


def evaluate_final(state, final):
    import torch
    from evaluate_data_comparison import evaluate_checkpoint, validation_documents
    from text_pretraining import make_windows
    from train_lm import restore, evaluate
    from torch.utils.tensorboard import SummaryWriter
    torch.set_num_threads(1)
    directory = ROOT / 'runs' / final['run']
    report = read_json(directory / 'report.json')
    documents, domains = validation_documents(ROOT)
    tokenizer_hash = checksum(ROOT / 'data/tokenizer.json')
    test_documents = read_json(ROOT / 'data/test.json')
    best_check = next(check for check in report['checks']
                      if check['global_step'] == report['best_step'])
    best_lineage_tokens = state['selected_origin']['lineage_tokens'] + best_check.get('tokens_seen', 0)
    models = {}
    checkpoints = [('baseline', ORIGIN, ORIGIN_TOKENS),
                   ('final_last', directory / 'last.pt', TOTAL_TOKENS),
                   ('final_best', directory / 'best.pt', best_lineage_tokens)]
    for label, checkpoint, lineage_tokens in checkpoints:
        validation = evaluate_checkpoint(checkpoint, documents, domains, tokenizer_hash)
        model, tokenizer, _ = restore(checkpoint, 'cuda')
        test = evaluate(model, make_windows(test_documents, tokenizer, 256), 128)
        del model
        torch.cuda.empty_cache()
        models[label] = dict(validation=validation, heldout_test=test,
                             checkpoint_lineage_tokens=lineage_tokens)
    if abs(models['final_best']['validation']['overall']['loss'] - report['best_validation']['loss']) > 1e-5:
        raise ValueError('Final independent evaluation differs from recorded validation')
    if abs(models['final_last']['validation']['overall']['loss'] - final['validation_loss']) > 1e-5:
        raise ValueError('Terminal independent evaluation differs from recorded validation')
    result = dict(status='completed', models=models, completed_lineage_tokens=TOTAL_TOKENS,
                  test_documents=len(test_documents),
                  test_scope='Existing heldout AI documents, evaluated after the recipe selection and full training budget.',
                  recipe_selection=state['selected_origin'])
    (ROOT / 'recipe-final-evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with SummaryWriter(str(directory / 'tensorboard'), filename_suffix='.conclusion') as writer:
        for name, values in models.items():
            writer.add_text(f'run/{name}_evaluation', json.dumps(values, ensure_ascii=False, indent=2), final['global_step'])
        writer.add_text('run/conclusion', json.dumps(dict(
            cumulative_training_tokens=TOTAL_TOKENS, best_validation=report['best_validation'],
            best_checkpoint_lineage_tokens=best_lineage_tokens,
            baseline_validation=models['baseline']['validation']['overall'],
            independent_test=models['final_best']['heldout_test'],
            selected_recipe=state['selected_origin']), ensure_ascii=False, indent=2), final['global_step'])


def main():
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    lock = (ROOT / 'recipe-experiment.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists():
        raise RuntimeError('Inspect the existing recipe experiment state before restarting')
    baseline_report = read_json(ORIGIN.parent / 'report.json')
    origin = dict(run=ORIGIN.parent.name, checkpoint=str(ORIGIN),
                  validation_loss=baseline_report['checks'][-1]['validation']['loss'],
                  lineage_tokens=ORIGIN_TOKENS, dropout=.1, optimizer='adam',
                  betas=[.9, .999], weight_decay=0.)
    state = dict(status='preparing', total_lineage_target=TOTAL_TOKENS,
                 origin=origin, runs=[], purpose='降低自有3360万模型的固定验证loss',
                 code_sha256={path.name: checksum(path) for path in CODE.glob('*.py')})
    record(state)
    try:
        pilot = DATA / 'pilot-25m'
        if wait_for_data(pilot)['target_tokens'] != PILOT_TOKENS:
            raise ValueError('Unexpected pilot token budget')
        first = run_training(state, '15-v100-dropout-01', origin, pilot, .1)
        second = run_training(state, '16-v100-dropout-00', origin, pilot, .0)
        dropout_winner = min([origin, first, second], key=lambda item: item['validation_loss'])
        state['dropout_selection'] = dropout_winner
        record(state)
        third = run_training(state, '17-v100-adam-reset', dropout_winner, pilot,
                             dropout_winner['dropout'], reset_optimizer=True)
        fourth = run_training(state, '18-v100-adamw-reset', dropout_winner, pilot,
                              dropout_winner['dropout'], reset_optimizer=True,
                              optimizer='adamw', beta2=.95, weight_decay=.1)
        selected = min([dropout_winner, third, fourth], key=lambda item: item['validation_loss'])
        state.update(selected_origin=selected, status='preparing_long_training')
        record(state)
        write_pilot_conclusion(state)
        wait_for_data(DATA / 'merged')
        remaining = TOTAL_TOKENS - selected['lineage_tokens']
        from prepare_longrun_data import build_training_budget
        budget_data = DATA / f'budget-{remaining}'
        build_training_budget(DATA / 'merged', budget_data, remaining, seed=20261123)
        from indexed_tokens import split_fingerprint
        (budget_data / 'READY.json').write_text(json.dumps(dict(
            verified=True, target_tokens=remaining,
            train_sha256=split_fingerprint(budget_data, 'train')), indent=2))
        final = run_training(state, '19-v100-long-pretraining-1b', selected, budget_data,
                             selected['dropout'], optimizer=selected['optimizer'],
                             beta2=selected['betas'][1], weight_decay=selected['weight_decay'], long_run=True)
        if final['lineage_tokens'] != TOTAL_TOKENS:
            raise ValueError('Cumulative training budget mismatch')
        state.update(status='evaluating', final=final)
        record(state)
        evaluate_final(state, final)
        state['status'] = 'completed'
        record(state)
    except Exception as error:
        state.update(status='failed', error=str(error))
        record(state)
        raise


if __name__ == '__main__':
    main()

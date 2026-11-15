"""Compare matched fresh/replay training endpoints on the frozen validation set."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from indexed_tokens import split_fingerprint
from text_pretraining import make_windows
from train_lm import evaluate, restore


ROOT = Path('/root/autodl-tmp/ownlm-industry-v2')
RUN_NAMES = {'replay': '13-v100-data-replay', 'fresh': '14-v100-data-fresh'}
DOMAINS = ('ai', 'education', 'science', 'news')
BOOTSTRAP_SEED = 20261120
BOOTSTRAP_REPLICATES = 5000
LOSS_TOLERANCE = 1e-5


def read_json(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_runs(root):
    parent = root / 'runs/10-v100-fresh200m-selected-lr'
    origin_hash = checksum(parent / 'last.pt')
    reference_validation_hash = checksum(root / 'data/validation.json')
    reference_tokenizer_hash = checksum(root / 'data/tokenizer.json')
    records = {}
    for arm, name in RUN_NAMES.items():
        directory = root / 'runs' / name
        data = root / 'data-comparison-v1' / arm
        metadata = read_json(directory / 'run.json')
        report = read_json(directory / 'report.json')
        manifest = read_json(data / 'train.index.json')
        source = read_json(data / 'source.json')
        config = report['training_config']
        index = np.load(data / 'train.windows.npy', mmap_mode='r')
        targets = int(index[:, 1].sum())
        require(metadata['status'] == 'completed' and not report['paused'], f'{arm} is incomplete')
        require(metadata['checkpoint_sha256'] == origin_hash, f'{arm} has a different starting checkpoint')
        require(config == metadata['training'], f'{arm} training metadata differs from report')
        require(config['precision'] == 'fp16' and config['evaluation_precision'] == 'float32',
                f'{arm} precision differs from experiment plan')
        require(config['keep_optimizer_state'] and config['batch_size'] == 128,
                f'{arm} optimizer/batch settings differ from experiment plan')
        require(manifest['context_length'] == 256 and np.all(index[:, 1] == 256),
                f'{arm} includes partial or differently sized windows')
        require(len(index) == manifest['windows'] and len(index) % 128 == 0,
                f'{arm} window count is inconsistent')
        require(targets == manifest['target_tokens'] == source['target_tokens']
                == config['train_token_budget'] == report['tokens_seen'], f'{arm} token budget mismatch')
        require(config['total_steps'] * 128 * 256 == targets
                and report['global_step'] == config['total_steps'], f'{arm} update count mismatch')
        require(report['checks'][-1]['global_step'] == report['global_step'], f'{arm} final check is missing')
        require(metadata['split_sha256']['train'] == split_fingerprint(data, 'train'),
                f'{arm} training data changed')
        require(checksum(data / 'validation.json') == reference_validation_hash
                == metadata['split_sha256']['validation'], f'{arm} validation differs')
        require(checksum(data / 'tokenizer.json') == reference_tokenizer_hash
                == metadata['tokenizer_sha256'], f'{arm} tokenizer differs')
        quotas = {domain: source['domains'][domain]['selected_windows'] for domain in DOMAINS}
        require(sum(quotas.values()) == len(index) and len(set(quotas.values())) == 1,
                f'{arm} domain window quotas differ')
        require(all(source['domains'][domain]['target_tokens'] == count * 256
                    for domain, count in quotas.items()), f'{arm} domain token quotas differ')
        records[arm] = dict(directory=directory, metadata=metadata, report=report,
                            config=config, targets=targets, quotas=quotas)
    replay, fresh = records['replay'], records['fresh']
    require(replay['config'] == fresh['config'], 'Training configurations differ between arms')
    require(replay['report']['model_config'] == fresh['report']['model_config'], 'Model configurations differ')
    require(replay['targets'] == fresh['targets'] and replay['quotas'] == fresh['quotas'],
            'Training budgets or domain quotas differ')
    initial_losses = [record['report']['checks'][0]['validation']['loss'] for record in records.values()]
    require(abs(initial_losses[0] - initial_losses[1]) <= LOSS_TOLERANCE, 'Initial validation losses differ')
    return records, dict(origin_checkpoint_sha256=origin_hash,
                         validation_sha256=reference_validation_hash,
                         tokenizer_sha256=reference_tokenizer_hash,
                         target_tokens_per_arm=replay['targets'],
                         updates_per_arm=replay['config']['total_steps'],
                         domain_windows_per_arm=replay['quotas'],
                         initial_validation_losses=initial_losses)


def validation_documents(root):
    documents = read_json(root / 'data/validation.json')
    legacy = read_json(root / 'data/validation_legacy.json')
    source = read_json(root / 'data/source.json')
    require(documents[:len(legacy)] == legacy, 'Legacy validation ordering changed')
    domains = ['ai'] * len(legacy)
    for entry in source['sources']:
        require(entry['domain'] in DOMAINS, 'Unexpected validation domain')
        domains.extend([entry['domain']] * entry['new_validation_documents'])
    require(len(domains) == len(documents) == 200, 'Validation document/domain ordering is inconsistent')
    return documents, domains


def aggregate(rows):
    tokens = sum(row['tokens'] for row in rows)
    loss_sum = math.fsum(row['loss_sum'] for row in rows)
    correct = sum(row['correct'] for row in rows)
    loss = loss_sum / tokens
    return dict(documents=len(rows), tokens=tokens, loss_sum=loss_sum, correct=correct,
                loss=loss, perplexity=math.exp(min(loss, 700)), token_accuracy=correct/tokens)


def evaluate_checkpoint(path, documents, domains, expected_tokenizer_hash):
    model, tokenizer, saved = restore(path, 'cuda')
    require(saved['tokenizer_sha256'] == expected_tokenizer_hash, 'Checkpoint tokenizer differs')
    model.float().eval()
    context = model.config['context_length']
    require(context == 256, 'Checkpoint context differs')
    rows = []
    for document_id, (text, domain) in enumerate(zip(documents, domains)):
        metrics = evaluate(model, make_windows([text], tokenizer, context), 128)
        tokens = metrics['tokens']
        rows.append(dict(document_id=document_id, domain=domain,
                         text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                         tokens=tokens, loss_sum=metrics['loss'] * tokens,
                         correct=round(metrics['token_accuracy'] * tokens),
                         loss=metrics['loss'], token_accuracy=metrics['token_accuracy']))
    result = dict(checkpoint=str(path), checkpoint_sha256=checksum(path),
                  global_step=saved['global_step'], overall=aggregate(rows),
                  domains={domain: aggregate([row for row in rows if row['domain'] == domain])
                           for domain in DOMAINS}, documents=rows)
    del model, saved
    torch.cuda.empty_cache()
    return result


def paired_bootstrap(replay_rows, fresh_rows):
    require([(row['document_id'], row['domain'], row['tokens'], row['text_sha256']) for row in replay_rows]
            == [(row['document_id'], row['domain'], row['tokens'], row['text_sha256']) for row in fresh_rows],
            'Per-document evaluation rows do not align')
    tokens = np.asarray([row['tokens'] for row in replay_rows], dtype=np.float64)
    differences = np.asarray([fresh['loss_sum'] - replay['loss_sum']
                              for replay, fresh in zip(replay_rows, fresh_rows)], dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_differences = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    sampled_tokens = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for domain in DOMAINS:
        indices = np.asarray([i for i, row in enumerate(replay_rows) if row['domain'] == domain])
        draws = generator.choice(indices, size=(BOOTSTRAP_REPLICATES, len(indices)), replace=True)
        sampled_differences += differences[draws].sum(axis=1)
        sampled_tokens += tokens[draws].sum(axis=1)
    estimates = sampled_differences / sampled_tokens
    interval = np.quantile(estimates, [0.025, 0.975]).tolist()
    return dict(delta_fresh_minus_replay=float(differences.sum()/tokens.sum()),
                percentile_95_interval=interval, replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED, resampling_unit='validation document, paired and stratified by domain',
                weighting='sum of token loss differences divided by sum of target tokens in each resample',
                interpretation='negative delta favors fresh; positive delta favors replay',
                limitation='Interval reflects resampling of the fixed validation documents. It does not cover training-seed or corpus-sampling variability.')


def main():
    torch.set_num_threads(1)
    require(torch.cuda.is_available(), 'CUDA is required for the planned FP32 evaluation')
    records, checks = validate_runs(ROOT)
    documents, domains = validation_documents(ROOT)
    parent = ROOT / 'runs/10-v100-fresh200m-selected-lr'
    checkpoints = {'baseline': parent / 'best.pt',
                   **{arm: record['directory'] / 'last.pt' for arm, record in records.items()}}
    models = {}
    for label, path in checkpoints.items():
        print(f'Evaluating {label} on {len(documents)} validation documents', flush=True)
        result = evaluate_checkpoint(path, documents, domains, checks['tokenizer_sha256'])
        expected = (read_json(parent / 'report.json')['best_validation']['loss'] if label == 'baseline'
                    else records[label]['report']['checks'][-1]['validation']['loss'])
        require(abs(result['overall']['loss'] - expected) <= LOSS_TOLERANCE,
                f'{label} per-document aggregation differs from recorded full validation')
        result['recorded_full_validation_loss'] = expected
        result['loss_difference_from_recorded'] = result['overall']['loss'] - expected
        models[label] = result
    require(all(abs(loss - models['baseline']['overall']['loss']) <= LOSS_TOLERANCE
                for loss in checks['initial_validation_losses']), 'Baseline differs from the arms initial evaluation')
    bootstrap = paired_bootstrap(models['replay']['documents'], models['fresh']['documents'])
    output = dict(status='completed', primary_comparison='final endpoint of each equal-budget arm',
                  evaluation_precision='float32', checks=checks, models=models, bootstrap=bootstrap,
                  data_scope='These sampled fresh and replay corpora share domain quotas and window length. Finer source distributions can differ.',
                  test_usage='Test data was not evaluated or used for selection')
    destination = ROOT / 'data-comparison-evaluation.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    temporary.replace(destination)
    summary = dict(checks=checks, bootstrap=bootstrap,
                   models={name: {key: result[key] for key in ('global_step', 'overall', 'domains')}
                           for name, result in models.items()}, data_scope=output['data_scope'])
    for arm, record in records.items():
        with SummaryWriter(str(record['directory'] / 'tensorboard'), filename_suffix='.comparison') as writer:
            writer.add_text('run/data_comparison', json.dumps(summary, ensure_ascii=False, indent=2),
                            record['report']['global_step'])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(destination, flush=True)


if __name__ == '__main__':
    main()

"""Replay a late checkpoint with identical batches and RNG to compare LR schedules."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from indexed_tokens import split_fingerprint
from lm_components import learning_rate_at_step
from text_pretraining import load_data, collate
from train_lm import restore, evaluate
from training_state import epoch_batches, restore_rng, preserve_rng, atomic_save, rng_state

ROOT = Path('/root/autodl-tmp/ownlm-industry-v2')
PARENT = ROOT / 'runs/10-v100-fresh200m-selected-lr'
ORIGIN = PARENT / 'diagnostics/plateau-origin.pt'
STEPS = 500


def main():
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.set_num_threads(1)
    data = ROOT / 'data-v4-200m'
    tokenizer, splits = load_data(data, 256)
    hashes = {name: split_fingerprint(data, name) for name in ('train', 'validation')}
    with ORIGIN.open('rb') as stream:
        origin_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    results = {}
    for number, schedule in [(11, 'cosine-replay'), (12, 'reheat-cooldown')]:
        model, _, saved = restore(ORIGIN, 'cuda')
        config = saved['training_config']
        assert saved['split_sha256'] == hashes
        assert saved['tokenizer_sha256'] == hashlib.sha256((data/'tokenizer.json').read_bytes()).hexdigest()
        assert saved['global_step'] == 5500 and saved['next_epoch'] == 1
        batch_ids = epoch_batches(len(splits['train']), config['batch_size'], 1, config['seed'])
        selected = batch_ids[saved['next_batch']:saved['next_batch'] + STEPS]
        assert len(selected) == STEPS
        optimizer = torch.optim.Adam(model.parameters(), lr=config['peak_lr'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler = torch.amp.GradScaler('cuda')
        scaler.load_state_dict(saved['scaler'])
        directory = ROOT / 'runs' / f'{number:02d}-v100-plateau-{schedule}'
        directory.mkdir()
        metadata = dict(status='running', origin_sha256=origin_hash, origin_global_step=5500,
                        purpose='相同权重、Adam、GradScaler、RNG和后续500批次的学习率对照',
                        schedule=schedule, steps=STEPS, batch_size=config['batch_size'],
                        reheat_peak=1e-4, final_cooldown_steps=100,
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (directory/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        (directory/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        final_lr = learning_rate_at_step(6000, config['total_steps'], config['warmup_steps'],
                                        config['peak_lr'], config['min_lr_ratio'])
        tokens = 0
        retries = 0
        started = time.monotonic()
        model.train()
        restore_rng(saved['rng'], 'cuda')
        with SummaryWriter(str(directory/'tensorboard')) as writer:
            writer.add_text('run/description', json.dumps(metadata, ensure_ascii=False, indent=2), 0)
            writer.add_scalar('validation/loss', saved['checks'][-1]['validation']['loss'], 0)
            for local_step, ids in enumerate(selected, 1):
                global_step = 5500 + local_step
                lr = learning_rate_at_step(global_step, config['total_steps'], config['warmup_steps'],
                                           config['peak_lr'], config['min_lr_ratio'])
                if schedule == 'reheat-cooldown':
                    lr = 1e-4
                    if local_step > 400:
                        fraction = (local_step - 400) / 100
                        lr = final_lr + (1e-4 - final_lr) * (1 + math.cos(math.pi*fraction)) / 2
                for group in optimizer.param_groups:
                    group['lr'] = lr
                x, y = collate([splits['train'][index] for index in ids])
                x, y = x.cuda(), y.cuda()
                for attempt in range(8):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits = model(x)
                        loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(gradient):
                        scaler.update(new_scale=scaler.get_scale()*.5)
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                    break
                else:
                    raise FloatingPointError('Persistent AMP overflow')
                retries += attempt
                tokens += int((y != -100).sum())
                if local_step % 10 == 0:
                    writer.add_scalar('optimization/loss', loss.item(), local_step)
                    writer.add_scalar('optimization/learning_rate', lr, local_step)
                    writer.add_scalar('progress/train_tokens_seen', tokens, local_step)
                    writer.add_scalar('time/elapsed_seconds', time.monotonic()-started, local_step)
                if local_step % 100 == 0:
                    with preserve_rng('cuda'):
                        metrics = evaluate(model, splits['validation'], 128)
                    writer.add_scalar('validation/loss', metrics['loss'], local_step)
                    writer.flush()
                    print(schedule, local_step, metrics['loss'], flush=True)
            seconds = time.monotonic()-started
            result = dict(validation=metrics, tokens=tokens, seconds=seconds, amp_retries=retries,
                          final_lr=lr, global_step=global_step)
            if schedule == 'cosine-replay':
                reference = next(c for c in json.loads((PARENT/'validation_checks.json').read_text())
                                 if c['global_step'] == 6000)
                difference = abs(metrics['loss']-reference['validation']['loss'])
                result['replay_loss_difference'] = difference
                if difference > 1e-5:
                    raise RuntimeError(f'Control replay differs by {difference}; investigate before interpreting')
            results[schedule] = result
            writer.add_text('run/result', json.dumps(result, indent=2), STEPS)
            writer.add_text('run/status', 'completed', STEPS)
        # Diagnostic weights are retained separately from the main run.
        atomic_save(dict(checkpoint_version=1, config=model.config, model=model.state_dict(), tokenizer=tokenizer.to_str(),
                         tokenizer_sha256=saved['tokenizer_sha256'], epoch=global_step/len(batch_ids),
                         global_step=global_step, selection_validation=metrics,
                         optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng_state('cuda'),
                         tokens_seen=saved['tokens_seen']+tokens, next_epoch=1, next_batch=6000,
                         training_config=dict(precision='fp16', schedule=schedule, diagnostic_complete=True),
                         continuation='Start a new plan with init-checkpoint and keep-optimizer-state'),
                    directory/'diagnostic.pt')
        metadata['status'] = 'completed'
        (directory/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        (directory/'report.json').write_text(json.dumps(result, indent=2))
        del model, optimizer, scaler, saved
        torch.cuda.empty_cache()
    (PARENT/'plateau-schedule-comparison.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()

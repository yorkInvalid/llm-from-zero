"""Train four LoRA snapshots and generate paired answers from the same base on CUDA."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import peft
import torch
import transformers
from peft import get_peft_model, LoraConfig, PeftModel
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CID = 'Qwen/Qwen2.5-7B'
REV = 'd149729398750b98c0af14eb82c78cfe92750796'
BASE = 'base'
STAGES = {563*n: 'epoch-'+str(n/4).rstrip('0').rstrip('.') for n in range(1,9)}
SEED = 20261116
MICRO, BATCH, MAX_LEN = 2, 4, 384
LR, WARMUP = 1e-4, 45
EOS, IM_END = 151643, 151645

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def rows(path):
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def verify_base(base, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    assert manifest['model'] == CID and manifest['revision'] == REV
    if (base / 'adapter_config.json').exists():
        raise ValueError('Expected pure model weights; an adapter config exists in the base directory')
    for name, spec in manifest['files'].items():
        item = base / name
        if item.stat().st_size != spec['bytes']:
            raise ValueError(f'File size mismatch: {item}')
        if 'sha256' in spec:
            actual = sha(item)
            expected = spec['sha256']
        else:
            content = item.read_bytes()
            actual = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            expected = spec['git_blob_sha1']
        if actual != expected:
            raise ValueError(f'Frozen base hash mismatch: {item}')
    return manifest


def prompt(s):
    return '<|im_start|>user\n' + s + '<|im_end|>\n<|im_start|>assistant\n'


def encode(data, tokenizer):
    output = []
    for r in data:
        pre = tokenizer.encode(prompt(r['prompt']), add_special_tokens=False)
        ans = tokenizer.encode(r['response'], add_special_tokens=False) + [EOS]
        assert len(pre) + len(ans) <= MAX_LEN and len(ans) > 1, r['id']
        assert r.get('input_tokens', len(pre)) == len(pre)
        assert r.get('supervised_tokens', len(ans)) == len(ans)
        output.append(dict(id=r['id'], source=r['source'], ids=pre + ans, split=len(pre), targets=len(ans)))
    return output


def batches(data, epoch):
    schedule = json.loads(args.schedule.read_text())
    by_id = {r['id']:i for i,r in enumerate(data)}
    out = []
    for part in schedule['quarters'][4*(epoch-1):4*epoch]:
        ids = [by_id[k] for k in part['ids']]
        assert len(ids) == 2250 and sum(data[i]['targets'] for i in ids)==part['supervised_tokens']
        out.extend(ids[j:j+BATCH] for j in range(0, len(ids), BATCH))
    assert sorted(k for ids in out for k in ids) == list(range(9000))
    return out


def active_logits(model, group, pad):
    width = max(len(r['ids']) for r in group)
    inputs = torch.tensor([r['ids'] + [pad] * (width-len(r['ids'])) for r in group], device='cuda')
    idx = torch.arange(width, device='cuda').unsqueeze(0)
    ends = torch.tensor([len(r['ids']) for r in group], device='cuda').unsqueeze(1)
    starts = torch.tensor([r['split'] for r in group], device='cuda').unsqueeze(1)
    mask = idx < ends
    target_mask = ((idx >= starts) & mask)[:, 1:]
    network = model.get_base_model()
    hidden = network.model(input_ids=inputs, attention_mask=mask.long(), use_cache=False).last_hidden_state
    target_values = inputs[:, 1:][target_mask]
    return network.lm_head(hidden[:, :-1][target_mask]).float(), target_values


@torch.inference_mode()
def validate(model, data, pad):
    model.eval()
    entries = []
    for i in range(0, len(data), MICRO):
        group = data[i:i+MICRO]
        logits, labels = active_logits(model, group, pad)
        nll = F.cross_entropy(logits, labels, reduction='none')
        at = 0
        for r in group:
            end = at + r['targets'] - 1
            assert labels[end].item() == EOS
            entries.append(dict(id=r['id'], source=r['source'], body_targets=end-at,
                                body_nll_sum=float(nll[at:end].sum()), end_nll=float(nll[end]),
                                end_top1_native=bool(torch.argmax(logits[end]).item() == EOS)))
            at = end + 1
    body_t = sum(r['body_targets'] for r in entries)
    body_loss = sum(r['body_nll_sum'] for r in entries)
    end_loss = sum(r['end_nll'] for r in entries)
    n = len(entries)
    return dict(rows=entries, targets=body_t+n, body_targets=body_t, examples=n,
                loss=(body_loss+end_loss)/(body_t+n), body_nll=body_loss/body_t, eos_nll=end_loss/n,
                eos_top1=sum(r['end_top1_native'] for r in entries))


def model_and_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side='left')
    assert tok.eos_token_id == EOS and tok.convert_tokens_to_ids('<|im_end|>') == IM_END
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, attn_implementation='sdpa',
                                               local_files_only=True).to('cuda')
    return base, tok


def checkpoint(model, opt, scaler, stage_dir, state):
    temp = stage_dir.with_name('.' + stage_dir.name + '-in-progress')
    temp.mkdir(exist_ok=True)
    model.peft_config['default'].base_model_name_or_path = CID
    model.peft_config['default'].revision = REV
    model.save_pretrained(temp, save_embedding_layers=False)
    torch.save({'optimizer':opt.state_dict(), 'scaler':scaler.state_dict()}, temp/'training_state.pt')
    state['adapter_sha256'] = sha(temp/'adapter_model.safetensors')
    save(temp/'state.json', state)
    temp.rename(stage_dir)


def train(args):
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    data_files = [args.data/'train.jsonl', args.data/'validation.jsonl']
    data_hashes = {p.name:sha(p) for p in data_files}
    frozen_base = verify_base(args.base, args.base_manifest)
    save(out/'base-verified.json', frozen_base)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base, tok = model_and_tokenizer(args.base)
    data, val = (encode(rows(p), tok) for p in data_files)
    assert len(data) == 9000 and len(val) == 120
    sched=json.loads(args.schedule.read_text())
    assert data_hashes['validation.jsonl']==sched['validation_sha256']
    if args.arm=='control': assert data_hashes['train.jsonl']==sched['control_sha256']
    assert {r['id']:r['targets'] for r in data}==sched['control_targets'], 'Unmatched supervised length at sample level'
    assert not (set(r['id'] for r in data) & set(r['id'] for r in val))
    assert sum(r['targets'] for r in data)==1353439
    assert len({r['id'] for r in data + val}) == len(data)+len(val)
    assert sum(p.numel() for p in base.parameters()) == 7615616512
    last = max((step for step, name in STAGES.items() if (out/name/'state.json').exists()), default=0)
    if last:
        ck = out/STAGES[last]
        state = json.loads((ck/'state.json').read_text())
        assert state['data_hashes'] == data_hashes and state['adapter_sha256'] == sha(ck/'adapter_model.safetensors')
        model = PeftModel.from_pretrained(base, ck, is_trainable=True)
    else:
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0., target_modules=['q_proj', 'k_proj', 'v_proj',
                          'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM')
        model = get_peft_model(base, lora)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    n_trainable = sum(p.numel() for p in trainable)
    assert n_trainable == 20185088 and all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    elapsed = 0.
    log = out/'updates.jsonl'
    if last:
        state_dict = torch.load(ck/'training_state.pt', map_location='cuda', weights_only=True)
        opt.load_state_dict(state_dict['optimizer'])
        scaler.load_state_dict(state_dict['scaler'])
        elapsed = state['train_seconds']
        del state_dict
        valid_logs = [r for r in rows(log) if r['step'] <= last]
        assert len(valid_logs) == last
        temp_log = log.with_name('updates.valid.tmp')
        temp_log.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in valid_logs))
        temp_log.replace(log)
    elif log.exists():
        log.rename(out/f'updates-before-first-checkpoint-{int(time.time())}.jsonl')
    report = dict(base=CID, revision=REV, arm=args.arm, schedule_sha256=sha(args.schedule), quarter_samples=2250, quarter_updates=563, quarter_end_batch=2, data_hashes=data_hashes, seed=SEED, train_examples=len(data),
                  validation_examples=len(val), input_plus_answer_tokens_per_epoch=sum(len(r['ids']) for r in data),
                  answer_tokens_per_epoch=sum(r['targets'] for r in data), base_parameters=7615616512,
                  trainable_parameters=n_trainable, dtype='FP16 base, FP32 adapters, AMP FP16', microbatch=MICRO,
                  effective_batch=BATCH, epochs=2, max_len=MAX_LEN, learning_rate=LR, warmup_steps=WARMUP,
                  optimizer='AdamW', weight_decay=.01, gradient_clip=1., schedule='warmup then constant',
                  device=torch.cuda.get_device_name(), torch=str(torch.__version__), transformers=transformers.__version__,
                  peft=peft.__version__, lora_rank=8, lora_alpha=16, loss='answer-only causal cross entropy including endoftext')
    save(out/'run-config.json', report)
    if not last:
        with torch.inference_mode():
            model.eval()
            sample = data[0]
            whole = torch.tensor([sample['ids']], device='cuda')
            labels = whole.clone()
            labels[:, :sample['split']] = -100
            ref = model(input_ids=whole, labels=labels, use_cache=False).loss.item()
            with torch.autocast('cuda', dtype=torch.float16):
                logits, targets = active_logits(model, [sample], tok.pad_token_id)
                part = F.cross_entropy(logits, targets).item()
            assert abs(ref-part) < .015, (ref, part)
            save(out/'reference-loss-check.json', dict(id=sample['id'], full=ref, answer_only=part))
            del logits, whole, labels, targets
        baseline = validate(model, val, tok.pad_token_id)
        save(out/'validation-base.json', baseline)
        print('BASE VALIDATION', baseline['loss'], flush=True)
    torch.cuda.reset_peak_memory_stats()
    step = 0
    model.train()
    with log.open('a', buffering=1) as log_file:
        for epoch in (1, 2):
            for indices in batches(data, epoch):
                step += 1
                if step <= last:
                    continue
                start = time.perf_counter()
                records = [data[j] for j in indices]
                total = sum(r['targets'] for r in records)
                lr = LR*min(step/WARMUP, 1.)
                for group in opt.param_groups:
                    group['lr'] = lr
                for retry in range(8):
                    opt.zero_grad(set_to_none=True)
                    batch_nll = 0.
                    for i in range(0, len(records), MICRO):
                        micro = records[i:i+MICRO]
                        with torch.autocast('cuda', dtype=torch.float16):
                            logits, targets = active_logits(model, micro, tok.pad_token_id)
                            loss_sum = F.cross_entropy(logits, targets, reduction='sum')
                        scaler.scale(loss_sum/total).backward()
                        batch_nll += loss_sum.detach().item()
                        del logits, targets, loss_sum
                    scaler.unscale_(opt)
                    grad = torch.nn.utils.clip_grad_norm_(trainable, 1.)
                    if not math.isfinite(batch_nll):
                        raise FloatingPointError(f'nonfinite loss on {step}')
                    scale = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    if scaler.get_scale() >= scale:
                        break
                else:
                    raise FloatingPointError(f'AMP overflow repeated at step {step}')
                torch.cuda.synchronize()
                elapsed += time.perf_counter()-start
                entry = dict(step=step, epoch=epoch, arm=args.arm, nonpad_input_tokens=sum(r['split'] for r in records), nonpad_total_tokens=sum(len(r['ids']) for r in records), ids=[r['id'] for r in records], targets=total, nll_sum=batch_nll,
                             loss=batch_nll/total, lr=lr, gradient_norm=float(grad), amp_retries=retry, train_seconds=elapsed)
                log_file.write(json.dumps(entry, ensure_ascii=False)+'\n')
                if step in (1, 5) or step%100 == 0:
                    print('UPDATE', step, round(elapsed, 1), round(batch_nll/total, 5), flush=True)
                if step in STAGES:
                    name = STAGES[step]
                    stats = validate(model, val, tok.pad_token_id)
                    save(out/f'validation-{name}.json', stats)
                    checkpoint(model, opt, scaler, out/name, dict(step=step, name=name, train_seconds=elapsed,
                               total_examples=(step//563)*2250, data_hashes=data_hashes, validation_loss=stats['loss'],
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
                    model.train()
                    print('CHECKPOINT', name, 'validation', stats['loss'], flush=True)
    final = dict(report, train_seconds=elapsed, steps=step, checkpoints=list(STAGES.values()))
    save(out/'completed-training.json', final)
    print('TRAINING COMPLETE', elapsed, flush=True)


@torch.inference_mode()
def generate(args):
    prompts = []
    for path in args.prompts:
        prompts.extend(rows(path) if path.suffix == '.jsonl' else json.loads(path.read_text()))
    assert len({q['id'] for q in prompts}) == len(prompts) and prompts
    hashes = {str(p):sha(p) for p in args.prompts}
    out = args.out
    base, tok = model_and_tokenizer(args.base)
    model = base
    gen_dir = out/'generations'
    gen_dir.mkdir(exist_ok=True, parents=True)
    args.stages = args.stages or [BASE] + list(STAGES.values())
    for stage in args.stages:
        path = gen_dir/(stage+'.jsonl')
        spec_path = gen_dir/(stage+'-meta.json')
        if path.exists() and spec_path.exists():
            meta = json.loads(spec_path.read_text())
            if meta['prompt_hashes'] == hashes and meta['output_sha256'] == sha(path):
                continue
        if stage != BASE:
            if model is base:
                model = PeftModel.from_pretrained(base, out/stage, adapter_name=stage.replace('.', '_'), is_trainable=False)
            else:
                model.load_adapter(out/stage, adapter_name=stage.replace('.', '_'))
                model.set_adapter(stage.replace('.', '_'))
        else:
            assert model is base, 'Generate the pure base first'
        model.eval()
        tmp = path.with_suffix('.tmp')
        start = time.perf_counter()
        endings = Counter()
        with tmp.open('w', buffering=1) as f:
            for i in range(0,len(prompts),args.gen_batch):
                sub = prompts[i:i+args.gen_batch]
                inputs = tok([prompt(q['prompt']) for q in sub], padding=True, return_tensors='pt',
                             add_special_tokens=False).to('cuda')
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1,
                                        eos_token_id=[EOS,IM_END], pad_token_id=tok.pad_token_id,
                                        repetition_penalty=1., use_cache=True)
                for q, seq in zip(sub, output[:, inputs.input_ids.shape[1]:].tolist()):
                    end = next((j for j,x in enumerate(seq) if x in (EOS,IM_END)), len(seq))
                    stop = seq[end] if end < len(seq) else None
                    ids = seq[:end+1]
                    record = dict(id=q['id'], prompt=q['prompt'], checkpoint=stage,
                                  text=tok.decode(seq[:end], clean_up_tokenization_spaces=False),
                                  raw=tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                                  token_ids=ids, generated_tokens=len(ids), ended_eos=stop == EOS,
                                  ended_any_stop=stop is not None, stop_token_id=stop)
                    endings[str(stop)] += 1
                    f.write(json.dumps(record, ensure_ascii=False)+'\n')
        torch.cuda.synchronize()
        tmp.replace(path)
        meta = dict(greedy=True, max_new_tokens=256, prompt_hashes=hashes, count=len(prompts), batch_size=args.gen_batch,
                    stop_ids=[EOS,IM_END], stop_counts=dict(endings), seconds=time.perf_counter()-start, output_sha256=sha(path))
        save(spec_path, meta)
        print('GENERATED', stage, len(prompts), meta['seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train','generate'])
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--base-manifest', type=Path, default=Path(__file__).with_name('base-manifest.json'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, nargs='+')
    parser.add_argument('--stages', nargs='+', choices=[BASE]+list(STAGES.values()))
    parser.add_argument('--arm', choices=['control','paired'],default='control')
    parser.add_argument('--schedule', type=Path, default=Path(__file__).with_name('schedule.json'))
    parser.add_argument('--gen-batch', type=int, default=8)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'This experiment requires CUDA'
    torch.set_num_threads(4)
    if args.command == 'train':
        assert args.data
        train(args)
    else:
        assert args.prompts
        generate(args)

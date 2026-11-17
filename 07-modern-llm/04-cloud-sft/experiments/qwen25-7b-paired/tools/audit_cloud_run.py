"""Verify paired runs and compute token budgets per quarter from the actual tokenized batches."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json

BASE_ID = 'Qwen/Qwen2.5-7B'
REV = 'd149729398750b98c0af14eb82c78cfe92750796'
STAGES = ['epoch-'+str(i/4).rstrip('0').rstrip('.') for i in range(1,9)]

def rows(p):
    with p.open(encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def j(p): return json.loads(p.read_text())
def h(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def audit_arm(arm,train_path,args,schedule,prompts):
    r=args.root/'run'/arm
    cfg=j(r/'completed-training.json')
    assert cfg['base']==BASE_ID and cfg['revision']==REV
    assert cfg['data_hashes']=={'train.jsonl':h(train_path),'validation.jsonl':h(args.validation)}
    assert cfg['schedule_sha256']==h(args.root/'schedule.json')
    assert cfg['steps']==4504 and cfg['arm']==arm and cfg['base_parameters']==7615616512 and cfg['trainable_parameters']==20185088
    tr={x['id']:x for x in rows(train_path)}
    assert len(tr)==9000 and {i:x['supervised_tokens'] for i,x in tr.items()}==schedule['control_targets']
    v={x['id']:x for x in rows(args.validation)}
    assert len(v)==120 and not (v.keys() & tr.keys())
    u=rows(r/'updates.jsonl')
    assert len(u)==4504 and [row['step'] for row in u]==list(range(1,4505))
    out=[]
    all_sha=[]
    for n,stage in enumerate(STAGES):
        window=u[563*n:563*(n+1)]
        ids=[i for row in window for i in row['ids']]
        sched=schedule['quarters'][n]
        assert ids==sched['ids'] and len(set(ids))==2250
        assert [len(row['ids']) for row in window]==[4]*562+[2]
        for row in window:
            assert row['arm']==arm and row['epoch']==n//4+1 and row['targets']==sum(tr[i]['supervised_tokens'] for i in row['ids'])
            assert row['nonpad_input_tokens']==sum(tr[i]['input_tokens'] for i in row['ids'])
            assert row['nonpad_total_tokens']==row['nonpad_input_tokens']+row['targets']
            assert row['lr']==1e-4*min(1,row['step']/45) and 0<row['loss']<1000
        ck=r/stage
        state=j(ck/'state.json')
        assert state['step']==(n+1)*563 and state['total_examples']==(n+1)*2250
        digest=h(ck/'adapter_model.safetensors')
        assert digest==state['adapter_sha256'] and (ck/'training_state.pt').stat().st_size>1000000
        adapter=j(ck/'adapter_config.json')
        assert adapter['base_model_name_or_path']==BASE_ID and adapter['revision']==REV and adapter['r']==8 and adapter['lora_alpha']==16
        all_sha.append(digest)
        obs=j(r/f'validation-{stage}.json')
        assert obs['loss']==state['validation_loss']
        assert {s['id']:s['body_targets']+1 for s in obs['rows']}=={i:x['supervised_tokens'] for i,x in v.items()}
        loss=sum(x['body_nll_sum']+x['end_nll'] for x in obs['rows'])/obs['targets']
        assert abs(loss-obs['loss'])<1e-10
        tok=sum(row['targets'] for row in window); assert tok==sched['supervised_tokens']
        inp=sum(row['nonpad_input_tokens'] for row in window)
        assert inp==sched['input_tokens']
        out.append(dict(stage=stage,examples=len(ids),updates=len(window),supervised_tokens=tok,nonpad_input_tokens=inp,
                        nonpad_total_tokens=inp+tok,online_ce=sum(row['nll_sum'] for row in window)/tok,
                        heldout_ce=loss,body_ce=obs['body_nll'],end_ce=obs['eos_nll'],train_seconds=state['train_seconds'],
                        adapter_sha256=digest,optimizer_state_bytes=(ck/'training_state.pt').stat().st_size,
                        peak_allocated_gib=state['peak_allocated_gib'],peak_reserved_gib=state['peak_reserved_gib']))
    for n in [0,1]:
        used=Counter(i for row in u[n*2252:(n+1)*2252] for i in row['ids'])
        assert used==Counter(tr.keys()) and sum(x['targets'] for x in u[n*2252:(n+1)*2252])==1353439
    assert len(set(all_sha))==8
    gen={}
    for stage in (['base'] if arm=='control' else [])+STAGES:
        gp=r/'generations'/(stage+'.jsonl')
        answers=rows(gp)
        assert len(answers)==len(prompts)==120 and {s['id']:s['prompt'] for s in answers}==prompts
        meta=j(gp.with_name(stage+'-meta.json'))
        assert meta['max_new_tokens']==256 and meta['output_sha256']==h(gp) and list(meta['prompt_hashes'].values())==[h(args.prompts)]
        for x in answers:
            assert x['checkpoint']==stage and len(x['token_ids'])==x['generated_tokens'] and len(x['token_ids'])<=256
            assert x['ended_eos']==(x['stop_token_id']==151643) and x['ended_any_stop']==(x['stop_token_id'] in (151643,151645))
            assert (x['token_ids'][-1]==x['stop_token_id']) if x['ended_any_stop'] else len(x['token_ids'])==256
        gen[stage]={'responses':len(answers),'sha256':h(gp),'seconds':meta['seconds'],'stops':meta['stop_counts']}
    return dict(supervised_tokens=2706878,examples=18000,train_seconds=cfg['train_seconds'],stages=out,generated=gen)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--control',type=Path,required=True);ap.add_argument('--paired',type=Path,required=True)
    ap.add_argument('--prompts',type=Path,required=True);ap.add_argument('--validation',type=Path,required=True)
    a=ap.parse_args(); schedule=j(a.root/'schedule.json');pro={x['id']:x['prompt'] for x in rows(a.prompts)}
    result={n:audit_arm(n,getattr(a,n),a,schedule,pro) for n in ('control','paired')}
    for c,t in zip(result['control']['stages'],result['paired']['stages']):
        assert all(c[k]==t[k] for k in ('stage','examples','updates','supervised_tokens','nonpad_input_tokens','nonpad_total_tokens'))
    ca={r['id']:r for r in rows(a.control)};pa={r['id']:r for r in rows(a.paired)}
    result['changed_prompts']=sum(ca[i]['prompt']!=pa[i]['prompt'] for i in ca)
    result['changed_responses']=sum(ca[i]['response']!=pa[i]['response'] for i in ca)
    result['equal_supervised_lengths_all_9000_rows']=True
    result['measured_unique_generations']=sum(m['responses'] for key in ('control','paired') for m in result[key]['generated'].values())
    assert result['measured_unique_generations']==2040
    (a.root/'verified-run.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Verified 2 arms, 16 distinct checkpoints, 9008 updates and 2040 generations.')

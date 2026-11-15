"""单 GPU / CPU / MPS 的可观察 SFT；云端使用独立 requirements-cloud.txt。"""
import argparse
import importlib.metadata
import json
import platform
import random
import shutil
from contextlib import nullcontext
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from sft_data import (HERE, MODEL_ID, REVISION, data_hash, read_rows, encode_row, collate,
                      prompt_ids, score, aggregate, tokenizer_for)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def device_for(name):
    if name=='auto':name='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device=torch.device(name)
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('没有可用 CUDA，检查实例是否分配 GPU 与镜像驱动')
    return device


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize()
    elif device.type=='mps':torch.mps.synchronize()


def environment(device):
    report={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'device':str(device),'compute_dtype':'bfloat16' if device.type=='cuda' else 'float32','model_id':MODEL_ID,'revision':REVISION,'data_hash':data_hash(),
            'packages':{name:importlib.metadata.version(name) for name in ['transformers','tokenizers','peft','accelerate']}}
    if device.type=='cuda':
        report.update(gpu=torch.cuda.get_device_name(),gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
                      bf16_supported=torch.cuda.is_bf16_supported())
    return report


def load_model(device,mode='base',checkpoint=None,train=False,config=None):
    # 全参数更新保留 FP32 权重/梯度/Adam；CUDA 前向使用 BF16 autocast。
    # LoRA 基座和统一的前后生成评测以 BF16 加载；适配器保留 PEFT 默认 FP32。
    dtype=torch.float32 if device.type!='cuda' or (train and mode=='full') else torch.bfloat16
    source=str(checkpoint) if mode=='full' and checkpoint else MODEL_ID
    kwargs={'dtype':dtype,'attn_implementation':'sdpa','trust_remote_code':False,'low_cpu_mem_usage':True}
    if source==MODEL_ID:kwargs['revision']=REVISION
    else:kwargs['local_files_only']=True
    model=AutoModelForCausalLM.from_pretrained(source,**kwargs)
    if mode=='lora':
        if checkpoint:model=PeftModel.from_pretrained(model,str(checkpoint),is_trainable=train)
        else:
            model=get_peft_model(model,LoraConfig(r=config['rank'],lora_alpha=config['alpha'],lora_dropout=0.,
                         target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                         bias='none',task_type='CAUSAL_LM'))
    model.to(device)
    model.config.use_cache=False
    if train and config['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train(train)
    return model


def forward_context(model):
    device=next(model.parameters()).device
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()


def preflight(device,mode,out=None,required_gib=None):
    if device.type!='cuda':
        raise RuntimeError('当前真实 4B 实验面向 A800 单卡，请使用 --device cuda；本机只执行 inspect 和小模型测试')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('当前 CUDA 设备不支持本配置要求的 BF16')
    if mode=='full' and torch.cuda.get_device_properties(device).total_memory/2**30<70:
        raise RuntimeError('全参数配置按单卡 80GB 准备；当前总显存不足 70GiB，请先调整方案')
    if out is not None:
        parent=Path(out).resolve().parent
        while not parent.exists():parent=parent.parent
        free=shutil.disk_usage(parent).free/2**30
        minimum=required_gib if required_gib is not None else (60 if mode=='full' else 12)
        if free<minimum:raise RuntimeError(f'磁盘可用 {free:.1f} GiB，小于本次至少 {minimum} GiB；全参数恢复状态很大，请先准备空间')
        print(f'磁盘可用 {free:.1f} GiB；本次检查下限 {minimum} GiB（不包含后续导出副本）',flush=True)


def batch_on(rows,tokenizer,device):
    return {k:v.to(device) for k,v in collate(rows,tokenizer.pad_token_id).items()}


@torch.no_grad()
def validation_loss(model,rows,tokenizer,device):
    previous=model.training;model.eval();total=count=0
    try:
        for row in rows:
            batch=batch_on([row],tokenizer,device)
            with forward_context(model):
                out=model(**batch,use_cache=False)
            n=int(batch['labels'][:,1:].ne(-100).sum())
            total+=out.loss.item()*n;count+=n
    finally:model.train(previous)
    return total/count


def update(model,optimizer,batches):
    """按有效回答 token 总数加权；不是简单平均不同长度 micro-batch 的 loss。"""
    optimizer.zero_grad(set_to_none=True)
    n_total=sum(int(b['labels'][:,1:].ne(-100).sum()) for b in batches)
    loss_sum=0.
    for batch in batches:
        n=int(batch['labels'][:,1:].ne(-100).sum())
        with forward_context(model):
            out=model(**batch,use_cache=False)
        (out.loss*(n/n_total)).backward()
        loss_sum+=out.loss.detach().float().item()*n
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True,foreach=False)
    optimizer.step()
    return loss_sum/n_total,float(norm),n_total


def save_checkpoint(path,model,optimizer,generator,step,config,tokenizer,device):
    path.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    state={'optimizer':optimizer.state_dict(),'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),'step':step}
    if device.type=='cuda':state['cuda_rng']=torch.cuda.get_rng_state_all()
    if device.type=='mps':state['mps_rng']=torch.mps.get_rng_state()
    torch.save(state,path/'training_state.pt')
    write_json(path/'run_metadata.json',{'config':config,'model_id':MODEL_ID,'revision':REVISION,
               'data_hash':data_hash(),'environment':environment(device),'step':step})
    # 最后写元数据；加载时缺少元数据会拒绝不完整 checkpoint。


def train(args):
    config=json.loads(Path(args.config).read_text())
    if args.steps is not None:config['steps']=args.steps
    if any(config[k]<1 for k in ['steps','batch_size','gradient_accumulation','save_every','eval_every']):
        raise ValueError('steps、batch_size、gradient_accumulation、save_every 和 eval_every 必须为正')
    device=device_for(args.device)
    out=Path(args.out)
    if out.exists():raise ValueError('输出目录已存在，请换一个新目录；恢复使用 --resume 指向 checkpoint')
    # 一次 checkpoint 约 45GiB（全参数），默认只在最后保存。
    checkpoints=(config['steps']-1)//config['save_every']+1
    preflight(device,config['mode'],out,required_gib=(50*checkpoints+10 if config['mode']=='full' else 12+checkpoints))
    start=0;state=None
    if args.resume:
        saved=json.loads((Path(args.resume)/'run_metadata.json').read_text())
        if saved['data_hash']!=data_hash() or saved['revision']!=REVISION or saved['model_id']!=MODEL_ID:raise ValueError('数据或基座版本变化，不能恢复同一次训练')
        for key,value in config.items():
            if key!='steps' and saved['config'][key]!=value:raise ValueError(f'恢复配置变化：{key}')
        start=saved['step']
        if config['steps']<=start:raise ValueError('总 steps 必须大于断点步数')
        state=torch.load(Path(args.resume)/'training_state.pt',map_location='cpu',weights_only=True)
    random.seed(config['seed']);torch.manual_seed(config['seed'])
    tokenizer=tokenizer_for()
    rows=[encode_row(tokenizer,r,config['max_length']) for r in read_rows('train')]
    val=[encode_row(tokenizer,r,config['max_length']) for r in read_rows('validation')]
    model=load_model(device,config['mode'],args.resume,train=True,config=config)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config['learning_rate'],weight_decay=.01,foreach=False)
    generator=torch.Generator().manual_seed(config['seed'])
    if state:
        optimizer.load_state_dict(state['optimizer']);generator.set_state(state['generator']);torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and 'cuda_rng' in state:torch.cuda.set_rng_state_all(state['cuda_rng'])
        if device.type=='mps' and 'mps_rng' in state:torch.mps.set_rng_state(state['mps_rng'])
        del state  # 释放从 CPU 载入的 Adam 状态副本。
    out.mkdir(parents=True)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(out/'tensorboard'))
    report=environment(device)
    report.update(config=config,weight_dtype=str(next(model.parameters()).dtype),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in model.parameters()),resumed_from=args.resume,
                  validation_before=validation_loss(model,val,tokenizer,device))
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    latest=None;synchronize(device);started=time.perf_counter();targets=0
    with (out/'metrics.jsonl').open('w') as log:
        try:
            for step in range(start+1,config['steps']+1):
                batches=[]
                for _ in range(config['gradient_accumulation']):
                    indices=torch.randint(len(rows),(config['batch_size'],),generator=generator).tolist()
                    batches.append(batch_on([rows[i] for i in indices],tokenizer,device))
                synchronize(device);t=time.perf_counter()
                loss,norm,tokens=update(model,optimizer,batches)
                synchronize(device);elapsed=time.perf_counter()-t;targets+=tokens
                entry={'step':step,'loss':loss,'gradient_norm_before_clip':norm,'answer_tokens':tokens,
                       'update_seconds':elapsed,'answer_tokens_per_second':tokens/elapsed,
                       'clipped':norm>1.0,'clip_scale':min(1.,1/(norm+1e-6))}
                if step%config['eval_every']==0 or step==config['steps']:
                    entry['validation_loss']=validation_loss(model,val,tokenizer,device)
                if device.type=='cuda':entry['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                for key,value in entry.items():
                    if key!='step':writer.add_scalar(key,value,step)
                log.write(json.dumps(entry)+'\n');log.flush();writer.flush()
                print(json.dumps(entry),flush=True)
                if step%config['save_every']==0 or step==config['steps']:
                    latest=out/f'checkpoint-{step:06d}'
                    save_checkpoint(latest,model,optimizer,generator,step,config,tokenizer,device)
        finally:writer.close()
    synchronize(device)
    report.update(final_checkpoint=str(latest),elapsed_training_seconds=time.perf_counter()-started,
                  answer_tokens_processed=targets,validation_after=validation_loss(model,val,tokenizer,device))
    if device.type=='cuda':
        report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    write_json(out/'summary.json',report)


@torch.inference_mode()
def evaluate(args):
    path=Path(args.out)
    if path.exists():raise ValueError('评估输出已存在，请用新文件名保留旧结果')
    if args.limit is not None and args.limit<1:raise ValueError('limit 必须为正')
    device=device_for(args.device)
    preflight(device,'eval')
    tokenizer=tokenizer_for()
    mode='base';checkpoint=None
    if args.checkpoint:
        checkpoint=Path(args.checkpoint)
        metadata=json.loads((checkpoint/'run_metadata.json').read_text())
        if metadata['revision']!=REVISION or metadata['data_hash']!=data_hash() or metadata['model_id']!=MODEL_ID:raise ValueError('权重版本或数据与当前实验不一致')
        mode=metadata['config']['mode']
    model=load_model(device,mode,checkpoint)
    rows=read_rows(args.split)
    if args.limit is not None:rows=rows[:args.limit]
    predictions=[]
    for row in rows:
        ids=torch.tensor([prompt_ids(tokenizer,row)],device=device)
        result=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=128,
                              do_sample=False,temperature=None,top_p=None,top_k=None,
                              use_cache=True,pad_token_id=tokenizer.pad_token_id)
        generated=result[0,ids.shape[1]:]
        text=tokenizer.decode(generated,skip_special_tokens=True)
        eos=model.generation_config.eos_token_id
        eos=eos if isinstance(eos,list) else [eos]
        entry={'id':row['id'],'text':row['text'],'target':row['answer'],'prediction':text,
               'stop_reason':'eos' if generated[-1].item() in eos else 'max_new_tokens',
               'score':score(text,row['answer'])}
        predictions.append(entry);print(row['id'],entry['score'],repr(text),flush=True)
    report={'environment':environment(device),'split':args.split,'ids':[r['id'] for r in rows],
            'checkpoint':args.checkpoint,'generation':{'max_new_tokens':128,'do_sample':False,'enable_thinking':False},
            'metrics':aggregate(predictions),'predictions':predictions}
    path=Path(args.out)
    if path.exists():raise ValueError('评估输出已存在，请用新文件名保留旧结果')
    path.parent.mkdir(parents=True,exist_ok=True);write_json(path,report)
    print(report['metrics'])


def compare(args):
    before=json.loads(Path(args.before).read_text());after=json.loads(Path(args.after).read_text())
    for key in ['split','ids','generation']:
        if before[key]!=after[key]:raise ValueError(f'两份评测的 {key} 不一致')
    for key in ['model_id','revision','data_hash','compute_dtype']:
        if before['environment'][key]!=after['environment'][key]:raise ValueError(f'{key} 不一致')
    result={key:{'before':before['metrics'][key],'after':after['metrics'][key],
                  'delta':after['metrics'][key]-before['metrics'][key]}
            for key in ['json_valid_rate','schema_valid_rate','exact_match_rate','field_accuracy']}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.out:write_json(args.out,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('doctor');p.add_argument('--device',default='auto')
    p=sub.add_parser('inspect');p.add_argument('--max-length',type=int,default=512)
    p=sub.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int);p.add_argument('--resume')
    p=sub.add_parser('eval');p.add_argument('--checkpoint');p.add_argument('--out',required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--limit',type=int)
    p.add_argument('--device',default='cuda')
    p=sub.add_parser('compare');p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--out')
    args=parser.parse_args()
    if args.command=='doctor':
        version=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
        if version<(2,6):raise RuntimeError('请用 PyTorch >= 2.6 的镜像')
        print(json.dumps(environment(device_for(args.device)),ensure_ascii=False,indent=2))
    elif args.command=='inspect':
        tok=tokenizer_for();encoded=[]
        for split in ['train','validation','test']:
            rows=[encode_row(tok,r,args.max_length) for r in read_rows(split)]
            print(split,len(rows),'最大长度',max(len(r['input_ids']) for r in rows));encoded+=rows
        first=encoded[0]
        print('prompt:',tok.decode(first['input_ids'][:first['prompt_length']]))
        print('参与loss:',tok.decode([x for x in first['labels'] if x!=-100]))
        print('prompt token:',first['prompt_length'],'answer+EOS token:',sum(x!=-100 for x in first['labels']))
    elif args.command=='train':train(args)
    elif args.command=='eval':evaluate(args)
    else:compare(args)
if __name__=='__main__':main()

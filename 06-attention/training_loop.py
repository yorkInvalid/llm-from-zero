"""Periodic validation and complete same-plan training checkpoints."""
import hashlib
import json
import math
import time
import torch
from torch import nn
from text_pretraining import collate
from lm_components import learning_rate_at_step
from training_monitor import TrainingMonitor
from training_progress import TrainingProgress
from training_state import atomic_save, cpu_tree, epoch_batches, rng_state, restore_rng, preserve_rng, validate_resume


def train(args,model,tokenizer,splits,optimizer,scaler,config,metadata,resumed=None):
    from train_lm import evaluate,observe_epoch,restore,inspect_batch
    from lm_report import write_observation_report
    batches_per_epoch=math.ceil(len(splits['train'])/args.batch_size)
    tok_json=(args.data_dir/'tokenizer.json').read_text()
    tok_hash=hashlib.sha256((args.data_dir/'tokenizer.json').read_bytes()).hexdigest()
    probe_ids=torch.randperm(len(splits['train']),generator=torch.Generator().manual_seed(31415)).tolist()[:args.train_probe_windows]
    probe=[splits['train'][i] for i in probe_ids]
    metadata['train_probe']=dict(windows=len(probe),seed=31415,meaning='fixed subset, not full training-set loss')
    train_tag = 'train' if args.full_train_eval else 'train_probe'
    metadata['train_metric_scope'] = 'full_training_set' if args.full_train_eval else 'fixed_training_probe'
    if resumed:
        validate_resume(resumed,config,metadata['split_sha256'],tok_hash)
        if resumed['global_step']>=config['total_steps']:
            raise ValueError('训练计划已完成；新增训练计划请使用 --init-checkpoint')
        optimizer.load_state_dict(resumed['optimizer'])
        scaler.load_state_dict(resumed['scaler'])
    with TrainingMonitor(args.run_dir/'tensorboard',metadata) as monitor:
        start=time.monotonic()
        def infer_payload(epoch,step,validation):
            return dict(config=model.config,model=cpu_tree(model.state_dict()),tokenizer=tok_json,
                        tokenizer_sha256=tok_hash,epoch=epoch,global_step=step,
                        training_config=config,selection_validation=validation)
        if resumed:
            history=resumed['history'];observations=resumed['observations'];checks=resumed['checks']
            best_payload=cpu_tree(resumed['best_payload']);best=best_payload['selection_validation']['loss']
            global_step=resumed['global_step'];start_epoch=resumed['next_epoch'];start_batch=resumed['next_batch']
            tokens_seen = resumed.get('tokens_seen')
            if tokens_seen is None:
                tokens_seen = (start_epoch - 1) * metadata['counts']['train']['tokens']
                prior = epoch_batches(len(splits['train']), args.batch_size, start_epoch, config['seed'])[:start_batch]
                tokens_seen += sum(splits['train'][i][1].numel() for batch in prior for i in batch)
            atomic_save(best_payload,args.run_dir/'best.pt')
            for metrics,observation in zip(history,observations):
                monitor.epoch(metrics['global_step'],metrics,observation,train_tag=train_tag,
                              log_train=args.full_train_eval,log_validation=False)
            for check in checks:
                for name in ('validation','train_probe'):
                    for metric,value in check[name].items():monitor.writer.add_scalar(f'{name}/{metric}',value,check['global_step'])
        else:
            print('正在评估初始模型…',flush=True)
            with preserve_rng(args.device):
                probe_metrics=evaluate(model,probe,args.batch_size)
                initial=dict(epoch=0,global_step=0,
                             train=evaluate(model,splits['train'],args.batch_size) if args.full_train_eval else probe_metrics,
                             train_scope=train_tag,
                             validation=evaluate(model,splits['validation'],args.batch_size))
                observation=observe_epoch(model,tokenizer,splits['validation'],0,prompts=args.observation_prompts)
            history=[initial];observations=[observation]
            checks=[dict(epoch=0,global_step=0,validation=initial['validation'],train_probe=probe_metrics)]
            global_step=0;start_epoch=1;start_batch=0
            tokens_seen=0
            best=initial['validation']['loss'];best_payload=infer_payload(0,0,initial['validation'])
            atomic_save(best_payload,args.run_dir/'best.pt')
            monitor.epoch(0,initial,observation,train_tag=train_tag)
            if args.full_train_eval:
                for metric,value in probe_metrics.items():monitor.writer.add_scalar(f'train_probe/{metric}',value,0)
            print(json.dumps(initial),flush=True)
        initial_eval_seconds=time.monotonic()-start
        monitor.writer.add_scalar('time/initial_evaluation_seconds',initial_eval_seconds,global_step)
        progress=TrainingProgress(config['total_steps'],args.epochs-start_epoch+1,monitor,
                                  initial_step=global_step,eval_every=args.eval_every,
                                  initial_epoch_eval_seconds=0. if resumed else initial_eval_seconds)
        if resumed:restore_rng(resumed['rng'],args.device)

        def log_tokens():
            monitor.writer.add_scalar('progress/train_tokens_seen', tokens_seen, global_step)
            monitor.writer.add_scalar('progress/train_token_budget', config['train_token_budget'], global_step)
            monitor.writer.add_scalar('progress/token_fraction', tokens_seen/config['train_token_budget'], global_step)
        log_tokens()

        def save_full(next_epoch,next_batch):
            atomic_save(dict(checkpoint_version=1,config=model.config,model=model.state_dict(),
                             tokenizer=tok_json,tokenizer_sha256=tok_hash,training_config=config,
                             split_sha256=metadata['split_sha256'],optimizer=optimizer.state_dict(),
                             scaler=scaler.state_dict(),rng=rng_state(args.device),global_step=global_step,
                             tokens_seen=tokens_seen,
                             epoch=global_step/batches_per_epoch,next_epoch=next_epoch,next_batch=next_batch,
                             best_payload=best_payload,history=history,observations=observations,checks=checks),
                        args.run_dir/'last.pt')
        if not resumed:save_full(1,0)
        paused=False
        interval_start=time.monotonic();interval_tokens=0
        for epoch in range(start_epoch,args.epochs+1):
            model.train()
            batches=epoch_batches(len(splits['train']),args.batch_size,epoch,config['seed'])
            for batch_index in range(start_batch if epoch==start_epoch else 0,len(batches)):
                step_start=time.monotonic();global_step+=1
                lr=learning_rate_at_step(global_step,config['total_steps'],config['warmup_steps'],
                                         config['peak_lr'],config['min_lr_ratio'],
                                         schedule=config['schedule'],cooldown_steps=config.get('cooldown_steps',0))
                for group in optimizer.param_groups:group['lr']=lr
                x,y=collate([splits['train'][i] for i in batches[batch_index]])
                x,y=x.to(args.device),y.to(args.device)
                for attempt in range(8):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=args.precision=='fp16'):
                        logits=model(x)
                        loss=nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite forward loss')
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    grad=nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(grad):
                        if args.precision!='fp16':raise FloatingPointError('Non-finite gradients')
                        scaler.update(new_scale=scaler.get_scale()*.5);continue
                    scaler.step(optimizer);scaler.update();break
                else:raise FloatingPointError('AMP overflow persisted for 8 attempts')
                valid_tokens=(y!=-100).sum().item()
                interval_tokens+=valid_tokens
                tokens_seen+=valid_tokens
                progress.step(time.monotonic()-step_start,loss.item(),epoch)
                monitor.writer.add_scalar('optimization/amp_scale',scaler.get_scale(),global_step)
                monitor.writer.add_scalar('optimization/amp_retries',attempt,global_step)
                if global_step%args.log_every==0 or global_step==1 or batch_index==len(batches)-1:
                    log_tokens()
                    monitor.step(global_step,loss.item(),lr,grad.item(),interval_tokens/max(time.monotonic()-interval_start,1e-9))
                    interval_start=time.monotonic();interval_tokens=0
                end_epoch=batch_index==len(batches)-1
                stopping=0<args.stop_after_step<=global_step<config['total_steps']
                if global_step%args.eval_every==0 or end_epoch or stopping:
                    check_start=time.monotonic();fraction=global_step/batches_per_epoch
                    print(f'\nStep {global_step}: 正在评估固定训练探针、完整验证集并保存断点…',flush=True)
                    with preserve_rng(args.device):
                        validation=evaluate(model,splits['validation'],args.batch_size)
                        probe_metrics=evaluate(model,probe,args.batch_size)
                        observation=observe_epoch(model,tokenizer,splits['validation'],fraction,prompts=args.observation_prompts)
                        train_metrics=(evaluate(model,splits['train'],args.batch_size)
                                       if args.full_train_eval else probe_metrics) if end_epoch else None
                    check=dict(epoch=fraction,global_step=global_step,tokens_seen=tokens_seen,
                               validation=validation,train_probe=probe_metrics)
                    checks.append(check)
                    for name in ('train_probe','validation'):
                        for metric,value in check[name].items():monitor.writer.add_scalar(f'{name}/{metric}',value,global_step)
                    for i,g in enumerate(observation['generations']):
                        monitor.writer.add_text(f'generation/prompt_{i}',f"Step {global_step}\n\n{g['prompt']}\n\n{g['text']}",global_step)
                    if end_epoch:
                        metrics=dict(epoch=epoch,global_step=global_step,train=train_metrics,
                                     train_scope=train_tag,validation=validation)
                        history.append(metrics);observations.append(observation)
                        monitor.epoch(global_step,metrics,observation,train_tag=train_tag,
                                      log_train=args.full_train_eval,log_validation=False)
                    if validation['loss']<best:
                        best=validation['loss'];best_payload=infer_payload(fraction,global_step,validation)
                        atomic_save(best_payload,args.run_dir/'best.pt')
                    (args.run_dir/'validation_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
                    (args.run_dir/'observations.json').write_text(json.dumps(dict(history=history,observations=observations),ensure_ascii=False,indent=2))
                    save_full(epoch+1 if end_epoch else epoch,0 if end_epoch else batch_index+1)
                    monitor.writer.flush();print(json.dumps(check),flush=True)
                    if end_epoch:progress.evaluation(time.monotonic()-check_start)
                    else:progress.checkpoint(time.monotonic()-check_start)
                    interval_start=time.monotonic();interval_tokens=0
                if stopping:paused=True;break
            if paused:break
        best_model,_,saved=restore(args.run_dir/'best.pt',args.device)
        if not paused and tokens_seen != config['train_token_budget']:
            raise RuntimeError('completed updates do not match the declared training token budget')
        report=dict(history=history,checks=checks,best_epoch=saved['epoch'],best_step=saved['global_step'],
                    best_validation=saved['selection_validation'],seconds=time.monotonic()-start,
                    parameters=sum(p.numel() for p in model.parameters()),device=args.device,
                    model_config=model.config,training_config=config,global_step=global_step,paused=paused,
                    tokens_seen=tokens_seen,train_metric_scope=metadata['train_metric_scope'])
        (args.run_dir/'report.json').write_text(json.dumps(report,indent=2))
        (args.run_dir/'predictions.json').write_text(json.dumps(inspect_batch(tokenizer,splits['validation'],best_model),ensure_ascii=False,indent=2))
        write_observation_report(args.run_dir,history,observations,saved['epoch'],train_label=train_tag)
        monitor.writer.add_text('run/result',json.dumps(dict(best_step=saved['global_step'],validation=saved['selection_validation']),indent=2),global_step)
        monitor.last_step=global_step
        if paused:monitor.metadata['final_status']='paused'
        progress.close(completed=not paused)
        monitor.writer.add_scalar('time/total_wall_seconds',time.monotonic()-start,global_step)
        print(f"{'已保存中途断点' if paused else '训练完成'}: {args.run_dir}; best.pt 用于推理，last.pt 用于完整续训",flush=True)

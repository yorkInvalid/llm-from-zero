"""Atomic checkpoints and deterministic batch order for same-plan continuation."""
from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np
import torch


def rng_state(device):
    n = np.random.get_state()
    return dict(cpu=torch.get_rng_state(), python=random.getstate(),
                numpy=(n[0], torch.tensor(n[1].astype('int64')), n[2], n[3], n[4]),
                cuda=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
                mps=torch.mps.get_rng_state() if str(device)=='mps' else None)


def restore_rng(state, device):
    torch.set_rng_state(state['cpu'].cpu())
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],n[1].cpu().numpy().astype('uint32'),n[2],n[3],n[4]))
    if str(device).startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    if str(device)=='mps':torch.mps.set_rng_state(state['mps'].cpu())


@contextmanager
def preserve_rng(device):
    state=rng_state(device)
    try:yield
    finally:restore_rng(state,device)


def epoch_batches(length, batch_size, epoch, seed=42):
    order=torch.randperm(length,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
    return [order[i:i+batch_size] for i in range(0,length,batch_size)]


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(payload,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(cpu_tree(payload),temporary)
    temporary.replace(path)


def checkpoint_optimizer_config(saved):
    """Read the actual saved recipe, including checkpoints predating recipe metadata."""
    groups = saved.get('optimizer', {}).get('param_groups', [])
    if not groups:
        raise ValueError('完整续训需要包含优化器配置的 checkpoint')
    group = groups[0]
    recipe = dict(optimizer=saved.get('training_config', {}).get(
                      'optimizer', 'adamw' if group.get('decoupled_weight_decay', False) else 'adam'),
                  betas=list(group['betas']), weight_decay=group['weight_decay'],
                  optimizer_eps=group['eps'])
    if any((list(other['betas']), other['weight_decay'], other['eps']) !=
           (recipe['betas'], recipe['weight_decay'], recipe['optimizer_eps']) for other in groups):
        raise ValueError('当前训练入口要求所有参数组使用相同优化器配方')
    return recipe


def validate_resume(saved,config,hashes,tokenizer_hash):
    if saved.get('checkpoint_version')!=1:
        raise ValueError('不是完整训练 checkpoint；旧权重请使用 --init-checkpoint')
    for key in ('total_steps','warmup_steps','peak_lr','min_lr_ratio','schedule','batch_size',
                'seed','device','precision','eval_every','train_probe_windows'):
        if saved['training_config'].get(key)!=config.get(key):
            raise ValueError(f'断点续训配置不匹配: {key}；改变训练计划请使用 --init-checkpoint')
    if saved['split_sha256']!=hashes or saved['tokenizer_sha256']!=tokenizer_hash:
        raise ValueError('断点续训要求相同数据与 tokenizer 哈希')
    if saved['training_config'].get('full_train_eval', True) != config.get('full_train_eval', True):
        raise ValueError('断点续训配置不匹配: full_train_eval')
    if ('train_token_budget' in saved['training_config']
            and saved['training_config']['train_token_budget'] != config['train_token_budget']):
        raise ValueError('断点续训配置不匹配: train_token_budget')
    if saved['training_config'].get('cooldown_steps', 0) != config.get('cooldown_steps', 0):
        raise ValueError('断点续训配置不匹配: cooldown_steps')
    if saved['config'].get('dropout', .1) != config.get('dropout', saved['config'].get('dropout', .1)):
        raise ValueError('断点续训配置不匹配: dropout')
    recipe = checkpoint_optimizer_config(saved)
    for key, value in recipe.items():
        if config.get(key, value) != value:
            raise ValueError(f'断点续训配置不匹配: {key}')


def reuse_optimizer_state(optimizer, scaler, saved, precision):
    if saved.get('checkpoint_version') != 1 or not saved.get('optimizer', {}).get('state'):
        raise ValueError('保留优化器需要包含 Adam 状态的完整 checkpoint')
    if saved['training_config']['precision'] != precision:
        raise ValueError('保留优化器状态时须保持相同 precision')
    optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler'])

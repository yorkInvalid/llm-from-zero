"""Reproducible local audit: data invariants, model numerics, context ablation and initialization pilot."""
import argparse
import json
import math
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from causal_lm import CausalLM
from train_lm import restore, evaluate
from text_pretraining import load_data, collate
from position_encoding import apply_rope

ROOT = Path(__file__).parent / 'artifacts/book-lm'


def scaled_attention(model):
    # Change only attention initialization; all other tensors remain bit-identical.
    with torch.no_grad():
        for layer in model.layers:
            for group in (layer.attention.W_q, layer.attention.W_k, layer.attention.W_v):
                for p in group:
                    p.mul_((model.config['d_model'] ** -.5) / .1)
            layer.attention.W_o.mul_((model.config['d_model'] ** -.5) / .1)


@torch.no_grad()
def numerics(model, x):
    model.eval()
    stats = []
    z = model.embedding(x).transpose(-2, -1)
    for layer in model.layers:
        norm = layer.norm1(z.transpose(-2,-1)).transpose(-2,-1)
        a = layer.attention
        q, k = a.W_q[0] @ norm, a.W_k[0] @ norm
        positions = torch.arange(x.shape[-1], device=x.device)
        q, k = apply_rope(q, positions), apply_rope(k, positions)
        scores = k.transpose(-2,-1) @ q / math.sqrt(a.d_k)
        branch, w = a(norm, padding_mask=x==0, causal=True)
        entropy = -(w.clamp_min(1e-30).log()*w).sum(-2)
        before = z.square().mean().sqrt().item()
        z = layer(z, x==0)
        stats.append(dict(residual_rms=before, attention_rms=branch.square().mean().sqrt().item(),
                          score_std=scores.std().item(), mean_attention_entropy=entropy.mean().item(),
                          output_rms=z.square().mean().sqrt().item()))
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=160)
    args=parser.parse_args()
    torch.set_num_threads(4)
    data=ROOT/'industry-data'; out=ROOT/'industry-audit';out.mkdir(exist_ok=True)
    tok,splits=load_data(data,256)
    docs={n:json.loads((data/f'{n}.json').read_text()) for n in ('train','validation')}
    report={'data':{}}
    for n, ds in docs.items():
        ids=[tok.encode(d).ids for d in ds]
        assert all(tok.decode(i)==d for i,d in zip(ids,ds))
        assert sum(len(y) for _,y in splits[n])==sum(len(i)+1 for i in ids)
        for x,y in splits[n]: assert torch.equal(x[1:],y[:-1])
        counts=torch.bincount(torch.cat([y for _,y in splits[n]]),minlength=tok.get_vocab_size())
        top=counts.topk(12)
        report['data'][n]=dict(documents=len(ds),characters=sum(map(len,ds)),tokens=sum(map(len,ids)),
            max_document_chars=max(map(len,ds)),roundtrip=True,shift=True,coverage=True,
            top_tokens=[{'id':i,'text':tok.decode([i]),'count':c} for i,c in zip(top.indices.tolist(),top.values.tolist())],
            replacement_chars=sum(d.count('\ufffd') for d in ds),
            ascii_comma=sum(d.count(',') for d in ds), chinese_comma=sum(d.count('，') for d in ds))
    report['data']['exact_overlap']=len(set(docs['train'])&set(docs['validation']))
    # Deterministic sample distributed across validation windows, not used for training.
    rng=random.Random(42)
    val=[splits['validation'][i] for i in sorted(rng.sample(range(len(splits['validation'])),32))]
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    m,_,_=restore(ROOT/'industry-33m-continue/best.pt',device)
    x,y=collate(splits['train'][:2]);x=x.to(device)
    report['current_numerics']=numerics(m,x)
    # Causal invariance: perturb future input, earlier logits must remain unchanged.
    altered=x.clone();altered[:,128:]=3
    with torch.no_grad():
        report['future_leak_max_diff']=(m(x)[:,:128]-m(altered)[:,:128]).abs().max().item()
    # One-token context retains current token but removes earlier history; RoPE relative shifts invariant.
    with torch.no_grad():
        context=[]
        for n in range(8):
            inp,target=val[n];inp=inp.to(device)
            full=m(inp)
            positions=list(range(16,len(inp),16))
            short=m(inp[positions,None])[:,0,:]
            labels=target[positions].to(device)
            context.append([nn.functional.cross_entropy(full[positions],labels,reduction='sum').item(),
                            nn.functional.cross_entropy(short,labels,reduction='sum').item(),len(positions)])
        count=sum(r[2] for r in context)
        report['context_probe']=dict(positions=count,full_loss=sum(r[0] for r in context)/count,
                                    current_token_only_loss=sum(r[1] for r in context)/count)
    report['current_validation_probe']=evaluate(m,val,8)
    del m;torch.mps.empty_cache()
    report['initialization_pilot']={}
    # Both pilots start from identical random parameters and see identical batch sequence/dropout RNG.
    order=[rng.sample(range(len(splits['train'])),8) for _ in range(args.steps)]
    for kind in ('legacy','fan_in'):
        torch.manual_seed(123)
        m=CausalLM(tok.get_vocab_size(),d_model=512,num_layers=8,num_heads=8,context_length=256,attention_init='legacy').to(device)
        if kind=='fan_in': scaled_attention(m)
        stat=numerics(m,x)
        initial=evaluate(m,val,8)
        opt=torch.optim.Adam(m.parameters(),lr=3e-4)
        writer=SummaryWriter(str(out/kind/'tensorboard'))
        torch.manual_seed(999)
        for step,indices in enumerate(order,1):
            m.train(); a,b=collate([splits['train'][i] for i in indices]);a,b=a.to(device),b.to(device)
            opt.zero_grad(set_to_none=True)
            logits=m(a);loss=nn.functional.cross_entropy(logits.reshape(-1,tok.get_vocab_size()),b.flatten(),ignore_index=-100)
            loss.backward();grad=nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            if step%20==0:
                writer.add_scalar('optimization/loss',loss.item(),step);writer.flush()
                print(kind,step,loss.item(),flush=True)
        final=evaluate(m,val,8)
        report['initialization_pilot'][kind]=dict(steps=args.steps,seed=123,initial=initial,final=final,numerics=stat)
        writer.add_scalar('validation/loss',initial['loss'],0);writer.add_scalar('validation/loss',final['loss'],args.steps);writer.close()
        del m,opt,logits,loss;torch.mps.empty_cache()
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['initialization_pilot'],indent=2),flush=True)


if __name__=='__main__': main()

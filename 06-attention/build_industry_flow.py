"""Export actual checkpoint tensors and data transformations for the inline walkthrough."""
import json
from pathlib import Path
import torch
from torch import nn
from train_lm import restore
from text_pretraining import make_windows, collate

ROOT=Path(__file__).parent/'artifacts/book-lm'
OUT=Path('output/industry-flow')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model,tok,saved=restore(ROOT/'industry-33m-continue/best.pt',device)
    text=json.loads((ROOT/'industry-data/validation.json').read_text())[0]
    encoded=tok.encode(text)
    samples=make_windows([text],tok,256)
    selected=[samples[0],samples[-1]]
    x,y=collate(selected); x,y=x.to(device),y.to(device)
    stats=[]
    hooks=[]
    for i,layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(lambda m,a,o,i=i:stats.append(dict(layer=i+1,shape=list(o.shape),rms=o.square().mean().sqrt().item()))))
    with torch.no_grad():
        logits=model(x)
        for h in hooks: h.remove()
        emb=model.embedding(x).cpu()
        losses=nn.functional.cross_entropy(logits.reshape(-1,tok.get_vocab_size()),y.flatten(),ignore_index=-100,reduction='none').reshape(y.shape)
        n=model.layers[0].norm1(emb.to(device)).transpose(-2,-1)
        _,weights=model.layers[0].attention(n,padding_mask=x==0,causal=True)
        probs=logits.softmax(-1)
        rows=[]
        for i in range(32):
            ps,ids=probs[0,i].topk(5)
            rows.append(dict(position=i,input_id=x[0,i].item(),label_id=y[0,i].item(),
                             label=tok.decode([y[0,i].item()]),loss=losses[0,i].item(),
                             true_probability=probs[0,i,y[0,i]].item(),
                             predictions=[dict(id=t,text=tok.decode([t]),probability=p) for t,p in zip(ids.tolist(),ps.tolist())]))
        prompt_results=[]
        for prompt in ('人工智能','机器学习的主要方法','数据','深度学习是一种'):
            prefix=torch.tensor([1,*tok.encode(prompt).ids],device=device)
            next_probs=model(prefix)[-1].softmax(-1);ps,ids=next_probs.topk(5)
            generated=model.generate(prefix,max_new_tokens=32)
            prompt_results.append(dict(prompt=prompt,text=tok.decode(generated.tolist()),top5=[dict(text=tok.decode([t]),id=t,probability=p) for t,p in zip(ids.tolist(),ps.tolist())]))
    for h in hooks:h.remove()
    # A diagnostic one-step update in memory; checkpoint is never overwritten.
    model.eval(); opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    before=model.embedding.weight.detach().clone()
    loss=nn.functional.cross_entropy(model(x).reshape(-1,tok.get_vocab_size()),y.flatten(),ignore_index=-100)
    loss.backward(); grad=model.embedding.weight.grad.norm().item();opt.step()
    update=(model.embedding.weight-before).norm().item()
    result=dict(checkpoint='industry-33m-continue/best.pt',config=saved['config'],text=text,
                utf8=list(text[:12].encode('utf-8')),utf8_text=text[:12],
                ids=[1,*encoded.ids,2],tokens=['<BOS>',*encoded.tokens,'<EOS>'],
                decoded=['<BOS>',*[tok.decode([t]) for t in encoded.ids],'<EOS>'],
                windows=[dict(start=i*256,length=len(a),input=a.tolist(),labels=b.tolist()) for i,(a,b) in enumerate(samples)],
                batch=dict(input=x.cpu().tolist(),labels=y.cpu().tolist(),lengths=[len(s[0]) for s in selected]),
                embedding=emb[0,:16,:8].tolist(),layers=stats[:8],
                attention=weights[0,0,:16,:16].cpu().tolist(),rows=rows,
                batch_loss=losses[y!=-100].mean().item(),valid_labels=(y!=-100).sum().item(),
                update_demo=dict(loss=loss.item(),embedding_gradient_norm=grad,embedding_update_norm=update),
                predictions=prompt_results)
    (OUT/'flow-data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    raw_path = OUT/'raw-example.json'
    source = json.loads((ROOT/'industry-data/source.json').read_text())
    if not raw_path.exists():
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
        fs = HfFileSystem()
        with fs.open(f"datasets/BAAI/IndustryCorpus2@{source['revision']}/{source['shard']}", 'rb') as stream:
            raw_rows = pq.ParquetFile(stream).read_row_group(0, columns=['text','quality_score']).to_pylist()
        row = next(r for r in raw_rows if ' '.join(r['text'].split()) == text)
        raw_path.write_text(json.dumps(row, ensure_ascii=False))
    result['raw'] = json.loads(raw_path.read_text())
    if ' '.join(result['raw']['text'].split()) != text:
        raise ValueError('源文档与当前验证样本不匹配，请移除旧 raw-example.json 后重建')
    result['source'] = source
    template = (Path(__file__).parent/'assets/industry-flow.template.html').read_text()
    payload = json.dumps(result, ensure_ascii=False).replace('<', '\\u003c')
    (OUT/'flow.html').write_text(template.replace('__PAYLOAD__', payload))
    print('Exported',len(encoded.ids),'tokens',len(samples),'windows',flush=True)

if __name__=='__main__':main()

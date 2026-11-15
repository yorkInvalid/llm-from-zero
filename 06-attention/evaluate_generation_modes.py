"""Compare fixed greedy and seeded nucleus sampling before/after pretraining."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from train_lm import restore


@torch.no_grad()
def sample(model, prefix, max_new_tokens=64, temperature=.8, top_p=.9, seed=42):
    generator=torch.Generator(device=prefix.device).manual_seed(seed)
    output=prefix.clone()
    cache=None
    for _ in range(max_new_tokens):
        if cache is None or cache[0]['padding_mask'].shape[-1]>=model.config['context_length']:
            logits,cache=model(output[-model.config['context_length']:],use_cache=True)
        else:
            logits,cache=model(output[-1:],use_cache=True,kv_cache=cache)
        scores=logits[-1].float()/temperature
        scores[[model.config['pad_id'],1]]=-torch.inf
        probabilities,indices=scores.softmax(-1).sort(descending=True)
        remove=probabilities.cumsum(0)>top_p
        remove[1:]=remove[:-1].clone()
        remove[0]=False
        probabilities[remove]=0
        selected=torch.multinomial(probabilities,1,generator=generator)
        token=indices[selected]
        output=torch.cat((output,token))
        if token.item()==2:break
    return output


def repetition(tokens,n=4):
    windows=[tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return 1-len(set(windows))/len(windows) if windows else 0.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(1)
    prompts=['人工智能','机器学习的主要方法','数据','科学研究的意义',
             '机器学习是一种','教育的目的是帮助学生']
    results={'method':dict(max_new_tokens=64,temperature=.8,top_p=.9,seed=42,
                           repetition='fraction of repeated 4-token windows in generated tokens, excluding EOS',
                           caveat='Small fixed prompt panel; repetition reduction alone is not factuality or fluency.')}
    tokenizer_hash=None
    for label,path in [('before',args.before),('after',args.run_dir/'best.pt')]:
        model,tokenizer,saved=restore(path,args.device)
        if tokenizer_hash is not None and saved['tokenizer_sha256']!=tokenizer_hash:
            raise ValueError('tokenizer mismatch')
        tokenizer_hash=saved['tokenizer_sha256']
        model.eval()
        rows=[]
        for prompt in prompts:
            prefix=torch.tensor([1,*tokenizer.encode(prompt).ids],device=args.device)
            for mode in ('greedy','top_p'):
                output=(model.generate(prefix,max_new_tokens=64) if mode=='greedy' else sample(model,prefix))
                generated=output[len(prefix):].tolist()
                rows.append(dict(prompt=prompt,mode=mode,text=tokenizer.decode(output.tolist()),
                                 generated_tokens=len(generated),ended_with_eos=bool(generated and generated[-1]==2),
                                 repeated_4gram_fraction=repetition([i for i in generated if i!=2])))
        with path.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        results[label]=dict(checkpoint_sha256=checksum,rows=rows)
        del model,saved
        if args.device=='cuda':torch.cuda.empty_cache()
    destination=args.run_dir/'generation-modes-comparison.json'
    destination.write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(destination)


if __name__=='__main__':main()

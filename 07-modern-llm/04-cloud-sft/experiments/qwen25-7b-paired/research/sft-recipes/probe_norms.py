"""Compare two small original RMSNorm tensors in published checkpoint pairs. No GPU.
Only ranged safetensors requests are accepted, never full weight shards. Python 3.10+.
"""
import concurrent.futures
import hashlib
import json
import math
import struct
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

PAIRS = [
    ("Qwen2.5: base -> instruct (SFT and later alignment)",
     ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"),
     ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")),
    ("DeepSeek: Qwen2.5-Math-7B -> R1-Distill-Qwen-7B (SFT)",
     ("Qwen/Qwen2.5-Math-7B", "b101308fe89651ea5ce025f25317fea6fc07e96e"),
     ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "916b56a44061fd5cd7d6a8fb632557ed4f724f60")),
    ("MiMo: 7B-Base -> 7B-SFT (original, not SFT-0525)",
     ("XiaomiMiMo/MiMo-7B-Base", "d892dcc601b8a242e7b3e7d3bb9f55abbe72c0ad"),
     ("XiaomiMiMo/MiMo-7B-SFT", "0a9f1180d99e686ce56b75fee637db62407ceb1b")),
    ("MiMo: Qwen3.5-9B -> MiMo-V2.6-Distill-Qwen-9B (SFT)",
     ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
     ("XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", "1f09ddcc1a0ea45751ae0dfe2af20d533168c46f")),
]
SUFFIXES = ("layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight")

def fetch(url, lo=None, hi=None):
    size = hi-lo+1 if lo is not None else 8_000_000
    if size <= 0 or size > 8_000_000: raise ValueError("oversize read")
    headers = {"Accept-Encoding":"identity"}
    if lo is not None:
        headers["Range"] = f"bytes={lo}-{hi}"
        url += f"?range_probe={lo}-{hi}"  # separate keys for caches that ignore Vary: Range
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=18) as res:
                if lo is not None:
                    if res.status != 206 or not res.headers.get("Content-Range","").startswith(f"bytes {lo}-{hi}/"):
                        raise ValueError("server did not honor requested range")
                data=res.read(size+1)
            if len(data)>size or (lo is not None and len(data)!=size): raise ValueError("length mismatch")
            return data
        except (URLError, TimeoutError):
            if attempt==2: raise
            time.sleep(0.4*(attempt+1))

def extract(model):
    ident,rev=model
    base=f"https://huggingface.co/{ident}/resolve/{rev}/"
    index=json.loads(fetch(base+"model.safetensors.index.json"))["weight_map"]
    cache={}; out={}
    for suf in SUFFIXES:
        keys=[k for k in ("model."+suf, "model.language_model."+suf) if k in index]
        if len(keys)!=1: raise ValueError((ident,suf,keys))
        key=keys[0]; shard=index[key]; url=base+shard
        if shard not in cache:
            n=struct.unpack("<Q",fetch(url,0,7))[0]
            cache[shard]=(8+n,json.loads(fetch(url,8,7+n)))
        start,h=cache[shard]; spec=h[key]; a,b=spec["data_offsets"]; lo,hi=start+a,start+b-1
        assert len(spec["shape"])==1 and spec["dtype"] in ("BF16","F16","F32")
        raw=fetch(url,lo,hi)
        out[suf]=(raw,dict(key=key,dtype=spec["dtype"],shape=spec["shape"],shard=shard,
                           byte_range_inclusive=[lo,hi],sha256=hashlib.sha256(raw).hexdigest()))
    return out

def floats(raw,dtype):
    if dtype=="BF16":
        return [struct.unpack(">f",struct.pack(">I",u<<16))[0] for u, in struct.iter_unpack("<H",raw)]
    return [v for v, in struct.iter_unpack("<e" if dtype=="F16" else "<f",raw)]

def main():
    models=[m for _,a,b in PAIRS for m in [a,b]]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fs={m:pool.submit(extract,m) for m in models}
        for label,base,tuned in PAIRS:
            x,y=fs[base].result(),fs[tuned].result()
            row=dict(comparison=label,before={"model":base[0],"revision":base[1]},
                     after={"model":tuned[0],"revision":tuned[1]},tensors=[])
            for s in SUFFIXES:
                a,m=x[s]; b,n=y[s]
                if (m["dtype"],m["shape"])!=(n["dtype"],n["shape"]):raise ValueError("not like-for-like")
                aa=floats(a,m["dtype"]); bb=floats(b,n["dtype"])
                assert len(aa)==len(bb)==m["shape"][0] and all(math.isfinite(t) for t in aa+bb)
                row["tensors"].append(dict(before=m,after=n,changed=sum(i!=j for i,j in zip(aa,bb)),
                                           elements=len(aa),max_abs_delta=max(abs(i-j) for i,j in zip(aa,bb))))
            results.append(row)
            print(label,[(z["changed"],z["elements"]) for z in row["tensors"]],file=sys.stderr,flush=True)
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__": main()

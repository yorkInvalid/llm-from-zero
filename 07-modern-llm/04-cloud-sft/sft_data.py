"""单轮、非思考模式 SFT 的严格回答 mask 与确定性评分。"""
import hashlib
import json
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from make_data import SYSTEM
HERE=Path(__file__).resolve().parent
MODEL_ID='Qwen/Qwen3-4B'
REVISION='1cfa9a7208912126459214e8b04321603b3df60c'
FIELDS=('product','quantity','unit','delivery_city')

def tokenizer_for(local_files_only=False):
    """固定版本的分词器只从本地快照加载，避免 Transformers 的额外远端查询。"""
    files=['config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt']
    paths=[Path(hf_hub_download(MODEL_ID,filename,revision=REVISION,
                               local_files_only=local_files_only)) for filename in files]
    # 保留 snapshots/<revision> 目录；resolve() 会把缓存符号链接解析到 blobs。
    return AutoTokenizer.from_pretrained(str(paths[0].parent),local_files_only=True,
                                         trust_remote_code=False)

def read_rows(split):
    path=HERE/'data'/f'{split}.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def data_hash():
    h=hashlib.sha256()
    for split in ['train','validation','test']:
        h.update((HERE/'data'/f'{split}.jsonl').read_bytes())
    h.update(SYSTEM.encode())
    return h.hexdigest()

def prompt_messages(row):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':row['text']}]

def prompt_ids(tokenizer,row):
    return tokenizer.apply_chat_template(prompt_messages(row),tokenize=True,
                                         add_generation_prompt=True,enable_thinking=False)

def encode_row(tokenizer,row,max_length=512):
    prefix=prompt_ids(tokenizer,row)
    answer=json.dumps(row['answer'],ensure_ascii=False,separators=(',',':'))
    full=tokenizer.apply_chat_template(prompt_messages(row)+[{'role':'assistant','content':answer}],
                                       tokenize=True,add_generation_prompt=False,enable_thinking=False)
    if full[:len(prefix)]!=prefix:
        raise ValueError('模板的生成前缀与完整对话不一致，不能安全确定回答边界')
    # 监督 JSON 回答和真实 EOS，去掉 EOS 后的模板换行。
    end=full.index(tokenizer.eos_token_id,len(prefix))+1
    ids=full[:end]
    if len(ids)>max_length:
        raise ValueError(f"{row['id']}: {len(ids)} > max_length={max_length}；拒绝静默截断答案")
    labels=[-100]*len(prefix)+ids[len(prefix):]
    if not any(x!=-100 for x in labels):raise ValueError('无有效回答目标')
    return {'input_ids':ids,'labels':labels,'prompt_length':len(prefix),'id':row['id']}

def collate(rows,pad_id):
    n=max(len(r['input_ids']) for r in rows)
    return {'input_ids':torch.tensor([r['input_ids']+[pad_id]*(n-len(r['input_ids'])) for r in rows]),
            'labels':torch.tensor([r['labels']+[-100]*(n-len(r['labels'])) for r in rows]),
            'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(n-len(r['input_ids'])) for r in rows])}

def strict_object(text):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('重复 JSON key')
            result[k]=v
        return result
    def reject_constant(value):raise ValueError('非标准 JSON 常量')
    return json.loads(text.strip(),object_pairs_hook=unique,parse_constant=reject_constant)

def score(text,target):
    try:obj=strict_object(text)
    except (ValueError,TypeError):return dict(json_valid=False,schema_valid=False,exact=False,correct_fields=0)
    is_dict=isinstance(obj,dict)
    schema=is_dict and set(obj)==set(FIELDS)
    if schema:
        schema=all(obj[k] is None or type(obj[k]) is str for k in ['product','unit','delivery_city'])
        schema=schema and (obj['quantity'] is None or type(obj['quantity']) is int)
    correct=sum(k in obj and type(obj[k]) is type(target[k]) and obj[k]==target[k] for k in FIELDS) if is_dict else 0
    return dict(json_valid=True,schema_valid=bool(schema),exact=bool(schema and correct==4),correct_fields=correct)

def aggregate(predictions):
    n=len(predictions)
    if not n:raise ValueError('评测集不能为空')
    return {'count':n,'json_valid_rate':sum(r['score']['json_valid'] for r in predictions)/n,
            'schema_valid_rate':sum(r['score']['schema_valid'] for r in predictions)/n,
            'exact_match_rate':sum(r['score']['exact'] for r in predictions)/n,
            'field_accuracy':sum(r['score']['correct_fields'] for r in predictions)/(4*n)}

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sft_data import tokenizer_for,read_rows,encode_row,collate,score
from run import update, preflight
from unittest.mock import patch
from types import SimpleNamespace

def tiny():
    torch.manual_seed(12)
    c=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
                  num_attention_heads=2,num_key_value_heads=1,head_dim=8,attention_dropout=0.,use_cache=False)
    c._attn_implementation='eager'
    return Qwen3ForCausalLM(c)

class SFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer=tokenizer_for(local_files_only=True)

    def test_offline_tokenizer_never_queries_model_info(self):
        with patch('huggingface_hub.model_info',side_effect=AssertionError('离线加载不得查询远端模型信息')) as model_info:
            tokenizer=tokenizer_for(local_files_only=True)
        model_info.assert_not_called()
        row=read_rows('train')[0]
        self.assertEqual(encode_row(tokenizer,row),encode_row(self.tokenizer,row))
        self.assertEqual(tokenizer.eos_token_id,self.tokenizer.eos_token_id)

    def test_hardware_and_disk_preflight(self):
        with self.assertRaisesRegex(RuntimeError,'cuda'):
            preflight(torch.device('cpu'),'lora')
        with patch('torch.cuda.is_bf16_supported',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'BF16'):
                preflight(torch.device('cuda'),'lora')
        with patch('torch.cuda.is_bf16_supported',return_value=True), patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=80*2**30)):
            with tempfile.TemporaryDirectory() as path, patch('run.shutil.disk_usage',return_value=SimpleNamespace(free=59*2**30)):
                with self.assertRaisesRegex(RuntimeError,'磁盘'):
                    preflight(torch.device('cuda'),'full',Path(path)/'run',required_gib=60)
            preflight(torch.device('cuda'),'full')

    def test_split_groups_and_texts_disjoint(self):
        groups=[];texts=[]
        for split in ['train','validation','test']:
            rows=read_rows(split)
            groups.append({r['group_id'] for r in rows});texts.append({r['text'] for r in rows})
            self.assertEqual(len(texts[-1]),len(rows))
        for i in range(3):
            for j in range(i):self.assertFalse(groups[i]&groups[j]);self.assertFalse(texts[i]&texts[j])

    def test_real_template_mask_and_eos(self):
        rows=[encode_row(self.tokenizer,r) for r in read_rows('train')]
        for row in rows:
            n=row['prompt_length']
            self.assertTrue(all(x==-100 for x in row['labels'][:n]))
            self.assertEqual(row['labels'][n:],row['input_ids'][n:])
            self.assertEqual(row['labels'][-1],self.tokenizer.eos_token_id)
            text=self.tokenizer.decode(row['labels'][n:-1])
            self.assertIsInstance(json.loads(text),dict)
            self.assertNotIn('<think>',text)
        batch=collate([min(rows,key=lambda r:len(r['input_ids'])),max(rows,key=lambda r:len(r['input_ids']))],self.tokenizer.pad_token_id)
        self.assertTrue((batch['labels'][batch['attention_mask']==0]==-100).all())
        with self.assertRaises(ValueError):encode_row(self.tokenizer,read_rows('train')[0],max_length=10)

    def test_score_strict_types_schema_and_duplicate_keys(self):
        target={'product':'纸箱','quantity':2,'unit':'个','delivery_city':None}
        self.assertTrue(score(json.dumps(target),target)['exact'])
        for bad in ['```json\n'+json.dumps(target)+'\n```','{"quantity":2,"quantity":2}',
                    json.dumps({**target,'extra':'x'}),json.dumps({**target,'quantity':'2'}),
                    json.dumps({**target,'delivery_city':'null'})]:
            self.assertFalse(score(bad,target)['exact'])
        self.assertFalse(score(json.dumps({**target,'quantity':True}),target)['schema_valid'])
        self.assertFalse(score('NaN',target)['json_valid'])

    def test_loss_shift_and_ignored_prompt(self):
        model=tiny()
        ids=torch.tensor([[1,2,3,4,5]])
        labels=torch.tensor([[-100,-100,-100,4,5]])
        out=model(input_ids=ids,labels=labels)
        manual=torch.nn.functional.cross_entropy(out.logits[:,2:4].reshape(-1,32),torch.tensor([4,5]))
        torch.testing.assert_close(out.loss,manual)

    def test_accumulation_matches_token_weighted_batch(self):
        a=tiny();b=copy.deepcopy(a)
        rows=[{'input_ids':[1,2,3,4],'labels':[-100,-100,3,4]},
              {'input_ids':[1,2,5,6,7],'labels':[-100,-100,5,6,7]}]
        oa=torch.optim.SGD(a.parameters(),lr=.01);ob=torch.optim.SGD(b.parameters(),lr=.01)
        la,_,_=update(a,oa,[collate([r],0) for r in rows])
        lb,_,_=update(b,ob,[collate(rows,0)])
        self.assertAlmostEqual(la,lb,places=5)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-5)

    def test_lora_freezes_base_and_reload_matches(self):
        model=get_peft_model(tiny(),LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        batch=collate([{'input_ids':[1,2,3,4],'labels':[-100,-100,3,4]}],0)
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.01)
        update(model,opt,[batch])
        changed=[]
        for n,p in model.named_parameters():
            if not torch.equal(p,before[n]):changed.append(n)
            if 'lora_' not in n:self.assertFalse(p.requires_grad);self.assertTrue(torch.equal(p,before[n]))
        self.assertTrue(changed);self.assertTrue(all('lora_' in n for n in changed))
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored=PeftModel.from_pretrained(tiny(),path)
            model.eval();restored.eval()
            with torch.no_grad():torch.testing.assert_close(model(**batch).logits,restored(**batch).logits)

    def test_full_weight_reload_matches(self):
        model=tiny();batch=collate([{'input_ids':[1,2,3],'labels':[-100,2,3]}],0)
        opt=torch.optim.AdamW(model.parameters(),lr=.01);update(model,opt,[batch])
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored=Qwen3ForCausalLM.from_pretrained(path,local_files_only=True,attn_implementation='eager')
            with torch.no_grad():torch.testing.assert_close(model(**batch).logits,restored(**batch).logits)

if __name__=='__main__':unittest.main()

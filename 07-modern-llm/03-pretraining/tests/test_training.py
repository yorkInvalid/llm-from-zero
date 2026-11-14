"""离线验证标签对齐、因果性、数据隔离及精确恢复。"""
import tempfile
import unittest
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_lab import (load_assets, new_model, windows, collate, manual_loss,
                          batch_for, train_step, save_resume, load_resume)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.corpus, cls.tokenizer = load_assets()

    def test_documents_disjoint_and_targets_cover_once(self):
        names = list(self.corpus)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                self.assertFalse(set(self.corpus[a]) & set(self.corpus[b]))
        paragraph = self.corpus['train_a'][1]
        expected = self.tokenizer.encode(paragraph).ids + [2]
        actual = [int(t) for block in windows([paragraph], self.tokenizer, length=7) for t in block[1:]]
        self.assertEqual(actual, expected)

    def test_hf_shift_and_padding(self):
        model = new_model(512).eval()
        batch = collate([torch.tensor([1,30,40,2]), torch.tensor([1,20,2])])
        self.assertEqual(batch['labels'][1,-1].item(), -100)
        self.assertEqual(batch['labels'][0].tolist(), batch['input_ids'][0].tolist())
        with torch.no_grad():
            out = model(**batch)
        torch.testing.assert_close(out.loss, manual_loss(out.logits,batch['labels']))
        modified = out.logits.clone()
        modified[:,-1] = torch.randn_like(modified[:,-1])*100
        modified[1,2] = torch.randn_like(modified[1,2])*100
        torch.testing.assert_close(out.loss, manual_loss(modified,batch['labels']))

    def test_no_future_token_leak(self):
        model = new_model(512).eval()
        a=torch.tensor([[1,20,30,40]])
        b=a.clone();b[0,3]=41
        with torch.no_grad():
            x=model(a).logits; y=model(b).logits
        torch.testing.assert_close(x[:,:3],y[:,:3],atol=0,rtol=0)

    def test_backward_then_step(self):
        model=new_model(512)
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        p=model.model.embed_tokens.weight
        original=p.detach().clone()
        out=model(**collate([torch.tensor([1,20,30,2])]))
        out.loss.backward()
        self.assertGreater(p.grad.norm().item(),0)
        self.assertTrue(torch.equal(original,p))
        opt.step()
        self.assertFalse(torch.equal(original,p))

    def test_resume_reproduces_next_update(self):
        samples=windows(self.corpus['train_a'][:3],self.tokenizer,length=8)
        model=new_model(512)
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        generator=torch.Generator().manual_seed(123)
        train_step(model,opt,batch_for(samples,generator,batch_size=2))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resume.pt'
            save_resume(path,model,opt,generator,1)
            batch=batch_for(samples,generator,batch_size=2)
            train_step(model,opt,batch)
            restored, opt2, gen2, step=load_resume(path)
            self.assertEqual(step,1)
            batch2=batch_for(samples,gen2,batch_size=2)
            self.assertTrue(torch.equal(batch['input_ids'],batch2['input_ids']))
            train_step(restored,opt2,batch2)
            for p,q in zip(model.parameters(),restored.parameters()):
                torch.testing.assert_close(p,q,atol=0,rtol=0)


if __name__ == '__main__': unittest.main()

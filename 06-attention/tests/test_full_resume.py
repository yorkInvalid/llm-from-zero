import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from tokenizers import Tokenizer,models,pre_tokenizers
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_state import cpu_tree, validate_resume
from indexed_tokens import TokenWindowWriter

SCRIPT=Path(__file__).resolve().parents[1]/'train_lm.py'

class FullResumeTests(unittest.TestCase):
    def test_indexed_probe_resume_and_new_plan_keep_optimizer(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);data=p/'data';data.mkdir()
            tok=Tokenizer(models.WordLevel({'<PAD>':0,'<BOS>':1,'<EOS>':2,'a':3,'b':4,'c':5,'<UNK>':6},unk_token='<UNK>'))
            tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(data/'tokenizer.json'))
            writer=TokenWindowWriter(data,context_length=4)
            for text in ['a b c a b c','b c a b c a','c a b c a b']:
                writer.add_document([1,*tok.encode(text).ids,2])
            writer.finish()
            (data/'validation.json').write_text(json.dumps(['a c b a c b','b a c b a c']))
            common=[sys.executable,str(SCRIPT),'train','--data-dir',str(data),'--epochs','2',
                    '--batch-size','2','--context-length','4','--d-model','8','--num-layers','1','--num-heads','2',
                    '--eval-every','2','--train-probe-windows','2','--learning-rate','.003',
                    '--no-full-train-eval','--observation-prompts','a']
            if os.environ.get('RESUME_TEST_DEVICE')=='cuda':common+=['--device','cuda','--precision','fp16']
            def run(name,*extra):
                result=subprocess.run(common+['--run-dir',str(p/name),*extra],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            run('full');run('partial','--stop-after-step','2')
            # Historical checkpoints saved optimizer values without recipe metadata.
            legacy = torch.load(p/'partial/last.pt', map_location='cpu', weights_only=True)
            for key in ('optimizer', 'betas', 'weight_decay', 'optimizer_eps', 'dropout', 'cooldown_steps'):
                legacy['training_config'].pop(key, None)
            torch.save(legacy, p/'partial/last.pt')
            run('resumed','--resume-checkpoint',str(p/'partial/last.pt'))
            a=torch.load(p/'full/last.pt',map_location='cpu',weights_only=True)
            b=torch.load(p/'resumed/last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(a['tokens_seen'],42);self.assertEqual(b['tokens_seen'],42)
            self.assertEqual(a['checks'],b['checks'])
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            events=EventAccumulator(str(p/'full/tensorboard')).Reload()
            self.assertNotIn('train/loss',events.Tags()['scalars'])
            self.assertEqual([v.step for v in events.Scalars('train_probe/loss')],[0,2,3,4,6])
            run('continued','--init-checkpoint',str(p/'full/last.pt'),'--keep-optimizer-state')
            c=torch.load(p/'continued/last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(c['tokens_seen'],42)
            for state in c['optimizer']['state'].values():self.assertEqual(state['step'].item(),12)
            meta=json.loads((p/'continued/run.json').read_text())
            self.assertEqual(meta['training_start'],'continue_weights_and_optimizer_new_schedule')

    def test_saved_snapshot_does_not_follow_live_cpu_weights(self):
        weight=torch.tensor([1.])
        snapshot=cpu_tree({'model':{'weight':weight}})
        weight.add_(1.)
        self.assertEqual(snapshot['model']['weight'].item(),1.)

    def test_interruption_preserves_updates_optimizer_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);data=p/'data';data.mkdir()
            tok=Tokenizer(models.WordLevel({'<PAD>':0,'<BOS>':1,'<EOS>':2,'a':3,'b':4,'c':5,'<UNK>':6},unk_token='<UNK>'))
            tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(data/'tokenizer.json'))
            for name,docs in [('train',['a b c a b c','b c a b c a','c a b c a b']),('validation',['a c b a c b','b a c b a c'])]:
                (data/f'{name}.json').write_text(json.dumps(docs))
            common=[sys.executable,str(SCRIPT),'train','--data-dir',str(data),'--epochs','2',
                    '--batch-size','2','--context-length','4','--d-model','8','--num-layers','1','--num-heads','2',
                    '--eval-every','2','--train-probe-windows','2','--learning-rate','.003','--observation-prompts','a']
            if os.environ.get('RESUME_TEST_DEVICE')=='cuda':
                common+=['--device','cuda','--precision','fp16']
            def run(name,*extra):
                r=subprocess.run(common+['--run-dir',str(p/name),*extra],capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            run('full')
            run('interrupted','--stop-after-step','2')
            self.assertEqual(json.loads((p/'interrupted/run.json').read_text())['status'],'paused')
            run('resumed','--resume-checkpoint',str(p/'interrupted/last.pt'))
            a=torch.load(p/'full/last.pt',weights_only=True);b=torch.load(p/'resumed/last.pt',weights_only=True)
            self.assertEqual(a['global_step'],6);self.assertEqual(a['global_step'],b['global_step'])
            for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
            for key,state in a['optimizer']['state'].items():
                for n,value in state.items():torch.testing.assert_close(value,b['optimizer']['state'][key][n],rtol=0,atol=0)
            self.assertEqual(a['checks'],b['checks'])
            self.assertEqual([c['global_step'] for c in a['checks']],[0,2,3,4,6])
            self.assertTrue(torch.equal(a['rng']['cpu'],b['rng']['cpu']))
            self.assertEqual(a['scaler'],b['scaler'])
            for x,y in zip(a['rng']['cuda'],b['rng']['cuda']):
                self.assertTrue(torch.equal(x,y))
            bad=dict(a['split_sha256'],train='changed')
            with self.assertRaisesRegex(ValueError,'数据'):
                validate_resume(a,a['training_config'],bad,a['tokenizer_sha256'])
            bad=dict(a['training_config'],batch_size=4)
            with self.assertRaisesRegex(ValueError,'batch_size'):
                validate_resume(a,bad,a['split_sha256'],a['tokenizer_sha256'])

if __name__=='__main__':unittest.main()

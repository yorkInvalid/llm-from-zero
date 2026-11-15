import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attention import MultiHeadAttention
from position_encoding import apply_rope
from training_progress import TrainingProgress
from causal_lm import CausalLM


class DiagnosticTests(unittest.TestCase):
    def test_attention_matches_independent_sdpa_and_gradients(self):
        torch.manual_seed(11)
        a=MultiHeadAttention(32,4,8,8,use_rope=True).double()
        x=torch.randn(2,32,19,dtype=torch.float64,requires_grad=True)
        actual,_=a(x,causal=True)
        positions=torch.arange(19)
        heads=[]
        for h in range(4):
            q=apply_rope(a.W_q[h]@x,positions).transpose(-1,-2)
            k=apply_rope(a.W_k[h]@x,positions).transpose(-1,-2)
            v=(a.W_v[h]@x).transpose(-1,-2)
            heads.append(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(-1,-2))
        expected=a.W_o@torch.cat(heads,dim=-2)
        torch.testing.assert_close(actual,expected)
        g1=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
        g2=torch.autograd.grad(expected.square().sum(),x)[0]
        torch.testing.assert_close(g1,g2)

    def test_fan_in_only_scales_attention_and_keeps_checkpoint_config(self):
        torch.manual_seed(42)
        old=CausalLM(300,d_model=128,num_layers=2,attention_init='legacy')
        torch.manual_seed(42)
        new=CausalLM(300,d_model=128,num_layers=2,attention_init='fan_in')
        self.assertEqual(new.config['attention_init'],'fan_in')
        for name,p in old.named_parameters():
            scale=128**-.5/.1 if '.attention.W_' in name else 1.
            torch.testing.assert_close(dict(new.named_parameters())[name],p*scale)
        rebuilt=CausalLM(**new.config)
        rebuilt.load_state_dict(new.state_dict())
        x=torch.tensor([[1,5,6,7]])
        new.eval();rebuilt.eval()
        torch.testing.assert_close(new(x),rebuilt(x))

    def test_new_default_and_old_checkpoint_compatibility(self):
        import tempfile
        from tokenizers import Tokenizer, models
        from train_lm import restore
        self.assertEqual(CausalLM(300).config['attention_init'], 'fan_in')
        old = CausalLM(300, d_model=32, num_layers=1, attention_init='legacy').eval()
        config = {k:v for k,v in old.config.items() if k != 'attention_init'}
        tokenizer = Tokenizer(models.WordLevel({'<UNK>':0}, unk_token='<UNK>'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.pt'
            torch.save(dict(config=config, model=old.state_dict(), tokenizer=tokenizer.to_str()),path)
            loaded, _, _ = restore(path, 'cpu')
            self.assertEqual(loaded.config['attention_init'], 'legacy')
            x = torch.tensor([[1,4,6,8]])
            torch.testing.assert_close(old(x), loaded(x))

    def test_progress_separates_updates_and_evaluation(self):
        monitor=MagicMock()
        with patch('sys.stderr',new=io.StringIO()):
            p=TrainingProgress(2,1,monitor)
            p.step(2.,4.,1)
            calls={c.args[0]:c.args[1] for c in monitor.writer.add_scalar.call_args_list}
            self.assertEqual(calls['time/eta_updates_seconds'],2.)
            self.assertEqual(calls['time/eta_includes_evaluation'],0)
            p.step(2.,3.,1);p.evaluation(5.);p.close()
            calls={c.args[0]:c.args[1] for c in monitor.writer.add_scalar.call_args_list}
            self.assertEqual(calls['time/evaluation_seconds'],5.)
            self.assertEqual(calls['time/eta_total_seconds'],0)

    def test_resume_eta_uses_session_updates_and_measured_initial_evaluation(self):
        monitor=MagicMock()
        with patch('sys.stderr',new=io.StringIO()):
            p=TrainingProgress(100,1,monitor,initial_step=49,eval_every=50,
                               initial_epoch_eval_seconds=7.)
            p.step(2.,4.,1)
            calls={c.args[0]:c.args[1] for c in monitor.writer.add_scalar.call_args_list}
            self.assertEqual(calls['time/eta_updates_seconds'],100.)
            self.assertEqual(calls['time/eta_total_seconds'],107.)
            p.close(completed=False)
            calls={c.args[0]:c.args[1] for c in monitor.writer.add_scalar.call_args_list}
            self.assertEqual(calls['time/eta_total_seconds'],107.)


if __name__=='__main__':unittest.main()

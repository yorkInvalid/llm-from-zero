"""归一化、门控、因果性、学习率边界、兼容性和实时日志。"""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from tokenizers import Tokenizer, models
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_lm import CausalLM, CausalBlock
from lm_components import RMSNorm, SwiGLU, learning_rate_at_step
from train_lm import restore
from training_monitor import TrainingMonitor


class ModernLMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_rms_formula_dimensions_and_stability(self):
        norm = RMSNorm(2)
        x = torch.tensor([[[3., 4.], [0., 0.]]], requires_grad=True)
        expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(norm(x), expected)
        self.assertNotEqual(norm(x)[0, 0].mean().item(), 0)
        norm(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(norm.weight.grad.abs().sum().item(), 0)
        large = torch.tensor([[60000., 60000.]], dtype=torch.float16)
        self.assertTrue(torch.isfinite(norm(large)).all())

    def test_pre_norm_zero_branches_are_identity_and_gradients_pass(self):
        for norm_type in ('layer', 'rms'):
            layer = CausalBlock(16, 4, 0., norm_style='pre', norm_type=norm_type)
            with torch.no_grad():
                layer.attention.W_o.zero_()
                layer.ffn.down.weight.zero_()
            x = torch.randn(16, 4, requires_grad=True)
            output = layer(x, None)
            torch.testing.assert_close(output, x)
            output.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_swiglu_gate_and_projection(self):
        ffn = SwiGLU(4)
        x = torch.randn(2, 4, 3, requires_grad=True)
        tokens = x.transpose(-2, -1)
        gate = nn.functional.linear(tokens, ffn.gate.weight)
        up = nn.functional.linear(tokens, ffn.up.weight)
        expected = nn.functional.linear(gate * torch.sigmoid(gate) * up, ffn.down.weight)
        torch.testing.assert_close(ffn(x), expected.transpose(-2, -1))
        ffn(x).square().sum().backward()
        self.assertTrue(all(p.grad.abs().sum() > 0 for p in ffn.parameters()))
        with torch.no_grad():
            ffn.gate.weight.zero_()
        torch.testing.assert_close(ffn(x), torch.zeros_like(x))

    def test_modern_causality_final_norm_and_rope(self):
        model = CausalLM(20, d_model=16, num_layers=2).eval()
        self.assertIsInstance(model.final_norm, RMSNorm)
        self.assertTrue(all(layer.attention.use_rope for layer in model.layers))
        x = torch.tensor([1, 3, 4, 5])
        other = torch.tensor([1, 3, 9, 8])
        torch.testing.assert_close(model(x)[:2], model(other)[:2])
        # RoPE 模式输入处不再叠加 sin/cos，位置处理发生在 Q/K 内。
        seen = []
        handle = model.layers[0].register_forward_pre_hook(lambda m, args: seen.append(args[0]))
        try:
            model(x)
        finally:
            handle.remove()
        torch.testing.assert_close(seen[0], model.embedding(x).T)

    def test_learning_rate_boundaries(self):
        rates = [learning_rate_at_step(s, 10, 3, .001) for s in range(1, 11)]
        self.assertAlmostEqual(rates[0], .001 / 3)
        self.assertAlmostEqual(rates[2], .001)
        self.assertAlmostEqual(rates[-1], .0001)
        self.assertTrue(all(a >= b for a, b in zip(rates[2:], rates[3:])))
        self.assertAlmostEqual(learning_rate_at_step(1, 1, 0, .001), .001)
        self.assertAlmostEqual(learning_rate_at_step(1, 4, 0, .001), .001)
        with self.assertRaises(ValueError):
            learning_rate_at_step(0, 10, 3, .001)

    def test_legacy_and_modern_checkpoint_roundtrip(self):
        for legacy in (True, False):
            options = dict(norm_style='post', norm_type='layer', position_encoding='sinusoidal', ffn_type='relu') if legacy else {}
            model = CausalLM(8, d_model=16, num_layers=1, **options).eval()
            config = dict(model.config)
            if legacy:
                for key in options:
                    config.pop(key)
            tokenizer = Tokenizer(models.WordLevel({'<unk>': 0}, unk_token='<unk>'))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pt'
                torch.save(dict(config=config, model=model.state_dict(), tokenizer=tokenizer.to_str()), path)
                restored, _, _ = restore(path, 'cpu')
            x = torch.tensor([1, 3, 4])
            torch.testing.assert_close(restored(x), model(x))

    def test_tensorboard_records_real_scalars_and_text(self):
        metrics = dict(epoch=0, train=dict(loss=2., perplexity=7., token_accuracy=.1),
                       validation=dict(loss=3., perplexity=20., token_accuracy=.05))
        observation = dict(generations=[dict(prompt='Alice', text='Alice was')],
                           predictions=dict(rows=[]))
        with tempfile.TemporaryDirectory() as directory:
            with TrainingMonitor(directory, {'seed': 42}) as monitor:
                monitor.step(1, 2.5, .001, 1.2, 100)
                monitor.epoch(1, metrics, observation)
                # writer 尚未关闭时就能读到，证明实时 flush 生效。
                events = EventAccumulator(directory).Reload()
                self.assertAlmostEqual(events.Scalars('optimization/learning_rate')[0].value, .001)
                self.assertEqual(events.Scalars('validation/loss')[0].value, 3.)
                self.assertIn('generation/prompt_0/text_summary', events.Tags()['tensors'])


if __name__ == '__main__':
    unittest.main()

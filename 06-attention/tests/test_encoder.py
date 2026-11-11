"""多层 Encoder 的独立参数、逐层计算、梯度与训练记录。"""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder import Encoder
from train import train


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_invalid_layer_count(self):
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                Encoder(num_layers=count)

    def test_layers_have_independent_registered_parameters(self):
        model = Encoder()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3870)
        groups = [dict(layer.named_parameters()) for layer in model.layers]
        for name in groups[0]:
            self.assertEqual(len({group[name].data_ptr() for group in groups}), 3)
        before = groups[1]['norm2.weight'].detach().clone()
        with torch.no_grad():
            groups[0]['norm2.weight'].add_(1)
        torch.testing.assert_close(groups[1]['norm2.weight'], before)

    def test_forward_matches_manual_stack_and_supports_variable_length(self):
        for count, length, rope in ((1, 1, False), (3, 3, False), (2, 5, True)):
            with self.subTest(count=count, length=length, rope=rope):
                model = Encoder(num_layers=count, use_rope=rope)
                X = torch.randn(10, length)
                output, weights = model(X)
                H = X
                for index, layer in enumerate(model.layers):
                    H, expected_weights = layer(H)
                    torch.testing.assert_close(weights[index], expected_weights)
                    self.assertEqual(layer.attention_sublayer.attention.use_rope, rope)
                torch.testing.assert_close(output, H)
                self.assertEqual(output.shape, X.shape)
                self.assertEqual(len(weights), count)

    def test_gradients_and_updates_reach_every_layer(self):
        model = Encoder()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        X = torch.randn(10, 3, requires_grad=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        output, _ = model(X)
        (output - torch.randn_like(output)).square().mean().backward()
        self.assertTrue(torch.isfinite(X.grad).all())
        self.assertGreater(X.grad.abs().sum().item(), 0)
        for name, p in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)
        optimizer.step()
        for name, p in model.named_parameters():
            self.assertFalse(torch.equal(p, before[name]), name)

    def test_training_records_final_weights_for_every_layer(self):
        for mode in ('none', 'sinusoidal', 'rope'):
            with self.subTest(mode=mode):
                with contextlib.redirect_stdout(io.StringIO()):
                    model, losses, history = train(mode, num_steps=2, num_layers=2)
                torch.manual_seed(42)
                I = torch.randn(10, 3)
                from position_encoding import sinusoidal_encoding
                X = I + sinusoidal_encoding(I) if mode == 'sinusoidal' else I
                with torch.no_grad():
                    output, weights = model(X)
                target = I.mean(dim=1, keepdim=True).expand_as(I)
                self.assertAlmostEqual(losses[-1], (output-target).square().mean().item(), places=6)
                self.assertEqual(len(history), 3)
                self.assertEqual(tuple(history[-1].shape), (2, 2, 3, 3))
                torch.testing.assert_close(history[-1], torch.stack(weights))
                self.assertFalse(history[-1].requires_grad)
                self.assertNotEqual(history[0].data_ptr(), history[-1].data_ptr())

    def test_plotting_selects_each_layer_and_unique_names(self):
        from visualization import plot_encoder_training
        history = [torch.rand(3, 2, 3, 3) for _ in range(2)]
        with patch('visualization.plot_training', return_value=('curve', 'heatmap')) as plot:
            paths = plot_encoder_training([1., 0.5], history, 'unused', name='none')
        self.assertEqual(len(paths), 3)
        for index, call in enumerate(plot.call_args_list):
            self.assertEqual(call.kwargs['name'], f'none-3layers-layer{index+1}')
            for step in range(2):
                torch.testing.assert_close(call.args[1][step], history[step][index])


if __name__ == '__main__':
    unittest.main()

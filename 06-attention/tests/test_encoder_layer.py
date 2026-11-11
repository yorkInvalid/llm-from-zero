"""第二次残差与独立 LayerNorm 的回归测试。"""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder_layer import EncoderLayer


class EncoderLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_second_norm_has_independent_parameters(self):
        model = EncoderLayer()
        first = model.attention_sublayer.norm
        second = model.norm2
        for name in ('weight', 'bias'):
            self.assertIsNot(getattr(first, name), getattr(second, name))
        first_weight = first.weight.detach().clone()
        with torch.no_grad():
            second.weight.fill_(2)
        torch.testing.assert_close(first.weight, first_weight)

    def test_zero_ffn_preserves_z_before_norm_not_original_x(self):
        model = EncoderLayer()
        with torch.no_grad():
            model.ffn.fc2.weight.zero_()
            model.ffn.fc2.bias.zero_()
            model.norm2.weight.copy_(torch.linspace(0.5, 1.5, 10))
            model.norm2.bias.copy_(torch.linspace(-0.2, 0.2, 10))
        X = torch.randn(10, 3)
        Z, _ = model.attention_sublayer(X)
        captured = []
        handle = model.norm2.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        try:
            output, _ = model(X)
        finally:
            handle.remove()
        torch.testing.assert_close(captured[0], Z.T)
        expected = (Z - Z.mean(dim=0, keepdim=True)) / torch.sqrt(
            Z.var(dim=0, unbiased=False, keepdim=True) + model.norm2.eps)
        expected = expected * model.norm2.weight[:, None] + model.norm2.bias[:, None]
        torch.testing.assert_close(output, expected)

    def test_second_residual_keeps_gradient_when_ffn_is_zero(self):
        model = EncoderLayer().double()
        with torch.no_grad():
            model.ffn.fc2.weight.zero_()
            model.ffn.fc2.bias.zero_()
        X = torch.randn(10, 3, dtype=torch.double, requires_grad=True)
        output, _ = model(X)
        probe = torch.randn_like(output)
        actual_grad = torch.autograd.grad((output * probe).sum(), X)[0]
        Z, _ = model.attention_sublayer(X)
        expected = model.norm2(Z.T).T
        expected_grad = torch.autograd.grad((expected * probe).sum(), X)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertGreater(actual_grad.abs().sum().item(), 0)

    def test_encoder_shapes_and_finite_constant_inputs(self):
        for dim, length in ((4, 1), (10, 3), (6, 6)):
            with self.subTest(dim=dim, length=length):
                model = EncoderLayer(d_model=dim, d_ff=dim * 4)
                X = torch.ones(dim, length)
                output, weights = model(X)
                self.assertEqual(output.shape, X.shape)
                self.assertEqual(tuple(weights.shape), (2, length, length))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()

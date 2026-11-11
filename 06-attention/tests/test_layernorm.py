"""验证按 token 的 Post-LN、可学习仿射参数及梯度。"""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_sublayer import AttentionSublayer


class LayerNormTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_post_norm_matches_formula_and_preserves_weights(self):
        for length in (1, 3, 10):
            with self.subTest(length=length):
                model = AttentionSublayer()
                with torch.no_grad():
                    model.norm.weight.copy_(torch.linspace(0.5, 1.5, 10))
                    model.norm.bias.copy_(torch.linspace(-0.3, 0.3, 10))
                X = torch.randn(10, length)
                original = X.clone()
                B, expected_weights = model.attention(X)
                Y = X + B
                centered = Y - Y.mean(dim=0, keepdim=True)
                expected = centered / torch.sqrt(centered.square().mean(dim=0, keepdim=True) + model.norm.eps)
                expected = expected * model.norm.weight[:, None] + model.norm.bias[:, None]
                Z, weights = model(X)
                self.assertEqual(Z.shape, X.shape)
                torch.testing.assert_close(Z, expected)
                torch.testing.assert_close(weights, expected_weights)
                torch.testing.assert_close(X, original)

    def test_tokens_normalized_independently_and_constant_token_is_finite(self):
        model = AttentionSublayer(d_in=4, num_heads=1, d_k=2, d_v=2)
        with torch.no_grad():
            model.attention.W_o.zero_()
        X = torch.tensor([[1., 10., 5.], [2., 30., 5.], [3., 50., 5.], [4., 70., 5.]])
        Z, _ = model(X)
        torch.testing.assert_close(Z.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
        # epsilon 会让方差略小于 1，不应断言精确等于 1。
        var = X.var(dim=0, unbiased=False)
        torch.testing.assert_close(Z.var(dim=0, unbiased=False), var / (var + model.norm.eps))
        torch.testing.assert_close(Z[:, 2], torch.zeros(4))
        changed = X.clone()
        changed[:, 1] = torch.tensor([8., 2., -9., 4.])
        changed_Z, _ = model(changed)
        torch.testing.assert_close(changed_Z[:, [0, 2]], Z[:, [0, 2]])

    def test_affine_parameters_and_attention_receive_updates(self):
        model = AttentionSublayer()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 420)
        X = torch.randn(10, 3, requires_grad=True)
        target = torch.randn(10, 3)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        Z, _ = model(X)
        # 不用 Z.sum()：初始归一化输出和为零，会掩盖梯度传播情况。
        (Z - target).square().mean().backward()
        self.assertTrue(torch.isfinite(X.grad).all())
        self.assertGreater(X.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertFalse(torch.equal(parameter, before[name]), name)


if __name__ == '__main__':
    unittest.main()

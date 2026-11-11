"""FFN 数值、位置独立性与 Encoder 组合测试。"""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder_layer import EncoderLayer
from feed_forward import FeedForward


class FeedForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_known_weights_biases_and_relu(self):
        ffn = FeedForward(2, 3)
        with torch.no_grad():
            ffn.fc1.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., -1.]]))
            ffn.fc1.bias.copy_(torch.tensor([1., -2., 1.]))
            ffn.fc2.weight.copy_(torch.tensor([[1., 2., 3.], [-1., 1., 2.]]))
            ffn.fc2.bias.copy_(torch.tensor([0.5, -0.5]))
        X = torch.tensor([[1., -2.], [3., 1.]])
        original = X.clone()
        # fc1 后 [[2,-1],[1,-1],[-3,2]]；ReLU 将负数归零。
        expected = torch.tensor([[4.5, 6.5], [-1.5, 3.5]])
        torch.testing.assert_close(ffn(X), expected)
        torch.testing.assert_close(X, original)

    def test_shapes_with_different_feature_and_sequence_sizes(self):
        for dim, hidden, length in ((10, 40, 3), (4, 7, 1), (6, 9, 6)):
            with self.subTest(dim=dim, length=length):
                self.assertEqual(FeedForward(dim, hidden)(torch.randn(dim, length)).shape,
                                 (dim, length))

    def test_tokens_are_independent_and_share_parameters(self):
        ffn = FeedForward()
        X = torch.randn(10, 3)
        F = ffn(X)
        changed = X.clone()
        changed[:, 1] += torch.arange(10.)
        torch.testing.assert_close(ffn(changed)[:, [0, 2]], F[:, [0, 2]])
        # 单独计算每个 token，结果必须与一起计算一致。
        separate = torch.cat([ffn(X[:, i:i+1]) for i in range(3)], dim=1)
        torch.testing.assert_close(F, separate)
        permutation = [2, 0, 1]
        torch.testing.assert_close(ffn(X[:, permutation]), F[:, permutation])

    def test_encoder_composition_preserves_attention_weights(self):
        for use_rope in (False, True):
            with self.subTest(use_rope=use_rope):
                model = EncoderLayer(use_rope=use_rope)
                X = torch.randn(10, 3)
                Z, expected_weights = model.attention_sublayer(X)
                F, weights = model(X)
                Y = Z + model.ffn(Z)
                centered = Y - Y.mean(dim=0, keepdim=True)
                expected = centered / torch.sqrt(Y.var(dim=0, unbiased=False, keepdim=True) + model.norm2.eps)
                expected = expected * model.norm2.weight[:, None] + model.norm2.bias[:, None]
                torch.testing.assert_close(F, expected)
                torch.testing.assert_close(weights, expected_weights)
                self.assertEqual(tuple(F.shape), (10, 3))

    def test_registered_parameters_receive_gradients_and_updates(self):
        model = EncoderLayer()
        self.assertEqual(sum(p.numel() for p in model.ffn.parameters()), 850)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1290)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        X = torch.randn(10, 3, requires_grad=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        F, _ = model(X)
        (F - torch.randn_like(F)).square().mean().backward()
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


if __name__ == '__main__':
    unittest.main()

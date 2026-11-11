"""运行：.venv/bin/python -m unittest discover -s 06-attention/tests -v"""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention import MultiHeadAttention
from attention_sublayer import AttentionSublayer
from position_encoding import sinusoidal_encoding
from train import forward_with_residual, train


class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_known_output_is_added_after_projection(self):
        # Q/K 全零使注意力均匀；V 是恒等映射，W_o = 2I。
        # 两个 token [1, 3]、[5, 7] 的平均是 [3, 5]，B = [6, 10]。
        model = MultiHeadAttention(d_in=2, num_heads=1, d_k=2, d_v=2)
        with torch.no_grad():
            model.W_q[0].zero_()
            model.W_k[0].zero_()
            model.W_v[0].copy_(torch.eye(2))
            model.W_o.copy_(2 * torch.eye(2))
        X = torch.tensor([[1., 5.], [3., 7.]])
        original = X.clone()
        Y, weights = forward_with_residual(model, X)
        torch.testing.assert_close(Y, torch.tensor([[7., 11.], [13., 17.]]))
        torch.testing.assert_close(weights, torch.full((1, 2, 2), 0.5))
        torch.testing.assert_close(X, original)

    def test_zero_branch_preserves_input_and_identity_gradient(self):
        model = MultiHeadAttention()
        with torch.no_grad():
            model.W_o.zero_()
        X = torch.randn(10, 3, requires_grad=True)
        Y, _ = forward_with_residual(model, X)
        torch.testing.assert_close(Y, X)
        Y.sum().backward()
        torch.testing.assert_close(X.grad, torch.ones_like(X))

    def test_gradients_reach_both_paths(self):
        model = MultiHeadAttention()
        X = torch.randn(10, 3, requires_grad=True)
        B, _ = model(X)
        branch_grad = torch.autograd.grad(B.sum(), X)[0]
        Y, _ = forward_with_residual(model, X)
        Y.sum().backward()
        torch.testing.assert_close(X.grad, 1 + branch_grad)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_training_uses_post_norm_and_updates_parameters(self):
        # 独立重建初始化及一步 Adam，核对实际训练入口，避免只测试加法函数。
        for mode in ('none', 'sinusoidal', 'rope'):
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                I = torch.randn(10, 3)
                X = I + sinusoidal_encoding(I) if mode == 'sinusoidal' else I
                target = I.mean(dim=1, keepdim=True).repeat(1, 3)
                reference = AttentionSublayer(use_rope=mode == 'rope')
                optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
                B, _ = reference.attention(X)
                Y = X + B
                mean = Y.mean(dim=0, keepdim=True)
                var = Y.var(dim=0, unbiased=False, keepdim=True)
                Z = (Y - mean) / torch.sqrt(var + reference.norm.eps)
                Z = Z * reference.norm.weight[:, None] + reference.norm.bias[:, None]
                expected_loss = ((Z - target) ** 2).mean()
                expected_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    Z_after, weights_after = reference(X)
                    expected_after = ((Z_after - target) ** 2).mean().item()
                with contextlib.redirect_stdout(io.StringIO()):
                    actual, losses, weights = train(mode, num_steps=1)
                self.assertEqual(len(losses), 2)
                self.assertEqual(len(weights), 2)
                self.assertAlmostEqual(losses[0], expected_loss.item(), places=6)
                self.assertAlmostEqual(losses[1], expected_after, places=6)
                torch.testing.assert_close(weights[-1], weights_after)
                for name, value in actual.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[name])

    def test_all_modes_learn_fixed_target(self):
        for mode in ('none', 'sinusoidal', 'rope'):
            with self.subTest(mode=mode):
                with contextlib.redirect_stdout(io.StringIO()):
                    _, losses, weights = train(mode)
                self.assertEqual(len(losses), 301)
                self.assertEqual(len(weights), 301)
                self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
                # 新增 LayerNorm 改变了优化过程；要求 300 步至少降低 99%。
                self.assertLess(losses[-1], losses[0] * 0.01)
                history = torch.stack(weights)
                self.assertEqual(tuple(history.shape), (301, 2, 3, 3))
                torch.testing.assert_close(history.sum(dim=2), torch.ones(301, 2, 3))
                self.assertTrue(((history >= 0) & (history <= 1)).all())


if __name__ == '__main__':
    unittest.main()

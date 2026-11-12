"""列向量布局的因果方向、未来隔离与 Padding Mask 的独立组合。"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention import MultiHeadAttention


class CausalMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_hand_calculated_prefix_means(self):
        model = MultiHeadAttention(d_in=2, num_heads=1, d_k=2, d_v=2).eval()
        with torch.no_grad():
            model.W_q[0].zero_()
            model.W_k[0].zero_()
            model.W_v[0].copy_(torch.eye(2))
            model.W_o.copy_(torch.eye(2))
        X = torch.tensor([[1., 3., 8.], [2., 4., 12.]])
        output, weights = model(X, causal=True)
        torch.testing.assert_close(weights, torch.tensor([
            [[1., .5, 1/3], [0., .5, 1/3], [0., 0., 1/3]]]))
        torch.testing.assert_close(output, torch.tensor([[1., 2., 4.], [2., 3., 6.]]))
        # 默认双向；末尾 token 可以影响开头，不能无意中启用 causal。
        bidirectional, normal_weights = model(X)
        torch.testing.assert_close(normal_weights, torch.full((1, 3, 3), 1/3))
        torch.testing.assert_close(bidirectional, torch.tensor([[4., 4., 4.], [6., 6., 6.]]))

    def test_future_changes_cannot_change_prefix_including_rope_and_batch(self):
        for rope in (False, True):
            for shape in ((10, 5), (2, 10, 5)):
                with self.subTest(rope=rope, shape=shape):
                    model = MultiHeadAttention(use_rope=rope, dropout=.3).eval()
                    X = torch.randn(*shape)
                    changed = X.clone()
                    changed[..., 3:] += torch.randn_like(changed[..., 3:]) * 10
                    output, weights = model(X, causal=True)
                    other, other_weights = model(changed, causal=True)
                    torch.testing.assert_close(output[..., :3], other[..., :3])
                    torch.testing.assert_close(weights[..., :3], other_weights[..., :3])
                    if len(shape) == 3:
                        for row in range(shape[0]):
                            single, single_weights = model(X[row], causal=True)
                            torch.testing.assert_close(output[row], single)
                            torch.testing.assert_close(weights[row], single_weights)

    def test_padding_and_causal_union_with_empty_query_is_finite(self):
        model = MultiHeadAttention(use_rope=True, dropout=.2)
        X = torch.randn(3, 10, 4, requires_grad=True)
        # 右 padding、左 padding、中间 padding。
        padding = torch.tensor([[False, False, True, True],
                                [True, True, False, False],
                                [False, True, False, True]])
        output, weights = model(X, padding_mask=padding, causal=True)
        positions = torch.arange(4)
        blocked = padding[..., None] | (positions[:, None] > positions[None, :])
        blocked_heads = blocked[:, None].expand_as(weights)
        torch.testing.assert_close(weights[blocked_heads], torch.zeros_like(weights[blocked_heads]))
        expected_sums = (~blocked.all(dim=-2)).float()[:, None].expand(3, 2, 4)
        torch.testing.assert_close(weights.sum(dim=-2), expected_sums)
        torch.testing.assert_close(output[1, :, :2], torch.zeros(10, 2))
        self.assertTrue(torch.isfinite(output).all())
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(X.grad).all())
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_future_input_has_zero_gradient_from_prefix_loss(self):
        model = MultiHeadAttention().eval()
        X = torch.randn(10, 5, requires_grad=True)
        output, _ = model(X, causal=True)
        output[:, :3].square().sum().backward()
        torch.testing.assert_close(X.grad[:, 3:], torch.zeros(10, 2))
        self.assertGreater(X.grad[:, :3].abs().sum().item(), 0)

    def test_default_equals_explicit_false_with_padding(self):
        model = MultiHeadAttention().eval()
        X = torch.randn(2, 10, 4)
        mask = torch.tensor([[False, False, True, True], [False, False, False, True]])
        default = model(X, padding_mask=mask)
        explicit = model(X, padding_mask=mask, causal=False)
        for first, second in zip(default, explicit):
            torch.testing.assert_close(first, second)

    def test_single_position_and_all_pad_rejection(self):
        model = MultiHeadAttention().eval()
        output, weights = model(torch.randn(10, 1), causal=True)
        self.assertEqual(tuple(output.shape), (10, 1))
        torch.testing.assert_close(weights, torch.ones(2, 1, 1))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 10, 3), padding_mask=torch.tensor([
                [False, False, False], [True, True, True]]), causal=True)


if __name__ == '__main__':
    unittest.main()

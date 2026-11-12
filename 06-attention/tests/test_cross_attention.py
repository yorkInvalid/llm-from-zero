"""Q 来自目标状态、K/V 来自源 memory 的矩形注意力测试。"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention import MultiHeadAttention


class CrossAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_rectangular_attention_matches_manual_formula(self):
        model = MultiHeadAttention(d_in=2, num_heads=1, d_k=2, d_v=2).double().eval()
        with torch.no_grad():
            for matrix in (model.W_q[0], model.W_k[0], model.W_v[0], model.W_o):
                matrix.copy_(torch.eye(2, dtype=torch.double))
        X = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.double)
        memory = torch.tensor([[1., 2., -1.], [2., -1., 1.]], dtype=torch.double)
        output, weights = model(X, memory=memory)
        expected_weights = torch.softmax(memory.T @ X / (2 ** .5), dim=0)
        torch.testing.assert_close(weights[0], expected_weights)
        torch.testing.assert_close(output, memory @ expected_weights)
        self.assertEqual(tuple(weights.shape), (1, 3, 2))
        # 第一个 query 也能读到最后一个源位置，没有因果遮罩。
        self.assertGreater(weights[0, 2, 0].item(), 0)

    def test_batch_matches_separate_calls_and_source_padding_is_ignored(self):
        model = MultiHeadAttention(dropout=.2).eval()
        X = torch.randn(2, 10, 3)
        memory = torch.randn(2, 10, 5)
        mask = torch.tensor([[False, False, False, False, True],
                             [False, False, True, True, True]])
        output, weights = model(X, memory=memory, padding_mask=mask)
        self.assertEqual(tuple(output.shape), (2, 10, 3))
        self.assertEqual(tuple(weights.shape), (2, 2, 5, 3))
        torch.testing.assert_close(weights.sum(dim=-2), torch.ones(2, 2, 3))
        for row, length in ((0, 4), (1, 2)):
            single, single_weights = model(X[row], memory=memory[row, :, :length])
            torch.testing.assert_close(output[row], single)
            torch.testing.assert_close(weights[row, :, :length], single_weights)
            torch.testing.assert_close(weights[row, :, length:], torch.zeros_like(weights[row, :, length:]))
        changed = memory.clone()
        changed.masked_fill_(mask.unsqueeze(1), 1000.)
        torch.testing.assert_close(model(X, memory=changed, padding_mask=mask)[0], output)
        changed[0, :, 0] += 10
        new_output, _ = model(X, memory=changed, padding_mask=mask)
        self.assertFalse(torch.allclose(new_output[0], output[0]))
        torch.testing.assert_close(new_output[1], output[1])

    def test_gradients_reach_both_inputs_and_parameters_but_not_source_pad(self):
        model = MultiHeadAttention(dropout=.1)
        X = torch.randn(2, 10, 3, requires_grad=True)
        memory = torch.randn(2, 10, 4, requires_grad=True)
        mask = torch.tensor([[False, False, True, True], [False, False, False, True]])
        output, _ = model(X, memory=memory, padding_mask=mask)
        output.square().sum().backward()
        for gradient in (X.grad, memory.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        masked = mask.unsqueeze(1).expand_as(memory)
        torch.testing.assert_close(memory.grad[masked], torch.zeros_like(memory.grad[masked]))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_self_attention_matches_cross_with_same_input(self):
        model = MultiHeadAttention().eval()
        X = torch.randn(10, 4)
        mask = torch.tensor([False, False, False, True])
        for actual, expected in zip(model(X, memory=X, padding_mask=mask),
                                    model(X, padding_mask=mask)):
            torch.testing.assert_close(actual, expected)

    def test_single_source_and_independent_attention_instances(self):
        cross = MultiHeadAttention().eval()
        self_attention = MultiHeadAttention()
        for left, right in zip(cross.parameters(), self_attention.parameters()):
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        X = torch.randn(10, 3)
        output, weights = cross(X, memory=torch.randn(10, 1))
        torch.testing.assert_close(weights, torch.ones(2, 1, 3))
        torch.testing.assert_close(output[:, 0], output[:, 2])

    def test_invalid_cross_configuration_is_rejected(self):
        model = MultiHeadAttention()
        X = torch.randn(2, 10, 3)
        memory = torch.randn(2, 10, 4)
        for invalid in (torch.randn(10, 4), torch.randn(1, 10, 4),
                        torch.randn(2, 9, 4), torch.randn(2, 10, 0), memory.double()):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                model(X, memory=invalid)
        with self.assertRaises(ValueError):
            model(X, memory=memory, causal=True)
        with self.assertRaises(ValueError):
            MultiHeadAttention(use_rope=True)(X, memory=memory)
        with self.assertRaises(ValueError):
            model(X, memory=memory, padding_mask=torch.zeros(2, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(X, memory=memory, padding_mask=torch.ones(2, 4, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()

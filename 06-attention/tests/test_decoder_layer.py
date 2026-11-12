"""Decoder 的三条分支、双 mask、因果性与 Encoder 梯度连接。"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder_layer import DecoderLayer
from encoder import Encoder


def manual_norm(X, norm):
    centered = X - X.mean(dim=-2, keepdim=True)
    scaled = centered / torch.sqrt(X.var(dim=-2, unbiased=False, keepdim=True) + norm.eps)
    return scaled * norm.weight[:, None] + norm.bias[:, None]


class DecoderLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_three_branches_match_explicit_calculation(self):
        model = DecoderLayer().double().eval()
        with torch.no_grad():
            for i, norm in enumerate((model.norm1, model.norm2, model.norm3)):
                norm.weight.copy_(torch.linspace(.5, 1.5, 10) + i * .1)
                norm.bias.fill_(i * .2)
        X = torch.randn(10, 3, dtype=torch.double)
        memory = torch.randn(10, 5, dtype=torch.double)
        B, self_weights = model.self_attention(X, causal=True)
        H1 = manual_norm(X + B, model.norm1)
        C, cross_weights = model.cross_attention(H1, memory=memory)
        H2 = manual_norm(H1 + C, model.norm2)
        expected = manual_norm(H2 + model.ffn(H2), model.norm3)
        output, weights = model(X, memory)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(weights['self'], self_weights)
        torch.testing.assert_close(weights['cross'], cross_weights)
        self.assertEqual(tuple(output.shape), (10, 3))
        self.assertEqual(tuple(weights['cross'].shape), (2, 5, 3))

    def test_independent_parameters_and_full_dropout_keep_residuals(self):
        model = DecoderLayer(dropout=1.)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1710)
        for left, right in zip(model.self_attention.parameters(), model.cross_attention.parameters()):
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        for name in ('weight', 'bias'):
            self.assertEqual(len({getattr(n, name).data_ptr() for n in
                                  (model.norm1, model.norm2, model.norm3)}), 3)
        X = torch.randn(10, 3)
        output, _ = model(X, torch.randn(10, 4))
        expected = manual_norm(manual_norm(manual_norm(X, model.norm1), model.norm2), model.norm3)
        torch.testing.assert_close(output, expected)

    def test_future_target_changes_do_not_affect_prefix(self):
        for rope in (False, True):
            with self.subTest(rope=rope):
                model = DecoderLayer(use_rope=rope, dropout=.2).eval()
                X = torch.randn(2, 10, 5)
                memory = torch.randn(2, 10, 4)
                output, weights = model(X, memory)
                changed = X.clone()
                changed[..., 3:] += torch.randn_like(changed[..., 3:]) * 10
                other, other_weights = model(changed, memory)
                torch.testing.assert_close(output[..., :3], other[..., :3])
                for name in ('self', 'cross'):
                    torch.testing.assert_close(weights[name][..., :3], other_weights[name][..., :3])
                self.assertEqual(model.self_attention.use_rope, rope)
                self.assertFalse(model.cross_attention.use_rope)

    def test_source_and_target_masks_are_routed_separately(self):
        model = DecoderLayer().eval()
        X = torch.randn(2, 10, 4)
        memory = torch.randn(2, 10, 5)
        target_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
        source_mask = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
        output, weights = model(X, memory, target_mask, source_mask)
        for row, t, s in ((0, 3, 4), (1, 2, 3)):
            single, _ = model(X[row, :, :t], memory[row, :, :s])
            torch.testing.assert_close(output[row, :, :t], single)
            torch.testing.assert_close(weights['self'][row, :, t:], torch.zeros_like(weights['self'][row, :, t:]))
            torch.testing.assert_close(weights['cross'][row, :, s:], torch.zeros_like(weights['cross'][row, :, s:]))
        changed_X = X.masked_fill(target_mask[:, None], 100.)
        changed_memory = memory.masked_fill(source_mask[:, None], -100.)
        other, _ = model(changed_X, changed_memory, target_mask, source_mask)
        valid = (~target_mask)[:, None].expand_as(output)
        torch.testing.assert_close(output[valid], other[valid])
        changed_memory[0, :, 0] += 20
        changed_output, _ = model(X, changed_memory, target_mask, source_mask)
        self.assertFalse(torch.allclose(output[0, :, :3], changed_output[0, :, :3]))

    def test_gradients_connect_decoder_to_actual_encoder(self):
        encoder = Encoder(num_layers=1)
        decoder = DecoderLayer()
        source = torch.randn(10, 4, requires_grad=True)
        target = torch.randn(10, 5, requires_grad=True)
        source_mask = torch.tensor([False, False, False, True])
        memory, _ = encoder(source, padding_mask=source_mask)
        output, _ = decoder(target, memory, source_padding_mask=source_mask)
        (output[:, :3] - torch.randn(10, 3)).square().mean().backward()
        torch.testing.assert_close(target.grad[:, 3:], torch.zeros(10, 2))
        torch.testing.assert_close(source.grad[:, 3:], torch.zeros(10, 1))
        self.assertGreater(source.grad[:, :3].abs().sum().item(), 0)
        for model in (encoder, decoder):
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_left_padding_has_finite_outputs_and_gradients(self):
        model = DecoderLayer()
        X = torch.randn(10, 4, requires_grad=True)
        output, weights = model(X, torch.randn(10, 3),
                                target_padding_mask=torch.tensor([True, True, False, False]))
        torch.testing.assert_close(weights['self'][..., :2], torch.zeros(2, 4, 2))
        self.assertTrue(torch.isfinite(output).all())
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(X.grad).all())

    def test_memory_required_and_all_pad_rejected(self):
        model = DecoderLayer()
        X, memory = torch.randn(10, 3), torch.randn(10, 4)
        with self.assertRaises(ValueError):
            model(X, None)
        with self.assertRaises(ValueError):
            model(X, memory, target_padding_mask=torch.ones(3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(X, memory, source_padding_mask=torch.ones(4, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()

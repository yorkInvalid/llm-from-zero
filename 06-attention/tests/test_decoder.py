"""Decoder 堆叠、memory 共享、双 mask、因果性和端到端参数更新。"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder import Decoder
from encoder import Encoder


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_invalid_depth_and_independent_parameters(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                Decoder(num_layers=count)
        model = Decoder()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5130)
        groups = [dict(layer.named_parameters()) for layer in model.layers]
        for name in groups[0]:
            self.assertEqual(len({group[name].data_ptr() for group in groups}), 3)
        before = groups[1]['norm1.weight'].detach().clone()
        with torch.no_grad():
            groups[0]['norm1.weight'].add_(1)
        torch.testing.assert_close(groups[1]['norm1.weight'], before)

    def test_manual_stack_matches_and_every_layer_receives_same_memory(self):
        for count, batch, rope in ((1, False, False), (3, True, True)):
            with self.subTest(count=count, batch=batch, rope=rope):
                model = Decoder(num_layers=count, use_rope=rope).eval()
                X = torch.randn(*((2,) if batch else ()), 10, 3)
                memory = torch.randn(*((2,) if batch else ()), 10, 5)
                seen = []
                handles = [layer.register_forward_pre_hook(
                    lambda module, args: seen.append(args[1])) for layer in model.layers]
                try:
                    output, weights = model(X, memory)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(len(seen), count)
                self.assertTrue(all(item is memory for item in seen))
                H = X
                for index, layer in enumerate(model.layers):
                    H, expected_weights = layer(H, memory)
                    for name in ('self', 'cross'):
                        torch.testing.assert_close(weights[index][name], expected_weights[name])
                    self.assertEqual(layer.self_attention.use_rope, rope)
                    self.assertFalse(layer.cross_attention.use_rope)
                torch.testing.assert_close(output, H)
                self.assertEqual(output.shape, X.shape)

    def test_full_stack_is_causal_for_both_attention_outputs(self):
        model = Decoder(use_rope=True, dropout=.2).eval()
        X, memory = torch.randn(2, 10, 5), torch.randn(2, 10, 4)
        output, weights = model(X, memory)
        changed = X.clone()
        changed[..., 3:] += torch.randn_like(changed[..., 3:]) * 10
        other, other_weights = model(changed, memory)
        torch.testing.assert_close(output[..., :3], other[..., :3])
        for first, second in zip(weights, other_weights):
            for name in ('self', 'cross'):
                torch.testing.assert_close(first[name][..., :3], second[name][..., :3])

    def test_masks_reach_all_layers_and_batch_matches_unpadded(self):
        model = Decoder(dropout=.2).eval()
        X, memory = torch.randn(2, 10, 4), torch.randn(2, 10, 5)
        target_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
        source_mask = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
        output, weights = model(X, memory, target_mask, source_mask)
        for row, t, s in ((0, 3, 4), (1, 2, 3)):
            single, single_weights = model(X[row, :, :t], memory[row, :, :s])
            torch.testing.assert_close(output[row, :, :t], single)
            for layer_weights, single_layer in zip(weights, single_weights):
                torch.testing.assert_close(layer_weights['self'][row, :, :t, :t], single_layer['self'])
                torch.testing.assert_close(layer_weights['cross'][row, :, :s, :t], single_layer['cross'])
                for name, length in (('self', t), ('cross', s)):
                    padded_rows = layer_weights[name][row, :, length:]
                    torch.testing.assert_close(padded_rows, torch.zeros_like(padded_rows))
        changed_X = X.masked_fill(target_mask[:, None], 100.)
        changed_memory = memory.masked_fill(source_mask[:, None], -100.)
        other, _ = model(changed_X, changed_memory, target_mask, source_mask)
        valid = (~target_mask)[:, None].expand_as(output)
        torch.testing.assert_close(output[valid], other[valid])

    def test_optimizer_updates_every_decoder_and_encoder_layer(self):
        encoder = Encoder(num_layers=2)
        decoder = Decoder(num_layers=3, dropout=.1)
        parameters = list(encoder.parameters()) + list(decoder.parameters())
        before = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.Adam(parameters, lr=.01)
        source = torch.randn(10, 4, requires_grad=True)
        target = torch.randn(10, 5, requires_grad=True)
        mask = torch.tensor([False, False, False, True])
        memory, _ = encoder(source, padding_mask=mask)
        output, _ = decoder(target, memory, source_padding_mask=mask)
        (output[:, :3] - torch.randn(10, 3)).square().mean().backward()
        torch.testing.assert_close(source.grad[:, 3:], torch.zeros(10, 1))
        torch.testing.assert_close(target.grad[:, 3:], torch.zeros(10, 2))
        self.assertGreater(source.grad[:, :3].abs().sum().item(), 0)
        for p in parameters:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        optimizer.step()
        for old, new in zip(before, parameters):
            self.assertFalse(torch.equal(old, new))

    def test_left_padding_and_required_memory(self):
        model = Decoder()
        X = torch.randn(10, 4, requires_grad=True)
        output, weights = model(X, torch.randn(10, 3),
                                target_padding_mask=torch.tensor([True, True, False, False]))
        self.assertTrue(torch.isfinite(output).all())
        for layer_weights in weights:
            torch.testing.assert_close(layer_weights['self'][..., :2], torch.zeros(2, 4, 2))
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(X.grad).all())
        with self.assertRaises(ValueError):
            model(X, None)


if __name__ == '__main__':
    unittest.main()

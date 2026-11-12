"""Dropout 的插入位置、train/eval 开关、PAD 兼容及训练集成。"""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention import MultiHeadAttention
from attention_sublayer import AttentionSublayer
from encoder_layer import EncoderLayer
from token_encoder import TokenClassifier, TokenEncoder
from train_tokens import train_tokens


class DropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_eval_matches_zero_dropout_and_parameter_count_is_unchanged(self):
        zero = TokenClassifier(7, padding_idx=6)
        dropped = TokenClassifier(7, padding_idx=6, dropout=.5)
        dropped.load_state_dict(zero.state_dict())
        self.assertEqual(sum(p.numel() for p in zero.parameters()),
                         sum(p.numel() for p in dropped.parameters()))
        ids = torch.tensor([[0, 1, 2], [3, 4, 6]])
        dropped.eval()
        torch.testing.assert_close(zero(ids)[0], dropped(ids)[0])
        torch.testing.assert_close(dropped(ids)[0], dropped(ids)[0])
        zero.train()
        torch.testing.assert_close(zero(ids)[0], zero(ids)[0])

    def test_train_is_stochastic_even_inside_no_grad(self):
        model = TokenClassifier(7, dropout=.5)
        ids = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 0]])
        model.train()
        with torch.no_grad():
            first, _ = model(ids)
            second, _ = model(ids)
        self.assertFalse(torch.equal(first, second))
        model.eval()
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m, nn.Dropout)))
        model.train()
        self.assertTrue(all(m.training for m in model.modules() if isinstance(m, nn.Dropout)))

    def test_input_dropout_is_after_position_encoding(self):
        for mode in ('none', 'sinusoidal', 'rope'):
            with self.subTest(mode=mode):
                model = TokenEncoder(7, dropout=1., position_encoding=mode)
                captured = []
                handle = model.encoder.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach().clone()))
                try:
                    model(torch.tensor([0, 1, 2]))
                finally:
                    handle.remove()
                torch.testing.assert_close(captured[0], torch.zeros(10, 3))

    def test_residual_paths_survive_full_branch_dropout(self):
        X = torch.randn(10, 3)
        attention = AttentionSublayer(dropout=1.)
        output, _ = attention(X)
        torch.testing.assert_close(output, attention.norm(X.T).T)
        layer = EncoderLayer(dropout=1.)
        Z, _ = layer.attention_sublayer(X)
        output, _ = layer(X)
        torch.testing.assert_close(output, layer.norm2(Z.T).T)

    def test_attention_uses_dropped_weights_but_returns_probabilities(self):
        model = MultiHeadAttention(d_in=2, num_heads=1, d_k=2, d_v=2, dropout=.5)
        with torch.no_grad():
            model.W_q[0].zero_()
            model.W_k[0].zero_()
            model.W_v[0].copy_(torch.eye(2))
            model.W_o.copy_(torch.eye(2))
        X = torch.randn(2, 8)
        mask = torch.tensor([False] * 6 + [True] * 2)
        captured = []
        handle = model.attention_dropout.register_forward_hook(
            lambda module, args, output: captured.append(output.detach().clone()))
        try:
            output, weights = model(X, padding_mask=mask)
        finally:
            handle.remove()
        used = captured[0]
        torch.testing.assert_close(output, X @ used)
        torch.testing.assert_close(weights.sum(dim=-2), torch.ones(1, 8))
        torch.testing.assert_close(used[mask], torch.zeros(2, 8))
        self.assertTrue(((used == 0) | torch.isclose(used, weights[0] / .5)).all())
        self.assertGreater((used[:6] == 0).sum().item(), 0)
        self.assertGreater((used[:6] != 0).sum().item(), 0)

    def test_training_switches_modes_and_updates_with_dropout(self):
        modes = []
        def factory(*args, **kwargs):
            model = TokenClassifier(*args, **kwargs)
            model.register_forward_pre_hook(
                lambda module, args: modes.append((module.training, torch.is_grad_enabled())))
            return model
        with patch('train_tokens.TokenClassifier', side_effect=factory):
            with contextlib.redirect_stdout(io.StringIO()):
                model, losses, snapshots = train_tokens(batch=True, num_steps=3, dropout=.1)
        self.assertEqual(modes, [(False, False), (True, True)] * 3 + [(False, False)])
        self.assertFalse(model.training)
        self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
        self.assertFalse(torch.equal(snapshots['before'], snapshots['after']))

    def test_eval_padding_invariance_with_nonzero_dropout_configuration(self):
        model = TokenClassifier(7, padding_idx=6, dropout=.2, position_encoding='rope').eval()
        batched, _ = model(torch.tensor([[0, 1, 2], [3, 4, 6]]))
        single, _ = model(torch.tensor([3, 4]))
        torch.testing.assert_close(batched[1, :2], single)

    def test_invalid_probability(self):
        for p in (-.1, 1.1):
            with self.subTest(p=p), self.assertRaises(ValueError):
                TokenClassifier(7, dropout=p)


if __name__ == '__main__':
    unittest.main()

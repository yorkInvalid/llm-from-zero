"""Token ID 查表、位置编码与真实训练中 Embedding 参数更新。"""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from position_encoding import sinusoidal_encoding
from token_encoder import TokenEncoder, TokenClassifier
from train_tokens import train_tokens, VOCAB, LABELS


class TokenEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_lookup_and_position_encoding_at_encoder_boundary(self):
        ids = torch.tensor([2, 0, 2])
        for mode in ('none', 'sinusoidal', 'rope'):
            with self.subTest(mode=mode):
                model = TokenEncoder(4, num_layers=2, position_encoding=mode)
                with torch.no_grad():
                    model.embedding.weight.copy_(torch.arange(40.).reshape(4, 10) / 40)
                captured = []
                handle = model.encoder.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach().clone()))
                try:
                    output, weights = model(ids)
                finally:
                    handle.remove()
                expected = model.embedding.weight[ids].T
                torch.testing.assert_close(expected[:, 0], expected[:, 2])
                if mode == 'sinusoidal':
                    expected = expected + sinusoidal_encoding(expected)
                torch.testing.assert_close(captured[0], expected)
                self.assertEqual(tuple(output.shape), (10, 3))
                self.assertEqual(len(weights), 2)
                for layer in model.encoder.layers:
                    self.assertEqual(layer.attention_sublayer.attention.use_rope, mode == 'rope')

    def test_backward_only_computes_gradients_step_changes_parameters(self):
        model = TokenClassifier(5, num_layers=2)
        ids = torch.tensor([0, 1, 0, 2])
        original_ids = ids.clone()
        labels = torch.tensor([0, 1, 0, 2])
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        logits, _ = model(ids)
        self.assertEqual(tuple(logits.shape), (4, 3))
        nn.CrossEntropyLoss()(logits, labels).backward()
        embedding = model.token_encoder.embedding
        self.assertGreater(embedding.weight.grad[:3].abs().sum().item(), 0)
        # 没有参与查表的行没有梯度；这是首次 Adam 更新，没有历史动量或 weight decay。
        torch.testing.assert_close(embedding.weight.grad[3:], torch.zeros(2, 10))
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, before[name])
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        optimizer.step()
        for name, p in model.named_parameters():
            self.assertFalse(torch.equal(p, before[name]), name)
        torch.testing.assert_close(embedding.weight[3:], before['token_encoder.embedding.weight'][3:])
        torch.testing.assert_close(ids, original_ids)
        self.assertFalse(ids.requires_grad)

    def test_invalid_ids_and_position_mode(self):
        model = TokenEncoder(4)
        for ids in (torch.tensor([]), torch.tensor([[[1, 2]]]), torch.tensor([-1]), torch.tensor([4])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                model(ids)
        with self.assertRaises(TypeError):
            model(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            TokenEncoder(4, position_encoding='unknown')

    def test_training_updates_embedding_and_learns_fixed_labels(self):
        for mode in ('none', 'sinusoidal', 'rope'):
            with self.subTest(mode=mode):
                with contextlib.redirect_stdout(io.StringIO()):
                    model, losses, snapshots = train_tokens(mode)
                self.assertEqual(len(losses), 201)
                self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
                self.assertLess(losses[-1], losses[0] * 0.1)
                torch.testing.assert_close(snapshots['before'], snapshots['after_backward'])
                self.assertGreater(snapshots['first_gradient'].abs().sum().item(), 0)
                self.assertFalse(torch.equal(snapshots['before'], snapshots['after_first_step']))
                torch.testing.assert_close(model.token_encoder.embedding.weight, snapshots['after'])
                with torch.no_grad():
                    logits, _ = model(torch.arange(len(VOCAB)))
                torch.testing.assert_close(logits.argmax(dim=1), torch.tensor(LABELS))
                self.assertAlmostEqual(nn.CrossEntropyLoss()(logits, torch.tensor(LABELS)).item(), losses[-1], places=6)

    def test_zero_steps_does_not_update_embedding(self):
        with contextlib.redirect_stdout(io.StringIO()):
            _, losses, snapshots = train_tokens(num_steps=0)
        self.assertEqual(len(losses), 1)
        torch.testing.assert_close(snapshots['before'], snapshots['after'])
        self.assertNotIn('first_gradient', snapshots)


if __name__ == '__main__':
    unittest.main()

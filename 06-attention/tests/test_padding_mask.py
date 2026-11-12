"""Batch 隔离、PAD key 遮罩、多层不变性与忽略标签的梯度验证。"""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention import MultiHeadAttention
from encoder import Encoder
from token_encoder import TokenClassifier
from train_tokens import (train_tokens, make_padded_batch, token_classification_loss,
                          PAD_ID, IGNORE_INDEX)


class PaddingMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_masked_key_row_is_zero_not_query_column(self):
        model = MultiHeadAttention(d_in=2, num_heads=1, d_k=2, d_v=2)
        with torch.no_grad():
            model.W_q[0].zero_()
            model.W_k[0].zero_()
            model.W_v[0].copy_(torch.eye(2))
            model.W_o.copy_(torch.eye(2))
        X = torch.tensor([[1., 3., 999.], [2., 6., -999.]])
        output, weights = model(X, padding_mask=torch.tensor([False, False, True]))
        expected_weights = torch.tensor([[[.5, .5, .5], [.5, .5, .5], [0., 0., 0.]]])
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(output, torch.tensor([[2., 2., 2.], [4., 4., 4.]]))

    def test_batch_matches_separate_unpadded_sequences_in_all_layers(self):
        for mode in ('none', 'sinusoidal', 'rope'):
            with self.subTest(mode=mode):
                model = TokenClassifier(7, padding_idx=PAD_ID, position_encoding=mode)
                ids = torch.tensor([[0, 1, 2], [3, 4, PAD_ID]])
                logits, all_weights = model(ids)
                self.assertEqual(tuple(logits.shape), (2, 3, 3))
                self.assertEqual(tuple(all_weights[0].shape), (2, 2, 3, 3))
                self.assertTrue(torch.isfinite(logits).all())
                for row, length in ((0, 3), (1, 2)):
                    separate, separate_weights = model(ids[row, :length])
                    torch.testing.assert_close(logits[row, :length], separate)
                    for batched, single in zip(all_weights, separate_weights):
                        torch.testing.assert_close(batched[row, :, :length, :length], single)
                        torch.testing.assert_close(batched[row].sum(dim=-2), torch.ones(2, 3))
                        if length < 3:
                            torch.testing.assert_close(batched[row, :, length:, :], torch.zeros(2, 1, 3))
                # 改变另一个样本，当前样本的有效预测不变。
                changed = ids.clone()
                changed[0] = torch.tensor([5, 5, 5])
                torch.testing.assert_close(model(changed)[0][1, :2], logits[1, :2])
                # PAD 向量即使被手动设为非零，有效预测也不应变化。
                with torch.no_grad():
                    model.token_encoder.embedding.weight[PAD_ID].fill_(20)
                torch.testing.assert_close(model(ids)[0][1, :2], logits[1, :2])

    def test_masked_input_has_no_gradient_from_valid_outputs(self):
        model = Encoder(num_layers=3, use_rope=True)
        X = torch.randn(2, 10, 4, requires_grad=True)
        mask = torch.tensor([[False, False, True, True], [False, False, False, True]])
        output, _ = model(X, padding_mask=mask)
        valid = (~mask).unsqueeze(1).expand_as(output)
        (output * torch.randn_like(output))[valid].sum().backward()
        torch.testing.assert_close(X.grad[~valid], torch.zeros_like(X.grad[~valid]))
        self.assertGreater(X.grad[valid].abs().sum().item(), 0)

    def test_loss_ignores_pad_logits_and_matches_valid_token_average(self):
        logits = torch.randn(2, 3, 3, requires_grad=True)
        labels = torch.tensor([[0, 1, 2], [0, 1, IGNORE_INDEX]])
        valid = labels != IGNORE_INDEX
        loss = token_classification_loss(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits[valid], labels[valid])
        torch.testing.assert_close(loss, expected)
        changed = logits.detach().clone()
        changed[~valid] = torch.tensor([100., -100., 50.])
        torch.testing.assert_close(token_classification_loss(changed, labels), loss)
        loss.backward()
        torch.testing.assert_close(logits.grad[~valid], torch.zeros(1, 3))
        with self.assertRaises(ValueError):
            token_classification_loss(logits, torch.full_like(labels, IGNORE_INDEX))

    def test_invalid_masks_and_all_pad_sample_are_rejected(self):
        model = MultiHeadAttention()
        X = torch.randn(2, 10, 3)
        for mask in (torch.zeros(3, dtype=torch.bool),
                     torch.tensor([[False, False, False], [True, True, True]])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                model(X, padding_mask=mask)
        with self.assertRaises(TypeError):
            model(X, padding_mask=torch.zeros(2, 3))
        classifier = TokenClassifier(7, padding_idx=PAD_ID)
        with self.assertRaises(ValueError):
            classifier(torch.tensor([[0, 1], [PAD_ID, PAD_ID]]))

    def test_batch_gradient_matches_separate_sequence_losses(self):
        model = TokenClassifier(7, padding_idx=PAD_ID, num_layers=2, position_encoding='rope')
        ids, labels = make_padded_batch()
        logits, _ = model(ids)
        token_classification_loss(logits, labels).backward()
        batch_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        model.zero_grad()
        loss = sum(token_classification_loss(model(ids[row, :length])[0], labels[row, :length]) * length
                   for row, length in ((0, 3), (1, 2))) / 5
        loss.backward()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, batch_grads[name], atol=2e-6, rtol=2e-4)
        torch.testing.assert_close(model.token_encoder.embedding.weight.grad[PAD_ID], torch.zeros(10))

    def test_padded_training_converges_and_pad_embedding_stays_fixed(self):
        ids, labels = make_padded_batch()
        valid = labels != IGNORE_INDEX
        for mode in ('none', 'sinusoidal', 'rope'):
            with self.subTest(mode=mode):
                with contextlib.redirect_stdout(io.StringIO()):
                    model, losses, snapshots = train_tokens(mode, batch=True)
                self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
                self.assertLess(losses[-1], losses[0] * .1)
                logits, _ = model(ids)
                torch.testing.assert_close(logits.argmax(dim=-1)[valid], labels[valid])
                torch.testing.assert_close(snapshots['before'][PAD_ID], snapshots['after'][PAD_ID])
                self.assertGreater((snapshots['after'][:PAD_ID] - snapshots['before'][:PAD_ID]).abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()

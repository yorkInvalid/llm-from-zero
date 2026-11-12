"""标签正确性、数据隔离、PAD 统计和上下文基线验证。"""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_task import (IGNORE_INDEX, PAD_ID, TokenwiseBaseline, collate_sequences,
                          evaluate, make_splits, repetition_labels)
from token_encoder import TokenClassifier
from train_context import train_model


class ContextTaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_labels_depend_on_other_positions(self):
        self.assertEqual(repetition_labels([0, 1, 0]), [1, 0, 1])
        self.assertEqual(repetition_labels([0, 1, 2]), [0, 0, 0])
        self.assertEqual(repetition_labels([2, 2, 2]), [1, 1, 1])
        self.assertEqual(repetition_labels([0, 0, 1, 1, 2]), [1, 1, 1, 1, 0])

    def test_splits_reproducible_and_disjoint_even_under_permutation(self):
        train, validation = make_splits(100, 40)
        self.assertEqual((train, validation), make_splits(100, 40))
        train_keys = {tuple(sorted(seq)) for seq in train}
        val_keys = {tuple(sorted(seq)) for seq in validation}
        self.assertEqual(len(train_keys), 100)
        self.assertEqual(len(val_keys), 40)
        self.assertFalse(train_keys & val_keys)
        self.assertGreater(len({len(seq) for seq in train}), 1)
        for data in (train, validation):
            # 同一个 ID 在该数据集中确实会对应两种标签。
            observed = {}
            for seq in data:
                for token, label in zip(seq, repetition_labels(seq)):
                    observed.setdefault(token, set()).add(label)
            self.assertTrue(all(labels == {0, 1} for labels in observed.values()))

    def test_collation_right_pads_inputs_and_ignores_labels(self):
        ids, labels = collate_sequences([[0, 1, 0], [2, 3]])
        torch.testing.assert_close(ids, torch.tensor([[0, 1, 0], [2, 3, PAD_ID]]))
        torch.testing.assert_close(labels, torch.tensor([[1, 0, 1], [0, 0, IGNORE_INDEX]]))
        for invalid in ([], [[]], [[PAD_ID]], [[-1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                collate_sequences(invalid)

    def test_evaluation_uses_token_weighting_and_does_not_update_model(self):
        class ConstantModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(2))

            def forward(self, ids):
                return self.bias.expand(*ids.shape, 2), []
        model = ConstantModel().train()
        data = [[0, 1, 0], [2, 3]]
        together = evaluate(model, data, batch_size=2)
        separate = evaluate(model, data, batch_size=1)
        for key in together:
            self.assertAlmostEqual(together[key], separate[key], places=6)
        self.assertAlmostEqual(together['accuracy'], 3 / 5)
        self.assertAlmostEqual(together['balanced_accuracy'], .5)
        self.assertAlmostEqual(together['sequence_accuracy'], .5)
        self.assertAlmostEqual(together['loss'], torch.log(torch.tensor(2.)).item())
        self.assertEqual(together['valid_tokens'], 5)
        self.assertTrue(model.training)
        self.assertIsNone(model.bias.grad)
        torch.testing.assert_close(model.bias, torch.zeros(2))

    def test_baseline_cannot_change_prediction_for_same_token_with_new_context(self):
        model = TokenwiseBaseline().eval()
        first, _ = model(torch.tensor([[0, 1, 0]]))
        second, _ = model(torch.tensor([[0, 1, 2]]))
        torch.testing.assert_close(first[0, 0], second[0, 0])
        # token 0 的正确标签却分别为 1 和 0，基线无法同时答对。

    def test_training_records_both_splits_and_updates_model(self):
        train, validation = make_splits(16, 8)
        model = TokenClassifier(9, num_classes=2, d_model=16, num_layers=1,
                                padding_idx=PAD_ID, position_encoding='none', dropout=.1)
        before = model.token_encoder.embedding.weight.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            history = train_model(model, train, validation, epochs=2, batch_size=8)
        self.assertEqual([row['epoch'] for row in history], [0, 1, 2])
        self.assertFalse(model.training)
        self.assertFalse(torch.equal(before, model.token_encoder.embedding.weight))
        for row in history:
            for split, data in (('train', train), ('validation', validation)):
                self.assertTrue(torch.isfinite(torch.tensor(row[split]['loss'])))
                self.assertEqual(row[split]['valid_tokens'], sum(map(len, data)))
        final = evaluate(model, validation, batch_size=8)
        self.assertEqual(final, history[-1]['validation'])


if __name__ == '__main__':
    unittest.main()

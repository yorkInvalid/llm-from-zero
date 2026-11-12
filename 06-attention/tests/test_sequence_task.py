"""倒序数据、加权验证、真实学习和保存后的生成一致性。"""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sequence_task import make_splits, collate_sequences
from train_seq2seq import build_model, train_model, evaluate, load_model


class SequenceTaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_disjoint_unique_reproducible_splits(self):
        train, val = make_splits(100, 20)
        self.assertEqual((train, val), make_splits(100, 20))
        self.assertEqual(len(set(train + val)), 120)
        self.assertFalse(set(train) & set(val))
        self.assertTrue(all(2 <= len(s) <= 6 and all(3 <= t < 13 for t in s) for s in train + val))
        with self.assertRaises(ValueError):
            make_splits(10, 10, min_length=1, max_length=1)

    def test_shift_eos_and_padding(self):
        source, target, labels = collate_sequences([(3, 5, 7), (4, 8)])
        self.assertEqual(source.tolist(), [[3, 5, 7], [4, 8, 0]])
        self.assertEqual(target.tolist(), [[1, 7, 5, 3], [1, 8, 4, 0]])
        self.assertEqual(labels.tolist(), [[7, 5, 3, 2], [8, 4, 2, 0]])
        for bad in ([], [()], [(0, 3)]):
            with self.assertRaises(ValueError):
                collate_sequences(bad)

    def test_evaluation_weighting_modes_and_generation_metrics(self):
        model = build_model(d_model=16, num_layers=1, dropout=.2)
        samples = [(3, 4), (5, 6, 7), (4, 3, 5, 6)]
        model.encoder.encoder.layers[0].eval()
        modules = list(model.modules())
        modes = [m.training for m in modules]
        separate = evaluate(model, samples, batch_size=1)
        batched = evaluate(model, samples, batch_size=3)
        self.assertAlmostEqual(separate['loss'], batched['loss'], places=5)
        self.assertEqual(separate['token_accuracy'], batched['token_accuracy'])
        # 一个完全正确、一个有 EOS 但内容错、一个没有 EOS。
        predictions = [torch.tensor([1, 4, 3, 2]), torch.tensor([1, 3, 2]), torch.tensor([1, 4])]
        with patch.object(model, 'generate', side_effect=predictions):
            metrics = evaluate(model, samples, generate=True)
        self.assertAlmostEqual(metrics['exact_match'], 1 / 3)
        self.assertAlmostEqual(metrics['eos_rate'], 2 / 3)
        self.assertEqual([m.training for m in modules], modes)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_overfit_small_task_and_checkpoint_roundtrip(self):
        samples = [(3, 4), (4, 5), (5, 3), (3, 5, 4)]
        model = build_model(d_model=16, num_layers=1)
        with contextlib.redirect_stdout(io.StringIO()):
            history = train_model(model, samples, samples, epochs=100,
                                  batch_size=4, learning_rate=.005)
        self.assertLess(history[-1]['train']['loss'], history[0]['train']['loss'] * .1)
        metrics = evaluate(model, samples, generate=True, max_new_tokens=5)
        self.assertEqual(metrics['exact_match'], 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(dict(model=model.state_dict(), model_config=model.config), path)
            restored_model = load_model(path)
        restored = evaluate(restored_model, samples, generate=True, max_new_tokens=5)
        self.assertEqual(metrics, restored)


if __name__ == '__main__':
    unittest.main()

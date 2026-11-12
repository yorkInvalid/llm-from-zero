"""自监督标签、无泄漏划分、因果性、真实更新及续写。"""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_lm import CausalLM
from text_pretraining import prepare, load_data, collate, split_paragraphs, make_windows
from train_lm import evaluate, restore, observe_epoch


class TextPretrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_split_tokenizer_roundtrip_and_shift(self):
        paragraphs = [f'Paragraph {i}: Alice reads a different story about a rabbit every evening.' for i in range(30)]
        text = '\n\n'.join(paragraphs + [paragraphs[0]])
        splits = split_paragraphs(text)
        self.assertEqual([len(splits[k]) for k in ('train', 'validation', 'test')], [24, 3, 3])
        self.assertFalse(set(splits['train']) & set(splits['test']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.txt'
            path.write_text(text)
            prepare(directory, path)
            tokenizer, data = load_data(directory, 8)
            self.assertEqual(tokenizer.decode(tokenizer.encode('未见过的文字 🐰').ids), '未见过的文字 🐰')
            paragraph = paragraphs[0]
            windows = make_windows([paragraph], tokenizer, 8)
            expected = tokenizer.encode(paragraph).ids + [2]
            self.assertEqual(torch.cat([y for x, y in windows]).tolist(), expected)
            for x, y in windows:
                torch.testing.assert_close(x[1:], y[:-1])
            x, y = collate([windows[0], (torch.tensor([1, 3]), torch.tensor([3, 2]))])
            self.assertTrue((x[1, 2:] == 0).all())
            self.assertTrue((y[1, 2:] == -100).all())
            model = CausalLM(tokenizer.get_vocab_size(), d_model=16, num_layers=1, context_length=8).eval()
            checkpoint = Path(directory) / 'model.pt'
            torch.save(dict(config=model.config, model=model.state_dict(), tokenizer=tokenizer.to_str()), checkpoint)
            restored, loaded_tokenizer, _ = restore(checkpoint, 'cpu')
            torch.testing.assert_close(model(x), restored(x))
            self.assertEqual(loaded_tokenizer.encode(paragraph).ids, tokenizer.encode(paragraph).ids)

    def test_observation_preserves_training_state_rng_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.txt'
            path.write_text('\n\n'.join(
                f'Paragraph {i}: Alice reads a different story about a rabbit every evening.' for i in range(30)))
            prepare(directory, path)
            tokenizer, data = load_data(directory, 8)
            model = CausalLM(tokenizer.get_vocab_size(), d_model=16, num_layers=1, context_length=8)
            model.layers[0].norm1.eval()
            modes = [m.training for m in model.modules()]
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            rng = torch.get_rng_state().clone()
            first = observe_epoch(model, tokenizer, data['validation'], 0, max_new_tokens=3)
            second = observe_epoch(model, tokenizer, data['validation'], 0, max_new_tokens=3)
            self.assertEqual(first, second)
            torch.testing.assert_close(torch.get_rng_state(), rng)
            self.assertEqual([m.training for m in model.modules()], modes)
            for name, p in model.named_parameters():
                torch.testing.assert_close(p, before[name])
                self.assertIsNone(p.grad)
            for row in first['predictions']['rows']:
                if row['label_id'] >= 0:
                    self.assertTrue(0 <= row['label_probability'] <= 1)
                    self.assertEqual(row['label_in_top3'], row['label_id'] in [c['id'] for c in row['top3']])

    def test_causal_prefix_and_pad_invariance(self):
        model = CausalLM(20, d_model=16, num_layers=1, context_length=8).eval()
        x = torch.tensor([1, 3, 4, 5])
        other = torch.tensor([1, 3, 8, 9])
        torch.testing.assert_close(model(x)[:2], model(other)[:2])
        padded = torch.tensor([[1, 3, 4, 5], [1, 6, 0, 0]])
        torch.testing.assert_close(model(padded)[1, :2], model(torch.tensor([1, 6])))

    def test_learning_evaluation_and_checkpoint(self):
        model = CausalLM(8, d_model=16, num_layers=1, context_length=8, dropout=0.)
        data = [(torch.tensor([1, 3, 4]), torch.tensor([3, 4, 2]))]
        initial = evaluate(model, data)['loss']
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(60):
            x, y = collate(data)
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x).reshape(-1, 8), y.reshape(-1)).backward()
            optimizer.step()
        self.assertLess(evaluate(model, data)['loss'], initial * .1)
        self.assertTrue(model.training)
        self.assertEqual(model.generate(torch.tensor([1]), max_new_tokens=6).tolist(), [1, 3, 4, 2])
        self.assertTrue(model.training)
        self.assertEqual(model.generate(torch.tensor([1]), max_new_tokens=0).tolist(), [1])

    def test_evaluation_token_weighting(self):
        model = CausalLM(10, d_model=16, num_layers=1)
        samples = [(torch.tensor([1, 3]), torch.tensor([3, 2])),
                   (torch.tensor([1, 4, 5, 6]), torch.tensor([4, 5, 6, 2]))]
        a, b = evaluate(model, samples, 1), evaluate(model, samples, 2)
        self.assertAlmostEqual(a['loss'], b['loss'], places=5)
        self.assertEqual(a['tokens'], 6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()

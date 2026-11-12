"""统一封装与原组件等价，自动 mask、参数注册和旧权重兼容。"""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformer import Transformer
from train_seq2seq import build_model, load_model
from generation import greedy_generate


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_forward_matches_components_and_routes_distinct_pad_ids(self):
        model = Transformer(9, 7, source_padding_idx=8, target_padding_idx=0,
                            num_encoder_layers=1, num_decoder_layers=2).eval()
        source = torch.tensor([[3, 4, 8], [4, 5, 6]])
        target = torch.tensor([[1, 3, 0], [1, 4, 5]])
        logits, weights = model(source, target, return_attention=True)
        memory, expected_encoder = model.encoder(source)
        expected, expected_decoder = model.decoder(target, memory, source == 8)
        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(weights, dict(encoder=expected_encoder, decoder=expected_decoder))
        torch.testing.assert_close(model(source, target), logits)
        single = model(source[0, :2], target[0, :2])
        torch.testing.assert_close(single, logits[0, :2])
        self.assertEqual(logits.shape, (2, 3, 7))

    def test_single_optimizer_updates_both_sides(self):
        model = build_model(d_model=16, num_layers=1)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        logits = model(torch.tensor([3, 4]), torch.tensor([1, 4, 3]))
        torch.nn.functional.cross_entropy(logits, torch.tensor([4, 3, 2])).backward()
        optimizer.step()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertFalse(torch.equal(before[name], p), name)
        self.assertTrue(any(n.startswith('encoder.') for n in before))
        self.assertTrue(any(n.startswith('decoder.') for n in before))

    def test_generation_and_legacy_checkpoint(self):
        model = build_model(d_model=16, num_layers=1, dropout=.2)
        source = torch.tensor([3, 4])
        expected = greedy_generate(model.encoder, model.decoder, source, 1, 2, 4)
        torch.testing.assert_close(model.generate(source, 1, 2, 4), expected)
        self.assertTrue(all(m.training for m in model.modules()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            torch.save(dict(encoder=model.encoder.state_dict(), decoder=model.decoder.state_dict(),
                            model_config=dict(d_model=16, num_layers=1, dropout=.2)), path)
            restored = load_model(path)
        torch.testing.assert_close(restored.generate(source, 1, 2, 4), expected)
        for name, p in restored.state_dict().items():
            torch.testing.assert_close(p, model.state_dict()[name])


if __name__ == '__main__':
    unittest.main()

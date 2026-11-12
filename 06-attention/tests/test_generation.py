"""自回归前缀、停止条件、源 mask 和推理状态恢复。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation import greedy_generate
from token_encoder import TokenEncoder
from token_decoder import TokenDecoderLM


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder = TokenEncoder(8, num_layers=1, padding_idx=0, dropout=.5)
        self.decoder = TokenDecoderLM(7, num_layers=2, padding_idx=0, dropout=.5)
        self.source = torch.tensor([3, 4, 0])

    def generate(self, **kwargs):
        return greedy_generate(self.encoder, self.decoder, self.source, 1, 2, **kwargs)

    def test_prefix_grows_memory_is_reused_and_eos_stops(self):
        prefixes, memories = [], []
        def scripted(ids, memory, source_padding_mask=None):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(self.decoder.training)
            self.assertFalse(self.encoder.training)
            torch.testing.assert_close(source_padding_mask, torch.tensor([False, False, True]))
            prefixes.append(ids.tolist())
            memories.append(memory)
            logits = torch.full((ids.numel(), 7), -10.)
            # 非末位置故意预测另一个词，验证只读末位置。
            logits[:, 6] = 10.
            logits[-1, [3, 4, 2][len(prefixes) - 1]] = 20.
            return logits, []
        with patch.object(self.encoder, 'forward', wraps=self.encoder.forward) as enc:
            with patch.object(self.decoder, 'forward', side_effect=scripted):
                result = self.generate(max_new_tokens=10)
        self.assertEqual(result.tolist(), [1, 3, 4, 2])
        self.assertEqual(prefixes, [[1], [1, 3], [1, 3, 4]])
        self.assertEqual(enc.call_count, 1)
        self.assertTrue(all(m is memories[0] for m in memories))

    def force_head(self, token):
        with torch.no_grad():
            self.decoder.lm_head.weight.zero_()
            self.decoder.lm_head.bias.zero_()
            self.decoder.lm_head.bias[token] = 10.

    def test_real_model_immediate_eos_and_length_limit(self):
        self.force_head(2)
        self.assertEqual(self.generate().tolist(), [1, 2])
        self.force_head(3)
        self.assertEqual(self.generate(max_new_tokens=3).tolist(), [1, 3, 3, 3])
        with patch.object(self.encoder, 'forward') as enc:
            self.assertEqual(self.generate(max_new_tokens=0).tolist(), [1])
            enc.assert_not_called()

    def test_bos_and_pad_are_not_generated(self):
        self.force_head(3)
        with torch.no_grad():
            self.decoder.lm_head.bias[0] = 30.
            self.decoder.lm_head.bias[1] = 20.
        self.assertEqual(self.generate(max_new_tokens=2).tolist(), [1, 3, 3])

    def test_determinism_no_updates_and_exact_mode_restoration(self):
        self.encoder.encoder.layers[0].eval()  # 混合状态也应恢复。
        modules = list(self.encoder.modules()) + list(self.decoder.modules())
        modes = [m.training for m in modules]
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        before = [p.detach().clone() for p in parameters]
        first = self.generate(max_new_tokens=4)
        second = self.generate(max_new_tokens=4)
        torch.testing.assert_close(first, second)
        self.assertEqual([m.training for m in modules], modes)
        for old, p in zip(before, parameters):
            torch.testing.assert_close(old, p)
            self.assertIsNone(p.grad)
        with patch.object(self.decoder, 'forward', side_effect=RuntimeError('test failure')):
            with self.assertRaises(RuntimeError):
                self.generate()
        self.assertEqual([m.training for m in modules], modes)

    def test_invalid_arguments(self):
        for limit in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                self.generate(max_new_tokens=limit)
        for bos, eos in ((1, 1), (0, 2), (1, 0), (-1, 2), (1, 7), (True, 2)):
            with self.assertRaises(ValueError):
                greedy_generate(self.encoder, self.decoder, self.source, bos, eos)
        for source in (torch.tensor([]), torch.tensor([[1, 2]]), torch.tensor([0, 0]),
                       torch.tensor([-1]), torch.tensor([8])):
            with self.assertRaises(ValueError):
                greedy_generate(self.encoder, self.decoder, source, 1, 2)
        with self.assertRaises(TypeError):
            greedy_generate(self.encoder, self.decoder, torch.tensor([1.]), 1, 2)


if __name__ == '__main__':
    unittest.main()

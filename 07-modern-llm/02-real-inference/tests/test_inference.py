"""离线小随机 Qwen 的算法验证；不用于展示语言能力。"""
import sys
import unittest
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_utils import manual_generate, check_cache, choose_token


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(9)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8,
                             bos_token_id=1, eos_token_id=31, pad_token_id=0)
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.ids = torch.tensor([[1, 4, 6, 2]])

    def test_cache_logits_and_lengths(self):
        self.assertEqual(len(check_cache(self.model, self.ids, steps=4)), 4)

    def test_manual_matches_generate_and_input_work(self):
        with torch.inference_mode():
            expected = self.model.generate(self.ids, attention_mask=torch.ones_like(self.ids),
                                           max_new_tokens=5, do_sample=False, eos_token_id=None)
        plain = manual_generate(self.model, self.ids, max_new_tokens=5, use_cache=False, stop_at_eos=False)
        cached = manual_generate(self.model, self.ids, max_new_tokens=5, use_cache=True, stop_at_eos=False)
        self.assertTrue(torch.equal(expected, cached['ids']))
        self.assertTrue(torch.equal(plain['ids'], cached['ids']))
        self.assertEqual([r['input_tokens'] for r in plain['trace']], [4,5,6,7,8])
        self.assertEqual([r['input_tokens'] for r in cached['trace']], [4,1,1,1,1])
        self.assertEqual([r['cache_tokens'] for r in cached['trace']], [4,5,6,7,8])
        self.assertEqual(cached['stop_reason'], 'max_new_tokens')

    def test_eos_stops_after_first_predicted_token(self):
        old = self.model.generation_config.eos_token_id
        first = manual_generate(self.model, self.ids, max_new_tokens=1)['trace'][0]['token_id']
        try:
            self.model.generation_config.eos_token_id = [first]
            result = manual_generate(self.model, self.ids, max_new_tokens=10)
            self.assertEqual(result['stop_reason'], 'eos')
            self.assertEqual(result['new_ids'].shape[1], 1)
        finally:
            self.model.generation_config.eos_token_id = old

    def test_sampling_reproducibility_and_top_k(self):
        logits = torch.tensor([[0., 1., 2., 3.]])
        def draws(seed):
            generator = torch.Generator().manual_seed(seed)
            return [choose_token(logits, sample=True, top_k=2, generator=generator).item() for _ in range(20)]
        self.assertEqual(draws(42), draws(42))
        self.assertTrue(set(draws(42)).issubset({2,3}))
        self.assertEqual(choose_token(logits, sample=True, top_k=1).item(), 3)
        with self.assertRaises(ValueError):
            choose_token(logits, sample=True, temperature=0)

    def test_reject_unsupported_batch(self):
        with self.assertRaises(ValueError):
            manual_generate(self.model, self.ids.repeat(2,1))


if __name__ == '__main__':
    unittest.main()

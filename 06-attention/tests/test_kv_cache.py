"""缓存与完整重算等价、位置偏移、PAD、窗口重建与独立请求。"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_lm import CausalLM


class KVCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def model(self, legacy=False, context=16):
        options = dict(norm_style='post', norm_type='layer', position_encoding='sinusoidal', ffn_type='relu') if legacy else {}
        return CausalLM(20, d_model=16, num_layers=2, context_length=context, **options).eval()

    @torch.no_grad()
    def test_logits_match_prefill_single_and_chunk_decode(self):
        ids = torch.tensor([1, 4, 5, 6, 7, 8])
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                model = self.model(legacy)
                full = model(ids)
                prefill, cache = model(ids[:3], use_cache=True)
                torch.testing.assert_close(prefill, full[:3])
                old = [[k.clone() for k in c['k']] for c in cache]
                next_logits, extended = model(ids[3:4], use_cache=True, kv_cache=cache)
                torch.testing.assert_close(next_logits, full[3:4])
                chunk, final = model(ids[4:], use_cache=True, kv_cache=extended)
                torch.testing.assert_close(chunk, full[4:])
                for layer, old_keys in zip(final, old):
                    self.assertEqual(layer['padding_mask'].shape, (6,))
                    for k, previous in zip(layer['k'], old_keys):
                        torch.testing.assert_close(k[..., :3], previous)
                        self.assertFalse(k.requires_grad)
                self.assertEqual(cache[0]['padding_mask'].shape, (3,))  # 不原地改写调用方缓存。

    @torch.no_grad()
    def test_batch_padding_matches_full_and_pad_append(self):
        model = self.model()
        ids = torch.tensor([[0, 1, 4, 5, 6], [1, 7, 0, 0, 0]])
        full = model(ids)
        first, cache = model(ids[:, :3], use_cache=True)
        last, cache = model(ids[:, 3:], use_cache=True, kv_cache=cache)
        torch.testing.assert_close(torch.cat((first, last), dim=1), full)
        for layer in cache:
            torch.testing.assert_close(layer['padding_mask'], ids == 0)

    def test_generation_equivalence_and_window_rebuild(self):
        for legacy in (False, True):
            model = self.model(legacy, context=4)
            # 强制持续输出普通 token，以覆盖滑动窗口，不被 EOS 提前截断。
            with torch.no_grad():
                model.lm_head.weight.zero_()
                model.lm_head.bias.zero_()
                model.lm_head.bias[3] = 10
            lengths = []
            handle = model.embedding.register_forward_pre_hook(lambda m, args: lengths.append(args[0].shape[-1]))
            try:
                cached = model.generate(torch.tensor([1, 4, 5]), max_new_tokens=5)
            finally:
                handle.remove()
            self.assertEqual(lengths, [3, 1, 4, 4, 4])
            uncached = model.generate(torch.tensor([1, 4, 5]), max_new_tokens=5, use_cache=False)
            torch.testing.assert_close(cached, uncached)

    def test_random_model_generations_and_state_restoration(self):
        for legacy in (False, True):
            model = self.model(legacy, context=8).train()
            model.layers[0].norm1.eval()
            modes = [m.training for m in model.modules()]
            before = {n: p.clone() for n, p in model.state_dict().items()}
            for ids in (torch.tensor([1, 3]), torch.arange(1, 12)):
                a = model.generate(ids, max_new_tokens=12, use_cache=True)
                b = model.generate(ids, max_new_tokens=12, use_cache=False)
                torch.testing.assert_close(a, b)
            self.assertEqual([m.training for m in model.modules()], modes)
            self.assertEqual(model.generate(torch.tensor([1]), max_new_tokens=0).tolist(), [1])
            for n, p in model.state_dict().items():
                torch.testing.assert_close(p, before[n])
            self.assertFalse(any('cache' in name for name in model.state_dict()))

    def test_eos_and_single_token_work_per_step(self):
        model = self.model()
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias.zero_()
            model.lm_head.bias[3] = 10
        lengths = []
        handle = model.embedding.register_forward_pre_hook(lambda m, args: lengths.append(args[0].numel()))
        try:
            model.generate(torch.tensor([1, 4, 5]), max_new_tokens=4)
        finally:
            handle.remove()
        self.assertEqual(lengths, [3, 1, 1, 1])
        with torch.no_grad():
            model.lm_head.bias[2] = 20
        self.assertEqual(model.generate(torch.tensor([1]), max_new_tokens=4).tolist(), [1, 2])

    def test_cache_guards_and_normal_training(self):
        model = self.model(context=4)
        ids = torch.tensor([1, 3, 4])
        with self.assertRaises(ValueError):
            model(ids, use_cache=True)  # eval 但梯度开启。
        with torch.no_grad():
            _, cache = model(ids, use_cache=True)
            with self.assertRaises(ValueError):
                model(torch.tensor([5, 6]), use_cache=True, kv_cache=cache)
            with self.assertRaises(ValueError):
                model(ids, kv_cache=cache)
            with self.assertRaises(ValueError):
                model(torch.tensor([5]), use_cache=True, kv_cache=cache[:1])
            model.train()
            with self.assertRaises(ValueError):
                model(ids, use_cache=True)
        model(ids).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()

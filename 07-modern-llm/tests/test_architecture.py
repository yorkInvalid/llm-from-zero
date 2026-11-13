from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_architecture import load_config, describe
from gqa_demo import TinyGQA


class ArchitectureTests(unittest.TestCase):
    def test_real_config_dimensions_and_storage(self):
        config = load_config()
        result = describe(config)
        self.assertEqual(result['shapes']['q_projection'], [1, 1024, 2048])
        self.assertEqual(result['shapes']['k_heads'], [1, 8, 1024, 128])
        self.assertEqual(result['storage']['kv_cache_mib'], 112)
        self.assertEqual(result['storage']['kv_ratio_to_mha'], .5)
        self.assertEqual(describe(config, tokens=2048)['storage']['kv_cache_mib'], 224)
        untied = describe({**config, 'tie_word_embeddings': False})
        self.assertEqual(untied['parameters']['total'] - result['parameters']['total'],
                         config['vocab_size'] * config['hidden_size'])

    def test_parameter_formula_matches_small_module(self):
        # 根据实际组件注册数核对公式，避免只验证硬编码总数。
        config = dict(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, num_hidden_layers=2, intermediate_size=32,
                      vocab_size=30, tie_word_embeddings=True)
        embedding = torch.nn.Embedding(30, 16)
        attention = TinyGQA()
        mlp = torch.nn.ModuleList([torch.nn.Linear(16,32,bias=False),
                                  torch.nn.Linear(16,32,bias=False), torch.nn.Linear(32,16,bias=False)])
        per_layer = sum(p.numel() for p in attention.parameters()) + sum(p.numel() for p in mlp.parameters())
        per_layer += 2 * 16 + 2 * 8
        expected = embedding.weight.numel() + 2 * per_layer + 16
        self.assertEqual(describe(config)['parameters']['total'], expected)

    def test_gqa_equals_mha_with_repeated_kv_projections(self):
        torch.manual_seed(42)
        gqa, mha = TinyGQA(), TinyGQA(kv_heads=4)
        with torch.no_grad():
            mha.q_proj.weight.copy_(gqa.q_proj.weight)
            mha.o_proj.weight.copy_(gqa.o_proj.weight)
            for target, source in ((mha.k_proj,gqa.k_proj),(mha.v_proj,gqa.v_proj)):
                target.weight.copy_(source.weight.reshape(2,8,16).repeat_interleave(2,0).reshape(32,16))
        x = torch.randn(2, 5, 16)
        a, aw = gqa(x)
        b, bw = mha(x)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(aw['weights'],bw['weights'])
        self.assertLess(sum(p.numel() for p in gqa.parameters()), sum(p.numel() for p in mha.parameters()))

    def test_causality_and_gradients(self):
        torch.manual_seed(42)
        model = TinyGQA()
        x = torch.randn(1,5,16,requires_grad=True)
        output,_ = model(x)
        changed = x.detach().clone()
        changed[:,3:] += 10
        torch.testing.assert_close(output[:,:3],model(changed)[0][:,:3])
        output[:,:3].square().sum().backward()
        torch.testing.assert_close(x.grad[:,3:],torch.zeros_like(x.grad[:,3:]))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            TinyGQA(query_heads=3,kv_heads=2)
        with self.assertRaises(ValueError):
            describe(load_config(),tokens=0)


if __name__ == '__main__':
    unittest.main()

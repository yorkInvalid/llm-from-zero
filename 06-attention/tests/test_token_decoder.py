"""目标查表、位置编码、词表投影、PAD/因果隔离和交叉熵更新。"""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from position_encoding import sinusoidal_encoding
from token_decoder import TokenDecoder, TokenDecoderLM
from token_encoder import TokenEncoder


class TokenDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_embedding_and_position_at_decoder_boundary(self):
        ids, memory = torch.tensor([2, 1, 2]), torch.randn(10, 4)
        for mode in ('none', 'sinusoidal', 'rope'):
            with self.subTest(mode=mode):
                model = TokenDecoder(6, num_layers=2, position_encoding=mode)
                with torch.no_grad():
                    model.embedding.weight.copy_(torch.arange(60.).reshape(6, 10) / 60)
                captured = []
                handle = model.decoder.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach().clone()))
                try:
                    hidden, weights = model(ids, memory)
                finally:
                    handle.remove()
                expected = model.embedding.weight[ids].T
                if mode == 'sinusoidal':
                    expected = expected + sinusoidal_encoding(expected)
                torch.testing.assert_close(captured[0], expected)
                self.assertEqual(hidden.shape, (10, 3))
                self.assertEqual(len(weights), 2)
                for layer in model.decoder.layers:
                    self.assertEqual(layer.self_attention.use_rope, mode == 'rope')
                    self.assertFalse(layer.cross_attention.use_rope)

    def test_head_is_per_position_linear_without_softmax(self):
        model = TokenDecoderLM(7, num_layers=1).eval()
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.weight[:, :7] = torch.eye(7)
            model.lm_head.bias.fill_(-20)
        for ids, memory in ((torch.tensor([1, 2, 3]), torch.randn(10, 5)),
                            (torch.tensor([[1, 2], [2, 3]]), torch.randn(2, 10, 4))):
            hidden, _ = model.token_decoder(ids, memory)
            logits, _ = model(ids, memory)
            self.assertEqual(logits.shape, (*ids.shape, 7))
            torch.testing.assert_close(logits, hidden[..., :7, :].transpose(-2, -1) - 20)
            self.assertTrue((logits < 0).all())
        self.assertNotEqual(model.lm_head.weight.data_ptr(),
                            model.token_decoder.embedding.weight.data_ptr())

    def test_padding_matches_unpadded_sequences_and_ignores_source_pad(self):
        encoder = TokenEncoder(9, num_layers=1, padding_idx=0).eval()
        model = TokenDecoderLM(7, num_layers=2, padding_idx=0, dropout=.2).eval()
        source = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0]])
        target = torch.tensor([[1, 4, 5], [1, 3, 0]])
        source_mask = source == 0
        memory, _ = encoder(source)
        logits, weights = model(target, memory, source_mask)
        for row, s, t in ((0, 3, 3), (1, 2, 2)):
            single_memory, _ = encoder(source[row, :s])
            single_logits, _ = model(target[row, :t], single_memory)
            torch.testing.assert_close(logits[row, :t], single_logits)
            for layer in weights:
                torch.testing.assert_close(layer['self'][row, :, t:],
                                           torch.zeros_like(layer['self'][row, :, t:]))
                torch.testing.assert_close(layer['cross'][row, :, s:],
                                           torch.zeros_like(layer['cross'][row, :, s:]))
        changed, _ = model(target, memory.masked_fill(source_mask[:, None], 999), source_mask)
        torch.testing.assert_close(logits, changed)

    def test_future_token_ids_cannot_change_prefix_logits(self):
        for mode in ('none', 'sinusoidal', 'rope'):
            model = TokenDecoderLM(8, num_layers=2, position_encoding=mode).eval()
            memory = torch.randn(10, 3)
            first, _ = model(torch.tensor([1, 2, 3, 4]), memory)
            second, _ = model(torch.tensor([1, 2, 6, 7]), memory)
            torch.testing.assert_close(first[:2], second[:2])

    def test_cross_entropy_updates_embeddings_head_and_both_stacks(self):
        encoder = TokenEncoder(9, num_layers=2, padding_idx=0)
        model = TokenDecoderLM(7, num_layers=2, padding_idx=0)
        parameters = list(encoder.parameters()) + list(model.parameters())
        before = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.Adam(parameters, lr=.01)
        source = torch.tensor([[2, 3, 0], [3, 4, 5]])
        # BOS=1，EOS=2，PAD=0；输入与标签错开一位。
        target = torch.tensor([[1, 3, 4], [1, 5, 0]])
        labels = torch.tensor([[3, 4, 2], [5, 2, 0]])
        memory, _ = encoder(source)
        logits, _ = model(target, memory, source == 0)
        logits.retain_grad()
        loss = nn.CrossEntropyLoss(ignore_index=0)(logits.reshape(-1, 7), labels.reshape(-1))
        loss.backward()
        torch.testing.assert_close(logits.grad[1, 2], torch.zeros(7))
        for old, p in zip(before, parameters):
            torch.testing.assert_close(old, p)  # backward 只算梯度。
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        for embedding in (encoder.embedding, model.token_decoder.embedding):
            torch.testing.assert_close(embedding.weight.grad[0], torch.zeros(10))
        optimizer.step()
        for old, p in zip(before, parameters):
            self.assertFalse(torch.equal(old, p))
        torch.testing.assert_close(model.token_decoder.embedding.weight[0], torch.zeros(10))
        torch.testing.assert_close(encoder.embedding.weight[0], torch.zeros(10))
        # 未查表的行不更新；输出头与查表不共享权重。
        torch.testing.assert_close(model.token_decoder.embedding.weight[6],
                                   before[len(list(encoder.parameters()))][6])

    def test_input_dropout_and_eval(self):
        model = TokenDecoder(7, num_layers=1, dropout=1.)
        captured = []
        handle = model.decoder.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        ids, memory = torch.tensor([1, 2]), torch.randn(10, 3)
        try:
            model(ids, memory)
            model.eval()
            model(ids, memory)
        finally:
            handle.remove()
        torch.testing.assert_close(captured[0], torch.zeros_like(captured[0]))
        X = model.embedding(ids).T
        torch.testing.assert_close(captured[1], X + sinusoidal_encoding(X))

    def test_invalid_inputs(self):
        model = TokenDecoder(7, padding_idx=0)
        memory = torch.randn(10, 3)
        for ids in (torch.tensor([]), torch.tensor([[[1]]]), torch.tensor([-1]),
                    torch.tensor([7]), torch.tensor([0, 0])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                model(ids, memory)
        with self.assertRaises(TypeError):
            model(torch.tensor([1.]), memory)
        with self.assertRaises(ValueError):
            TokenDecoder(7, position_encoding='unknown')
        with self.assertRaises(ValueError):
            model(torch.tensor([1]), None)


if __name__ == '__main__':
    unittest.main()

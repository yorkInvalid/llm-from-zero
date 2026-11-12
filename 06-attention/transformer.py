"""统一 Encoder–Decoder 接口，内部复用已有教学组件。"""

from torch import nn

from generation import greedy_generate
from token_encoder import TokenEncoder
from token_decoder import TokenDecoderLM


class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model=32,
                 num_encoder_layers=2, num_decoder_layers=2, num_heads=2,
                 d_k=16, d_v=16, d_ff=128, position_encoding='sinusoidal',
                 source_padding_idx=None, target_padding_idx=None, dropout=0.0):
        super().__init__()
        self.config = dict(source_vocab_size=source_vocab_size, target_vocab_size=target_vocab_size,
                           d_model=d_model, num_encoder_layers=num_encoder_layers,
                           num_decoder_layers=num_decoder_layers, num_heads=num_heads,
                           d_k=d_k, d_v=d_v, d_ff=d_ff, position_encoding=position_encoding,
                           source_padding_idx=source_padding_idx,
                           target_padding_idx=target_padding_idx, dropout=dropout)
        common = dict(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v,
                      d_ff=d_ff, position_encoding=position_encoding, dropout=dropout)
        self.encoder = TokenEncoder(source_vocab_size, num_layers=num_encoder_layers,
                                    padding_idx=source_padding_idx, **common)
        self.decoder = TokenDecoderLM(target_vocab_size, num_layers=num_decoder_layers,
                                      padding_idx=target_padding_idx, **common)

    def forward(self, source_ids, target_input_ids, return_attention=False):
        """返回 (...,T,target_vocab_size) logits；目标右移和 loss 由训练代码处理。

        return_attention=True 时返回 (logits, {'encoder': ..., 'decoder': ...})。
        两侧 ID 均支持单序列或 batch；PAD mask 分别从各自 ID 生成。
        """
        memory, encoder_weights = self.encoder(source_ids)
        pad = self.encoder.embedding.padding_idx
        source_mask = source_ids == pad if pad is not None else None
        logits, decoder_weights = self.decoder(target_input_ids, memory, source_mask)
        if return_attention:
            return logits, {'encoder': encoder_weights, 'decoder': decoder_weights}
        return logits

    def generate(self, source_ids, bos_id, eos_id, max_new_tokens=50):
        """单序列贪心生成，返回包含 BOS 的 ID；复用现有状态恢复逻辑。"""
        return greedy_generate(self.encoder, self.decoder, source_ids,
                               bos_id, eos_id, max_new_tokens)

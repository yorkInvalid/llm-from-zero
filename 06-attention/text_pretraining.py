"""文本清洗、先划分后分词、段落窗口和标签错位。"""

import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

URL = 'https://www.gutenberg.org/cache/epub/11/pg11.txt'
PAD, BOS, EOS = 0, 1, 2


def split_paragraphs(text):
    if '*** START OF' in text:
        text = text.split('*** START OF', 1)[1].split('***', 1)[1]
        text = text.split('*** END OF', 1)[0]
    paragraphs = []
    seen = set()
    for part in re.split(r'\n\s*\n', text):
        part = ' '.join(part.split())
        if part.startswith('CHAPTER I.') and 'CHAPTER II.' in part:
            continue  # Gutenberg 示例书籍的目录块，不当作正文。
        if len(part) >= 40 and part not in seen:
            seen.add(part)
            paragraphs.append(part)
    if len(paragraphs) < 20:
        raise ValueError('至少需要 20 个去重后、长度不低于 40 字符的段落')
    # 保留书籍顺序，尾部为留出集；不把重叠窗口随机分到不同集合。
    a, b = int(len(paragraphs) * .8), int(len(paragraphs) * .9)
    return dict(train=paragraphs[:a], validation=paragraphs[a:b], test=paragraphs[b:])


def prepare(directory, text_file=None, vocab_size=512):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if vocab_size < 259:
        raise ValueError('ByteLevel BPE 词表至少为 259（256 字节 + 3 特殊 token）')
    if text_file:
        raw = Path(text_file).read_bytes()
        source = str(Path(text_file).resolve())
    else:
        with urlopen(URL, timeout=30) as response:
            raw = response.read()
        source = URL
    text = raw.decode('utf-8-sig')
    splits = split_paragraphs(text)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(splits['train'], trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=['<PAD>', '<BOS>', '<EOS>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    (directory / 'original.txt').write_text(text, encoding='utf-8')
    tokenizer.save(str(directory / 'tokenizer.json'))
    for name, paragraphs in splits.items():
        (directory / f'{name}.json').write_text(json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = dict(source=source, sha256=hashlib.sha256(raw).hexdigest(),
                    bytes=len(raw), paragraphs={k: len(v) for k, v in splits.items()},
                    vocab_size=tokenizer.get_vocab_size(),
                    split_rule='unique paragraphs, original order, 80/10/10; tokenizer trained on train only',
                    eos_meaning='end of paragraph document, not arbitrary window boundary')
    (directory / 'source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


def make_windows(paragraphs, tokenizer, context_length):
    if context_length < 1:
        raise ValueError('context_length 须为正')
    samples = []
    for paragraph in paragraphs:
        ids = [BOS, *tokenizer.encode(paragraph).ids, EOS]
        for start in range(0, len(ids) - 1, context_length):
            piece = ids[start:start + context_length + 1]
            samples.append((torch.tensor(piece[:-1]), torch.tensor(piece[1:])))
    return samples


def collate(samples):
    inputs, labels = zip(*samples)
    return (pad_sequence(inputs, batch_first=True, padding_value=PAD),
            pad_sequence(labels, batch_first=True, padding_value=-100))


def load_data(directory, context_length, names=('train', 'validation')):
    directory = Path(directory)
    tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    splits = {name: make_windows(json.loads((directory / f'{name}.json').read_text()), tokenizer,
                                context_length) for name in names}
    return tokenizer, splits

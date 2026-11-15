"""Bounded Chinese IndustryCorpus2 pilot (requires HF access and pyarrow).

Read the first row group of one pinned high-quality AI shard, shuffle unique
whole documents, and split before training BPE. This is not a corpus-wide sample.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
from text_pretraining import prepare

REVISION = '249ea52f1ee412303e9741ae69fdd2109c0f8e1c'
SHARD = 'artificial_intelligence_machine_learning/chinese/high/rank_00012.parquet'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'artifacts/book-lm/industry-data')
    parser.add_argument('--characters', type=int, default=1000000)
    args = parser.parse_args()
    if args.characters < 10000:
        parser.error('characters must be at least 10000')
    directory = args.data_dir
    if directory.exists():
        parser.error('use a new data directory to preserve existing tokenizer/checkpoints')
    fs = HfFileSystem()
    source = f'datasets/BAAI/IndustryCorpus2@{REVISION}/{SHARD}'
    with fs.open(source, 'rb') as stream:
        parquet = pq.ParquetFile(stream)
        rows = parquet.read_row_group(0, columns=['text']).to_pylist()
    documents = sorted({' '.join(r['text'].split()) for r in rows if isinstance(r.get('text'), str)})
    documents = [d for d in documents if 200 <= len(d) <= 20000]
    random.Random(42).shuffle(documents)
    chosen, size = [], 0
    for document in documents:
        chosen.append(document)
        size += len(document)
        if size >= args.characters:
            break
    if size < args.characters:
        raise ValueError(f'First row group only provides {size} characters; reduce --characters')
    directory.mkdir(parents=True)
    raw_path = directory / 'sample.txt'
    raw_path.write_text('\n\n'.join(chosen), encoding='utf-8')
    metadata = prepare(directory, raw_path, vocab_size=8192)
    metadata.update(dataset='BAAI/IndustryCorpus2', revision=REVISION, shard=SHARD,
                    row_group=0, seed=42, characters=size,
                    sampling='unique normalized documents, 200..20000 chars; shuffled first row group; bounded prefix',
                    document_hashes=[hashlib.sha256(d.encode()).hexdigest() for d in chosen])
    (directory / 'source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in metadata.items() if k != 'document_hashes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

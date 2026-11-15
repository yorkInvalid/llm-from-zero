"""Build a bounded fresh-token experiment, retaining the existing tokenizer/holdouts."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

from tokenizers import Tokenizer
from expand_industry import SHARDS, REVISION, canonical, clean
from indexed_tokens import TokenWindowWriter, split_fingerprint


def document_hash(text):
    return hashlib.sha256(canonical(text).encode()).hexdigest()


class HoldoutGuard:
    """Conservative anchors: any shared 100-character canonical span is rejected."""
    def __init__(self, documents):
        self.exact = set()
        self.anchors = set()
        for text in documents:
            c = canonical(text)
            self.exact.add(hashlib.sha256(c.encode()).hexdigest())
            self.anchors.update(c[i:i + 50] for i in range(max(0, len(c) - 49)))

    def overlaps(self, text):
        c = canonical(text)
        return (hashlib.sha256(c.encode()).hexdigest() in self.exact
                or any(c[i:i + 50] in self.anchors for i in range(0, len(c) - 49, 50)))


def main():
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-tokens', type=int, default=100_000_000)
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--exclude-data', type=Path, action='append', default=[],
                        help='Also exclude exact document hashes from previous indexed corpora')
    args = parser.parse_args()
    if args.output.exists() or args.target_tokens < 4:
        parser.error('use a new output directory and at least four target tokens')
    args.output.mkdir(parents=True)
    previous = args.previous_data
    tokenizer = Tokenizer.from_file(str(previous / 'tokenizer.json'))
    old_train = json.loads((previous / 'train.json').read_text())
    holdouts = []
    for name in ('validation', 'validation_legacy', 'test'):
        holdouts.extend(json.loads((previous / f'{name}.json').read_text()))
        shutil.copyfile(previous / f'{name}.json', args.output / f'{name}.json')
    shutil.copyfile(previous / 'tokenizer.json', args.output / 'tokenizer.json')
    guard = HoldoutGuard(holdouts)
    seen = {document_hash(t) for t in old_train} | guard.exact
    excluded_corpora = []
    for directory in args.exclude_data:
        count_before = len(seen)
        with gzip.open(directory / 'training_documents.jsonl.gz', 'rt', encoding='utf-8') as stream:
            for line in stream:
                seen.add(json.loads(line)['canonical_sha256'])
        excluded_corpora.append(dict(directory=str(directory), added_hashes=len(seen)-count_before,
                                     train_sha256=split_fingerprint(directory, 'train')))
    del old_train, holdouts
    writer = TokenWindowWriter(args.output)
    fs = HfFileSystem()
    domains = list(SHARDS)
    stats, sources = {}, []
    started = time.monotonic()

    def progress():
        value = dict(status='building', target_tokens=args.target_tokens,
                     completed_tokens=writer.targets, documents=writer.documents,
                     seconds=time.monotonic() - started, domains=stats)
        temp = args.output / 'build-progress.tmp'
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        temp.replace(args.output / 'build-progress.json')

    with gzip.open(args.output / 'training_documents.jsonl.gz', 'wt', encoding='utf-8') as raw:
        for domain_number, domain in enumerate(domains):
            quota = args.target_tokens // len(domains) + (domain_number < args.target_tokens % len(domains))
            stat = Counter(target_tokens=0, documents=0, characters=0, truncated_documents=0)
            stats[domain] = stat
            directory = SHARDS[domain].rsplit('/', 1)[0]
            prefix = f'datasets/BAAI/IndustryCorpus2@{REVISION}/'
            files = [p for p in fs.ls(prefix + directory, detail=False) if p.endswith('.parquet')]
            rng = random.Random(args.seed + domain_number)
            rng.shuffle(files)
            done = False
            for filename in files:
                shard = filename.split(f'{REVISION}/', 1)[-1]
                with fs.open(filename, block_size=8 * 1024 * 1024) as stream:
                    parquet = pq.ParquetFile(stream)
                    groups = list(range(parquet.num_row_groups))
                    rng.shuffle(groups)
                    for group in groups:
                        print(f'{domain}: {shard} row_group={group}, tokens={stat["target_tokens"]:,}/{quota:,}', flush=True)
                        source = dict(domain=domain, shard=shard, row_group=group, accepted_documents=0, target_tokens=0)
                        sources.append(source)
                        row_number = 0
                        for batch in parquet.iter_batches(batch_size=128, row_groups=[group], columns=['text']):
                            accepted = []
                            for row in batch.to_pylist():
                                row_number += 1
                                text = clean(row.get('text'))
                                if text is None:
                                    stat['rejected_quality'] += 1
                                    continue
                                digest = document_hash(text)
                                if digest in seen:
                                    stat['rejected_exact_duplicate'] += 1
                                    continue
                                if guard.overlaps(text):
                                    stat['rejected_holdout_anchor'] += 1
                                    continue
                                seen.add(digest)
                                accepted.append((text, digest, row_number - 1))
                            encodings = tokenizer.encode_batch([r[0] for r in accepted])
                            for (text, digest, row_id), encoding in zip(accepted, encodings):
                                ids = [1, *encoding.ids, 2]
                                used = writer.add_document(ids, max_targets=quota - stat['target_tokens'])
                                stat.update(target_tokens=used, documents=1, characters=len(text),
                                            truncated_documents=int(used < len(ids) - 1))
                                source['accepted_documents'] += 1
                                source['target_tokens'] += used
                                raw.write(json.dumps(dict(domain=domain, shard=shard, row_group=group,
                                                          row=row_id, canonical_sha256=digest,
                                                          used_targets=used, text=text), ensure_ascii=False) + '\n')
                                if stat['target_tokens'] == quota:
                                    done = True
                                    break
                            progress()
                            if done:
                                break
                        if done:
                            break
                if done:
                    break
            if not done:
                raise RuntimeError(f'insufficient filtered {domain} data: {stat["target_tokens"]}/{quota}')
    index = writer.finish()
    assert writer.targets == args.target_tokens
    split_hashes = {name: split_fingerprint(args.output, name)
                    for name in ('train', 'validation', 'validation_legacy', 'test')}
    metadata = dict(version=args.output.name, dataset='BAAI/IndustryCorpus2', revision=REVISION,
                    seed=args.seed, target_tokens=writer.targets, documents=writer.documents,
                    domains=stats, sources=sources, index=index, split_sha256=split_hashes,
                    tokenizer_sha256=hashlib.sha256((args.output / 'tokenizer.json').read_bytes()).hexdigest(),
                    tokenizer='frozen original 8192 BPE; no refit',
                    sampling='equal target-token quotas across 4 high-quality Chinese domains; seeded file/row-group order; not corpus-wide representative',
                    cleaning='v2 quality filters; canonical exact dedup including prior train; conservative heldout 50-character anchors catching any shared 100-character span; no semantic near-dedup guarantee',
                    budget='fresh accepted document prefixes; one pass totals exactly target_tokens; final document per domain may be truncated without synthetic EOS',
                    validation='v2 validation and legacy validation copied byte-for-byte',
                    test='copied and used only for exclusion filtering; never evaluated or used for selection',
                    previous_data_sha256=json.loads((previous / 'source.json').read_text())['split_sha256'],
                    excluded_corpora=excluded_corpora,
                    seconds=time.monotonic() - started)
    (args.output / 'source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    progress()
    done = json.loads((args.output / 'build-progress.json').read_text())
    done['status'] = 'completed'
    (args.output / 'build-progress.json').write_text(json.dumps(done, ensure_ascii=False, indent=2))
    print(json.dumps({k: metadata[k] for k in ('version', 'target_tokens', 'documents', 'domains', 'seconds')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

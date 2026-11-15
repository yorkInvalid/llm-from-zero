"""Select matched full-length windows for the fresh-versus-replay experiment."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from expand_industry import SHARDS
from indexed_tokens import IndexedTokenWindows, TokenWindowWriter, split_fingerprint


DOMAINS = tuple(SHARDS)
CONTEXT = 256
WINDOWS_PER_DOMAIN = 43_008


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(source, output, seed, arm):
    started = time.monotonic()
    source = Path(source)
    output = Path(output)
    if output.exists():
        raise ValueError('Output must be a new directory')
    parent_metadata = json.loads((source / 'source.json').read_text())
    original = IndexedTokenWindows(source, 'train', CONTEXT)
    documents = []
    position = 0
    with gzip.open(source / 'training_documents.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            row.pop('text')
            row['token_start'] = position
            documents.append(row)
            position += row['used_targets'] + 1
    if position != len(original.tokens):
        raise ValueError('Raw document positions do not match the source token file')
    starts = np.array([doc['token_start'] for doc in documents], dtype=np.int64)
    domain_ids = np.array([DOMAINS.index(doc['domain']) for doc in documents], dtype=np.uint8)
    document_ids = np.searchsorted(starts, original.index[:, 0], side='right') - 1
    offsets = original.index[:, 0] - starts[document_ids]
    lengths = np.array([doc['used_targets'] for doc in documents], dtype=np.int64)
    if (np.any(offsets % CONTEXT) or np.any(offsets + original.index[:, 1] > lengths[document_ids])):
        raise ValueError('A source window crosses its original document boundary')
    selected_groups = []
    domain_counts = {}
    for number, domain in enumerate(DOMAINS):
        eligible = np.flatnonzero((original.index[:, 1] == CONTEXT) & (domain_ids[document_ids] == number))
        if len(eligible) < WINDOWS_PER_DOMAIN:
            raise ValueError(f'Insufficient complete windows in {domain}: {len(eligible)}')
        selected = np.random.default_rng(seed + number).choice(eligible, WINDOWS_PER_DOMAIN, replace=False)
        selected.sort()
        selected_groups.append(selected)
        domain_counts[domain] = dict(eligible_full_windows=len(eligible), selected_windows=len(selected),
                                     target_tokens=len(selected) * CONTEXT)
    selected = np.concatenate(selected_groups)
    selected_docs = np.unique(document_ids[selected])
    output.mkdir(parents=True)
    for name in ('tokenizer', 'validation', 'validation_legacy', 'test'):
        shutil.copyfile(source / f'{name}.json', output / f'{name}.json')
    writer = TokenWindowWriter(output, context_length=CONTEXT)
    counts = Counter()
    for window_id in selected:
        start = int(original.index[window_id, 0])
        ids = original.tokens[start:start + CONTEXT + 1]
        writer.add_document(ids)
        counts['windows_starting_with_bos'] += int(ids[0] == 1)
        counts['windows_with_eos_target'] += int(np.any(ids[1:] == 2))
        counts['eos_targets'] += int(np.count_nonzero(ids[1:] == 2))
    manifest = writer.finish()
    # A stored record is one selected window. Multiple records can come from one article.
    manifest['documents'] = len(selected_docs)
    manifest['stored_window_records'] = len(selected)
    manifest['document_layout'] = 'independent selected original windows, each storing 257 token IDs'
    (output / 'train.index.json').write_text(json.dumps(manifest, indent=2))
    provenance = np.column_stack((selected, document_ids[selected], offsets[selected]))
    np.save(output / 'train.provenance.npy', provenance)
    with gzip.open(output / 'source_documents.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for document_id in selected_docs:
            stream.write(json.dumps(dict(source_document_id=int(document_id), **documents[document_id]), ensure_ascii=False) + '\n')
    hashes = {name: split_fingerprint(output, name) for name in ('train', 'validation', 'validation_legacy', 'test')}
    metadata = dict(version=output.name, arm=arm, parent_data_directory=str(source.resolve()),
                    parent_train_sha256=split_fingerprint(source, 'train'),
                    dataset=parent_metadata['dataset'], revision=parent_metadata['revision'],
                    seed=seed, context_length=CONTEXT, target_tokens=writer.targets,
                    windows=len(selected), documents=len(selected_docs), domains=domain_counts,
                    split_sha256=hashes, tokenizer_sha256=sha256(output / 'tokenizer.json'),
                    ordering='domain blocks in fixed order ai, education, science, news; source-window index ascending within domain; training applies identical seeded shuffle to both arms',
                    sampling=f'{WINDOWS_PER_DOMAIN} original 256-target windows sampled without replacement per domain; same length rule in both arms',
                    window_semantics='all 257 token IDs copied from an existing document-bounded source window; original x/y offset retained; no synthetic BOS, EOS or cross-document packing',
                    provenance_columns=['source_window_id', 'source_document_id', 'window_offset_within_document'],
                    provenance_sha256=sha256(output / 'train.provenance.npy'),
                    source_documents_sha256=sha256(output / 'source_documents.jsonl.gz'),
                    special_tokens=dict(counts),
                    source_total_windows=len(original), source_partial_windows_discarded=int(np.count_nonzero(original.index[:, 1] < CONTEXT)),
                    limitations='Full-window selection omits partial tails and articles shorter than 256 targets, changes document-length and EOS exposure relative to ordinary corpus training. Both arms use the same rule. Domain quotas and length are matched; source article content and finer distributions differ. Existing cleaning and exact-document dedup do not guarantee semantic near-dedup.',
                    parent_cleaning=parent_metadata['cleaning'],
                    parent_excluded_corpora=parent_metadata.get('excluded_corpora', []),
                    test='copied byte-for-byte; never evaluated or used to select configurations',
                    builder_sha256=sha256(__file__))
    (output / 'source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    actual = IndexedTokenWindows(output, 'train', CONTEXT)
    assert len(actual) == WINDOWS_PER_DOMAIN * len(DOMAINS)
    assert actual.target_tokens == WINDOWS_PER_DOMAIN * len(DOMAINS) * CONTEXT
    assert np.all(actual.index[:, 1] == CONTEXT)
    assert len(np.unique(selected)) == len(selected)
    for name in ('tokenizer', 'validation', 'validation_legacy', 'test'):
        assert (output / f'{name}.json').read_bytes() == (source / f'{name}.json').read_bytes()
    assert int(actual.tokens.min()) >= 0 and int(actual.tokens.max()) < 8192
    sample_ids = np.random.default_rng(912).choice(len(selected), 257, replace=False)
    for new_id in sample_ids:
        source_start = int(original.index[selected[new_id], 0])
        target_start = int(actual.index[new_id, 0])
        assert np.array_equal(original.tokens[source_start:source_start+257], actual.tokens[target_start:target_start+257])
    audit = dict(verified=True, target_tokens=actual.target_tokens, windows=len(actual), original_documents=len(selected_docs),
                 domain_counts=domain_counts, full_length_windows=True, unique_source_windows=True,
                 source_document_boundaries_verified=True, holdout_and_tokenizer_bytes_unchanged=True,
                 sampled_original_window_comparisons=len(sample_ids), token_id_bounds=[int(actual.tokens.min()),int(actual.tokens.max())],
                 special_tokens=dict(counts), split_sha256=hashes, seconds=time.monotonic()-started)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    shutil.copyfile(__file__, output / Path(__file__).name)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261119)
    parser.add_argument('--arm', choices=('replay', 'fresh'), required=True)
    args = parser.parse_args()
    build(args.source, args.output, args.seed, args.arm)

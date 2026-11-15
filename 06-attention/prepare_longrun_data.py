"""Merge audited document corpora and prepare bounded pretraining exposures."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from expand_industry import canonical
from indexed_tokens import IndexedTokenWindows, TokenWindowWriter, split_fingerprint


CONTEXT = 256
DOMAINS = ('ai', 'education', 'science', 'news')
BOOK_ROOT = Path(__file__).resolve().parent / 'artifacts/book-lm'
DEFAULT_PARENTS = (
    BOOK_ROOT / 'industry-expanded-v3-100m',
    BOOK_ROOT / 'industry-expanded-v4-200m',
    BOOK_ROOT / 'industry-data-comparison-v1/fresh-pool-v2',
)
FROZEN_FILES = ('tokenizer.json', 'validation.json', 'validation_legacy.json', 'test.json')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def gzip_writer(path):
    binary = gzip.GzipFile(filename=str(path), mode='wb', compresslevel=1, mtime=0)
    return io.TextIOWrapper(binary, encoding='utf-8')


def copy_frozen(source, output):
    for filename in FROZEN_FILES:
        shutil.copyfile(source / filename, output / filename)


def verify_frozen(source, reference):
    for filename in FROZEN_FILES:
        if (source / filename).read_bytes() != (reference / filename).read_bytes():
            raise ValueError(f'Frozen file differs: {source / filename}')


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def split_hashes(directory):
    return {name: split_fingerprint(directory, name)
            for name in ('train', 'validation', 'validation_legacy', 'test')}


def verify_document_windows(rows, documents):
    starts, targets = documents[:, 0], documents[:, 1]
    document_ids = np.searchsorted(starts, rows.index[:, 0], side='right') - 1
    offsets = rows.index[:, 0] - starts[document_ids]
    expected_lengths = np.minimum(CONTEXT, targets[document_ids] - offsets)
    expected_windows = int(((targets + CONTEXT - 1) // CONTEXT).sum())
    if (len(rows) != expected_windows or np.any(document_ids < 0)
            or np.any(offsets % CONTEXT) or np.any(expected_lengths != rows.index[:, 1])
            or np.any(rows.tokens[starts] != 1)):
        raise ValueError('A token window does not match its document boundary')
    counts = np.bincount(document_ids, minlength=len(documents))
    if not np.array_equal(counts, (targets + CONTEXT - 1) // CONTEXT):
        raise ValueError('A document has missing or repeated windows')


def merge_corpora(parents, output, reference):
    started = time.monotonic()
    output = Path(output)
    if output.exists():
        raise ValueError('Merged output must be a new directory')
    output.mkdir(parents=True)
    copy_frozen(reference, output)
    heldout_hashes = set()
    for name in ('validation', 'validation_legacy', 'test'):
        heldout_hashes.update(hashlib.sha256(canonical(text).encode()).hexdigest()
                              for text in read_json(reference / f'{name}.json'))
    seen = set()
    documents, indices, parent_records = [], [], []
    domains = {domain: Counter() for domain in DOMAINS}
    token_offset = 0
    with (output / 'train.tokens.u32').open('xb') as tokens_out, \
            gzip_writer(output / 'training_documents.jsonl.gz') as raw_out:
        for parent_number, parent in enumerate(map(Path, parents)):
            print(f'Checking and merging {parent.name}', flush=True)
            metadata = read_json(parent / 'source.json')
            verify_frozen(parent, reference)
            for name, expected in metadata['split_sha256'].items():
                if split_fingerprint(parent, name) != expected:
                    raise ValueError(f'Parent split hash differs: {parent}/{name}')
            rows = IndexedTokenWindows(parent, 'train', CONTEXT)
            parent_documents = []
            position = 0
            with gzip.open(parent / 'training_documents.jsonl.gz', 'rt', encoding='utf-8') as stream:
                for parent_document_id, line in enumerate(stream):
                    row = json.loads(line)
                    digest = hashlib.sha256(canonical(row['text']).encode()).hexdigest()
                    if digest != row['canonical_sha256'] or digest in seen or digest in heldout_hashes:
                        raise ValueError('Invalid, duplicate, or heldout training document')
                    seen.add(digest)
                    domain_number = DOMAINS.index(row['domain'])
                    used = row['used_targets']
                    parent_documents.append((position, used, domain_number, parent_number, parent_document_id))
                    documents.append((position + token_offset, used, domain_number, parent_number, parent_document_id))
                    row.update(source_parent=parent.name, source_parent_number=parent_number,
                               source_document_id=parent_document_id, merged_document_id=len(documents)-1)
                    raw_out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    domains[row['domain']].update(target_tokens=used, documents=1,
                                                  eos_targets=int(rows.tokens[position+used] == 2))
                    position += used + 1
            parent_documents = np.asarray(parent_documents, dtype=np.int64)
            if position != len(rows.tokens) or len(parent_documents) != rows.manifest['documents']:
                raise ValueError('Parent document metadata does not match its token file')
            verify_document_windows(rows, parent_documents)
            adjusted_index = np.array(rows.index, copy=True)
            adjusted_index[:, 0] += token_offset
            indices.append(adjusted_index)
            with (parent / rows.manifest['token_file']).open('rb') as stream:
                shutil.copyfileobj(stream, tokens_out, length=16*1024*1024)
            parent_records.append(dict(directory=str(parent.resolve()), name=parent.name,
                                       target_tokens=rows.target_tokens, windows=len(rows),
                                       documents=len(parent_documents), stored_tokens=len(rows.tokens),
                                       token_offset=token_offset,
                                       train_sha256=metadata['split_sha256']['train'],
                                       token_file_sha256=sha256(parent / rows.manifest['token_file']),
                                       raw_documents_sha256=sha256(parent / 'training_documents.jsonl.gz'),
                                       source_sha256=sha256(parent / 'source.json')))
            token_offset += len(rows.tokens)
    documents = np.asarray(documents, dtype=np.int64)
    index = np.concatenate(indices)
    np.save(output / 'train.windows.npy', index)
    np.save(output / 'train.documents.npy', documents)
    targets = int(index[:, 1].sum())
    manifest = dict(format='indexed-token-v1', context_length=CONTEXT, dtype='<u4',
                    stored_tokens=token_offset, target_tokens=targets, windows=len(index),
                    documents=len(documents), token_file='train.tokens.u32', index_file='train.windows.npy')
    write_json(output / 'train.index.json', manifest)
    actual = IndexedTokenWindows(output, 'train', CONTEXT)
    verify_document_windows(actual, documents)
    for record, parent in zip(parent_records, map(Path, parents)):
        source_rows = IndexedTokenWindows(parent, 'train', CONTEXT)
        offset = record['token_offset']
        for start in range(0, len(source_rows.tokens), 4_000_000):
            stop = min(start + 4_000_000, len(source_rows.tokens))
            if not np.array_equal(source_rows.tokens[start:stop], actual.tokens[offset+start:offset+stop]):
                raise ValueError('Merged token bytes differ from their parent')
    if targets != 349_520_000 or len(set(v['target_tokens'] for v in domains.values())) != 1:
        raise ValueError('Unexpected merged budget/domain proportions')
    hashes = split_hashes(output)
    source = dict(version=output.name, dataset='BAAI/IndustryCorpus2',
                  revision=read_json(Path(parents[0])/'source.json')['revision'],
                  context_length=CONTEXT, target_tokens=targets, documents=len(documents), windows=len(index),
                  domains={name: dict(value) for name, value in domains.items()}, parents=parent_records,
                  document_index_columns=['token_start', 'used_targets', 'domain_number', 'parent_number', 'parent_document_id'],
                  document_index_sha256=sha256(output/'train.documents.npy'),
                  raw_documents_sha256=sha256(output/'training_documents.jsonl.gz'),
                  tokenizer_sha256=sha256(output/'tokenizer.json'), split_sha256=hashes,
                  sampling='All accepted documents from the three audited parents; four equal target-token domain totals',
                  window_semantics='Original document-bounded windows including partial tails, with original source tokens copied exactly',
                  cleaning='Inherited parent cleaning, exact canonical dedup and heldout passage exclusion; canonical hashes and exact heldout exclusion rechecked during merge',
                  limitations='Source corpora have finite domain coverage. Exact dedup does not establish semantic near-dedup. Twelve parent quota-end documents retain their original truncation.',
                  test='Retained for exclusion auditing only', builder_sha256=sha256(__file__))
    write_json(output/'source.json', source)
    audit = dict(verified=True, target_tokens=targets, windows=len(index), documents=len(documents),
                 all_source_token_bytes_equal=True, all_window_document_boundaries_verified=True,
                 exact_duplicate_documents=0, exact_heldout_overlaps=0,
                 canonical_hashes_recomputed=len(documents),
                 partial_windows=int(np.count_nonzero(index[:,1] < CONTEXT)),
                 eos_targets=sum(value['eos_targets'] for value in domains.values()),
                 token_id_bounds=[int(actual.tokens.min()), int(actual.tokens.max())],
                 frozen_files_equal=True, split_sha256=hashes, seconds=time.monotonic()-started)
    if audit['token_id_bounds'][0] < 0 or audit['token_id_bounds'][1] >= 8192:
        raise ValueError('Token ID outside frozen vocabulary')
    write_json(output/'audit.json', audit)
    shutil.copyfile(__file__, output/Path(__file__).name)
    print(json.dumps(audit, indent=2), flush=True)
    return source


def select_document_exposures(documents, target_tokens, seed):
    """Return (source document, targets, domain, cycle) with exact domain quotas."""
    exposures = []
    for domain_number, domain in enumerate(DOMAINS):
        quota = target_tokens // len(DOMAINS) + (domain_number < target_tokens % len(DOMAINS))
        remaining = quota
        eligible = np.flatnonzero(documents[:, 2] == domain_number)
        cycle = 0
        while remaining:
            order = np.random.default_rng(np.random.SeedSequence([seed, domain_number, cycle])).permutation(eligible)
            for document_id in order:
                used = min(int(documents[document_id, 1]), remaining)
                exposures.append((int(document_id), used, domain_number, cycle))
                remaining -= used
                if not remaining:
                    break
            cycle += 1
    return np.asarray(exposures, dtype=np.int64)


def build_training_budget(base_dir, output_dir, target_tokens, seed=20261122, compact=False):
    """Prepare exact token exposures; compact=True writes a standalone pilot corpus."""
    started = time.monotonic()
    base, output = Path(base_dir), Path(output_dir)
    if output.exists() or target_tokens < 4:
        raise ValueError('Use a new output directory and at least four targets')
    parent = read_json(base/'source.json')
    if split_fingerprint(base, 'train') != parent['split_sha256']['train']:
        raise ValueError('Merged training data changed')
    if sha256(base/'train.documents.npy') != parent['document_index_sha256']:
        raise ValueError('Merged document index changed')
    rows = IndexedTokenWindows(base, 'train', CONTEXT)
    documents = np.load(base/'train.documents.npy', mmap_mode='r')
    exposures = select_document_exposures(documents, target_tokens, seed)
    if compact:
        if len(np.unique(exposures[:,0])) != len(exposures):
            raise ValueError('Compact pilot expects sampling without repeated source documents')
        exposures = exposures[np.argsort(exposures[:,0])]
    output.mkdir(parents=True)
    copy_frozen(base, output)
    starts, lengths, provenance = [], [], []
    writer = TokenWindowWriter(output, context_length=CONTEXT) if compact else None
    domains = {domain: Counter() for domain in DOMAINS}
    for exposure_number, (document_id, used, domain_number, cycle) in enumerate(exposures):
        document_id, used, domain_number, cycle = map(int, (document_id, used, domain_number, cycle))
        source_start, source_targets = map(int, documents[document_id, :2])
        target_start = writer.position if compact else source_start
        offsets = np.arange(0, used, CONTEXT, dtype=np.int64)
        starts.append(target_start + offsets)
        lengths.append(np.minimum(CONTEXT, used-offsets))
        provenance.extend((document_id, int(offset), exposure_number) for offset in offsets)
        if compact:
            writer.add_document(rows.tokens[source_start:source_start+used+1])
        domains[DOMAINS[domain_number]].update(target_tokens=used, document_exposures=1,
                                              quota_truncated_documents=int(used < source_targets),
                                              eos_targets=int(rows.tokens[source_start+used] == 2))
    index = np.column_stack((np.concatenate(starts), np.concatenate(lengths)))
    if compact:
        manifest = writer.finish()
        if not np.array_equal(np.load(output/'train.windows.npy'), index):
            raise ValueError('Compact writer window layout differs from the selected documents')
        selected = {int(row[0]): int(row[1]) for row in exposures}
        with gzip.open(base/'training_documents.jsonl.gz', 'rt', encoding='utf-8') as stream, \
                gzip_writer(output/'training_documents.jsonl.gz') as raw_out:
            for line in stream:
                row = json.loads(line)
                document_id = row['merged_document_id']
                if document_id in selected:
                    row['source_used_targets'] = row['used_targets']
                    row['used_targets'] = selected[document_id]
                    raw_out.write(json.dumps(row, ensure_ascii=False)+'\n')
    else:
        link_or_copy(base/'train.tokens.u32', output/'train.tokens.u32')
        np.save(output/'train.windows.npy', index)
        manifest = dict(format='indexed-token-v1', context_length=CONTEXT, dtype='<u4',
                        stored_tokens=len(rows.tokens), target_tokens=int(index[:,1].sum()),
                        windows=len(index), documents=len(exposures), token_file='train.tokens.u32',
                        index_file='train.windows.npy', document_layout='indexed exposures into the original document token pool')
        write_json(output/'train.index.json', manifest)
        link_or_copy(base/'training_documents.jsonl.gz', output/'source_documents.jsonl.gz')
    np.save(output/'train.exposures.npy', exposures)
    np.save(output/'train.provenance.npy', np.asarray(provenance, dtype=np.int64))
    actual = IndexedTokenWindows(output, 'train', CONTEXT)
    if actual.target_tokens != target_tokens:
        raise ValueError('Budget does not match selected targets')
    provenance = np.asarray(provenance, dtype=np.int64)
    for first in range(0, len(actual), 8192):
        last = min(first+8192, len(actual))
        block = provenance[first:last]
        source_starts = documents[block[:,0],0] + block[:,1]
        target_starts = actual.index[first:last,0]
        for offset in range(CONTEXT+1):
            active = actual.index[first:last,1] >= offset
            if not np.array_equal(rows.tokens[source_starts[active]+offset], actual.tokens[target_starts[active]+offset]):
                raise ValueError('A selected window differs from its original token sequence')
    verify_frozen(output, base)
    hashes = split_hashes(output)
    unique_documents = len(np.unique(exposures[:,0]))
    source = dict(version=output.name, dataset=parent['dataset'], revision=parent['revision'],
                  seed=seed, context_length=CONTEXT, target_tokens=target_tokens, windows=len(actual),
                  documents=len(exposures), unique_documents=unique_documents,
                  domains={name: dict(value) for name,value in domains.items()},
                  source_pool=str(base.resolve()), source_pool_target_tokens=rows.target_tokens,
                  source_pool_train_sha256=parent['split_sha256']['train'],
                  effective_source_epochs=target_tokens/rows.target_tokens,
                  maximum_domain_cycles=int(exposures[:,3].max())+1,
                  repeated_document_exposures=len(exposures)-unique_documents,
                  exposure_columns=['source_document_id','used_targets','domain_number','source_cycle'],
                  provenance_columns=['source_document_id','window_offset','exposure_number'],
                  exposure_sha256=sha256(output/'train.exposures.npy'),
                  provenance_sha256=sha256(output/'train.provenance.npy'),
                  tokenizer_sha256=sha256(output/'tokenizer.json'), split_sha256=hashes,
                  sampling='Seeded document permutation per domain and source cycle. Every complete cycle visits each document once. Domain quotas differ by at most one target.',
                  ordering='Original source-document order after seeded selection' if compact else 'Domain blocks, independently shuffled document order within each source cycle',
                  training_epoch_semantics='The training loader shuffles all selected windows in one epoch. Source cycles describe exposure counts, not chronological loader epochs.',
                  window_semantics='Original document prefixes including partial tails and genuine EOS; last quota document can be shortened without synthetic EOS',
                  compact_token_storage=compact,
                  limitations='Repeated exposure is explicit. Source distributions and cleaning limitations are inherited from the merged corpus.',
                  test='Copied for exclusion provenance only', builder_sha256=sha256(__file__))
    write_json(output/'source.json', source)
    audit = dict(verified=True, target_tokens=target_tokens, windows=len(actual),
                 document_exposures=len(exposures), unique_documents=unique_documents,
                 repeated_document_exposures=len(exposures)-unique_documents,
                 all_windows_compared_to_source=True, frozen_files_equal=True,
                 partial_windows=int(np.count_nonzero(actual.index[:,1] < CONTEXT)),
                 eos_targets=sum(value['eos_targets'] for value in domains.values()),
                 split_sha256=hashes, seconds=time.monotonic()-started)
    write_json(output/'audit.json', audit)
    shutil.copyfile(__file__, output/Path(__file__).name)
    print(json.dumps(audit, indent=2), flush=True)
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BOOK_ROOT/'industry-longrun-v1')
    parser.add_argument('--base-dir', type=Path)
    parser.add_argument('--budget', type=int)
    parser.add_argument('--seed', type=int, default=20261122)
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    if args.budget is not None:
        if args.base_dir is None:
            parser.error('--budget requires --base-dir')
        build_training_budget(args.base_dir, args.output, args.budget, args.seed, args.compact)
    else:
        merge_corpora(DEFAULT_PARENTS, args.output/'merged', BOOK_ROOT/'industry-expanded-v2')
        build_training_budget(args.output/'merged', args.output/'pilot-25m', 25_000_000,
                              args.seed, compact=True)

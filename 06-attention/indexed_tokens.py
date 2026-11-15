"""Document-preserving token windows backed by compact, read-only NumPy files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class TokenWindowWriter:
    def __init__(self, directory, name='train', context_length=256):
        if context_length < 1:
            raise ValueError('context_length must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name, self.context_length = name, context_length
        self.stream = (self.directory / f'{name}.tokens.u32').open('xb')
        self.windows, self.position, self.targets, self.documents = [], 0, 0, 0

    def add_document(self, ids, max_targets=None):
        targets = len(ids) - 1
        if max_targets is not None:
            targets = min(targets, max_targets)
        if targets < 1:
            return 0
        # A budget-truncated document keeps its genuine prefix; no artificial EOS.
        values = np.asarray(ids[:targets + 1], dtype='<u4')
        self.stream.write(values.tobytes())
        for start in range(0, targets, self.context_length):
            self.windows.append((self.position + start, min(self.context_length, targets - start)))
        self.position += len(values)
        self.targets += targets
        self.documents += 1
        return targets

    def finish(self):
        self.stream.close()
        if not self.windows:
            raise ValueError('empty dataset')
        np.save(self.directory / f'{self.name}.windows.npy', np.asarray(self.windows, dtype=np.int64))
        manifest = dict(format='indexed-token-v1', context_length=self.context_length,
                        dtype='<u4', stored_tokens=self.position, target_tokens=self.targets,
                        windows=len(self.windows), documents=self.documents,
                        token_file=f'{self.name}.tokens.u32', index_file=f'{self.name}.windows.npy')
        (self.directory / f'{self.name}.index.json').write_text(json.dumps(manifest, indent=2))
        return manifest


class IndexedTokenWindows:
    def __init__(self, directory, name, context_length):
        directory = Path(directory)
        self.manifest = json.loads((directory / f'{name}.index.json').read_text())
        if self.manifest['format'] != 'indexed-token-v1' or self.manifest['context_length'] != context_length:
            raise ValueError('indexed dataset format/context mismatch')
        self.tokens = np.memmap(directory / self.manifest['token_file'], mode='r', dtype='<u4')
        self.index = np.load(directory / self.manifest['index_file'], mmap_mode='r')
        self.target_tokens = int(self.index[:, 1].sum())
        if (len(self.tokens) != self.manifest['stored_tokens'] or len(self.index) != self.manifest['windows']
                or self.target_tokens != self.manifest['target_tokens']):
            raise ValueError('indexed dataset is incomplete')
        if np.any(self.index[:, 1] < 1) or np.any(self.index[:, 1] > context_length):
            raise ValueError('invalid window lengths')
        if np.any(self.index[:, 0] < 0) or np.any(self.index[:, 0] + self.index[:, 1] >= len(self.tokens)):
            raise ValueError('invalid token offsets')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        start, targets = map(int, self.index[index])
        ids = torch.from_numpy(self.tokens[start:start + targets + 1].astype(np.int64))
        return ids[:-1], ids[1:]


def dataset_counts(rows):
    tokens = rows.target_tokens if hasattr(rows, 'target_tokens') else sum(y.numel() for _, y in rows)
    return dict(windows=len(rows), tokens=int(tokens))


def split_fingerprint(directory, name):
    directory = Path(directory)
    path = directory / f'{name}.index.json'
    if not path.exists():
        with (directory / f'{name}.json').open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes())
    for filename in (manifest['token_file'], manifest['index_file']):
        with (directory / filename).open('rb') as stream:
            digest.update(hashlib.file_digest(stream, 'sha256').digest())
    return digest.hexdigest()

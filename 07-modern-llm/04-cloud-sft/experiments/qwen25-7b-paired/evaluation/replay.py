"""Replay the archived blind decisions against the 17 frozen generation files, on CPU."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE_SHA = 'afee891d4d73602bc2122d483a433fac84ea9e8f464a542d7f6add14b26b4409'
AUDITED_SHA = '9fedb22b3fe40dc390458398a813ae89ee7824c54bb12da4f2c17fe32606ed25'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label}: expected {expected}, got {actual}')


def replay(archive, work):
    same(sha(archive), ARCHIVE_SHA, '云端测量包 SHA-256')
    frozen_code = json.loads((ROOT/'grading-code-freeze.json').read_text())['sha256']
    same(sha(ROOT/'grade.py'), frozen_code, '冻结评分脚本 SHA-256')
    frozen_selection = json.loads((ROOT/'selection.json').read_text())

    # grade.py 所有生成物写在脚本旁，使用副本保留原始冻结材料。
    local = work/'evaluation'
    local.mkdir()
    for name in ('freeze.py', 'freeze.json', 'core.tsv', 'items.jsonl', 'prompts.jsonl',
                 'SCORING.md', 'grade.py', 'select_dev.py'):
        shutil.copyfile(ROOT/name, local/name)
    shutil.copytree(ROOT/'review'/'decisions', local/'review'/'decisions')

    stages = [f'epoch-{i/4:g}' for i in range(1, 9)]
    with tarfile.open(archive, 'r:gz') as tar:
        for arm in ('control', 'paired'):
            for stage in (['base'] if arm=='control' else []) + stages:
                name = f'run/{arm}/generations/{stage}.jsonl'
                member = tar.getmember(name)
                if not member.isfile():
                    raise ValueError(f'{name}: not a regular file')
                dest = work/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open('xb') as target:
                    shutil.copyfileobj(src, target)

    for action in ('prepare', 'score'):
        subprocess.run([sys.executable, '-B', 'grade.py', action, '--run', str(work/'run')],
                       cwd=local, check=True)
    scored = local/'paired-scores.jsonl'
    same(sha(scored), frozen_selection['scored_file_sha256'], '逐题配对评分 SHA-256')
    same(sha(local/'review/audited.jsonl'), AUDITED_SHA, '盲评审计输出 SHA-256')
    subprocess.run([sys.executable, '-B', 'select_dev.py', str(scored), '--out', 'selected.json'],
                   cwd=local, check=True, stdout=subprocess.DEVNULL)
    selected = json.loads((local/'selected.json').read_text())
    same(selected['arms'], frozen_selection['arms'], '开发集各轮计数/选轮')
    rows = [json.loads(line) for line in scored.read_text().splitlines()]
    print(f'验证通过：{len(rows)} 条配对评分；文件 SHA-256 = {sha(scored)}')
    for arm in ('control', 'paired'):
        target = [row for row in rows if row['arm']==arm and row['split']=='confirm' and row['epoch']==2]
        print(f'{arm}: 开发集选 {selected["arms"][arm]["selected_epoch"]} 轮；'
              f'第 2 轮独立确认集两问同时通过 {sum(r["pair_pass"] for r in target)}/{len(target)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='下载的 pair-measurements.tar.gz')
    parser.add_argument('--out', type=Path, help='可选：保留还原出的文件；填写尚不存在的目录')
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    if args.out:
        work = args.out.resolve()
        work.mkdir(parents=True, exist_ok=False)
        replay(archive, work)
        print(f'输出保存在 {work}')
    else:
        with tempfile.TemporaryDirectory(prefix='qwen-paired-eval-') as tmp:
            replay(archive, Path(tmp))


if __name__ == '__main__':
    main()

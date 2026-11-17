"""Read P28 from the historical cloud return. No model, GPU, or third party package needed."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STAGES = [('纯预训练底座', 'control', 'base'),
          ('原数据 SFT 两轮', 'control', 'epoch-2'),
          ('双问 SFT 两轮', 'paired', 'epoch-2')]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['short', 'reason', 'both'], default='short')
    args = parser.parse_args()
    ids = {'short': ['P28S'], 'reason': ['P28R'], 'both': ['P28S', 'P28R']}[args.mode]
    archive = ROOT / 'evidence/pair-measurements.tar.gz'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != 'afee891d4d73602bc2122d483a433fac84ea9e8f464a542d7f6add14b26b4409':
        raise ValueError('云端回传包 SHA-256 不符')
    with tarfile.open(archive, 'r:gz') as tar:
        for qid in ids:
            for index, (label, arm, stage) in enumerate(STAGES):
                name = f'run/{arm}/generations/{stage}.jsonl'
                with tar.extractfile(name) as f:
                    r = next(v for l in f if (v := json.loads(l))['id'] == qid)
                if index == 0:
                    print(f"\n{qid} | 原题\n{r['prompt']}")
                print(f"\n[{label}]\n{r['text']}\n"
                      f"输出 token 数={r['generated_tokens']}；前 5 个 ID={r['token_ids'][:5]}；"
                      f"停止 ID={r['stop_token_id']}")
if __name__ == '__main__':
    main()

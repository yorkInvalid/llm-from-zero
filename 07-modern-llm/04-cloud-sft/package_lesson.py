"""只打包课程文件，不包含 runs、dist、虚拟环境或模型缓存。"""
from pathlib import Path
import hashlib
import zipfile
HERE=Path(__file__).resolve().parent
ALLOWED={'.py','.md','.txt','.json','.jsonl','.ipynb','.png','.svg'}
def main():
    out=HERE/'dist';out.mkdir(exist_ok=True)
    path=out/'autodl-sft-qwen3-4b-a800.zip'
    files=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in ALLOWED
           and not any(part in {'runs','dist','__pycache__','.ipynb_checkpoints'} for part in p.relative_to(HERE).parts)]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,Path('cloud-sft-a800')/p.relative_to(HERE))
    print(path);print('files:',len(files),'bytes:',path.stat().st_size)
    print('sha256:',hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()

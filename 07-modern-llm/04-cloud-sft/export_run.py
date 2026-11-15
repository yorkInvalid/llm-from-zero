"""导出明确指定的运行及报告，避免复制整个缓存或账号信息。"""
import argparse
import zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reports',nargs='*',default=[]);p.add_argument('--out',required=True)
    a=p.parse_args();run=Path(a.run).resolve();out=Path(a.out).resolve()
    if not (run/'summary.json').is_file():raise ValueError('请选择已经正常完成、含 summary.json 的运行目录')
    if out==run or run in out.parents:raise ValueError('ZIP 不能放在待打包运行目录内部')
    if out.exists():raise ValueError('输出 ZIP 已存在，请使用新名字')
    reports=[Path(f).resolve() for f in a.reports]
    for f in reports:
        if not f.is_file():raise FileNotFoundError(f)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for f in run.rglob('*'):
            if f.is_file() and not f.is_symlink():z.write(f,Path('run')/f.relative_to(run))
        for f in reports:z.write(f,Path('reports')/f.name)
        for folder in ['data','configs']:
            for f in (HERE/folder).glob('*'):
                if f.is_file():z.write(f,Path('reproduce')/folder/f.name)
        for f in list(HERE.glob('*.py'))+[HERE/'requirements-cloud.txt',HERE/'CLOUD_RUN.md']:
            z.write(f,Path('reproduce')/f.name)
    print(out,'bytes=',out.stat().st_size)
if __name__=='__main__':main()

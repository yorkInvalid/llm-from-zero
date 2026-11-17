"""Archive checkpoints without delaying parameter updates. Full optimizer state stays in the archives."""
import hashlib,json,sys,tarfile,time
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'downloads'
out.mkdir(exist_ok=True)
def write_pack(name,items):
    dst=out/name
    if dst.exists():return
    tmp=dst.with_suffix('.partial')
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
        for p in items:tar.add(p,p.relative_to(root))
    tmp.rename(dst)
    with dst.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    (out/(name+'.sha256')).write_text(digest+'  '+name+'\n')
def checkpointed():
 for arm in ['control','paired']:
    for i in range(1,9):
        stage='epoch-'+str(i/4).rstrip('0').rstrip('.')
        ck=root/'run'/arm/stage
        if (ck/'state.json').exists():write_pack(arm+'-'+stage+'.tar.gz',[ck])
if sys.argv[1]=='watch':
    while not (root/'FINISHED').exists():
        checkpointed()
        time.sleep(5)
    checkpointed()
else:
    files=[p for p in (root/'run').rglob('*') if p.is_file() and p.suffix not in ('.safetensors','.pt')]
    files += [root/n for n in ['experiment.py','schedule.json','prompts.jsonl','run-pair.sh','pack.py']]
    write_pack('pair-measurements.tar.gz',files)

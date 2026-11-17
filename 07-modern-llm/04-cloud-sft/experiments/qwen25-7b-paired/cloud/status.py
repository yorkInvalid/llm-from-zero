import json,pathlib,subprocess,time
from datetime import datetime
root=pathlib.Path(__file__).resolve().parent
while not (root/'FINISHED').exists():
    tab=[]
    for arm in ['control','paired']:
        out=root/'run'/arm
        log=out/'updates.jsonl'
        step=0
        timing=0
        if log.is_file():
            with log.open() as f:
                lines=f.readlines()
                if lines:
                    try:r=json.loads(lines[-1])
                    except ValueError:r=json.loads(lines[-2])
                    step=r['step']
                    timing=r['train_seconds']
        saved=sum(1 for d in out.glob('epoch-*/state.json'))
        generated=sum(1 for d in (out/'generations').glob('*-meta.json'))
        tab.append(f'{arm}:{step}/4504 {timing:.0f}s ck{saved} gen{generated}')
    title='Pair SFT | '+' | '.join(tab)+' | '+datetime.now().strftime('%H:%M:%S')
    print(chr(27)+']0;'+title+chr(7)+title,flush=True)
    time.sleep(30)
print('PAIR TRAINING AND GENERATION COMPLETE')

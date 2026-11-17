"""Choose a snapshot only on development prompts. This program never aggregates confirmation responses."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import grade


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('scored',type=Path)
    ap.add_argument('--arm',choices=['control','paired','both'],default='both')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    frozen,_=grade.frozen()
    expected={r['id'] for r in frozen.values() if r['split']=='dev'}
    arms=['control','paired'] if args.arm=='both' else [args.arm]
    tallies={arm:{e:{} for e in grade.EPOCHS} for arm in arms}
    for r in grade.lines(args.scored):
        if r['item_id'] not in expected or r['arm'] not in arms: continue
        bucket=tallies[r['arm']][r['epoch']]
        if r['split']!='dev' or r['item_id'] in bucket: raise ValueError('duplicate/mismatched dev split')
        bucket[r['item_id']]=int(r['pair_pass'])
    doc={'utc':datetime.now(timezone.utc).isoformat(),'development_cores':len(expected),
         'selection_rule':'most paired passes on 30 development cores; earliest trained checkpoint on tie',
         'prompt_sha256':grade.file_hash(grade.ROOT/'prompts.jsonl'),'scored_file_sha256':grade.file_hash(args.scored),'arms':{}}
    for arm,buckets in tallies.items():
        if any(set(b)!=expected for b in buckets.values()):raise ValueError((arm,'missing development answers'))
        counts={e:sum(b.values()) for e,b in buckets.items()}
        selected=max(grade.EPOCHS,key=counts.get)
        doc['arms'][arm]={'joint_by_epoch':counts,'selected_epoch':selected,'selected_correct':counts[selected]}
    payload=json.dumps(doc,ensure_ascii=False,indent=2)+'\n'
    args.out.write_text(payload,encoding='utf-8')
    print(payload)


if __name__=='__main__':main()

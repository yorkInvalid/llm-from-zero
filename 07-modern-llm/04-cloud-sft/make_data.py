"""确定性合成教学数据；非真实客户信息，不能代表生产场景效果。"""
import hashlib
import json
import random
from pathlib import Path
HERE=Path(__file__).resolve().parent
SYSTEM='你是采购信息抽取助手。只输出一个JSON对象，不要解释或Markdown。字段严格为product、quantity、unit、delivery_city。product和delivery_city为字符串或null，quantity为整数或null，unit为字符串或null。提取最终确认的采购需求，忽略库存和历史订单；未明确给出的字段填null，不要推测。'
PRODUCTS=['保温杯','文件夹','台灯','蓝牙耳机','鼠标','键盘','纸箱','帆布袋','笔记本','雨伞','水杯','充电器']
CITIES=['杭州','成都','武汉','苏州','深圳','宁波','西安','长沙','厦门','南京','青岛','合肥']

def build():
    rng=random.Random(20261114)
    manifest={'origin':'deterministic synthetic educational examples; no real customer data',
              'seed':20261114,'split_rule':'scenario assigned before paraphrase; templates held out by split',
              'limitations':'synthetic templates; not evidence of real-world extraction quality', 'files':{}}
    for split,groups in [('train',80),('validation',16),('test',16)]:
        rows=[]
        for index in range(groups):
            product=rng.choice(PRODUCTS);city=rng.choice(CITIES);qty=rng.randint(2,2000)
            unit=rng.choice(['个','件','套'])
            # 刻意包含缺失字段，JSON null 不能被当字符串 "null"。
            target={'product':product,'quantity':qty,'unit':unit,'delivery_city':city}
            if index%5==0:target['delivery_city']=None
            if index%7==0:target['quantity']=None;target['unit']=None
            if index%11==0:target['product']=None
            p=target['product'] or '产品还没定'
            amount=f"{qty}{unit}" if target['quantity'] is not None else '数量和单位还没定'
            destination=f"送到{city}" if target['delivery_city'] is not None else '收货城市待确认'
            if split=='train':
                texts=[f'采购记录：{p}；{amount}；{destination}。',
                       f'这次要采购{p}，{amount}，{destination}。仓库现有库存100件，不是本次采购量。']
            elif split=='validation':
                texts=[f'请记录本次需求。货品：{p}。需求量：{amount}。交付要求：{destination}。',
                       f'本次采购已确认：{p} / {amount} / {destination}。去年订单为50件，请忽略。']
            else:
                texts=[f'最终确认如下——{p}，{amount}，{destination}。旧记录写了3件，以本次确认为准。',
                       f'订单信息请从这里提取：交付地点要求{destination}；需要的东西是{p}；本次量为{amount}。']
            scenario=f'{split}-{index:03d}'
            for variant,text in enumerate(texts):
                rows.append({'id':f'{scenario}-{variant}','group_id':scenario,'text':text,
                             'answer':target,'origin':'synthetic'})
        path=HERE/'data'/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        manifest['files'][split]={'rows':len(rows),'groups':groups,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (HERE/'data/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print({k:v['rows'] for k,v in manifest['files'].items()})
if __name__=='__main__':build()

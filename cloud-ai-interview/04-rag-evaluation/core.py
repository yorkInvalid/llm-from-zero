"""只在需要时打开：复用第三课的真实模型，增加证据覆盖检查。"""
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / '03-rag-datawhale'
sys.path.insert(0, str(PREVIOUS))
import knowledge


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 禁止同名实验覆盖；失败也保留现场。
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def valid_name(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('实验名字只能用英文字母、数字、短横线或下划线。')
    return name


def cases():
    return read_json(ROOT / 'cases.json')['cases']


def locate_evidence(case, docs):
    """教师标注短原文 → 在固定原文中的字符范围。不是把标准答案给模型。"""
    by_name = {d['source']: d['text'] for d in docs}
    spans = []
    for ev in case['evidence']:
        text = by_name[ev['source']]
        if text.count(ev['quote']) != 1:
            raise ValueError(f"{case['id']} 证据无法唯一定位；请核对原资料与 cases.json。")
        start = text.index(ev['quote'])
        spans.append({**ev, 'start': start, 'end': start + len(ev['quote'])})
    return spans


def span_covered(span, hits):
    """允许同一份文件的两个相邻片段拼起来覆盖证据，不靠关键词猜。"""
    position = span['start']
    intervals = sorted((h['start'], h['end']) for h in hits if h['source'] == span['source'])
    for start, end in intervals:
        if end <= position:
            continue
        if start > position:
            break
        position = max(position, end)
        if position >= span['end']:
            return True
    return False


def score_evidence(spans, hits):
    flags = [span_covered(span, hits) for span in spans]
    return {'covered': sum(flags), 'required': len(flags),
            'complete': all(flags) if flags else None, 'flags': flags}


def summary(rows):
    eligible = [r for r in rows if r['case']['answerable']]
    return {'answerable_count': len(eligible),
            'complete_count': sum(r['score']['complete'] is True for r in eligible),
            'coverage_rate': (sum(r['score']['complete'] is True for r in eligible) / len(eligible)) if eligible else None,
            'unanswerable_count': len(rows)-len(eligible),
            'mean_context_characters': round(sum(r['context_characters'] for r in rows)/len(rows), 1) if rows else 0}


def prepare(name):
    import chromadb
    from chromadb.config import Settings
    name = valid_name(name)
    target = ROOT / 'artifacts' / 'indexes' / name
    if target.exists():
        raise ValueError('同名索引已存在；继续用它评估，或换一个名字，不覆盖旧记录。')
    filenames = ['2. 提示原则 Guidelines.md', '3. 迭代优化 Iterative.md', '4. 文本概括 Summarizing.md']
    docs = [{'source': f, 'text': (PREVIOUS/'data/source'/f).read_text(encoding='utf-8')} for f in filenames]
    suite = cases()
    for case in suite:
        locate_evidence(case, docs)
    chunks = knowledge.split_documents(docs, 300, 40)
    print(f'固定三份资料 → {len(chunks)} 个片段 → 复用第三课的 BGE 缓存', flush=True)
    vectors = knowledge.embed([c['text'] for c in chunks])
    target.mkdir(parents=True)
    db = chromadb.PersistentClient(path=str(target), settings=Settings(anonymized_telemetry=False))
    coll = db.create_collection('learning_materials', embedding_function=None, metadata={'hnsw:space': 'cosine'})
    coll.add(ids=[c['id'] for c in chunks], documents=[c['text'] for c in chunks], embeddings=vectors,
             metadatas=[{'source': c['source'], 'start': c['start'], 'end': c['end']} for c in chunks])
    write_json(target/'documents.json', docs)
    write_json(target/'chunks.json', chunks)
    write_json(target/'manifest.json', {'documents_hash': digest(docs), 'model': knowledge.MODEL,
               'query_prefix': knowledge.QUERY_PREFIX, 'size': 300, 'overlap': 40, 'chunks': len(chunks)})
    return target


def evaluate(index_name, k):
    if not 1 <= k <= 10:
        raise ValueError('k 需在 1–10 之间。')
    path = ROOT/'artifacts/indexes'/valid_name(index_name)
    manifest = read_json(path/'manifest.json')
    docs = read_json(path/'documents.json')
    if digest(docs) != manifest['documents_hash']:
        raise ValueError('固定资料快照变化，请建立新索引。')
    if (manifest['model'], manifest['query_prefix']) != (knowledge.MODEL, knowledge.QUERY_PREFIX):
        raise ValueError('向量模型配置变化，请重建。')
    suite = cases()
    spans = [locate_evidence(c, docs) for c in suite]
    # 所有问题使用同一个真实模型；预先生成查询向量，不把加载时间当查询延迟。
    vectors = knowledge.embed([c['question'] for c in suite], is_query=True)
    coll = knowledge.collection(path)
    rows = []
    for case, evidence, vector in zip(suite, spans, vectors):
        started = perf_counter()
        found = coll.query(query_embeddings=[vector], n_results=min(k, coll.count()),
                           include=['documents', 'metadatas', 'distances'])
        elapsed = (perf_counter()-started)*1000
        hits = [{'id': id_, 'text': text, **meta, 'cosine_distance': float(dist)}
                for id_, text, meta, dist in zip(found['ids'][0],found['documents'][0],found['metadatas'][0],found['distances'][0])]
        rows.append({'case': case, 'evidence': evidence, 'hits': hits, 'score': score_evidence(evidence,hits),
                     'context_characters': len(knowledge.context_text(hits)), 'query_ms': round(elapsed,2)})
    return {'kind':'retrieval_evaluation','source':'real_local_bge_and_chroma','index':index_name,
            'index_manifest':manifest,'suite_hash':digest(suite),'k':k,'rows':rows,'summary':summary(rows),
            'answer_generation':'not_called'}


def compare(before, after):
    if before['index_manifest'] != after['index_manifest'] or before['suite_hash'] != after['suite_hash']:
        raise ValueError('资料、切分、模型或问题集不同，不能把结果解释为只改变 k。')
    if [r['case']['id'] for r in before['rows']] != [r['case']['id'] for r in after['rows']]:
        raise ValueError('两次评估的问题顺序或集合不一致。')
    rows=[]
    for a,b in zip(before['rows'],after['rows']):
        x,y=a['score']['complete'],b['score']['complete']
        change='不计分' if x is None else ('改善' if not x and y else '退步' if x and not y else '不变')
        rows.append({'id':a['case']['id'],'change':change,'before':a,'after':b})
    return rows

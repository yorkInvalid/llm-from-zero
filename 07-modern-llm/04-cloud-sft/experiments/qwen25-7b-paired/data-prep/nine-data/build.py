"""Build the pinned ModelScope COIG-CQIA SFT split. Run with tokenizers installed."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re
import subprocess
import unicodedata

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
RAW = HERE / "modelscope-raw"
REPO = "https://www.modelscope.cn/datasets/m-a-p/COIG-CQIA"
REV = "9c816ffadbd53ef0149b9b1f7d850cddb5d6ebd4"
SEED = 20261116
TOKENIZER = WORK / "seven-weights/metadata/tokenizer.json"
OLD_BENCH = WORK / "eval/questions.json"
SOURCES = {
    "chinese_traditional": "传统文化", "coig_pc": "中文文本任务", "douban": "书影音",
    "exam": "综合考试", "finance": "财经文本", "logi_qa": "逻辑选择",
    "ruozhiba": "趣味问答", "segmentfault": "开发问答", "wiki": "科普百科",
    "xhs": "生活写作", "zhihu": "社区问答",
}
RESERVED = re.compile(r"<\|[^|\n]{1,60}\|>|\x00|\ufffd")
NORM = re.compile(r"[\W_]+", re.UNICODE)


def digest(data):
    return sha256(data).hexdigest()


def rank(text):
    return digest(f"{SEED}/{text}".encode())


def normalized(text):
    return NORM.sub("", unicodedata.normalize("NFKC", text).casefold())


def clean(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip()


def shingles(text, size):
    return {text[i:i + size] for i in range(len(text) - size + 1)}


def lines(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def group(path):
    if path.name == "mba_baike.jsonl":
        return "经管百科"
    if path.name.startswith("law_gee_"):
        return "法律考试"
    if path.parent.name == "wiki" and path.name in {"51zyzy.jsonl", "bkmy_symptom.jsonl"}:
        return "健康百科"
    return SOURCES[path.parent.name]


def write_rows(path, rows):
    s = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    path.write_text(s, encoding="utf-8")
    return digest(s.encode())


def mix(buckets):
    pools = [iter(sorted(buckets[s], key=lambda r: rank(r["id"]))) for s in sorted(buckets)]
    for group_rows in itertools.zip_longest(*pools):
        yield from (r for r in group_rows if r is not None)


def summarize(rows):
    counts = Counter(r["source"] for r in rows)
    stats = {}
    for key in ["input_tokens", "supervised_tokens", "total_tokens"]:
        values = sorted(r[key] for r in rows)
        stats[key] = dict(total=sum(values), min=values[0], median=(values[(len(values)-1)//2]+values[len(values)//2])/2,
                          p90=values[(9*len(values)-1)//10], max=values[-1], mean=round(sum(values)/len(values), 2))
    source_targets = Counter()
    for row in rows:
        source_targets[row["source"]] += row["supervised_tokens"]
    return dict(rows=len(rows), sources=dict(sorted(counts.items())), supervised_tokens_by_source=dict(sorted(source_targets.items())), tokens=stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-bench", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if subprocess.check_output(["git", "-C", str(RAW), "rev-parse", "HEAD"], text=True).strip() != REV:
        raise RuntimeError("ModelScope checkout revision mismatch")
    tok = Tokenizer.from_file(str(TOKENIZER))
    assert tok.token_to_id("<|endoftext|>") == 151643
    benchmark_files = [OLD_BENCH, *args.extra_bench]
    benches = [normalized(r["prompt"]) for f in benchmark_files
               for r in (lines(f) if f.suffix == ".jsonl" else json.loads(f.read_text()))]
    bench_shingles = set().union(*(shingles(b, 16) for b in benches))
    previous = [WORK / "7b-experiment/data" / f for f in ["train.jsonl", "validation.jsonl"]]
    old_prompts = {normalized(r["prompt"]) for path in previous for r in lines(path)}
    seen_prompts = set()
    seen_answers = set()
    buckets = defaultdict(list)
    reasons = Counter()
    raw_files = {}
    for path in sorted(RAW.glob("*/*.jsonl")):
        if path.parent.name not in SOURCES or path.name in {"bkmy_medicine.jsonl"}:
            continue
        rel = path.relative_to(RAW).as_posix()
        content = path.read_bytes()
        raw_hash = digest(content)
        git_blob = subprocess.check_output(["git", "-C", str(RAW), "rev-parse", f"HEAD:{rel}"], text=True).strip()
        if path.name == "mba_baike.jsonl":
            pointer = subprocess.check_output(["git", "-C", str(RAW), "show", f"HEAD:{rel}"], text=True)
            assert f"oid sha256:{raw_hash}" in pointer, "Fetch finance/mba_baike.jsonl via ModelScope LFS"
        else:
            from hashlib import sha1
            assert sha1(f"blob {len(content)}\x00".encode() + content).hexdigest() == git_blob, rel
        examined = len(content.splitlines())
        raw_files[rel] = {"sha256": raw_hash, "lines": examined, "git_blob": git_blob}
        for idx, line in enumerate(content.splitlines(), 1):
            raw = json.loads(line)
            if any(not isinstance(raw.get(k), str) for k in ["instruction", "input", "output"]):
                reasons["nontext"] += 1
                continue
            inst, extra, ans = (clean(raw[k]) for k in ["instruction","input","output"])
            prompt = "\n\n".join(s for s in [inst, extra] if s)
            if not inst or not ans or max(len(prompt), len(ans)) > 2400 or RESERVED.search(prompt + ans):
                reasons["empty_long_or_reserved"] += 1
                continue
            if not re.search(r"[\u3400-\u9fff]", prompt):
                reasons["no_chinese_in_prompt"] += 1
                continue
            cp = normalized(prompt)
            ca = normalized(ans)
            if not cp or cp in old_prompts or cp in seen_prompts or (len(ca) >= 24 and ca in seen_answers):
                reasons["duplicate_or_old_prompt_or_long_answer_duplicate"] += 1
                continue
            if cp in benches or not bench_shingles.isdisjoint(shingles(cp,16)) or not bench_shingles.isdisjoint(shingles(ca,16)):
                reasons["eval_overlap"] += 1
                continue
            prefix = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
            n_in = len(tok.encode(prefix, add_special_tokens=False).ids)
            n_ans = len(tok.encode(ans, add_special_tokens=False).ids) + 1
            if n_in + n_ans > 384:
                reasons["over_384_tokens"] += 1
                continue
            seen_prompts.add(cp)
            if len(ca) >= 24:
                seen_answers.add(ca)
            src = group(path)
            row = dict(id=f"{rel[:-6]}-L{idx:05}", source=src, prompt=prompt, response=ans, source_file=rel,
                       source_line=idx, input_tokens=n_in, supervised_tokens=n_ans, total_tokens=n_in+n_ans,
                       domain=raw.get("domain"), task_type=raw.get("task_type"),
                       answer_from=raw.get("answer_from"), human_verified=raw.get("human_verified"))
            buckets[src].append(row)
    pool = list(mix(buckets))
    print("eligible", len(pool), {k:len(v) for k,v in buckets.items()}, flush=True)
    validation = pool[:120]
    # Intersect only candidate validation prompts sharing a shingle, then test exact Jaccard.
    val_sets = [shingles(normalized(r["prompt"]), 4) for r in validation]
    index = defaultdict(set)
    for i, grams in enumerate(val_sets):
        for gram in grams:
            index[gram].add(i)
    train = []
    for r in pool[len(validation):]:
        grams = shingles(normalized(r["prompt"]), 4)
        candidates = set().union(*(index.get(gram, ()) for gram in grams))
        if any(len(grams & val_sets[i]) >= .70 * len(grams | val_sets[i]) for i in candidates):
            reasons["validation_near_duplicate"] += 1
            continue
        train.append(r)
        if len(train) == 9000:
            break
    assert len(train) == 9000 and len(validation) == 120
    train.sort(key=lambda r: rank("train/" + r["id"]))
    validation.sort(key=lambda r: rank("validation/" + r["id"]))
    files = {name:write_rows(HERE / name, rows) for name,rows in [("train.jsonl",train),("validation.jsonl",validation)]}
    manifest = dict(repo=REPO, revision=REV, seed=SEED, raw_files=raw_files, schema="prompt = instruction + optional double-newline + input; response = output",
        normalization="Trim end whitespace, initial BOM, CRLF and CR. Prompt and >=24-character answer dedup keys use NFKC, casefold, and alphanumerics; short choices may legitimately repeat.",
        sampling="Within each source sort by seeded SHA-256 of file:line ID then interleave sources in rounds. First 120 form validation; next 9000 eligible form training. Max 384 total tokens, no truncation. Remove cross-split prompt 4-gram Jaccard >= 0.70.",
        eval_filter="Exclude normalized identical prompts or >=16 consecutive normalized characters shared by prompt or answer and a benchmark prompt. Prior experiment 800 training + 24 validation prompts excluded by normalized exact match.",
        benchmarks={p.name+":"+p.parent.name:digest(p.read_bytes()) for p in benchmark_files}, previous_inputs={p.name:digest(p.read_bytes()) for p in previous},
        eligible_by_source={k:len(v) for k,v in buckets.items()}, reasons=dict(reasons), source_rows_examined=sum(v["lines"] for v in raw_files.values()),
        tokenizer_sha256=digest(TOKENIZER.read_bytes()), native_eos=151643, file_sha256=files, train=summarize(train), validation=summarize(validation))
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files":files,"train":manifest["train"],"validation":manifest["validation"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()

"""只检索，不生成答案；保存真实结果，方便核对。"""
import argparse
import json
from datetime import datetime
from knowledge import ROOT, search


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?", default="提示词里使用分隔符有什么用？")
    p.add_argument("--index", default="first")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()
    hits = search(args.question, ROOT / "artifacts" / args.index, args.top_k)
    for h in hits:
        print(f"\n[{h['id']}] {h['source']} 字符 {h['start']}:{h['end']} 距离={h['cosine_distance']:.4f}\n{h['text']}")
    print("\n距离越小越接近；不是正确率，也不保证存在答案。")
    folder = ROOT / "artifacts" / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-search.json")
    path.write_text(json.dumps({"question": args.question, "index": args.index, "hits": hits}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已保存：", path)

# 两份训练集如何生成

训练直接读取 [冻结输入包](../inputs/frozen-inputs.tar.gz)。本目录保留构造这些输入时的本地脚本，目录名沿用原实验，便于核对相对路径。可以在临时目录里重建，不改动仓库原件。

- [nine-data/fetch.py](nine-data/fetch.py) 和 [build.py](nine-data/build.py)：按固定 revision 从魔搭取 COIG-CQIA，去重、排除旧题和过长样本、生成 9,000 条训练和 120 条验证。旧实验输入 `7b-experiment/data/`、`eval/questions.json` 只供排重。来源文件、选样和 SHA 见 [coig-manifest.json](../inputs/coig-manifest.json)，原数据卡见 [COIG-CQIA-README.md](../inputs/COIG-CQIA-README.md)。
- [pair-data/build.py](pair-data/build.py)、[match.py](pair-data/match.py)、[audit.py](pair-data/audit.py)：改写配对、匹配预算、检查新题重叠。[配对映射](pair-data/pairs.jsonl)记录 1,095 对，另有 6,810 行保持原样。[数据笔记](pair-data/README.md)解释具体处理。两组每轮均为 634,005 个提示 token、1,353,439 个监督 token。
- [pair-transfer/rebuild_paired.py](pair-transfer/rebuild_paired.py)：当时为云端传输准备的独立补丁程序（内嵌差量，不需要网络、GPU 或分词器）。只接受哈希匹配的原数据，重建后再验 SHA。`pair-train/apply-delta.py` 与 `paired.delta.xz` 是同一目标数据的另一种传法。

下面在**实验根目录**运行，检验紧凑传输与冻结双问组是否逐字节一致：

```bash
scratch=$(mktemp -d)
tar -xzf inputs/frozen-inputs.tar.gz -C "$scratch"
python3 data-prep/pair-transfer/rebuild_paired.py "$scratch/data/control/train.jsonl" "$scratch/rebuilt.jsonl"
cmp "$scratch/rebuilt.jsonl" "$scratch/data/paired/train.jsonl"
```

如要查看配对脚本怎样从原题得到两种回答，可按原目录布局运行下列命令（同一个 shell，沿用上面的 `scratch`；Python 环境需安装 `tokenizers`，冻结映射路径不用 SciPy）：

```bash
cp -R data-prep "$scratch/work"
mkdir -p "$scratch/work/seven-weights/metadata" "$scratch/work/pair-eval"
tar -xzf inputs/base-metadata.tar.gz -C "$scratch/work/seven-weights/metadata" tokenizer.json
cp "$scratch/data/control/"*.jsonl "$scratch/work/nine-data/"
cp cloud/schedule.json "$scratch/work/pair-train/"
cp evaluation/items.jsonl evaluation/prompts.jsonl "$scratch/work/pair-eval/"
python3 "$scratch/work/pair-data/build.py" --selection "$scratch/work/pair-data/pairs.jsonl"
cmp "$scratch/work/pair-data/paired.jsonl" "$scratch/data/paired/train.jsonl"
python3 "$scratch/work/pair-data/audit.py"
```

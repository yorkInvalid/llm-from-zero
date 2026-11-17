"""Replay the archived ModelScope experiment. Default: CPU-only integrity check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
CLOUD = ROOT / "cloud"
INPUTS = ROOT / "inputs"
MANIFEST = INPUTS / "base-manifest.json"
SCHEDULE = CLOUD / "schedule.json"
STAGES = ["epoch-" + t for t in ("0.25", "0.5", "0.75", "1", "1.25", "1.5", "1.75", "2")]
# Individual file hashes: independent of gzip, tar timestamps and file order.
TRAIN = "data/control/train.jsonl"
PAIRED = "data/paired/train.jsonl"
VALID = "data/control/validation.jsonl"
PROMPTS = "prompts.jsonl"
FROZEN = {
    TRAIN: "e314fe11f0a578009e67809ec7d67e3c72484756a25e6df50825eba8f41a279e",
    PAIRED: "911843fd787d5366efc63fd73af290ca40f37cc12da6365ccca898b34afc6965",
    VALID: "4647e3ff0315dbedd17495fa7bf3ff6bdf754c7ecf228d97d096e6c33d24a44b",
    PROMPTS: "69b73299803a7aad9e98891f0efe3d0df62bd9675e7f5e513a337a0f85fbd0b5",
}
SCHEDULE_HASH = "9b0127e3cf5ccc57775923cf97e3b702c993d177c1902431da597cbea26f6aae"
MANIFEST_HASH = "fa8c6e1317885d1d0935d7b38b8bdd90badc8adb2b156d84a299be8afadc582e"
SCRIPT_HASH = "a2de6b419955f00612a05f0a03490da43ad15b6b1c9c62aaaaa8d83b360f9a06"


def require(condition, error):
    if not condition:
        raise ValueError(error)


def checked(data, expected, name):
    require(hashlib.sha256(data).hexdigest() == expected, f"SHA-256 不符：{name}")
    return data


def frozen_inputs():
    files = {}
    with tarfile.open(INPUTS / "frozen-inputs.tar.gz", "r:gz") as tar:
        for entry in tar:
            # Nothing is extracted by tar: only these four exact names may be written.
            name = entry.name.removeprefix("./")
            if entry.isdir() and name in ("", ".", "data", "data/control", "data/paired"):
                continue
            require(name in FROZEN and name not in files and entry.isfile()
                    and 0 < entry.size <= 32 * 1024**2, f"非预期的归档成员：{entry.name}")
            with tar.extractfile(entry) as stream:
                files[name] = checked(stream.read(), FROZEN[name], name)
    require(files.keys() == FROZEN.keys(), "归档缺少冻结输入")
    return files


def inspect_schedule(files):
    schedule = json.loads(checked(SCHEDULE.read_bytes(), SCHEDULE_HASH, SCHEDULE.name))
    def records(name):
        return [json.loads(line) for line in files[name].splitlines() if line.strip()]
    prompts, valid = records(PROMPTS), records(VALID)
    require(len(prompts) == len({r["id"] for r in prompts}) == 120, "评测题数量错误")
    require(len(valid) == len({r["id"] for r in valid}) == 120, "验证集数量错误")
    require(len(schedule["quarters"]) == 8, "日程必须包含 8 个四分之一轮")
    for name in (TRAIN, PAIRED):
        rows = records(name)
        by_id = {r["id"]: r for r in rows}
        targets = {k: r["supervised_tokens"] for k, r in by_id.items()}
        require(len(rows) == len(by_id) == 9000 and targets == schedule["control_targets"]
                and sum(targets.values()) == 1353439, f"监督 token 预算不符：{name}")
        require(by_id.keys().isdisjoint(r["id"] for r in valid), "训练及验证 ID 重合")
        for epoch in range(2):
            seen = []
            for quarter in schedule["quarters"][4*epoch:4*epoch+4]:
                ids = quarter["ids"]
                require(len(ids) == 2250 and all(i in by_id for i in ids)
                        and sum(targets[i] for i in ids) == quarter["supervised_tokens"]
                        and sum(by_id[i]["input_tokens"] for i in ids) == quarter["input_tokens"],
                        f"四分之一轮顺序或 token 预算不符：{name}")
                seen.extend(ids)
            require(len(seen) == len(set(seen)) == 9000, f"每轮应恰好覆盖 9000 个样本：{name}")


def verify_base(base, manifest):
    require(base.is_dir() and not (base / "adapter_config.json").exists(),
            f"需要本地纯底座目录，不能是 LoRA 目录：{base}")
    for name, spec in manifest["files"].items():
        path = base / name
        require(path.is_file() and path.stat().st_size == spec["bytes"], f"文件缺失或长度不符：{path}")
        if "sha256" in spec:
            digest, expected = hashlib.sha256(), spec["sha256"]
        else:
            digest = hashlib.sha1(f"blob {spec['bytes']}\0".encode())
            expected = spec["git_blob_sha1"]
        with path.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024**2), b""):
                digest.update(block)
        require(digest.hexdigest() == expected, f"模型文件内容不符：{path}")
    print(f"纯底座校验通过：{manifest['model']} @ {manifest['revision']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-only", action="store_true", help="校验冻结输入、日程及可选底座（默认）")
    group.add_argument("--generate-base", action="store_true", help="只跑全部评测题的纯底座推理")
    group.add_argument("--run", action="store_true", help="在 GPU 上训练两组并生成所有检查点回答")
    parser.add_argument("--base", type=Path, help="已下载的固定版本纯底座目录")
    parser.add_argument("--output", type=Path, help="GPU 任务使用的新输出目录（必须不存在）")
    args = parser.parse_args()
    gpu = args.run or args.generate_base
    if gpu and (not args.base or not args.output):
        parser.error("GPU 运行必须指定 --base 和 --output")
    if not gpu and args.output:
        parser.error("校验不会写文件，无需 --output")
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("--output 已存在；请使用新目录，以免重用其他运行的权重或生成缓存")

    experiment = CLOUD / "experiment.py"
    checked(experiment.read_bytes(), SCRIPT_HASH, experiment.name)
    manifest = json.loads(checked(MANIFEST.read_bytes(), MANIFEST_HASH, MANIFEST.name))
    files = frozen_inputs()
    inspect_schedule(files)
    print("输入校验通过：两组各 9000 条，共用 120 条验证；120 条评测提示；每组两轮、8 个检查点。", flush=True)
    if args.base:
        base = args.base.expanduser().resolve()
        verify_base(base, manifest)
    if not gpu:
        return

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    require(sys.version_info >= (3, 11) and not sys.flags.optimize and not env.get("PYTHONOPTIMIZE"),
            "云端脚本需要 Python 3.11+；请勿用 -O 或 PYTHONOPTIMIZE 关闭校验。")
    subprocess.run([sys.executable, "-c", "import torch, transformers, peft; "
                    "assert torch.cuda.is_available(), 'CUDA GPU unavailable'"], check=True, env=env)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Only expose verified manifest members to Transformers (ignore unrelated files in a shared cache).
    verified = output / "verified-base"
    verified.mkdir()
    for name in manifest["files"]:
        (verified / name).symlink_to(base / name)
    data_dir = output / "inputs"
    for name, content in files.items():
        path = data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (data_dir / "data/paired/validation.jsonl").write_bytes(files[VALID])

    def call(command, arm, *options):
        argv = [sys.executable, str(experiment), command, "--arm", arm, "--base", str(verified),
                "--out", str(output / "run" / arm), *map(str, options)]
        print("+", shlex.join(argv), flush=True)
        subprocess.run(argv, check=True, env=env)
    if args.run:
        for arm in ("control", "paired"):
            call("train", arm, "--data", data_dir / "data" / arm, "--base-manifest", MANIFEST,
                 "--schedule", SCHEDULE)
    call("generate", "control", "--prompts", data_dir / PROMPTS, "--stages", "base",
         *(STAGES if args.run else []))
    if args.run:
        call("generate", "paired", "--prompts", data_dir / PROMPTS, "--stages", *STAGES)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        sys.exit(f"复现中止：{exc}")

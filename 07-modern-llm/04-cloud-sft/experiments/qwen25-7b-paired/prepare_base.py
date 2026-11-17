"""Prepare the frozen Qwen2.5-7B base on CPU. Run --help for the three modes."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

INPUTS = Path(__file__).resolve().parent / "inputs"
MODEL = "Qwen/Qwen2.5-7B"
REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
SHARDS = {f"model-{n:05d}-of-00004.safetensors" for n in range(1, 5)}
CHUNK = 4 * 1024 * 1024


def check_file(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Wrong size: {path}")
    if "sha256" in spec:
        digest, expected = hashlib.sha256(), spec["sha256"]
    else:
        digest = hashlib.sha1(f"blob {spec['bytes']}\0".encode())
        expected = spec["git_blob_sha1"]
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Wrong digest: {path} (expected {expected})")


def unpack_metadata(target, files):
    wanted = {n: info for n, info in files.items() if n not in SHARDS}
    with tarfile.open(INPUTS / "base-metadata.tar.gz") as archive:
        entries = archive.getmembers()
        if len(entries) != len(wanted) or {m.name for m in entries} != set(wanted):
            raise ValueError("Unexpected metadata archive members")
        for m in entries:
            if not m.isfile() or m.size != wanted[m.name]["bytes"]:
                raise ValueError(f"Invalid metadata member: {m.name}")
            path = target / m.name
            with archive.extractfile(m) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            check_file(path, wanted[m.name])


@contextmanager
def download_cache(output):
    # Cache only this target's four downloads, and resume after interruption.
    cache = output.parent / f".{output.name}.modelscope-partial"
    cache.mkdir(exist_ok=True)
    with (cache / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield cache


def fetch(cache, name, spec):
    partial = cache / name
    if not partial.exists() or partial.stat().st_size < spec["bytes"]:
        url = f"https://modelscope.cn/models/{MODEL}/resolve/master/{name}"
        subprocess.run(["curl", "--fail", "--location", "--show-error", "--progress-bar",
                        "--retry", "3", "--connect-timeout", "30", "--continue-at", "-",
                        "--output", str(partial), url], check=True)
    try:
        check_file(partial, spec)
    except ValueError:
        partial.unlink(missing_ok=True)  # Never reuse a download that failed verification.
        raise
    return partial


def publish(stage, output):
    # mkdir is exclusive even if another process raced us; never replace old files.
    output.mkdir()
    try:
        for p in stage.iterdir():
            p.rename(output / p.name)
    except BaseException:
        shutil.rmtree(output)
        raise


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--modelscope", action="store_true", help="download four shards with curl (~15.23 GB; resumable)")
    src.add_argument("--shards-dir", type=Path, help="use existing Qwen2.5-7B shards in this directory, offline")
    src.add_argument("--verify-only", type=Path, help="verify a complete existing base directory, without writing")
    parser.add_argument("--output", type=Path, help="new, non-existent model directory (required unless verifying)")
    args = parser.parse_args()
    manifest = json.loads((INPUTS / "base-manifest.json").read_text())
    files = manifest["files"]
    if ((manifest["model"], manifest["revision"]) != (MODEL, REVISION)
            or {n for n, s in files.items() if "sha256" in s} != SHARDS
            or any(Path(n).name != n or n in (".", "..") for n in files)):
        parser.error("Unexpected base manifest")
    if args.verify_only:
        if args.output is not None:
            parser.error("--verify-only cannot be combined with --output")
        if (args.verify_only / "adapter_config.json").exists():
            parser.error("An adapter config is present; expected a pure base")
        for name, spec in files.items():
            check_file(args.verify_only / name, spec)
        print(f"Verified {len(files)} files: {args.verify_only}")
        return
    if args.output is None:
        parser.error("provide --output with a new directory name")
    output = args.output.absolute()
    if os.path.lexists(output):
        parser.error(f"Refusing to replace an existing path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qwen-base-", dir=output.parent) as tmp:
        stage = Path(tmp)
        unpack_metadata(stage, files)
        if args.shards_dir is not None:
            for name in sorted(SHARDS):
                src = args.shards_dir / name
                if src.stat().st_size != files[name]["bytes"]:
                    raise ValueError(f"Wrong size: {src}")
                shutil.copyfile(src, stage / name)
                check_file(stage / name, files[name])
            publish(stage, output)
        else:
            with download_cache(output) as cache:
                for name in sorted(SHARDS):
                    print(f"Fetching/verifying {name}", flush=True)
                    os.link(fetch(cache, name, files[name]), stage / name)
                publish(stage, output)
                # Cache and output are on the same filesystem. Hard links avoid a
                # second 15 GB copy; remove only this helper's completed downloads.
                for name in SHARDS:
                    (cache / name).unlink()
    print(f"Prepared {len(files)} verified files: {output}")


if __name__ == "__main__":
    try:
        run()
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"prepare_base: {exc}") from None

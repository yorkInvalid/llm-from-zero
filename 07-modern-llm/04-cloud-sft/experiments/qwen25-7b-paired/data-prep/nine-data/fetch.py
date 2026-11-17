"""Download the exact source files for build.py from ModelScope."""
from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent / "modelscope-raw"
REPO = "https://www.modelscope.cn/datasets/m-a-p/COIG-CQIA"
REV = "9c816ffadbd53ef0149b9b1f7d850cddb5d6ebd4"
MBA = "finance/mba_baike.jsonl"

if not (ROOT / ".git").is_dir():
    subprocess.run(["git", "clone", "--no-checkout", REPO + ".git", str(ROOT)], check=True)
subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", REV],
               env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}, check=True)
pointer = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{MBA}"], text=True)
expected = re.search(r"oid sha256:([0-9a-f]{64})", pointer).group(1)
path = ROOT / MBA
if sha256(path.read_bytes()).hexdigest() != expected:
    part = path.with_suffix(".part")
    subprocess.run(["curl", "-fLsS", "--retry", "3", "-o", str(part), f"{REPO}/resolve/{REV}/{MBA}"], check=True)
    if sha256(part.read_bytes()).hexdigest() != expected:
        raise RuntimeError("ModelScope LFS payload does not match pinned Git object")
    part.replace(path)
print(f"{REPO} @ {REV}")

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -c 'import sys,torch; assert sys.version_info >= (3,10), "需要 Python 3.10+"; print("PyTorch:",torch.__version__,"CUDA:",torch.cuda.is_available(),flush=True)'
if [ ! -x .venv/bin/python ]; then python3 -m venv --system-site-packages .venv; fi
p=.venv/bin/python
"$p" -m pip --disable-pip-version-check install -q 'transformers==4.57.3' 'mcp==1.26.0'
if [ "${1:-}" = "--probe" ]; then exec "$p" demo.py --probe; fi
"$p" download.py
"$p" -u demo.py "$@" | tee last-run.txt

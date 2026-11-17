#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [[ ! -s run/control/completed-training.json ]]; do sleep 3; done
/tmp/pair-venv/bin/python experiment.py generate --arm control --base ../qwen-nine/verified-base --out run/control --prompts prompts.jsonl
touch EARLY_CONTROL_DONE
while [[ ! -e data/paired/VERIFIED ]]; do sleep 3; done
touch data/paired/READY

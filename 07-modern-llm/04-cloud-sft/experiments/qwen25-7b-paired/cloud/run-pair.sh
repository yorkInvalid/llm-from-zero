#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [[ ! -e env-ready ]]; do sleep 4; done
PY=/tmp/pair-venv/bin/python
base=../qwen-nine/verified-base
manifest=../qwen-nine/base-manifest.json
mkdir -p run/control run/paired data/control data/paired
ln -fs ../../../qwen-nine/data/train.jsonl data/control/train.jsonl
ln -fs ../../../qwen-nine/data/validation.jsonl data/control/validation.jsonl
ln -fs ../../../qwen-nine/data/validation.jsonl data/paired/validation.jsonl
for arm in control paired; do
    if [[ "$arm" == paired ]]; then
        while [[ ! -e data/paired/READY ]]; do sleep 4; done
    fi
    if [[ ! -f run/$arm/completed-training.json ]]; then
        echo "TRAIN START $arm $(date -Is)"
        "$PY" experiment.py train --arm "$arm" --base "$base" --base-manifest "$manifest" --data data/$arm --out run/$arm
    fi
done
while [[ ! -s prompts.jsonl ]]; do sleep 4; done
for arm in control paired; do
    stages=(epoch-0.25 epoch-0.5 epoch-0.75 epoch-1 epoch-1.25 epoch-1.5 epoch-1.75 epoch-2)
    if [[ "$arm" == control ]]; then stages=(base "${stages[@]}"); fi
    "$PY" experiment.py generate --arm "$arm" --base "$base" --out run/$arm --prompts prompts.jsonl --stages "${stages[@]}"
done
"$PY" pack.py results
touch FINISHED

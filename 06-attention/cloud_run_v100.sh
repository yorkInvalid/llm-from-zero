#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ownlm-industry-v2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec /root/autodl-tmp/ownlm-venv/bin/python -u train_lm.py train \
  --data-dir data --run-dir runs/04-v100-v2-continue-fanin-fp16-b128 \
  --init-checkpoint initial.pt --context-length 256 \
  --device cuda --precision fp16 --batch-size 128 --epochs 1 \
  --learning-rate 0.0001 \
  --run-label '04 · V100 · 四领域729万token · 自有模型继续预训练' \
  --run-note '从Mac fan_in最佳权重继续；冻结8192词表；8层512维3360万参数；FP16+GradScaler；batch128；Adam和学习率计划重新初始化；旧AI验证集单独对照；性能探针选择batch，非显存占满目标。' \
  --observation-prompts 人工智能 机器学习的主要方法 数据 科学研究的意义

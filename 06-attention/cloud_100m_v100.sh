#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ownlm-industry-v2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec /root/autodl-tmp/ownlm-venv/bin/python -u train_lm.py train \
  --data-dir data-v3-100m --run-dir runs/06-v100-v3-100m-1ep-fp16-b128 \
  --init-checkpoint runs/05-v100-v2-continue-3ep-eval50/last.pt --keep-optimizer-state \
  --context-length 256 --device cuda --precision fp16 --batch-size 128 \
  --epochs 1 --learning-rate 0.0001 --warmup-ratio 0.03 \
  --eval-every 250 --train-probe-windows 1024 --no-full-train-eval \
  --run-label '06 · V100 · 新增1亿token · 自有3360万参数模型 · 1遍' \
  --run-note 'AI/教育/科技/新闻各2500万训练目标token；冻结原8192词表、200篇混合验证及旧AI验证；排除旧训练文档精确重复、留出集长片段重叠。保持8层512维，从05末尾最佳权重和Adam/GradScaler继续，另设1遍warmup+cosine学习率计划。按需读取预分词窗口；只评估固定1024个训练窗口，每250步完整验证并保存last.pt；没有全量train/loss，查看train_probe/loss。预算计成功更新的有效目标token，严格1亿。测试集不参与评估。' \
  --observation-prompts 人工智能 机器学习的主要方法 数据 科学研究的意义

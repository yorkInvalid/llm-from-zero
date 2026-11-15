#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ownlm-industry-v2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec /root/autodl-tmp/ownlm-venv/bin/python -u train_lm.py train \
  --data-dir data --run-dir runs/05-v100-v2-continue-3ep-eval50 \
  --init-checkpoint runs/04-v100-v2-continue-fanin-fp16-b128/best.pt \
  --context-length 256 --device cuda --precision fp16 --batch-size 128 \
  --epochs 3 --learning-rate 0.0001 --eval-every 50 --train-probe-windows 512 \
  --run-label '05 · V100 · 自有3360万参数模型 · 追加3遍 · 每50步验证' \
  --run-note '沿用v2四领域729万token和8192词表，从04最佳权重继续。04未存优化器，本轮重新初始化Adam及3遍学习率计划；此后last.pt完整保存优化器、GradScaler、随机状态与下一批位置。每50步评估完整混合验证集和固定512个训练窗口，每遍另算全量train loss。测试集不参与。' \
  --observation-prompts 人工智能 机器学习的主要方法 数据 科学研究的意义

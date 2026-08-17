#!/usr/bin/env bash
# 生成提交 ZIP：179 张测试影像 -> submission/submission.zip（无子目录）
set -e

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTHONPATH=${PYTHONPATH}:`pwd`

python predict_hin.py \
    --config_path freenet.freenet_1_0_hin \
    --model_dir ./log/hin \
    --out-dir ./submission/pred \
    --zip ./submission/submission.zip \
    --tta none

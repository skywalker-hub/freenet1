#!/usr/bin/env bash
# 全量训练。每 20 个 epoch 在两个验证集上各评一次：
# indomain（航线见过、瓦片没见过）和 val_line（整条航线没见过）。
# 先跑 tools/make_splits.py 生成 splits/。

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTHONPATH=${PYTHONPATH}:`pwd`
config_path='freenet.freenet_1_0_hin'
model_dir='./log/hin/freenet_1_0_open'

python train_hin.py \
    --config_path=${config_path} \
    --model_dir=${model_dir}

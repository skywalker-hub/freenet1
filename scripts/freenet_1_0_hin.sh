#!/usr/bin/env bash
# 全量训练。注意此时 data.test 还指向训练瓦片，评估分数只是过拟合自检，
# 不能当泛化指标——等场景划分做出来之后把 split_file 指过去。

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTHONPATH=${PYTHONPATH}:`pwd`
config_path='freenet.freenet_1_0_hin'
model_dir='./log/hin/freenet_1_0_open'

python train_hin.py \
    --config_path=${config_path} \
    --model_dir=${model_dir}

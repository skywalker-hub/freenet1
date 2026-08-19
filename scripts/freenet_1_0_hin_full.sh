#!/usr/bin/env bash
# 全数据训练：383 张瓦片全部用于训练，不划验证集、不做任何评估。
#
# 什么时候用：方法已经定型、只想榨最后一点分数的时候。代价是训练过程中完全看不见
# 泛化情况，出了问题只能靠线上提交发现，而提交次数有限。日常迭代请用
# scripts/freenet_1_0_hin.sh，它保留两个验证集。
#
# model_dir 与带验证的那版分开。同一个目录下 resume_from_last=True 会读到上一轮
# 的 model-20000.pth，而 global_step 已经等于 num_iters，训练会立刻结束。

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTHONPATH=${PYTHONPATH}:`pwd`
config_path='freenet.freenet_1_0_hin'
model_dir='./log/hin/freenet_1_0_full'

# 训练集从 281 张变成 383 张，一个 epoch 从 141 步变成 191 步。要保持和基线相同的
# 数据遍历量（约 142 遍），步数得同比放大：191 x 142 = 27122，取整到 27200。
# num_iters 和 max_iters 必须相等，否则学习率会提前归零或者结束时没退火完，
# 所以这里用同一个变量喂给两处。
ITERS=27200

python train_hin.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    --val none \
    train.num_iters ${ITERS} \
    learning_rate.params.max_iters ${ITERS}

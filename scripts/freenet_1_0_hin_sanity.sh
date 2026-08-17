#!/usr/bin/env bash
# 链路自检：用 4 张覆盖类别最多的瓦片过拟合。
# 判据是 macro_F1_present 能冲到很高；冲不上去说明数据/标签/损失/指标里有 bug，
# 别急着往全量跑。macro_F1_all 这时必然低，因为另外十几个类不在这 4 张图里。

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTHONPATH=${PYTHONPATH}:`pwd`
config_path='freenet.freenet_1_0_hin'
model_dir='./log/hin/sanity'

python train_hin.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    data.train.params.limit 4 \
    data.train.params.pick "'diverse'" \
    data.train.params.batch_size 2 \
    data.train.params.class_weight "'none'" \
    data.test.params.limit 4 \
    data.test.params.pick "'diverse'" \
    train.num_iters 300 \
    train.eval_interval_epoch 50 \
    train.save_ckpt_interval_epoch 9999 \
    train.resume_from_last False \
    learning_rate.params.max_iters 300

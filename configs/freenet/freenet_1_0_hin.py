NUM_ITERS = 20000
# 整幅瓦片进网络，不做随机裁剪。裁到 256 的话，y0/x0 均匀采样会让覆盖概率呈三角形
# 分布：中心像元每个 epoch 有 99% 概率被采到，四角只有 0.0015%，而评估和提交都是在
# 完整 512 上做的。整图训练消除这个错配，也回到 FreeNet 不做 patch 训练的原意。
CROP = 512
# 512 分辨率下单样本激活约 1.5 GiB（fp32），batch 2 约需 4-6 GiB 显存。
# OOM 就降到 1，同时把 NUM_ITERS 翻倍以保持相同的数据遍历量。
BATCH_SIZE = 2

# 224 波段减去 tools/audit_dataset.py 认定的 27 个低质量波段
IN_CHANNELS = 197
# open 模式：31 个已知类 + 1 个 other（对应测试集的 Unknown）
NUM_CLASSES = 32

_data_params = dict(
    # None = 自动探测：仓库内 dataset2683 或仓库上一级的 dataset2683
    #（远程服务器为 /root/autodl-tmp/dataset2683）。要指定别处就填绝对路径。
    root=None,
    mode='open',
    crop=CROP,
    batch_size=BATCH_SIZE,
    num_workers=4,
    keep_bad_bands=False,
    seed=2333,
)

config = dict(
    model=dict(
        type='FreeNetHIN',
        params=dict(
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='HINTileLoader',
            params=dict(training=True,
                        split_file=None,
                        class_weight='inv_sqrt',
                        **_data_params)
        ),
        # 还没有做场景划分，先指到训练瓦片上，这个分数只能当过拟合自检。
        # 有了 split 文件之后把 split_file 指过去，并保持 class_weight='none'。
        test=dict(
            type='HINTileLoader',
            params=dict(training=False,
                        split_file=None,
                        class_weight='none',
                        **_data_params)
        )
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.0001
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.01,
            power=0.9,
            max_iters=NUM_ITERS),
    ),
    train=dict(
        forward_times=1,
        num_iters=NUM_ITERS,
        eval_per_epoch=True,
        # 一个 epoch 约 383/BATCH_SIZE = 191 步，20000 步约 105 个 epoch。
        # 评估要在完整瓦片上跑，每张读 112MB，所以别评得太勤。
        eval_interval_epoch=20,
        save_ckpt_interval_epoch=10,
        summary_grads=False,
        summary_weights=False,
        eval_after_train=True,
        resume_from_last=True,
    ),
    test=dict(),
)

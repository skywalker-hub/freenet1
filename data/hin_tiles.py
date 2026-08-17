"""HyperImageNet 竞赛子集的 SimpleCV 数据入口。

与 data/dataloader.py 里的三个官方 loader 并列，互不影响：官方那些是
"一幅整景 + 像素指示图采样"，这里是 "多张独立瓦片 + 随机裁剪"。

返回给 SimpleCV 训练循环的是 (image, target, weight)，正好对上
FreeNet.forward(x, y, w) 的签名：
    image   float32 (C, H, W)，已剔坏波段、逐波段标准化、nodata 置零
    target  int64   (H, W)，0..30 对应类别 1..31，31 是 other，255 忽略
    weight  float32 (H, W)，参与损失的像元的权重，忽略位置为 0
"""

import numpy as np
import torch
from simplecv import registry
from torch.utils.data.dataloader import DataLoader

from hin.data import TileDataset, class_pixel_counts, list_pairs
from hin.labels import DEFAULT_BAD_BANDS, IGNORE_INDEX, num_classes


def make_class_weight(pairs, mode, n_cls, scheme):
    """inv_sqrt: 1/sqrt(像元数)，归一化到均值 1。主指标是 macro-F1，不加权稀有类会被淹没。"""
    if scheme in (None, 'none'):
        return None
    if scheme != 'inv_sqrt':
        raise ValueError(f'unknown class_weight scheme {scheme!r}')
    counts = class_pixel_counts(pairs, mode, n_cls)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = w / w.mean()
    w[counts == 0] = 0.0
    return w.astype(np.float32)


class HINTileDataset(TileDataset):
    """在 hin.data.TileDataset 之上，把第三个返回值从文件名换成逐像元权重图。"""

    def __init__(self, pairs, mode, crop, training, bad_bands, seed, class_weight=None):
        super().__init__(pairs, mode, crop, training, bad_bands=bad_bands, seed=seed)
        self.class_weight = None if class_weight is None else torch.as_tensor(
            np.asarray(class_weight, dtype=np.float32))

    def __getitem__(self, idx):
        image, target, _ = super().__getitem__(idx)
        keep = target != IGNORE_INDEX
        if self.class_weight is None:
            weight = keep.float()
        else:
            safe = torch.where(keep, target, torch.zeros_like(target))
            weight = self.class_weight[safe] * keep.float()
        return image, target, weight


@registry.DATALOADER.register('HINTileLoader')
class HINTileLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        if self.crop % 8 != 0:
            raise ValueError(f'crop={self.crop} 必须是 8 的倍数，'
                             f'官方 FreeNet 的 top-down 融合写死了 scale_factor=2')

        n_cls = num_classes(self.mode)
        bad_bands = () if self.keep_bad_bands else DEFAULT_BAD_BANDS
        pairs = list_pairs(self.root, self.split_file, self.limit, self.pick)
        if not pairs:
            raise RuntimeError(f'在 {self.root} 下没有找到影像/标签配对')

        class_weight = make_class_weight(pairs, self.mode, n_cls, self.class_weight) \
            if self.training else None
        dataset = HINTileDataset(pairs, self.mode, self.crop, self.training,
                                 bad_bands, self.seed, class_weight)

        workers = min(self.num_workers, len(pairs))
        super(HINTileLoader, self).__init__(
            dataset,
            batch_size=self.batch_size if self.training else 1,
            shuffle=self.training,
            num_workers=workers,
            pin_memory=True,
            drop_last=self.training and len(pairs) > self.batch_size,
            timeout=0,
            worker_init_fn=None,
            # SimpleCV 的 Iterator 每个 epoch 都会重建 iter()。不开 persistent_workers
            # 的话 worker 进程会跟着反复销毁重建；瓦片少的时候一个 epoch 才几步，
            # 进程启动开销会盖过真正的读盘时间。
            persistent_workers=workers > 0,
        )

    def set_defalut(self):
        self.config.update(dict(
            # None = 自动探测：先找仓库内 dataset2683，再找仓库上一级
            #（远程服务器 /root/autodl-tmp/{freenet1, dataset2683} 的布局）
            root=None,
            split_file=None,
            limit=0,
            pick='head',
            mode='open',
            crop=256,
            batch_size=4,
            num_workers=4,
            keep_bad_bands=False,
            class_weight='none',
            training=True,
            seed=2333,
        ))

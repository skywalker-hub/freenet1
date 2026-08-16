"""多瓦片数据集。

与官方 FreeNet 的 data/ 不同：官方在单幅整景内用像素指示图划分训练/测试，
这里每个样本是一张独立的 512x512x224 瓦片（或它的一个裁剪块）。
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from hin.labels import DEFAULT_BAD_BANDS, IGNORE_INDEX, OTHER_INDEX
from tools.rawtiff import RawTiff, imread

TILE_SIZE = 512


def pick_diverse(names, labels, limit):
    """贪心集合覆盖：每次挑能新增最多类别的瓦片，平局时挑标注率高的。

    本数据集的瓦片普遍类别很少（134 张只含 1 个类），按文件名取前 N 张做自检
    很可能只覆盖到一两个类别，起不到检查作用。
    """
    stats = {}
    for n in names:
        raw = imread(labels[n])
        values, counts = np.unique(raw, return_counts=True)
        present = {int(v) for v in values if v > 0}
        labeled = int(counts[values > 0].sum())
        stats[n] = (present, labeled)

    chosen, covered = [], set()
    remaining = list(names)
    while remaining and len(chosen) < limit:
        best = max(remaining, key=lambda n: (len(stats[n][0] - covered), stats[n][1]))
        chosen.append(best)
        covered |= stats[best][0]
        remaining.remove(best)
    return sorted(chosen)


def list_pairs(root, split_file=None, limit=0, pick='head'):
    """返回 [(image_path, label_path), ...]。"""
    label_dir = os.path.join(root, 'Train_Labels')
    labels = {n: os.path.join(label_dir, n)
              for n in sorted(os.listdir(label_dir)) if n.lower().endswith('.tif')}

    images = {}
    for d in sorted(os.listdir(root)):
        if not d.lower().startswith('train_images'):
            continue
        sub = os.path.join(root, d)
        if not os.path.isdir(sub):
            continue
        for n in sorted(os.listdir(sub)):
            if n.lower().endswith('.tif'):
                images[n] = os.path.join(sub, n)

    names = sorted(set(images) & set(labels))
    if split_file:
        with open(split_file) as fh:
            wanted = set(json.load(fh))
        missing = wanted - set(names)
        if missing:
            raise FileNotFoundError(f'split 里有 {len(missing)} 个名字找不到对应数据，例如 {sorted(missing)[:3]}')
        names = [n for n in names if n in wanted]
    if limit and limit < len(names):
        if pick == 'diverse':
            names = pick_diverse(names, labels, limit)
        elif pick == 'head':
            names = names[:limit]
        else:
            raise ValueError(f'unknown pick strategy {pick!r}')
    return [(images[n], labels[n]) for n in names]


def list_test_images(root):
    test_dir = os.path.join(root, 'test')
    return [os.path.join(test_dir, n) for n in sorted(os.listdir(test_dir))
            if n.lower().endswith('.tif')]


def map_labels(raw, valid, mode):
    """原始标签 -> 训练目标索引。"""
    target = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
    known = raw > 0
    target[known] = (raw[known] - 1).astype(np.uint8)
    if mode == 'open':
        target[(~known) & valid] = OTHER_INDEX
    return target


def normalize_per_sample(image, valid):
    """按本样本自身的有效像元做逐波段标准化，nodata 归零。

    这是第一阶段的简化做法。辐亮度数据跨航线亮度差可达 15 倍，用全局统计量会把
    "这张图偏亮" 当成类别线索；等基线跑通后再对比 per-line / 光谱归一化等方案。
    """
    out = image.astype(np.float32, copy=True)
    if valid.any():
        v = out[valid]
        mean = v.mean(axis=0)
        std = v.std(axis=0)
        out -= mean
        out /= np.maximum(std, 1e-3)
    out[~valid] = 0.0
    return out


class TileDataset(Dataset):
    """训练时随机裁剪，评估时返回整幅瓦片。"""

    def __init__(self, pairs, mode='open', crop=256, training=True,
                 bad_bands=DEFAULT_BAD_BANDS, seed=2333):
        self.pairs = list(pairs)
        self.mode = mode
        self.crop = crop if training else TILE_SIZE
        self.training = training
        self.seed = seed
        self._rs = None

        keep = np.ones(224, dtype=bool)
        if bad_bands:
            keep[list(bad_bands)] = False
        self.keep_bands = np.where(keep)[0]

    @property
    def in_channels(self):
        return len(self.keep_bands)

    def __len__(self):
        return len(self.pairs)

    def _rng(self, idx):
        """训练时用一个持续推进的随机状态，这样同一张瓦片每次被取到的裁剪位置都不同。

        不能用 "按 epoch 重新播种" 的写法：DataLoader 开了 persistent_workers 之后，
        主进程改 dataset 的属性不会同步到 worker，裁剪位置会被永久冻住。
        """
        if not self.training:
            return np.random.RandomState(self.seed + idx)
        if self._rs is None:
            info = torch.utils.data.get_worker_info()
            worker_id = info.id if info is not None else 0
            self._rs = np.random.RandomState((self.seed + 7919 * worker_id) % (2 ** 31 - 1))
        return self._rs

    def __getitem__(self, idx):
        image_path, label_path = self.pairs[idx]
        tif = RawTiff(image_path)
        rng = self._rng(idx)

        crop = min(self.crop, tif.height, tif.width)
        y0 = rng.randint(0, tif.height - crop + 1) if self.training else 0
        x0 = rng.randint(0, tif.width - crop + 1) if self.training else 0

        # 只读裁剪覆盖到的行，避免每个样本都把 112MB 全部读进来
        image = tif.read(y0, y0 + crop)[:, x0:x0 + crop, :]
        image = image[:, :, self.keep_bands]
        label = imread(label_path)[y0:y0 + crop, x0:x0 + crop]

        valid = image.any(axis=2)
        image = normalize_per_sample(image, valid)
        target = map_labels(label, valid, self.mode)

        if self.training:
            k = rng.randint(4)
            if k:
                image = np.rot90(image, k, axes=(0, 1))
                target = np.rot90(target, k, axes=(0, 1))
            if rng.rand() < 0.5:
                image = image[:, ::-1]
                target = target[:, ::-1]

        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        target = torch.from_numpy(np.ascontiguousarray(target)).long()
        return image, target, os.path.basename(image_path)


def class_pixel_counts(pairs, mode, num_classes, exact_valid_mask=False):
    """统计目标索引空间下的像元数，用于类别加权。

    open 模式下 "other" 类的准确计数需要 nodata 掩膜，那要把整幅影像读一遍
    （每张 112MB）。默认走近似路径：把所有标签 0 都算进 other，会高估约 18%
    （nodata 的平均占比），对类别权重的量级影响很小。
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for image_path, label_path in pairs:
        raw = imread(label_path)
        if mode == 'open' and exact_valid_mask:
            valid = RawTiff(image_path).read().any(axis=2)
        else:
            valid = np.ones(raw.shape, dtype=bool)
        target = map_labels(raw, valid, mode)
        hit = target != IGNORE_INDEX
        counts += np.bincount(target[hit].ravel(), minlength=num_classes)
    return counts

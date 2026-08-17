"""测量类别对在光谱上到底分不分得开，以及逐样本归一化对可分性做了什么。

diagnose_classes.py 只能看到"Barren 被判成了 Deciduous Forest"，看不出这是
模型没学会，还是输入里本来就没有区分这两者的信息。这里直接量后者。

对每个类别累积其像元的光谱均值和方差，然后对关心的类别对算一个效应量：

    d = RMS_over_bands( |mu_a - mu_b| / pooled_std )

这是假设各波段独立时的 Mahalanobis 距离。d 远大于 1 表示两类的光谱云基本分开，
分类器有信息可用；d 接近或小于 1 表示两类在输入空间里就重叠，再怎么训练也分不开。

同一组类别对会在两个空间里各算一遍：

    raw    只做波段筛选的原始辐亮度
    norm   训练实际用的输入（逐样本逐波段标准化）

两者之差就是归一化的代价。逐样本归一化是为了消除跨航线 15 倍的亮度差异，但它
用的是每张瓦片自己的均值——如果某个类占了半张图，它就被推到零附近，而另一张图里
占半张的另一个类同样被推到零附近，两者在输入里就变得无法区分。Barren 只出现在
7 张瓦片、平均占 39% 面积，是这个失效模式的头号嫌疑。

用法：
    export PYTHONPATH=${PYTHONPATH}:`pwd`
    python tools/spectral_separability.py --tiles-per-class 5
"""

import argparse
import collections
import os

import numpy as np

from hin.labels import CLASS_NAMES, DEFAULT_BAD_BANDS
from hin.paths import find_tifs, resolve_data_root
from hin.preprocess import band_runs, preprocess, select_bands
from tools.rawtiff import RawTiff, imread

UNKNOWN = 32


def brightness_norm(selected, valid):
    """逐像元除以自己的光谱模长：抹掉亮度、保留光谱形状。

    和逐样本标准化的关键区别是它只看单个像元，不看整幅瓦片的构成。跨航线的
    光照/大气差异主要表现为整条谱线的整体缩放，除以模长就能消掉，而不会因为
    "这张图里 Barren 占了一半" 就把 Barren 推到原点。
    """
    out = selected.copy()
    norm = np.linalg.norm(out, axis=2, keepdims=True)
    np.divide(out, np.maximum(norm, 1e-6), out=out)
    out[~valid] = 0.0
    return out

# 来自 diagnose_classes.py 的实测混淆，格式 (真值, 被判成的类)
CONFUSED_PAIRS = (
    (3, 9),    # Barren      -> Deciduous Forest   51.8%
    (22, 32),  # Fallow      -> Unknown            86.7%
    (17, 32),  # Barley      -> Unknown           100.0%
    (23, 32),  # Strawberry  -> Unknown            70.7%
    (14, 12),  # Cotton      -> Shrubland          82.4%
    (26, 27),  # Citrus      -> Orange             56.7%
    (8, 7),    # Other Hay   -> Grassland          34.9%
)

# 模型学得不错的类别对，用来给上面的数字定标
CONTROL_PAIRS = (
    (1, 4),    # Water     vs Snow          F1 0.55 / 0.45
    (13, 27),  # Corn      vs Orange        F1 0.54 / 0.56
    (9, 10),   # Deciduous vs Evergreen     F1 0.44 / 0.42
    (24, 31),  # Peach     vs Pistachio     F1 0.51 / 0.51
)


class Accumulator:
    """按类别累积逐波段的一阶和二阶矩。"""

    def __init__(self, n_bands):
        self.n = collections.Counter()
        self.s1 = collections.defaultdict(lambda: np.zeros(n_bands, dtype=np.float64))
        self.s2 = collections.defaultdict(lambda: np.zeros(n_bands, dtype=np.float64))

    def add(self, cls, pixels):
        if pixels.size == 0:
            return
        self.n[cls] += pixels.shape[0]
        self.s1[cls] += pixels.sum(axis=0, dtype=np.float64)
        self.s2[cls] += np.einsum('ij,ij->j', pixels, pixels, dtype=np.float64)

    def moments(self, cls):
        n = self.n[cls]
        mu = self.s1[cls] / n
        var = np.maximum(self.s2[cls] / n - mu ** 2, 0.0)
        return mu, np.sqrt(var)

    def distance(self, a, b):
        """各波段独立假设下的 Mahalanobis 距离。"""
        mu_a, sd_a = self.moments(a)
        mu_b, sd_b = self.moments(b)
        pooled = np.sqrt((sd_a ** 2 + sd_b ** 2) / 2)
        ok = pooled > 1e-9
        if not ok.any():
            return 0.0
        return float(np.sqrt((((mu_a - mu_b)[ok] / pooled[ok]) ** 2).mean()))


def pick_tiles(names, classes_of, wanted, per_class, rng):
    """为每个关心的类别挑若干含有它的瓦片，去重后返回。"""
    chosen = []
    for c in wanted:
        holders = [n for n in names if c in classes_of[n]]
        if not holders:
            continue
        rng.shuffle(holders)
        chosen += holders[:per_class]
    return sorted(set(chosen))


def main():
    p = argparse.ArgumentParser(description='类别光谱可分性')
    p.add_argument('--root', default=None)
    p.add_argument('--tiles-per-class', type=int, default=5)
    p.add_argument('--max-pixels', type=int, default=20000,
                   help='每张瓦片每个类别最多采这么多像元，避免大类主导内存')
    p.add_argument('--seed', type=int, default=2333)
    args = p.parse_args()

    root = resolve_data_root(args.root)
    labels = {os.path.basename(x): x for x in find_tifs(os.path.join(root, 'Train_Labels'))}
    images = {}
    for d in sorted(os.listdir(root)):
        sub = os.path.join(root, d)
        if d.lower().startswith('train_images') and os.path.isdir(sub):
            images.update({os.path.basename(x): x for x in find_tifs(sub)})
    names = sorted(set(images) & set(labels))

    classes_of = {}
    for n in names:
        v = np.unique(imread(labels[n]))
        classes_of[n] = {int(x) for x in v if x > 0}

    wanted = sorted({c for pair in CONFUSED_PAIRS + CONTROL_PAIRS for c in pair} - {UNKNOWN})
    rng = np.random.RandomState(args.seed)
    tiles = pick_tiles(names, classes_of, wanted, args.tiles_per_class, rng)
    print(f'采样 {len(tiles)} 张瓦片，覆盖 {len(wanted)} 个关心的类别')

    runs, n_keep = band_runs(DEFAULT_BAD_BANDS)
    spaces = ('raw', 'tile', 'pixel')
    acc = {k: Accumulator(n_keep) for k in spaces}

    for i, name in enumerate(tiles, 1):
        tif = RawTiff(images[name])
        cube = tif.read()
        label = imread(labels[name])
        tile, valid = preprocess(cube, runs, n_keep)
        raw = select_bands(cube, runs, n_keep)
        pixel = brightness_norm(raw, valid)
        flat = {'raw': raw.reshape(-1, n_keep), 'tile': tile.reshape(-1, n_keep),
                'pixel': pixel.reshape(-1, n_keep)}

        present = [int(v) for v in np.unique(label) if v > 0]
        # 有效但未标注的像元，就是训练里被当作 Unknown 的那批
        for cls, mask in [(c, (label == c) & valid) for c in present] + \
                         [(UNKNOWN, (label == 0) & valid)]:
            idx = np.flatnonzero(mask.ravel())
            if idx.size > args.max_pixels:
                idx = rng.choice(idx, args.max_pixels, replace=False)
            for k in spaces:
                acc[k].add(cls, flat[k][idx])
        if i % 10 == 0:
            print(f'  {i}/{len(tiles)}')

    def show(title, pairs):
        print(f'\n{title}')
        print(f'{"类别对":<44}{"raw":>8}{"逐样本":>9}{"逐像元":>9}')
        for a, b in pairs:
            if acc['raw'].n[a] == 0 or acc['raw'].n[b] == 0:
                continue
            d = [acc[k].distance(a, b) for k in spaces]
            tag = f'{CLASS_NAMES[a]} vs {CLASS_NAMES[b]}'
            print(f'{tag:<44}{d[0]:>8.2f}{d[1]:>9.2f}{d[2]:>9.2f}')

    show('实测被混淆的类别对', CONFUSED_PAIRS)
    show('模型学得不错的类别对（定标用）', CONTROL_PAIRS)
    print('\nraw    = 只筛波段的原始辐亮度，可分性的上限'
          '\n逐样本 = 训练现在实际用的输入（每张瓦片按自身均值方差标准化）'
          '\n逐像元 = 候选方案，每个像元除以自己的光谱模长'
          '\n\nd 远大于 1 = 输入里有区分信息；d 接近 1 = 输入里本来就重叠，再训也分不开。')

    print(f'\n{"类别":<24}{"采到像元":>12}')
    for c in sorted(acc['raw'].n):
        print(f'{CLASS_NAMES[c]:<24}{acc["raw"].n[c]:>12,}')


if __name__ == '__main__':
    main()

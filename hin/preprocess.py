"""波段筛选与归一化。

训练、推理、离线分析都必须走这里的同一份实现，否则预处理对不上，
线上分数会莫名低于本地。单独成模块也让只做数据分析的工具不必装训练依赖。
"""

import numpy as np

NUM_BANDS = 224


def band_runs(bad_bands):
    """把要保留的波段压成若干连续区间，返回 (区间元组, 波段数)。

    坏波段是三段连续的水汽吸收/噪声窗口，所以保留的也是连续区间。按区间切片拷贝
    比 fancy indexing 快得多，还能顺带完成 float32 转换。
    """
    bad = set(bad_bands or ())
    keep = [b for b in range(NUM_BANDS) if b not in bad]
    if not keep:
        raise ValueError('bad_bands 把 224 个波段全排除了')
    runs, start = [], keep[0]
    for prev, cur in zip(keep, keep[1:]):
        if cur != prev + 1:
            runs.append((start, prev + 1))
            start = cur
    runs.append((start, keep[-1] + 1))
    return tuple(runs), len(keep)


def select_bands(raw, runs, n_keep):
    out = np.empty(raw.shape[:2] + (n_keep,), dtype=np.float32)
    offset = 0
    for start, stop in runs:
        width = stop - start
        out[:, :, offset:offset + width] = raw[:, :, start:stop]
        offset += width
    return out


def preprocess(raw, runs, n_keep):
    """挑波段 + 按本样本自身的有效像元做逐波段标准化，nodata 归零。

    辐亮度数据跨航线亮度差可达 15 倍，用全局统计量会把 "这张图偏亮" 当成类别
    线索，所以统计量只取自本样本。这是第一阶段的简化做法，等基线跑通后再对比
    per-line / 光谱归一化等方案。
    """
    out = select_bands(raw, runs, n_keep)

    valid = out.any(axis=2)
    n = int(valid.sum())
    if n == 0:
        out[:] = 0.0
        return out, valid

    # nodata 像元全波段为 0，整幅求和就等于有效像元求和，不必物化布尔掩膜的副本
    mean = out.sum(axis=(0, 1), dtype=np.float64) / n
    out -= mean.astype(np.float32)
    # 先把 nodata 归零再平方，平方和里就不含它们的贡献。反过来先平方再解析扣除的话，
    # 扣除项会比真正的平方和还大，抵消误差能到 0.04 个标准差。
    out[~valid] = 0.0
    sq = np.einsum('ijk,ijk->k', out, out, dtype=np.float64)
    out /= np.maximum(np.sqrt(sq / n), 1e-3).astype(np.float32)
    return out, valid

"""测试集推理：影像 -> 提交用的标签数组。

预处理必须和训练时逐字一致（同一批坏波段、同样的逐样本标准化），否则线上分数
会莫名其妙地低于本地验证分数。所以这里直接复用 hin.data 里的函数，不另写一份。
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from hin.data import normalize_per_sample
from hin.labels import DEFAULT_BAD_BANDS, target_index_to_submission_id
from tools.rawtiff import RawTiff

# nodata 像元在提交文件里写 0。训练标签上验证过：影像全零的像元其标签必然是 0，
# 而 0 是评测忽略的取值，写别的类别只会平白增加假阳性。
NODATA_FILL = 0


class ImageDataset(Dataset):
    """只读影像的推理数据集，返回 (image, valid, name)。"""

    def __init__(self, paths, bad_bands=DEFAULT_BAD_BANDS):
        self.paths = list(paths)
        keep = np.ones(224, dtype=bool)
        if bad_bands:
            keep[list(bad_bands)] = False
        self.keep_bands = np.where(keep)[0]

    @property
    def in_channels(self):
        return len(self.keep_bands)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = RawTiff(path).read()[:, :, self.keep_bands]
        valid = image.any(axis=2)
        image = normalize_per_sample(image, valid)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return image, torch.from_numpy(valid), os.path.basename(path)


TTA_CHOICES = ('none', 'flip', 'd4')


def _tta_indices(tta):
    if tta == 'none':
        return (0,)
    if tta == 'flip':
        return (0, 1)
    if tta == 'd4':
        return tuple(range(8))
    raise ValueError(f'unknown tta {tta!r}, 可选 {TTA_CHOICES}')


def _forward_transform(x, i):
    k, flip = divmod(i, 2)
    y = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if flip else y


def _inverse_transform(x, i):
    k, flip = divmod(i, 2)
    y = torch.flip(x, dims=(-1,)) if flip else x
    return torch.rot90(y, -k, dims=(-2, -1))


@torch.no_grad()
def predict_prob(model, image, tta='none', amp=False):
    """返回 (num_classes, H, W) 的平均概率。

    FreeNet 在 eval 模式下 forward 直接返回 softmax，所以多个 TTA 分支可以
    在概率空间上直接取平均。
    """
    batched = image.dim() == 4
    x = image if batched else image.unsqueeze(0)
    total = None
    for i in _tta_indices(tta):
        with torch.autocast('cuda', enabled=amp and x.is_cuda):
            prob = model(_forward_transform(x, i))
        prob = _inverse_transform(prob.float(), i)
        total = prob if total is None else total + prob
    total /= len(_tta_indices(tta))
    return total if batched else total[0]


def prob_to_submission(prob, valid, mode='open', dtype=np.int32):
    """(C, H, W) 概率 -> 提交用的类别 ID 图，nodata 处填 0。"""
    ids = np.asarray(target_index_to_submission_id(mode), dtype=np.int64)
    index = prob.argmax(dim=0).cpu().numpy()
    out = ids[index]
    valid = valid.cpu().numpy() if torch.is_tensor(valid) else np.asarray(valid)
    out[~valid] = NODATA_FILL
    return out.astype(dtype)

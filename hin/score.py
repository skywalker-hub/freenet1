"""按提交文件的口径打分：预测标签图 vs 真值标签图。

取值约定和官方一致——0 是未标注/nodata，不参与统计；1..31 是已知类；32 是 Unknown。

官方只说了主指标是 Macro F1-score，没有公布缺席类别怎么处理。所以这里同时给出
两个口径：macro_f1_present 只平均真值里出现过的类别，macro_f1_all 对全部 32 类
取平均（缺席类别按 0 计入）。自己做验证集时看前者，估计线上分数时看后者更保守。
"""

import numpy as np
import torch

from hin.labels import CLASS_NAMES
from hin.metrics import ConfusionMatrix

NUM_SUBMISSION_CLASSES = 32
UNLABELED = 0
UNKNOWN = 32


def train_label_to_gt(label, valid):
    """训练标签 -> 提交取值空间的真值。

    训练标签的 0 混了两种像元：nodata 黑边，以及有效但未标注的地表。测试集真值把
    后者标成 32(Unknown)，而 0 是评测忽略值。所以拿训练标签直接打分的话，Unknown
    这一类根本不会进入统计，32 类的 macro-F1 里少了最难的那一类，分数会系统性偏高。
    这里按训练时 open 模式的同一套假设把有效未标注像元还原成 32。
    """
    gt = np.asarray(label).astype(np.int64, copy=True)
    gt[(gt == UNLABELED) & np.asarray(valid)] = UNKNOWN
    return gt


def score_arrays(pairs):
    """pairs 是 (gt, pred) 二维数组的可迭代对象，取值都在 0..32。"""
    cm = ConfusionMatrix(NUM_SUBMISSION_CLASSES)
    abstain = 0
    for gt, pred in pairs:
        gt = np.asarray(gt)
        pred = np.asarray(pred)
        if gt.shape != pred.shape:
            raise ValueError(f'尺寸不一致：真值 {gt.shape} vs 预测 {pred.shape}')
        bad = pred[(pred < 0) | (pred > NUM_SUBMISSION_CLASSES)]
        if bad.size:
            raise ValueError(f'预测里有越界取值，例如 {bad[0]}，合法范围是 0..32')
        # 预测 0 表示我们判定该像元是 nodata。真值非 0 却被我们填了 0，说明
        # nodata 掩膜和真值对不上，这些像元一定算错，单独计数好定位问题。
        abstain += int(((pred == UNLABELED) & (gt != UNLABELED)).sum())
        # 平移到 0..31 的索引空间，真值 0 变成忽略值
        t = np.where(gt == UNLABELED, 255, gt - 1).astype(np.int64)
        p = np.clip(pred - 1, 0, NUM_SUBMISSION_CLASSES - 1).astype(np.int64)
        cm.update(torch.from_numpy(t), torch.from_numpy(p), 255)
    res = cm.compute()
    res['abstain'] = abstain
    return res


def format_report(res):
    lines = [f'{"ID":>4} {"class":<20} {"support":>12} {"P":>7} {"R":>7} {"F1":>7}']
    for i in range(NUM_SUBMISSION_CLASSES):
        lines.append(f'{i + 1:>4} {CLASS_NAMES[i + 1]:<20} {res["support"][i]:>12} '
                     f'{res["precision"][i]:>7.4f} {res["recall"][i]:>7.4f} '
                     f'{res["f1"][i]:>7.4f}')
    lines += [
        '',
        f'OA                  = {res["oa"]:.6f}',
        f'macro_F1_present    = {res["macro_f1_present"]:.6f}  '
        f'（只平均真值里出现的 {res["num_present_classes"]} 个类）',
        f'macro_F1_all        = {res["macro_f1_all"]:.6f}  （对全部 32 类平均）',
    ]
    if res.get('abstain'):
        lines.append(f'注意：有 {res["abstain"]:,} 个像元真值非 0 但预测填了 0，'
                     f'说明 nodata 掩膜与真值不一致，这些像元必然算错')
    return '\n'.join(lines)

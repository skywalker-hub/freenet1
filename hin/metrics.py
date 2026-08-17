"""混淆矩阵与 macro-F1。

比赛主指标是 macro-F1。训练集里有 3 个类别只出现在 1 条航线上，验证集很可能
不包含全部类别，所以这里把 "验证集中真实出现过的类别" 和 "全部类别" 两种
macro-F1 都算出来，避免用缺席类别的 0 分互相比较不同划分的结果。
"""

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes, device='cpu'):
        self.num_classes = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @classmethod
    def merge(cls, matrices):
        """把多个验证集的统计合成一个。

        各验证集单独看都缺类，缺的还不是同一批；合起来算 macro_F1_all，
        分母里的零类少了，才谈得上跟官方 179 张全类别测试集的分数对比。
        """
        out = cls(matrices[0].num_classes, device=matrices[0].mat.device)
        for m in matrices:
            out.mat += m.mat.to(out.mat.device)
        return out

    def update(self, target, pred, ignore_index=255):
        keep = target != ignore_index
        if not keep.any():
            return
        t = target[keep].to(self.mat.device).view(-1)
        p = pred[keep].to(self.mat.device).view(-1)
        idx = t * self.num_classes + p
        self.mat += torch.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

    def compute(self):
        mat = self.mat.double().cpu().numpy()
        tp = np.diag(mat)
        support = mat.sum(axis=1)
        predicted = mat.sum(axis=0)

        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

        present = support > 0
        total = mat.sum()
        return dict(
            oa=float(tp.sum() / total) if total else 0.0,
            macro_f1_all=float(f1.mean()),
            macro_f1_present=float(f1[present].mean()) if present.any() else 0.0,
            num_present_classes=int(present.sum()),
            f1=f1,
            precision=precision,
            recall=recall,
            support=support.astype(np.int64),
        )


def format_per_class(result, names, ids=None):
    """ids 是提交文件里的类别 ID（1..31 和 32），跟模型输出索引差 1，分开列出避免看混。"""
    ids = list(range(len(names))) if ids is None else ids
    lines = [f'{"idx":>4} {"提交ID":>6} {"class":<20} '
             f'{"support":>12} {"P":>7} {"R":>7} {"F1":>7}']
    for i, name in enumerate(names):
        lines.append(f'{i:>4} {ids[i]:>6} {name:<20} {result["support"][i]:>12} '
                     f'{result["precision"][i]:>7.4f} {result["recall"][i]:>7.4f} '
                     f'{result["f1"][i]:>7.4f}')
    return '\n'.join(lines)

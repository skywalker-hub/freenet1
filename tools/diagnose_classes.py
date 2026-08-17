"""定位学不会的类别：它们的像元到底被判成了什么。

训练日志只给每类的 P/R/F1，看得到"没学会"，看不到"错到哪去了"。而修法完全取决于
错误的去向：

    被 Unknown 吞掉        -> 开集那条假设太贪，要调 other 类的权重或阈值
    被某个相邻类抢走       -> 采样/权重问题，稀有类进 batch 的机会太少
    两个相似类互相消灭     -> 光谱可分性不足，得加特征或考虑先合并再细分

所以把完整混淆矩阵跑出来，对每个低分类别列出它的真值像元流向了哪几个类。

用法（要能 import 到仓库根下的 predict_hin / hin / module）：
    export PYTHONPATH=${PYTHONPATH}:`pwd`
    python tools/diagnose_classes.py --model_dir ./log/hin/freenet_1_0_open
"""

import argparse

import numpy as np
import torch
from simplecv.core.config import AttrDict
from simplecv.data.data_loader import make_dataloader
from simplecv.util.config import import_config

from data import hin_tiles  # noqa: F401  导入即注册 HINTileLoader
from hin.labels import IGNORE_INDEX, class_name, target_index_to_submission_id
from hin.metrics import ConfusionMatrix
from module import freenet_hin  # noqa: F401  导入即注册 FreeNetHIN
from predict_hin import find_checkpoint, load_model

# data 下会被当作验证集跑的键，以及日志里对应的名字
VAL_KEYS = (('test', 'indomain'), ('val_line', 'val_line'))


def accumulate(model, loader, n_cls, device):
    cm = ConfusionMatrix(n_cls, device=device)
    with torch.no_grad():
        for image, target, _ in loader:
            prob = model(image.to(device, non_blocking=True))
            cm.update(target.to(device, non_blocking=True), prob.argmax(dim=1), IGNORE_INDEX)
    return cm


def summarize(cm, tag, n_cls, mode):
    res = cm.compute()
    print(f'\n[{tag}] OA={res["oa"]:.4f}  macro_F1_all={res["macro_f1_all"]:.4f}  '
          f'macro_F1_present={res["macro_f1_present"]:.4f}  '
          f'present={res["num_present_classes"]}/{n_cls}')
    return res


def report_flows(cm, res, mode, n_cls, threshold, topk):
    """对每个低分类别，列出它的真值像元被预测成了什么。"""
    mat = cm.mat.cpu().numpy()
    ids = target_index_to_submission_id(mode)
    failing = [c for c in range(n_cls)
               if res['support'][c] > 0 and res['f1'][c] < threshold]
    if not failing:
        print(f'\n没有 F1 低于 {threshold} 的类别')
        return

    print(f'\n{"=" * 72}\nF1 < {threshold} 的 {len(failing)} 个类别，真值像元的去向')
    stolen = np.zeros(n_cls, dtype=np.int64)
    for c in failing:
        row = mat[c]
        total = row.sum()
        print(f'\n{ids[c]:>3} {class_name(mode, c)}  '
              f'support={total:,}  F1={res["f1"][c]:.4f}')
        for p in np.argsort(-row)[:topk]:
            if row[p] == 0:
                break
            mark = ' <- 正确' if p == c else ''
            print(f'      {row[p] / total:>6.1%}  判成 {ids[p]:>2} {class_name(mode, p)}{mark}')
            if p != c:
                stolen[p] += row[p]

    print(f'\n{"=" * 72}\n吸收这些像元最多的类别')
    for p in np.argsort(-stolen)[:8]:
        if stolen[p] == 0:
            break
        # 该类自己的预测里，有多大比例其实是从失败类别那里抢来的
        predicted = mat[:, p].sum()
        print(f'  {ids[p]:>3} {class_name(mode, p):<20} 吸收 {stolen[p]:>11,} '
              f'（占它全部预测的 {stolen[p] / max(predicted, 1):.1%}）')


def main():
    p = argparse.ArgumentParser(description='诊断学不会的类别')
    p.add_argument('--config_path', default='freenet.freenet_1_0_hin')
    p.add_argument('--model_dir', default='./log/hin/freenet_1_0_open')
    p.add_argument('--ckpt', default=None, help='直接指定 .pth，优先于 --model_dir')
    p.add_argument('--threshold', type=float, default=0.05,
                   help='F1 低于此值的类别算作"学不会"')
    p.add_argument('--topk', type=int, default=5, help='每个类别列出前几个去向')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    cfg = AttrDict.from_dict(import_config(args.config_path))
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    ckpt_path = find_checkpoint(args.model_dir, args.ckpt)
    model, step = load_model(cfg, ckpt_path, device)
    n_cls = cfg['model']['params']['num_classes']
    mode = 'open' if n_cls == 32 else 'closed'
    print(f'权重 {ckpt_path}（global_step={step}）  设备 {device}')

    mats = []
    for key, tag in VAL_KEYS:
        if key not in cfg['data']:
            continue
        loader = make_dataloader(cfg['data'][key])
        cm = accumulate(model, loader, n_cls, device)
        summarize(cm, tag, n_cls, mode)
        mats.append(cm)

    if not mats:
        raise RuntimeError('配置里没有任何验证集')

    # 单个验证集各自缺类，合并后覆盖面最广，最接近线上的 32 类口径
    merged = ConfusionMatrix.merge(mats) if len(mats) > 1 else mats[0]
    res = summarize(merged, 'combined', n_cls, mode)
    report_flows(merged, res, mode, n_cls, args.threshold, args.topk)


if __name__ == '__main__':
    main()

"""给已有的预测结果打分，不需要模型也不需要 GPU。

预测可以是目录，也可以直接是提交用的 ZIP——后者顺便验证了压缩包本身能不能被
正常解开、有没有多余的子目录。

    python tools/score_submission.py --pred submission/submission.zip \
        --gt dataset2683/Train_Labels
"""

import argparse
import os
import tempfile
import zipfile

from hin.data import find_tifs
from hin.score import format_report, score_arrays, train_label_to_gt
from tools.rawtiff import RawTiff, imread


def collect(pred, workdir):
    """返回 {文件名: 路径}。pred 是 ZIP 时先解开并检查有无子目录。"""
    if os.path.isdir(pred):
        return {os.path.basename(p): p for p in find_tifs(pred)}
    if not zipfile.is_zipfile(pred):
        raise ValueError(f'{pred} 既不是目录也不是 ZIP')
    with zipfile.ZipFile(pred) as zf:
        names = zf.namelist()
        nested = [n for n in names if '/' in n or '\\' in n]
        if nested:
            raise RuntimeError(f'压缩包里有子目录，官方评测可能失败：{nested[:3]}')
        zf.extractall(workdir)
    return {n: os.path.join(workdir, n) for n in sorted(names)
            if n.lower().endswith(('.tif', '.tiff'))}


def main():
    p = argparse.ArgumentParser(description='按官方口径给预测标签打分')
    p.add_argument('--pred', required=True, help='预测目录或提交 ZIP')
    p.add_argument('--gt', required=True, help='真值标签目录')
    p.add_argument('--images', default=None,
                   help='真值来自 Train_Labels 时给影像目录（可以直接给 dataset2683，'
                        '会递归查找）：据此区分 nodata 与有效未标注像元，'
                        '把后者按 32(Unknown) 计分')
    args = p.parse_args()

    gts = {os.path.basename(p): p for p in find_tifs(args.gt)}
    images = {os.path.basename(p): p for p in find_tifs(args.images)} if args.images else {}

    with tempfile.TemporaryDirectory() as workdir:
        preds = collect(args.pred, workdir)
        if not preds:
            raise FileNotFoundError(f'{args.pred} 里没有 TIF')

        pairs, missing = [], []
        for name, path in preds.items():
            if name not in gts:
                missing.append(name)
                continue
            gt = imread(gts[name])
            if args.images:
                valid = RawTiff(images[name]).read().any(axis=2)
                gt = train_label_to_gt(gt, valid)
            pairs.append((gt, imread(path)))

        if missing:
            print(f'{len(missing)} 张没有对应真值，已跳过，例如 {missing[:3]}')
        if not pairs:
            raise FileNotFoundError('没有任何预测能和真值对上，检查 --gt 路径')
        note = '有效未标注像元按 Unknown(32) 计分' if args.images \
            else '真值按原样使用（未标注像元不参与统计）'
        print(f'比对 {len(pairs)} 张，{note}\n')
        print(format_report(score_arrays(pairs)))


if __name__ == '__main__':
    main()

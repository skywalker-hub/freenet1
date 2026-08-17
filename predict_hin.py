"""生成提交用的 ZIP。

官方要求：压缩包里是测试集每张影像对应的一个预测标签 TIF，**不能有子文件夹**，
否则评测可能失败并占用提交次数。这个脚本负责保证这一点，并在打包前做一遍自检。

典型用法：
    python predict_hin.py --config_path freenet.freenet_1_0_hin --model_dir ./log/hin

在自己划分的验证集上打分：
    python predict_hin.py --config_path freenet.freenet_1_0_hin --model_dir ./log/hin \
        --split-file splits/val_indomain.json --no-zip
"""

import argparse
import json
import os
import time
import zipfile

import numpy as np
import torch
from simplecv.module.model_builder import make_model
from simplecv.util.config import import_config

from hin.data import find_tifs, list_pairs, resolve_data_root
from hin.labels import DEFAULT_BAD_BANDS
from hin.predict import TTA_CHOICES, ImageDataset, predict_prob, prob_to_submission
from hin.score import format_report, score_arrays, train_label_to_gt
from module import freenet_hin  # noqa: F401  导入即注册 FreeNetHIN
from tools.rawtiff import RawTiff, imread, imwrite

DTYPES = {'int32': np.int32, 'uint8': np.uint8}


def parse_args():
    p = argparse.ArgumentParser(description='HyperImageNet 竞赛子集推理与提交打包')
    p.add_argument('--config_path', default='freenet.freenet_1_0_hin',
                   help='configs/ 下的配置模块名，必须和训练时用的一致')
    p.add_argument('--model_dir', default='./log/hin',
                   help='训练输出目录，从中读取最新 checkpoint')
    p.add_argument('--ckpt', default=None, help='直接指定 .pth，优先于 --model_dir')
    p.add_argument('--image-dir', default=None,
                   help='待推理影像目录，默认是数据根目录下的 test/')
    p.add_argument('--split-file', default=None,
                   help='验证集划分 JSON（文件名列表）。给了就从训练集里取这些瓦片，'
                        '影像和真值都自动定位，训练影像分散在多个目录也没关系')
    p.add_argument('--out-dir', default='./submission/pred', help='预测 TIF 的输出目录')
    p.add_argument('--zip', dest='zip_path', default='./submission/submission.zip')
    p.add_argument('--no-zip', dest='zip_path', action='store_const', const=None,
                   help='只出 TIF 不打包，本地评估时用')
    p.add_argument('--gt-dir', default=None,
                   help='给了就顺便本地打分，目录里需有同名真值 TIF')
    p.add_argument('--gt-is-train-label', action='store_true',
                   help='真值来自 Train_Labels：把有效但未标注的像元当作 32(Unknown) 计分，'
                        '否则 Unknown 这一类不参与统计，macro-F1 会偏高。'
                        '用 --split-file 时自动开启')
    p.add_argument('--gt-raw', action='store_true',
                   help='配合 --split-file：真值按原样使用，不还原 Unknown')
    p.add_argument('--tta', default='none', choices=TTA_CHOICES,
                   help='none=单次前向；flip=加水平翻转；d4=8 个旋转翻转全做')
    p.add_argument('--dtype', default='int32', choices=sorted(DTYPES),
                   help='输出 TIF 的位深，int32 与官方标签一致')
    p.add_argument('--amp', action='store_true', help='半精度推理，省显存')
    p.add_argument('--limit', type=int, default=0, help='只跑前 N 张，用于试运行')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def find_checkpoint(model_dir, explicit=None):
    if explicit:
        return explicit
    info = os.path.join(model_dir, 'checkpoint_info.json')
    if not os.path.isfile(info):
        raise FileNotFoundError(f'{info} 不存在，先训练或用 --ckpt 指定权重')
    with open(info) as fh:
        name = json.load(fh)['last']['name']
    if not name:
        raise FileNotFoundError(f'{info} 里没有记录任何 checkpoint')
    return os.path.join(model_dir, name)


def load_model(config, ckpt_path, device):
    model = make_model(config['model'])
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    # 训练走 dp_train，权重可能带 DataParallel 的 module. 前缀
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, int(ckpt.get('global_step', -1))


def main():
    args = parse_args()
    config = import_config(args.config_path)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    test_params = config['data']['test']['params']
    mode = test_params.get('mode', 'open')
    bad_bands = () if test_params.get('keep_bad_bands', False) else DEFAULT_BAD_BANDS

    root = resolve_data_root(test_params.get('root'))
    if args.split_file:
        # 训练影像分散在 Train_Images01..05，交给 list_pairs 去配对
        source = args.split_file
        pairs = list_pairs(root, args.split_file)
        paths = [image for image, _ in pairs]
        gt_map = {os.path.basename(image): label for image, label in pairs}
        gt_is_train_label = not args.gt_raw
    else:
        source = args.image_dir or os.path.join(root, 'test')
        paths = find_tifs(source)
        gt_map = {os.path.basename(p): os.path.join(args.gt_dir, os.path.basename(p))
                  for p in paths} if args.gt_dir else {}
        gt_is_train_label = args.gt_is_train_label
    if not paths:
        raise FileNotFoundError(f'{source} 里没有找到待推理的 TIF（子目录也找过了）')
    if args.limit:
        paths = paths[:args.limit]

    dataset = ImageDataset(paths, bad_bands)
    n_cls = config['model']['params']['num_classes']
    if dataset.in_channels != config['model']['params']['in_channels']:
        raise ValueError(f'波段数对不上：数据 {dataset.in_channels} vs 模型 '
                         f'{config["model"]["params"]["in_channels"]}，'
                         f'检查 keep_bad_bands 是否和训练时一致')

    ckpt_path = find_checkpoint(args.model_dir, args.ckpt)
    model, step = load_model(config, ckpt_path, device)
    print(f'权重 {ckpt_path}（global_step={step}）  设备 {device}  TTA {args.tta}')
    print(f'待推理 {len(paths)} 张，模式 {mode}，{n_cls} 类，输入 {dataset.in_channels} 波段')

    os.makedirs(args.out_dir, exist_ok=True)
    dtype = DTYPES[args.dtype]
    written, start = [], time.time()
    for i in range(len(dataset)):
        image, valid, name = dataset[i]
        prob = predict_prob(model, image.to(device), args.tta, args.amp)
        pred = prob_to_submission(prob, valid, mode, dtype)
        out_path = os.path.join(args.out_dir, name)
        imwrite(out_path, pred)
        written.append((name, out_path, valid.numpy()))
        if (i + 1) % 20 == 0 or i + 1 == len(dataset):
            done = time.time() - start
            print(f'  {i + 1}/{len(dataset)}  {done / (i + 1):.2f} s/张  已用 {done / 60:.1f} 分钟')

    verify(written, paths, dtype)

    if gt_map:
        report(written, gt_map, gt_is_train_label)

    if args.zip_path:
        pack(written, args.zip_path)


def verify(written, paths, dtype):
    """打包前自检：数量、尺寸、取值范围。提交次数有限，别把明显的错误传上去。"""
    if len(written) != len(paths):
        raise RuntimeError(f'写出 {len(written)} 张，应有 {len(paths)} 张')
    hist = np.zeros(33, dtype=np.int64)
    for name, path, _ in written:
        arr = imread(path)
        if arr.shape != (512, 512):
            raise RuntimeError(f'{name} 尺寸是 {arr.shape}，应为 (512, 512)')
        if arr.dtype != dtype:
            raise RuntimeError(f'{name} dtype 是 {arr.dtype}，应为 {np.dtype(dtype)}')
        if arr.min() < 0 or arr.max() > 32:
            raise RuntimeError(f'{name} 取值超出 0..32：[{arr.min()}, {arr.max()}]')
        compression = RawTiff(path).compression
        if compression != 1:
            raise RuntimeError(
                f'{name} TIFF 压缩码是 {compression}，必须是 1（无压缩）。'
                f'CodaLab 的 tifffile 没有 imagecodecs，LZW 会读失败')
        hist += np.bincount(arr.ravel().astype(np.int64), minlength=33)
    used = [c for c in range(1, 33) if hist[c]]
    total = hist[1:].sum()
    print(f'\n自检通过：{len(written)} 张 512x512 {np.dtype(dtype)} 无压缩 TIFF，取值合法')
    print(f'  预测里出现了 {len(used)} 个类别：{used}')
    print(f'  nodata(0) 占 {hist[0] / hist.sum():.1%}，Unknown(32) 占有效像元的 '
          f'{hist[32] / max(total, 1):.1%}')


def report(written, gt_map, gt_is_train_label):
    pairs, missing = [], []
    for name, path, valid in written:
        gt_path = gt_map.get(name)
        if not gt_path or not os.path.isfile(gt_path):
            missing.append(name)
            continue
        gt = imread(gt_path)
        if gt_is_train_label:
            gt = train_label_to_gt(gt, valid)
        pairs.append((gt, imread(path)))
    if missing:
        print(f'\n[打分] {len(missing)} 张没有找到真值，已跳过')
    if not pairs:
        print('[打分] 没有可比对的真值，跳过')
        return
    note = '有效未标注像元已按 Unknown(32) 计分' if gt_is_train_label \
        else '真值按原样使用（未标注像元不参与统计）'
    print(f'\n[打分] 基于 {len(pairs)} 张真值，{note}：\n')
    print(format_report(score_arrays(pairs)))


def pack(written, zip_path):
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)) or '.', exist_ok=True)
    # ZIP 容器用 deflate 没问题；里面的 TIF 必须是无压缩（Compression=1）。
    # 评测失败过的那种 LZW 是 TIFF 编码，不是 ZIP 算法。
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, path, _ in written:
            # arcname 只给文件名，保证压缩包里没有任何子目录
            zf.write(path, arcname=name)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    nested = [n for n in names if '/' in n or '\\' in n]
    if nested:
        raise RuntimeError(f'压缩包里出现了子目录：{nested[:3]}')
    size = os.path.getsize(zip_path) / 2 ** 20
    print(f'\n已打包 {zip_path}（{len(names)} 个文件，{size:.1f} MiB，无子目录）')


if __name__ == '__main__':
    main()

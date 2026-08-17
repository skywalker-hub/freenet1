"""第一阶段的最小训练入口：把 FreeNet 在本数据集上跑起来。

先用 --smoke 确认模型能前反向、显存够用：
    python -m hin.train --smoke --crop 256 --batch-size 2

再用一两张图做过拟合自检，macro-F1 应该能冲到很高，说明数据、标签、损失、
指标这条链路是通的：
    python -m hin.train --limit 2 --crop 256 --batch-size 2 --iters 300 --eval-every 100

然后再放开到全量：
    python -m hin.train --crop 256 --batch-size 4 --iters 20000 --eval-every 1000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hin.data import TileDataset, class_pixel_counts, list_pairs, resolve_data_root
from hin.labels import DEFAULT_BAD_BANDS, IGNORE_INDEX, class_name, num_classes
from hin.metrics import ConfusionMatrix, format_per_class
from hin.model import build_model


def make_scaler(enabled):
    """torch 2.4 起 GradScaler 从 torch.cuda.amp 挪到了 torch.amp。"""
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=None,
                   help='数据根目录；不传则自动探测仓库内或仓库上一级的 dataset2683')
    p.add_argument('--out', default='runs/freenet_stage1')
    p.add_argument('--mode', choices=['closed', 'open'], default='open',
                   help='closed=31 类忽略标签 0；open=32 类，把有效但未标注的像元当作 other')
    p.add_argument('--split-file', default=None, help='训练用的瓦片名列表 json')
    p.add_argument('--val-split-file', default=None,
                   help='验证用的瓦片名列表 json；不给就在训练瓦片上评估（只能当过拟合自检）')
    p.add_argument('--limit', type=int, default=0, help='只取 N 张瓦片，0 表示全部')
    p.add_argument('--pick', choices=['head', 'diverse'], default='head',
                   help='配合 --limit：head 按文件名取前 N 张，'
                        'diverse 贪心挑覆盖类别最多的 N 张（自检时用这个）')
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw')
    p.add_argument('--class-weight', choices=['none', 'inv_sqrt'], default='none')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=0, help='0 表示只在结束时评估一次')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--keep-bad-bands', action='store_true', help='不剔除低质量波段')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=2333)
    p.add_argument('--smoke', action='store_true', help='用随机张量跑一次前反向，检查结构与显存')
    return p.parse_args()


def smoke_test(args, device):
    in_channels = 224 if args.keep_bad_bands else 224 - len(DEFAULT_BAD_BANDS)
    n_cls = num_classes(args.mode)
    model = build_model(in_channels, n_cls).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'in_channels={in_channels}  num_classes={n_cls}  params={n_params / 1e6:.2f}M')

    x = torch.randn(args.batch_size, in_channels, args.crop, args.crop, device=device)
    y = torch.randint(0, n_cls, (args.batch_size, args.crop, args.crop), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = make_scaler(args.amp and device.type == 'cuda')
    start = time.time()
    with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
        logit = model(x)
        loss = F.cross_entropy(logit, y, ignore_index=IGNORE_INDEX)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f'输出 {tuple(logit.shape)}  loss {loss.item():.4f}  '
          f'一次前反向 {time.time() - start:.2f}s')
    if device.type == 'cuda':
        print(f'峰值显存 {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB')


@torch.no_grad()
def evaluate(model, loader, n_cls, device, amp):
    model.eval()
    cm = ConfusionMatrix(n_cls, device=device)
    for image, target, _ in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
            logit = model(image)
        cm.update(target, logit.argmax(dim=1), IGNORE_INDEX)
    model.train()
    return cm.compute()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')

    if args.smoke:
        smoke_test(args, device)
        return

    os.makedirs(args.out, exist_ok=True)
    n_cls = num_classes(args.mode)
    bad_bands = () if args.keep_bad_bands else DEFAULT_BAD_BANDS

    root = resolve_data_root(args.root)
    print(f'数据根目录: {root}')
    pairs = list_pairs(root, args.split_file, args.limit, args.pick)
    if not pairs:
        raise RuntimeError(f'在 {root} 下没有找到影像/标签配对')
    if args.val_split_file:
        val_pairs = list_pairs(root, args.val_split_file)
    else:
        val_pairs = pairs
        print('警告：没有给 --val-split-file，评估跑在训练瓦片上，只能用来做过拟合自检')
    print(f'训练瓦片 {len(pairs)} 张，验证瓦片 {len(val_pairs)} 张，'
          f'mode={args.mode}，输出 {n_cls} 类，剔除 {len(bad_bands)} 个波段')

    workers = min(args.workers, max(len(pairs), 1))
    train_set = TileDataset(pairs, args.mode, args.crop, training=True,
                            bad_bands=bad_bands, seed=args.seed)
    eval_set = TileDataset(val_pairs, args.mode, args.crop, training=False,
                           bad_bands=bad_bands, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True,
                              drop_last=len(pairs) > args.batch_size,
                              persistent_workers=workers > 0)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False,
                             num_workers=min(workers, 2), pin_memory=True)
    if len(train_loader) == 0:
        raise RuntimeError(f'batch_size={args.batch_size} 大于瓦片数 {len(pairs)}，训练集为空')

    model = build_model(train_set.in_channels, n_cls).to(device)
    print(f'in_channels={train_set.in_channels}  '
          f'params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    weight = None
    if args.class_weight == 'inv_sqrt':
        counts = class_pixel_counts(pairs, args.mode, n_cls)
        w = 1.0 / np.sqrt(np.maximum(counts, 1))
        w = w / w.mean()
        w[counts == 0] = 0.0
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        print('类别权重: ' + ' '.join(f'{v:.2f}' for v in w))

    if args.optimizer == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (1 - it / args.iters) ** 0.9)
    scaler = make_scaler(args.amp and device.type == 'cuda')

    history = []
    step, running, t0 = 0, 0.0, time.time()
    model.train()
    while step < args.iters:
        for image, target, _ in train_loader:
            if step >= args.iters:
                break
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type,
                                enabled=args.amp and device.type == 'cuda'):
                logit = model(image)
                loss = F.cross_entropy(logit, target, weight=weight,
                                       ignore_index=IGNORE_INDEX)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += loss.item()
            step += 1

            if step % args.log_every == 0:
                print(f'iter {step:>6}/{args.iters}  loss {running / args.log_every:.4f}  '
                      f'lr {sched.get_last_lr()[0]:.2e}  '
                      f'{(time.time() - t0) / args.log_every:.2f}s/it', flush=True)
                running, t0 = 0.0, time.time()

            if args.eval_every and step % args.eval_every == 0:
                res = evaluate(model, eval_loader, n_cls, device, args.amp)
                print(f'  [eval @ {step}] OA {res["oa"]:.4f}  '
                      f'macroF1(出现的{res["num_present_classes"]}类) '
                      f'{res["macro_f1_present"]:.4f}  '
                      f'macroF1(全{n_cls}类) {res["macro_f1_all"]:.4f}', flush=True)
                history.append(dict(step=step, oa=res['oa'],
                                    macro_f1_present=res['macro_f1_present'],
                                    macro_f1_all=res['macro_f1_all']))

    res = evaluate(model, eval_loader, n_cls, device, args.amp)
    names = [class_name(args.mode, i) for i in range(n_cls)]
    print('\n' + format_per_class(res, names))
    print(f'\nOA {res["oa"]:.4f}  macroF1(出现的{res["num_present_classes"]}类) '
          f'{res["macro_f1_present"]:.4f}  macroF1(全{n_cls}类) {res["macro_f1_all"]:.4f}')

    torch.save(dict(model=model.state_dict(), args=vars(args),
                    in_channels=train_set.in_channels, num_classes=n_cls,
                    bad_bands=list(bad_bands)),
               os.path.join(args.out, 'last.pth'))
    with open(os.path.join(args.out, 'metrics.json'), 'w') as fh:
        json.dump(dict(history=history,
                       final=dict(oa=res['oa'],
                                  macro_f1_present=res['macro_f1_present'],
                                  macro_f1_all=res['macro_f1_all'],
                                  per_class_f1={names[i]: float(res['f1'][i])
                                                for i in range(n_cls)})),
                  fh, indent=1, ensure_ascii=False)
    print(f'\n已保存到 {args.out}')


if __name__ == '__main__':
    main()

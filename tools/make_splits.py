"""生成训练/验证划分。

产出三个 JSON 文件名列表，供 list_pairs(split_file=...) 和 predict_hin.py --split-file 使用：

    splits/train.json           训练
    splits/val_indomain.json    同域验证：航线仍在训练集里，但瓦片空间上不重叠
    splits/val_line.json        留出航线验证：整条航线不参与训练

为什么要两份验证集：测试集本身是混合的。60 条测试航线里 42 条也是训练航线，
179 张测试瓦片里 41 张与训练瓦片空间重叠（最高 99%）；剩下 18 条航线训练里没见过。
线上分数是这两种情况的加权混合，一个验证集代表不了，两者的差距正好量化域偏移强度。

三条硬约束：

1. 切分单位是空间组，不是瓦片。瓦片是从航线上滑窗切出来的，文件名里的
   Sx/Sy/Ex/Ey 就是切割窗口，相邻窗口有重叠。按瓦片随机切会让验证瓦片含有
   训练瓦片见过的像元，分数虚高。
2. 每个类别必须在训练集里保留至少一条航线。Tomato/Peach/Olive 各自只出现在
   一条航线上，整条留出去就等于该类零训练数据，macro-F1 白送 3/32。
3. 同域验证的组，其所属航线必须还有别的组留在训练集里，否则它就变成留出航线了。

约束 2 只要求类别"存在"是不够的。Strawberry 全数据集才 6 万像元，随便一张瓦片
就占掉大半；只查存在性会切出训练 4 千、验证 5 万的划分，等于为了测这个类而放弃学它。
所以约束是按像元比例的：每个类至少 --min-keep 比例的像元留在训练集。稀有类因此
基本进不了验证集——宁可训练好也不去测量它，反正 macro-F1 只算训练目标。
"""

import argparse
import collections
import json
import os
import re

import numpy as np

from hin.labels import CLASS_NAMES
from hin.paths import find_tifs, resolve_data_root
from tools.rawtiff import imread

BOX = re.compile(r'_Sx_(\d+)_Sy_(\d+)_Ex_(\d+)_Ey_(\d+)\.tiff?$', re.I)


def parse_name(name):
    m = BOX.search(name)
    if not m:
        raise ValueError(f'文件名里解析不出切割坐标: {name}')
    sx, sy, ex, ey = (int(v) for v in m.groups())
    return name[:m.start()], (sx, sy, ex, ey)


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def spatial_groups(names, lines, boxes):
    """同一航线内空间相邻（切割窗口相交）的瓦片并成一组。"""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_line = collections.defaultdict(list)
    for n in names:
        by_line[lines[n]].append(n)
    for group in by_line.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if overlaps(boxes[a], boxes[b]):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb

    merged = collections.defaultdict(list)
    for n in names:
        merged[find(n)].append(n)
    return [sorted(v) for v in merged.values()]


def load_counts(label_paths):
    """每张瓦片各类别的像元数。"""
    out = {}
    for name, path in label_paths.items():
        values, counts = np.unique(imread(path), return_counts=True)
        out[name] = collections.Counter(
            {int(v): int(c) for v, c in zip(values, counts) if v > 0})
    return out


class Budget:
    """跟踪已被抽走的像元，保证每类留在训练集里的比例不低于 min_keep。"""

    def __init__(self, counts, min_keep):
        self.counts = counts
        self.total = sum(counts.values(), collections.Counter())
        self.allowance = {c: (1.0 - min_keep) * n for c, n in self.total.items()}
        self.taken = collections.Counter()

    def affordable(self, names):
        probe = collections.Counter(self.taken)
        for n in names:
            probe += self.counts[n]
        return all(probe[c] <= self.allowance[c] for c in self.total)

    def take(self, names):
        for n in names:
            self.taken += self.counts[n]


def main():
    p = argparse.ArgumentParser(description='生成训练/验证划分')
    p.add_argument('--root', default=None, help='数据根目录，默认自动探测')
    p.add_argument('--out-dir', default='./splits')
    p.add_argument('--val-line-tiles', type=int, default=50,
                   help='留出航线验证集的瓦片预算')
    p.add_argument('--val-indomain-tiles', type=int, default=50,
                   help='同域验证集的瓦片预算')
    p.add_argument('--min-keep', type=float, default=0.75,
                   help='每个类别至少这么大比例的像元留在训练集')
    p.add_argument('--seed', type=int, default=2333)
    args = p.parse_args()

    root = resolve_data_root(args.root)
    label_paths = {os.path.basename(x): x for x in find_tifs(os.path.join(root, 'Train_Labels'))}
    names = sorted(label_paths)
    lines, boxes = {}, {}
    for n in names:
        lines[n], boxes[n] = parse_name(n)

    counts = load_counts(label_paths)
    groups = spatial_groups(names, lines, boxes)
    tiles_of_line = collections.defaultdict(list)
    for n in names:
        tiles_of_line[lines[n]].append(n)
    budget = Budget(counts, args.min_keep)

    print(f'{len(names)} 张瓦片，{len(tiles_of_line)} 条航线，{len(groups)} 个空间组，'
          f'{len(budget.total)} 个已知类；每类至少 {args.min_keep:.0%} 像元留在训练集')

    rng = np.random.RandomState(args.seed)

    # 留出航线：整条线一起走
    order = [l for l in sorted(tiles_of_line)]
    rng.shuffle(order)
    held_lines = []
    for line in order:
        if sum(len(tiles_of_line[l]) for l in held_lines) >= args.val_line_tiles:
            break
        tiles = tiles_of_line[line]
        # 允许略微超预算，否则大航线永远进不来；但别让单条线把预算撑爆
        if held_lines and len(tiles) > args.val_line_tiles * 0.5:
            continue
        if not budget.affordable(tiles):
            continue
        budget.take(tiles)
        held_lines.append(line)
    val_line = sorted(n for l in held_lines for n in tiles_of_line[l])

    # 同域：从没被留出的航线里挑空间组，且该航线得还有别的组留在训练集
    groups_left = collections.Counter()
    pool = []
    for g in groups:
        if lines[g[0]] in set(held_lines):
            continue
        groups_left[lines[g[0]]] += 1
        pool.append(g)
    rng.shuffle(pool)

    val_indomain = []
    for g in pool:
        if len(val_indomain) >= args.val_indomain_tiles:
            break
        line = lines[g[0]]
        if groups_left[line] <= 1:
            continue                      # 拿走就没有同航线的训练数据了
        if not budget.affordable(g):
            continue
        budget.take(g)
        groups_left[line] -= 1
        val_indomain += g
    val_indomain = sorted(val_indomain)

    train = sorted(set(names) - set(val_line) - set(val_indomain))
    os.makedirs(args.out_dir, exist_ok=True)
    for fname, split in (('train', train), ('val_indomain', val_indomain),
                         ('val_line', val_line)):
        with open(os.path.join(args.out_dir, f'{fname}.json'), 'w') as fh:
            json.dump(split, fh, indent=1)

    report(train, val_indomain, val_line, held_lines, lines, counts, args.out_dir)


def report(train, val_indomain, val_line, held_lines, lines, counts, out_dir):
    total = len(train) + len(val_indomain) + len(val_line)
    print(f'\n训练 {len(train)} 张（{len(train) / total:.0%}），'
          f'同域验证 {len(val_indomain)} 张（{len(val_indomain) / total:.0%}），'
          f'留出航线验证 {len(val_line)} 张（{len(val_line) / total:.0%}），'
          f'来自 {len(held_lines)} 条航线')

    # 空间泄漏自检：两个验证集里的瓦片都不该与训练瓦片重叠
    train_boxes = collections.defaultdict(list)
    for n in train:
        train_boxes[lines[n]].append(parse_name(n)[1])
    for tag, split in (('同域', val_indomain), ('留出航线', val_line)):
        leaks = [n for n in split
                 if any(overlaps(parse_name(n)[1], b) for b in train_boxes[lines[n]])]
        print(f'{tag}验证集与训练瓦片的空间重叠：{len(leaks)} 张'
              + (f' {leaks[:2]}' if leaks else '（无）'))

    px = {tag: sum((counts[n] for n in split), collections.Counter())
          for tag, split in (('train', train), ('indomain', val_indomain), ('line', val_line))}
    all_classes = sorted(sum(px.values(), collections.Counter()))

    print(f'\n{"ID":>3} {"class":<20} {"训练像元":>12} {"占比":>6} '
          f'{"同域":>10} {"留出航线":>10}')
    missing_in, missing_line = [], []
    for c in all_classes:
        a, b, d = px['train'][c], px['indomain'][c], px['line'][c]
        print(f'{c:>3} {CLASS_NAMES[c]:<20} {a:>12,} {a / (a + b + d):>6.0%} '
              f'{b:>10,} {d:>10,}')
        if b == 0:
            missing_in.append(c)
        if d == 0:
            missing_line.append(c)
    print(f'\n同域验证覆盖 {len(all_classes) - len(missing_in)}/{len(all_classes)} 类，'
          f'缺 {missing_in}')
    print(f'留出航线验证覆盖 {len(all_classes) - len(missing_line)}/{len(all_classes)} 类，'
          f'缺 {missing_line}')
    print(f'\n已写入 {out_dir}/{{train,val_indomain,val_line}}.json')


if __name__ == '__main__':
    main()

"""dataset2683 数据体检。

产出四个文件到 --out 目录：
  manifest.csv   每个瓦片一行，含航线、空间范围、行序是否正常、标注率、nodata 率、类别
  report.json    全部统计量的结构化结果
  report.md      人看的汇总
  groups.json    训练瓦片的空间连通分量（划分验证集时同一分量必须整体进同一折）

用法（在服务器上）：
  python -m tools.audit_dataset --root dataset2683 --out reports/audit --workers 8
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.rawtiff import RawTiff, RawTiffError, imread

NAME_RE = re.compile(r'^(?P<line>.+?)_Sx_(?P<sx>\d+)_Sy_(?P<sy>\d+)_Ex_(?P<ex>\d+)_Ey_(?P<ey>\d+)\.tif$',
                     re.IGNORECASE)
NUM_KNOWN_CLASSES = 31


def parse_name(path):
    m = NAME_RE.match(os.path.basename(path))
    if m is None:
        return None
    return dict(line=m.group('line'),
                sx=int(m.group('sx')), sy=int(m.group('sy')),
                ex=int(m.group('ex')), ey=int(m.group('ey')))


def scan_files(root):
    image_dirs = sorted(d for d in os.listdir(root)
                        if d.lower().startswith('train_images') and os.path.isdir(os.path.join(root, d)))
    train, test = [], []
    for d in image_dirs:
        for name in sorted(os.listdir(os.path.join(root, d))):
            if name.lower().endswith('.tif'):
                train.append(os.path.join(root, d, name))
    test_dir = os.path.join(root, 'test')
    if os.path.isdir(test_dir):
        for name in sorted(os.listdir(test_dir)):
            if name.lower().endswith('.tif'):
                test.append(os.path.join(test_dir, name))
    label_dir = os.path.join(root, 'Train_Labels')
    labels = {}
    if os.path.isdir(label_dir):
        for name in sorted(os.listdir(label_dir)):
            if name.lower().endswith('.tif'):
                labels[name] = os.path.join(label_dir, name)
    return train, test, labels, image_dirs


def probe_header(path):
    try:
        tif = RawTiff(path)
    except RawTiffError as exc:
        return dict(path=path, error=str(exc))
    return dict(path=path, height=tif.height, width=tif.width, bands=tif.bands,
                dtype=str(tif.dtype), contiguous=tif.is_contiguous,
                data_offset=tif.data_offset, rows_per_strip=tif.rows_per_strip)


def probe_label(path):
    arr = imread(path)
    values, counts = np.unique(arr, return_counts=True)
    return dict(path=path,
                shape=list(arr.shape),
                dtype=str(arr.dtype),
                counts={int(v): int(c) for v, c in zip(values, counts)})


def probe_image_stats(args):
    """抽样若干整行，统计 nodata、逐波段一二阶矩，以及基于相邻像元差分的噪声估计。"""
    path, stride = args
    tif = RawTiff(path)
    arr = tif.read_strided(stride, 1).astype(np.float32)
    valid_2d = arr.sum(axis=2) > 0
    flat = arr.reshape(-1, arr.shape[2])
    valid = valid_2d.ravel()
    n_valid = int(valid.sum())
    out = dict(path=path, n_sampled=int(flat.shape[0]), n_valid=n_valid,
               nodata_frac=1.0 - n_valid / max(flat.shape[0], 1))
    if n_valid == 0:
        return out

    v = flat[valid].astype(np.float64)
    out['sum'] = v.sum(axis=0).tolist()
    out['sumsq'] = (v ** 2).sum(axis=0).tolist()
    out['tile_mean'] = float(v.mean())

    # 相邻列都有效的像元对，其光谱差主要来自传感器噪声而非地物变化
    pair = valid_2d[:, 1:] & valid_2d[:, :-1]
    if pair.sum() >= 256:
        diff = np.abs(arr[:, 1:, :] - arr[:, :-1, :])[pair]
        out['noise_mad'] = np.median(diff, axis=0).tolist()
        out['noise_weight'] = int(pair.sum())
    return out


def overlap_fraction(a, b):
    ix = max(0, min(a['ex'], b['ex']) - max(a['sx'], b['sx']))
    iy = max(0, min(a['ey'], b['ey']) - max(a['sy'], b['sy']))
    if ix == 0 or iy == 0:
        return 0.0
    inter = ix * iy
    area_a = (a['ex'] - a['sx']) * (a['ey'] - a['sy'])
    area_b = (b['ex'] - b['sx']) * (b['ey'] - b['sy'])
    return inter / min(area_a, area_b)


def connected_components(items, min_overlap):
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_line = defaultdict(list)
    for idx, it in enumerate(items):
        by_line[it['line']].append(idx)
    for idxs in by_line.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if overlap_fraction(items[idxs[a]], items[idxs[b]]) > min_overlap:
                    ra, rb = find(idxs[a]), find(idxs[b])
                    if ra != rb:
                        parent[ra] = rb
    groups = defaultdict(list)
    for idx in range(len(items)):
        groups[find(idx)].append(idx)
    return list(groups.values())


def transferable_labels(train_items, test_items, label_paths):
    by_line = defaultdict(list)
    for it in train_items:
        by_line[it['line']].append(it)

    per_tile, total_covered, total_px = [], 0, 0
    label_cache = {}
    for t in test_items:
        h, w = t['ey'] - t['sy'], t['ex'] - t['sx']
        total_px += h * w
        candidates = [tr for tr in by_line.get(t['line'], []) if overlap_fraction(t, tr) > 0]
        if not candidates:
            continue
        acc = np.zeros((h, w), dtype=np.int32)
        for tr in candidates:
            name = os.path.basename(tr['path'])
            if name not in label_paths:
                continue
            if name not in label_cache:
                label_cache[name] = imread(label_paths[name])
            lab = label_cache[name]
            ox0, oy0 = max(t['sx'], tr['sx']), max(t['sy'], tr['sy'])
            ox1, oy1 = min(t['ex'], tr['ex']), min(t['ey'], tr['ey'])
            src = lab[oy0 - tr['sy']:oy1 - tr['sy'], ox0 - tr['sx']:ox1 - tr['sx']]
            dst = acc[oy0 - t['sy']:oy1 - t['sy'], ox0 - t['sx']:ox1 - t['sx']]
            np.copyto(dst, src, where=(src > 0))
        covered = int((acc > 0).sum())
        if covered:
            total_covered += covered
            per_tile.append(dict(tile=os.path.basename(t['path']),
                                 covered_frac=covered / (h * w),
                                 classes=sorted(int(c) for c in np.unique(acc) if c > 0)))
    per_tile.sort(key=lambda d: -d['covered_frac'])
    return dict(overall_frac=total_covered / max(total_px, 1),
                num_tiles=len(per_tile), per_tile=per_tile)


MAD_TO_SIGMA = 1.4826 / np.sqrt(2.0)


def band_quality(mean, noise_mad, min_snr=3.0, min_mean=20.0):
    """噪声用相邻像元差分的 MAD 估计，避免把地物差异算成噪声。"""
    sigma = np.maximum(noise_mad * MAD_TO_SIGMA, 1e-6)
    snr = mean / sigma
    bad = np.where((snr < min_snr) | (mean < min_mean))[0]
    return bad.tolist(), snr


def run_pool(fn, items, workers, desc):
    results = []
    if workers <= 1:
        for i, it in enumerate(items):
            results.append(fn(it))
            if (i + 1) % 50 == 0:
                print(f'  {desc}: {i + 1}/{len(items)}', flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(fn, items, chunksize=1)):
            results.append(res)
            if (i + 1) % 50 == 0:
                print(f'  {desc}: {i + 1}/{len(items)}', flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='dataset2683')
    ap.add_argument('--out', default='reports/audit')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--stride', type=int, default=8,
                    help='影像统计的空间抽样步长，8 表示每 8x8 取 1 个像元')
    ap.add_argument('--max-stat-tiles', type=int, default=0,
                    help='参与波段统计的瓦片数上限，0 表示全部')
    ap.add_argument('--group-overlap', type=float, default=0.05,
                    help='训练瓦片视为同一空间组的最小重合比例')
    ap.add_argument('--min-band-snr', type=float, default=3.0)
    ap.add_argument('--min-band-mean', type=float, default=20.0)
    ap.add_argument('--skip-image-stats', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = {}

    print('[1/6] 扫描文件')
    train_paths, test_paths, label_paths, image_dirs = scan_files(args.root)
    print(f'  训练影像 {len(train_paths)}  测试影像 {len(test_paths)}  标签 {len(label_paths)}')

    train_items, test_items, unparsed = [], [], []
    for p in train_paths:
        info = parse_name(p)
        (train_items.append({**info, 'path': p}) if info else unparsed.append(p))
    for p in test_paths:
        info = parse_name(p)
        (test_items.append({**info, 'path': p}) if info else unparsed.append(p))

    train_names = {os.path.basename(p) for p in train_paths}
    report['files'] = dict(
        image_dirs=image_dirs,
        num_train=len(train_paths), num_test=len(test_paths), num_labels=len(label_paths),
        unparsable_names=unparsed,
        images_without_label=sorted(train_names - set(label_paths)),
        labels_without_image=sorted(set(label_paths) - train_names),
        duplicate_train_basenames=[n for n, c in Counter(os.path.basename(p) for p in train_paths).items() if c > 1],
    )

    print('[2/6] 解析 TIFF 头')
    headers = run_pool(probe_header, train_paths + test_paths, args.workers, 'header')
    header_by_path = {h['path']: h for h in headers}
    broken = [h for h in headers if 'error' in h]
    non_contiguous = [h['path'] for h in headers if h.get('contiguous') is False]
    geometries = Counter((h.get('height'), h.get('width'), h.get('bands'), h.get('dtype'))
                         for h in headers if 'error' not in h)
    report['headers'] = dict(
        geometries={str(k): v for k, v in geometries.items()},
        num_non_contiguous=len(non_contiguous),
        non_contiguous=[os.path.basename(p) for p in non_contiguous],
        non_contiguous_in_test=[os.path.basename(p) for p in non_contiguous if os.sep + 'test' + os.sep in p],
        unreadable=broken,
    )
    print(f'  几何一致性: {dict(geometries)}')
    print(f'  行序错位文件: {len(non_contiguous)} 个（必须走 RawTiff.read，不能直接 memmap）')

    print('[3/6] 统计标签')
    label_list = [label_paths[n] for n in sorted(label_paths)]
    label_stats = run_pool(probe_label, label_list, args.workers, 'label')

    class_pixels = Counter()
    class_lines = defaultdict(set)
    class_tiles = Counter()
    label_by_name = {}
    label_dtypes = Counter()
    for st in label_stats:
        name = os.path.basename(st['path'])
        label_by_name[name] = st
        label_dtypes[st['dtype']] += 1
        info = parse_name(st['path'])
        line = info['line'] if info else '?'
        for cls, cnt in st['counts'].items():
            class_pixels[cls] += cnt
            if cls != 0:
                class_lines[cls].add(line)
                class_tiles[cls] += 1

    total_label_px = sum(class_pixels.values())
    labeled_frac = {n: 1.0 - st['counts'].get(0, 0) / max(sum(st['counts'].values()), 1)
                    for n, st in label_by_name.items()}
    fracs = np.array(list(labeled_frac.values()))
    unexpected = sorted(c for c in class_pixels if c < 0 or c > NUM_KNOWN_CLASSES)
    missing = sorted(set(range(1, NUM_KNOWN_CLASSES + 1)) - set(class_pixels))

    report['labels'] = dict(
        dtypes=dict(label_dtypes),
        total_pixels=total_label_px,
        unlabeled_frac=class_pixels.get(0, 0) / max(total_label_px, 1),
        unexpected_values=unexpected,
        missing_classes=missing,
        per_class={str(c): dict(pixels=class_pixels[c],
                                frac=class_pixels[c] / max(total_label_px, 1),
                                num_lines=len(class_lines[c]),
                                num_tiles=class_tiles[c])
                   for c in sorted(class_pixels) if c != 0},
        labeled_frac_percentiles={f'p{q}': float(np.percentile(fracs, q))
                                  for q in (0, 5, 10, 25, 50, 75, 90, 100)},
        tiles_below_1pct=int((fracs < 0.01).sum()),
        tiles_below_5pct=int((fracs < 0.05).sum()),
        tiles_below_10pct=int((fracs < 0.10).sum()),
        single_line_classes=sorted(int(c) for c in class_lines if len(class_lines[c]) == 1),
        two_line_classes=sorted(int(c) for c in class_lines if len(class_lines[c]) == 2),
    )

    print('[4/6] 分析训练/测试的空间关系')
    train_lines = {it['line'] for it in train_items}
    test_lines = {it['line'] for it in test_items}
    groups = connected_components(train_items, args.group_overlap)
    group_sizes = Counter(len(g) for g in groups)
    transfer = transferable_labels(train_items, test_items, label_paths)

    test_overlap_tiles = 0
    max_overlaps = []
    train_by_line = defaultdict(list)
    for it in train_items:
        train_by_line[it['line']].append(it)
    for t in test_items:
        best = max((overlap_fraction(t, tr) for tr in train_by_line.get(t['line'], [])), default=0.0)
        if best > 0:
            test_overlap_tiles += 1
            max_overlaps.append((best, os.path.basename(t['path'])))
    max_overlaps.sort(reverse=True)

    report['geometry'] = dict(
        num_train_lines=len(train_lines), num_test_lines=len(test_lines),
        shared_lines=len(train_lines & test_lines),
        test_only_lines=sorted(test_lines - train_lines),
        test_tiles_overlapping_train=test_overlap_tiles,
        top_overlaps=[dict(frac=round(f, 4), tile=n) for f, n in max_overlaps[:20]],
        num_train_groups=len(groups),
        train_group_size_hist={str(k): v for k, v in sorted(group_sizes.items())},
        transferable_labels=dict(overall_frac=transfer['overall_frac'],
                                 num_tiles=transfer['num_tiles'],
                                 top=transfer['per_tile'][:20]),
    )
    with open(os.path.join(args.out, 'groups.json'), 'w') as fh:
        json.dump([[os.path.basename(train_items[i]['path']) for i in g] for g in groups], fh, indent=1)

    image_stats_by_path = {}
    if args.skip_image_stats:
        print('[5/6] 跳过影像统计')
        report['spectra'] = None
    else:
        print(f'[5/6] 影像与波段统计 (stride={args.stride})')
        stat_paths = train_paths + test_paths
        if args.max_stat_tiles:
            stat_paths = stat_paths[:args.max_stat_tiles]
        stats = run_pool(probe_image_stats, [(p, args.stride) for p in stat_paths], args.workers, 'image')
        image_stats_by_path = {s['path']: s for s in stats}

        def accumulate(paths):
            n, s, ss, brightness = 0, None, None, []
            noise, noise_w = None, 0
            for p in paths:
                st = image_stats_by_path.get(p)
                if not st or 'sum' not in st:
                    continue
                arr_s = np.asarray(st['sum'])
                arr_ss = np.asarray(st['sumsq'])
                s = arr_s if s is None else s + arr_s
                ss = arr_ss if ss is None else ss + arr_ss
                n += st['n_valid']
                brightness.append(st['tile_mean'])
                if 'noise_mad' in st:
                    w = st['noise_weight']
                    contrib = np.asarray(st['noise_mad']) * w
                    noise = contrib if noise is None else noise + contrib
                    noise_w += w
            if n == 0:
                return None
            mean = s / n
            var = np.maximum(ss / n - mean ** 2, 0.0)
            noise = noise / noise_w if noise_w else np.zeros_like(mean)
            return mean, np.sqrt(var), np.asarray(brightness), noise

        tr = accumulate(train_paths)
        te = accumulate(test_paths)
        mean, std, tr_bright, noise_mad = tr
        bad, snr = band_quality(mean, noise_mad, args.min_band_snr, args.min_band_mean)
        spectra = dict(
            band_mean=mean.round(3).tolist(),
            band_std=std.round(3).tolist(),
            band_noise_sigma=(noise_mad * MAD_TO_SIGMA).round(4).tolist(),
            band_snr=snr.round(4).tolist(),
            suggested_bad_bands=bad,
            num_suggested_bad_bands=len(bad),
            train_tile_brightness=dict(min=float(tr_bright.min()), max=float(tr_bright.max()),
                                       median=float(np.median(tr_bright)),
                                       max_min_ratio=float(tr_bright.max() / max(tr_bright.min(), 1e-6))),
            nodata_frac=dict(
                mean=float(np.mean([image_stats_by_path[p]['nodata_frac'] for p in train_paths
                                    if p in image_stats_by_path])),
                percentiles={f'p{q}': float(np.percentile(
                    [image_stats_by_path[p]['nodata_frac'] for p in train_paths if p in image_stats_by_path], q))
                    for q in (50, 90, 99, 100)}),
        )
        if te is not None:
            te_mean, te_std, te_bright, _ = te
            spectra['test_band_mean'] = te_mean.round(3).tolist()
            spectra['train_test_band_mean_ratio'] = (te_mean / np.maximum(mean, 1e-6)).round(4).tolist()
            spectra['test_tile_brightness'] = dict(min=float(te_bright.min()), max=float(te_bright.max()),
                                                   median=float(np.median(te_bright)))
        report['spectra'] = spectra

    print('[6/6] 写出报告')
    group_of = {}
    for gid, g in enumerate(groups):
        for i in g:
            group_of[train_items[i]['path']] = gid

    with open(os.path.join(args.out, 'manifest.csv'), 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['split', 'name', 'line', 'sx', 'sy', 'ex', 'ey', 'group',
                         'row_order_ok', 'bands', 'dtype', 'labeled_frac', 'nodata_frac', 'classes'])
        for split, items in (('train', train_items), ('test', test_items)):
            for it in items:
                name = os.path.basename(it['path'])
                hdr = header_by_path.get(it['path'], {})
                st = label_by_name.get(name)
                classes = sorted(c for c in st['counts'] if c != 0) if st else []
                img_st = image_stats_by_path.get(it['path'])
                writer.writerow([
                    split, name, it['line'], it['sx'], it['sy'], it['ex'], it['ey'],
                    group_of.get(it['path'], ''),
                    int(bool(hdr.get('contiguous'))), hdr.get('bands', ''), hdr.get('dtype', ''),
                    round(labeled_frac.get(name, float('nan')), 5) if st else '',
                    round(img_st['nodata_frac'], 5) if img_st else '',
                    ' '.join(str(c) for c in classes),
                ])

    with open(os.path.join(args.out, 'report.json'), 'w') as fh:
        json.dump(report, fh, indent=1)

    write_markdown(os.path.join(args.out, 'report.md'), report)
    print(f'完成，报告在 {args.out}')


def write_markdown(path, r):
    L = ['# dataset2683 数据体检报告', '']

    f = r['files']
    L += ['## 1 文件与配对', '',
          f"- 训练影像 {f['num_train']}，测试影像 {f['num_test']}，标签 {f['num_labels']}",
          f"- 影像目录：{', '.join(f['image_dirs'])}",
          f"- 缺标签的训练影像：{len(f['images_without_label'])} {f['images_without_label'][:5]}",
          f"- 无对应影像的标签：{len(f['labels_without_image'])} {f['labels_without_image'][:5]}",
          f"- 文件名无法解析：{len(f['unparsable_names'])}",
          f"- 跨目录重名：{len(f['duplicate_train_basenames'])}", '']

    h = r['headers']
    L += ['## 2 TIFF 结构', '',
          f"- 几何/类型组合（(H,W,C,dtype) -> 数量）：{h['geometries']}",
          f"- **行序错位文件 {h['num_non_contiguous']} 个**"
          f"（其中测试集 {len(h['non_contiguous_in_test'])} 个），这些文件不能直接 memmap，"
          f"必须按 StripOffsets 重排行，否则影像相对标签竖向 roll。",
          f"- 无法读取：{len(h['unreadable'])}", '']
    if h['non_contiguous']:
        L += ['<details><summary>行序错位文件清单</summary>', '']
        L += [f'- {n}' for n in h['non_contiguous']]
        L += ['', '</details>', '']

    lb = r['labels']
    L += ['## 3 标签分布', '',
          f"- 标签 dtype：{lb['dtypes']}",
          f"- 标注为 0 的像元占比：**{100 * lb['unlabeled_frac']:.2f}%**",
          f"- 超出 0-31 的异常取值：{lb['unexpected_values'] or '无'}",
          f"- 训练集中完全缺失的类别：{lb['missing_classes'] or '无'}",
          f"- 只出现在 1 条航线的类别：**{lb['single_line_classes']}**（场景级切分会直接丢掉这些类）",
          f"- 只出现在 2 条航线的类别：{lb['two_line_classes']}",
          f"- 瓦片标注率分位：{ {k: round(100 * v, 1) for k, v in lb['labeled_frac_percentiles'].items()} }",
          f"- 标注率 <1% / <5% / <10% 的瓦片数：{lb['tiles_below_1pct']} / "
          f"{lb['tiles_below_5pct']} / {lb['tiles_below_10pct']}", '',
          '| 类别 | 像元数 | 占全部像元 | 出现航线数 | 出现瓦片数 |', '|---|---|---|---|---|']
    for c in sorted(lb['per_class'], key=lambda x: int(x)):
        d = lb['per_class'][c]
        L.append(f"| {c} | {d['pixels']} | {100 * d['frac']:.3f}% | {d['num_lines']} | {d['num_tiles']} |")
    L.append('')

    g = r['geometry']
    L += ['## 4 训练/测试空间关系', '',
          f"- 训练航线 {g['num_train_lines']} 条，测试航线 {g['num_test_lines']} 条，"
          f"**共享 {g['shared_lines']} 条**",
          f"- 测试集独有航线 {len(g['test_only_lines'])} 条",
          f"- **与训练瓦片空间交叠的测试瓦片：{g['test_tiles_overlapping_train']} 张**，"
          f"最高重合比例 {g['top_overlaps'][0]['frac'] if g['top_overlaps'] else 0}",
          f"- 训练瓦片空间连通分量 {g['num_train_groups']} 个，规模分布 {g['train_group_size_hist']}"
          f"（划分验证集时同一分量必须整体进同一折，见 groups.json）",
          f"- 可由训练标签直接搬运到测试瓦片的像元占测试集 "
          f"**{100 * g['transferable_labels']['overall_frac']:.2f}%**，涉及 "
          f"{g['transferable_labels']['num_tiles']} 张测试瓦片", '',
          '重合度最高的测试瓦片：', '']
    for d in g['top_overlaps'][:10]:
        L.append(f"- {100 * d['frac']:.1f}%  {d['tile']}")
    L.append('')

    s = r.get('spectra')
    if s:
        L += ['## 5 光谱与无效值', '',
              f"- nodata（全零光谱）占比：均值 {100 * s['nodata_frac']['mean']:.2f}%，"
              f"分位 { {k: round(100 * v, 1) for k, v in s['nodata_frac']['percentiles'].items()} }",
              f"- **建议剔除的低质量波段共 {s['num_suggested_bad_bands']} 个**"
              f"（噪声由相邻像元差分估计，判据：SNR<3 或平均辐亮度<20）：{s['suggested_bad_bands']}",
              f"- 训练瓦片整体亮度 min/median/max = "
              f"{s['train_tile_brightness']['min']:.0f} / {s['train_tile_brightness']['median']:.0f} / "
              f"{s['train_tile_brightness']['max']:.0f}，"
              f"最大最小比 **{s['train_tile_brightness']['max_min_ratio']:.1f}x**"
              f"（全局逐波段 mean/std 归一化在这个跨度下不足以消除域偏移）"]
        if 'test_tile_brightness' in s:
            t = s['test_tile_brightness']
            L.append(f"- 测试瓦片整体亮度 min/median/max = {t['min']:.0f} / {t['median']:.0f} / {t['max']:.0f}")
            ratio = np.asarray(s['train_test_band_mean_ratio'])
            L.append(f"- 测试/训练逐波段均值比：min {ratio.min():.2f}，median {np.median(ratio):.2f}，"
                     f"max {ratio.max():.2f}")
        L.append('')

    with open(path, 'w') as fh:
        fh.write('\n'.join(L))


if __name__ == '__main__':
    main()

"""把已有预测 TIF 按官方标签的布局重写并重新打包，不跑模型。

评测端 tifffile 没有 imagecodecs，LZW 或非 GDAL 条带布局都会读失败。
已有 ./submission/pred/*.tif 时：

    python tools/repack_submission.py
"""

import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hin.paths import find_tifs
from tools.rawtiff import RawTiff, imread, imwrite


def main():
    p = argparse.ArgumentParser(description='按官方 TIFF 布局重写预测并打包')
    p.add_argument('--pred', default='./submission/pred', help='已有预测 TIF 目录')
    p.add_argument('--zip', default='./submission/submission.zip')
    args = p.parse_args()

    paths = find_tifs(args.pred)
    if not paths:
        raise FileNotFoundError(f'{args.pred} 里没有 TIF')

    for i, path in enumerate(paths, 1):
        arr = imread(path)
        imwrite(path, arr.astype('int32'))
        tif = RawTiff(path)
        if tif.compression != 1 or tif.rows_per_strip != 4:
            raise RuntimeError(f'{path} 重写后布局仍不对')
        if i == 1 or i == len(paths):
            print(f'  {os.path.basename(path)}  strips={len(tif.strip_offsets)}  '
                  f'rps={tif.rows_per_strip}  compression={tif.compression}')
    print(f'已重写 {len(paths)} 张')

    os.makedirs(os.path.dirname(os.path.abspath(args.zip)) or '.', exist_ok=True)
    with zipfile.ZipFile(args.zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=os.path.basename(path))
    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
    nested = [n for n in names if '/' in n or '\\' in n]
    if nested:
        raise RuntimeError(f'压缩包里出现了子目录：{nested[:3]}')
    size = os.path.getsize(args.zip) / 2 ** 20
    print(f'已打包 {args.zip}（{len(names)} 个文件，{size:.1f} MiB，无子目录）')


if __name__ == '__main__':
    main()

"""数据集在磁盘上的位置与布局。

单独成模块，是为了让只管文件的工具（划分、审计）不用装训练依赖。
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 本地开发时数据在仓库里，远程服务器上数据是仓库的兄弟目录
# （/root/autodl-tmp/{freenet1, dataset2683}），按顺序探测，两边免配置。
DATA_ROOT_CANDIDATES = (
    os.path.join(_REPO_ROOT, 'dataset2683'),
    os.path.normpath(os.path.join(_REPO_ROOT, '..', 'dataset2683')),
)


def resolve_data_root(root=None):
    """显式传入的路径优先；不传就在候选位置里找。"""
    if root:
        if not os.path.isdir(os.path.join(root, 'Train_Labels')):
            raise FileNotFoundError(f'{root} 下没有 Train_Labels，不是数据根目录')
        return root
    for cand in DATA_ROOT_CANDIDATES:
        if os.path.isdir(os.path.join(cand, 'Train_Labels')):
            return cand
    raise FileNotFoundError('找不到 dataset2683，候选位置：'
                            + '，'.join(DATA_ROOT_CANDIDATES)
                            + '。请用 root 参数显式指定。')


def find_tifs(directory):
    """递归收集目录下的 TIF，按文件名排序。

    解压方式不同会导致层级不同——测试集可能是 test/*.tif，也可能是
    test/Test_Images/*.tif——递归查找可以两种都认。返回顺序只看文件名，
    不受目录层级影响，这样推理顺序在不同机器上一致。
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'{directory} 不是目录')
    found = {}
    for current, _dirs, files in os.walk(directory):
        for name in files:
            if not name.lower().endswith(('.tif', '.tiff')):
                continue
            path = os.path.join(current, name)
            if name in found:
                raise RuntimeError(f'{directory} 下有重名 TIF，无法确定用哪个：\n'
                                   f'  {found[name]}\n  {path}')
            found[name] = path
    return [found[n] for n in sorted(found)]

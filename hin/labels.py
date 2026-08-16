"""HyperImageNet 竞赛子集的类别定义与标签映射约定。

原始标签取值：
    0        Unlabeled（含 nodata 黑边，也含测试集里被判为 Unknown 的那些地物）
    1..31    已知类
    32       Unknown，只出现在测试集真值里，训练标签中不存在

训练用的目标索引空间：
    0..30    对应原始类别 1..31
    31       "other"，即有效像元但原始标签为 0（open 模式下才启用）
    255      忽略，不参与损失
"""

CLASS_NAMES = {
    1: 'Water', 2: 'Impervious Surface', 3: 'Barren', 4: 'Snow',
    5: 'Woody Wetland', 6: 'Herbaceous Wetland', 7: 'Grassland', 8: 'Other Hay',
    9: 'Deciduous Forest', 10: 'Evergreen Forest', 11: 'Mixed Forest', 12: 'Shrubland',
    13: 'Corn', 14: 'Cotton', 15: 'Soybean', 16: 'Sunflower', 17: 'Barley',
    18: 'Winter Wheat', 19: 'Alfalfa', 20: 'Dry Bean', 21: 'Tomato', 22: 'Fallow',
    23: 'Strawberry', 24: 'Peach', 25: 'Grape', 26: 'Citrus', 27: 'Orange',
    28: 'Olive', 29: 'Almond', 30: 'Walnut', 31: 'Pistachio', 32: 'Unknown',
}

NUM_KNOWN_CLASSES = 31
OTHER_INDEX = 31
IGNORE_INDEX = 255

# tools/audit_dataset.py 在全部 562 张影像上算出的低质量波段：
# 106-112 与 152-168 是 AVIRIS 的两个水汽吸收窗，221-223 是 SWIR 噪声尾巴。
DEFAULT_BAD_BANDS = tuple(range(106, 113)) + tuple(range(152, 169)) + (221, 222, 223)


def num_classes(mode):
    if mode == 'closed':
        return NUM_KNOWN_CLASSES
    if mode == 'open':
        return NUM_KNOWN_CLASSES + 1
    raise ValueError(f'unknown mode {mode!r}')


def target_index_to_submission_id(mode):
    """模型输出的索引 -> 提交文件里应该写的类别 ID。"""
    ids = list(range(1, NUM_KNOWN_CLASSES + 1))
    if mode == 'open':
        ids.append(32)
    return ids


def class_name(mode, index):
    return CLASS_NAMES[target_index_to_submission_id(mode)[index]]

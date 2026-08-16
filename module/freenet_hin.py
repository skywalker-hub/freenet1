"""面向 HyperImageNet 竞赛子集的 FreeNet。

继承官方 module/freenet.py，网络结构一行不改，只替换标签约定和损失归一化：

官方 loss 写的是 `F.cross_entropy(x, y.long() - 1, ignore_index=-1)`，这套
"标签减 1" 的约定只能表达 "0 = 忽略"。本任务要区分三种像元——nodata 必须忽略、
1..31 是已知类、有效但未标注的像元在 open 模式下要作为第 32 类参与监督——
所以改成由数据层预先映射好索引，模型侧只认 ignore_index=255。
"""

import torch.nn.functional as F
from simplecv import registry

from hin.labels import IGNORE_INDEX
from module.freenet import FreeNet


@registry.MODEL.register('FreeNetHIN')
class FreeNetHIN(FreeNet):
    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long(), weight=None,
                                 ignore_index=IGNORE_INDEX, reduction='none')
        losses = losses * weight
        return losses.sum() / weight.sum().clamp(min=1.0)

    def set_defalut_config(self):
        super(FreeNetHIN, self).set_defalut_config()
        self.config.update(dict(
            in_channels=197,
            num_classes=32,
        ))

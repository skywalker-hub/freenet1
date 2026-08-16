"""FreeNet 的纯 PyTorch 版本。

结构与 module/freenet.py 里的官方实现逐层对应，只做三件事：
去掉 simplecv 的 CVModule/registry 依赖、内联 SEBlock、把损失从模型里挪出去。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """与 simplecv.module.SEBlock 等价。"""

    def __init__(self, in_channels, reduction_ratio):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        v = F.adaptive_avg_pool2d(x, 1).flatten(1)
        score = self.seq(v)
        return x * score.view(score.size(0), score.size(1), 1, 1)


def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True),
    )


def repeat_block(block_channel, r, n):
    return nn.Sequential(*[
        nn.Sequential(SEBlock(block_channel, r),
                      conv3x3_gn_relu(block_channel, block_channel, r))
        for _ in range(n)
    ])


class FreeNet(nn.Module):
    def __init__(self, in_channels, num_classes,
                 block_channels=(96, 128, 192, 256), num_blocks=(1, 1, 1, 1),
                 inner_dim=128, reduction_ratio=1.0):
        super().__init__()
        r = int(16 * reduction_ratio)
        chans = [int(c * reduction_ratio / r) * r for c in block_channels]
        inner_dim = int(inner_dim * reduction_ratio)

        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(in_channels, chans[0], r),

            repeat_block(chans[0], r, num_blocks[0]),
            nn.Identity(),
            downsample2x(chans[0], chans[1]),

            repeat_block(chans[1], r, num_blocks[1]),
            nn.Identity(),
            downsample2x(chans[1], chans[2]),

            repeat_block(chans[2], r, num_blocks[2]),
            nn.Identity(),
            downsample2x(chans[2], chans[3]),

            repeat_block(chans[3], r, num_blocks[3]),
            nn.Identity(),
        ])
        self.reduce_1x1convs = nn.ModuleList([nn.Conv2d(c, inner_dim, 1) for c in chans])
        self.fuse_3x3convs = nn.ModuleList(
            [nn.Conv2d(inner_dim, inner_dim, 3, 1, 1) for _ in chans])
        self.cls_pred_conv = nn.Conv2d(inner_dim, num_classes, 1)

        self.in_channels = in_channels
        self.num_classes = num_classes

    @staticmethod
    def _top_down(top, lateral):
        return lateral + F.interpolate(top, size=lateral.shape[-2:], mode='nearest')

    def forward(self, x):
        feats = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feats.append(x)

        inner = [conv(f) for conv, f in zip(self.reduce_1x1convs, feats)]
        inner.reverse()

        out = [self.fuse_3x3convs[0](inner[0])]
        for i in range(len(inner) - 1):
            out.append(self.fuse_3x3convs[i + 1](self._top_down(out[i], inner[i + 1])))

        return self.cls_pred_conv(out[-1])


def build_model(in_channels, num_classes, **kwargs):
    return FreeNet(in_channels, num_classes, **kwargs)


if __name__ == '__main__':
    net = build_model(197, 32)
    n = sum(p.numel() for p in net.parameters())
    y = net(torch.randn(1, 197, 128, 128))
    print(f'params {n / 1e6:.2f}M  output {tuple(y.shape)}')

"""Feature refinement blocks for the HBB backbone."""

import torch
from torch import nn

from .block import Bottleneck, C2f, C3
from .conv import Conv


class FRFN(nn.Module):
    """Mix a subset of the input channels and gate a depthwise spatial branch."""

    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, 2 * hidden_dim), act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim_conv = dim // 4
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, padding=1, bias=False)

    def forward(self, features):
        batch, _, height, width = features.shape
        mixed = self.partial_conv3(features[:, :self.dim_conv])
        fused = torch.cat((mixed, features[:, self.dim_conv:]), dim=1)
        tokens = fused.flatten(2).transpose(1, 2)
        spatial, gate = self.linear1(tokens).chunk(2, dim=-1)
        spatial = spatial.transpose(1, 2).reshape(batch, -1, height, width)
        spatial = self.dwconv(spatial).flatten(2).transpose(1, 2)
        output = self.linear2(spatial * gate)
        return output.transpose(1, 2).reshape(batch, -1, height, width)


class Bottleneck_FRFN(nn.Module):
    def __init__(self, channels, kernel_size=3, shortcut=True):
        super().__init__()
        self.cv1 = Conv(channels, channels, kernel_size)
        self.cv2 = FRFN(channels)
        self.add = shortcut

    def forward(self, features):
        refined = self.cv2(self.cv1(features))
        return features + refined if self.add else refined


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        branch_channels = int(c2 * e)
        self.m = nn.Sequential(*[Bottleneck_FRFN(branch_channels, k, shortcut) for _ in range(n)])


class C3k2_FRFN(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        if c3k:
            blocks = [C3k(self.c, self.c, n=2, shortcut=shortcut, g=g) for _ in range(n)]
        else:
            blocks = [Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)]
        self.m = nn.ModuleList(blocks)

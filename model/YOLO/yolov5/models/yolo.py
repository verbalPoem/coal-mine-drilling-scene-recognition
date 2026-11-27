"""YOLOv5 oriented detection model."""

from copy import deepcopy
import math
from pathlib import Path

import torch
from torch import nn
import yaml
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils.torch_utils import fuse_conv_and_bn, initialize_weights

from models.common import C3, SPPF, Concat, Conv
from models.head import Detect


def build_graph(config, in_channels):
    channels, layers, saved = [in_channels], [], set()
    for i, (source, repeats, name, args) in enumerate(
        config["backbone"] + config["head"]
    ):
        count = max(round(repeats * config["depth_multiple"]), 1)
        if name in {"Conv", "SPPF", "C3"}:
            out = math.ceil(args[0] * config["width_multiple"] / 8) * 8
            args = [channels[source], out, *args[1:]]
            if name == "C3":
                args.insert(2, count)
                count = 1
            constructor = {"Conv": Conv, "SPPF": SPPF, "C3": C3}[name]
            layer = (
                constructor(*args)
                if count == 1
                else nn.Sequential(*(constructor(*args) for _ in range(count)))
            )
        elif name == "nn.Upsample":
            out = channels[source]
            size, scale, mode = args
            layer = nn.Upsample(
                None if size == "None" else size, scale_factor=scale, mode=mode
            )
        elif name == "Concat":
            out = sum(channels[j] for j in source)
            layer = Concat(*args)
        elif name == "Detect":
            layer = Detect(
                config["nc"], config["anchors"], [channels[j] for j in source]
            )
            out = layer.na * layer.no
        else:
            raise ValueError(f"Unknown YOLOv5 layer {name!r}")
        layer.i, layer.f, layer.type = i, source, name
        layer.np = sum(p.numel() for p in layer.parameters())
        layers.append(layer)
        for j in [source] if isinstance(source, int) else source:
            if j != -1:
                saved.add(j % i)
        if i == 0:
            channels.clear()
        channels.append(out)
    return nn.Sequential(*layers), sorted(saved)


class Model(BaseModel):
    def __init__(self, cfg, ch=3, nc=None):
        super().__init__()
        self.yaml = (
            deepcopy(cfg)
            if isinstance(cfg, dict)
            else yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
        )
        if nc is not None:
            self.yaml["nc"] = int(nc)
        self.nc = self.yaml["nc"]
        self.names = {i: str(i) for i in range(self.nc)}
        self.model, self.save = build_graph(self.yaml, ch)
        head = self.model[-1]
        head.stride = torch.tensor(
            [
                256 / t.shape[-2]
                for t in self._predict_once(torch.zeros(1, ch, 256, 256))
            ]
        )
        head.anchors /= head.stride.view(-1, 1, 1)
        self.stride = head.stride
        with torch.no_grad():
            for conv, stride in zip(head.m, head.stride):
                bias = conv.bias.view(head.na, -1)
                bias[:, 4] += math.log(8 / (640 / stride) ** 2)
                bias[:, 5:] += math.log(0.6 / (head.nc - 0.999999))
        initialize_weights(self)

    def _apply(self, fn):
        self.criterion = None
        super()._apply(fn)
        head = self.model[-1]
        head.stride = fn(head.stride)
        head.grid = [fn(t) for t in head.grid]
        if isinstance(head.anchor_grid, list):
            head.anchor_grid = [fn(t) for t in head.anchor_grid]
        self.stride = head.stride
        return self

    def fuse(self, verbose=True):
        for module in self.modules():
            if isinstance(module, Conv) and hasattr(module, "bn"):
                module.conv = fuse_conv_and_bn(module.conv, module.bn)
                del module.bn
                module.forward = module.forward_fuse
        return self

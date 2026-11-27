import math

import torch
from torch import nn
from torch.nn import functional as F


class LightRepLKDepthwise(nn.Module):
    """Large and small depthwise convolutions with an optional fused kernel."""

    def __init__(self, channels, kernel, small_kernel=3):
        super().__init__()
        self.channels = channels
        self.kernel = kernel
        self.small_kernel = small_kernel
        self.large = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel,
                padding=kernel // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.small = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                small_kernel,
                padding=small_kernel // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return (
            self.merged(x) if hasattr(self, "merged") else self.large(x) + self.small(x)
        )

    def merge_kernel(self):
        if self.training:
            raise RuntimeError("Kernel merge requires eval")
        if hasattr(self, "merged"):
            return

        def fuse(branch):
            conv, bn = branch
            scale = bn.weight / (bn.running_var + bn.eps).sqrt()
            return conv.weight * scale.view(
                -1, 1, 1, 1
            ), bn.bias - bn.running_mean * scale

        wk, bk = fuse(self.large)
        ws, bs = fuse(self.small)
        pad = (self.kernel - self.small_kernel) // 2
        weight = wk + torch.nn.functional.pad(ws, [pad, pad, pad, pad])
        merged = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel,
            padding=self.kernel // 2,
            groups=self.channels,
            bias=True,
        ).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            merged.weight.copy_(weight)
            merged.bias.copy_(bk + bs)
        self.merged = merged
        del self.large, self.small


class NativeRepLKAdapter(nn.Module):
    """Add gated multi-kernel context to a backbone stage."""

    def __init__(
        self,
        base: nn.Module,
        channels: int,
        kernel: int,
        *,
        blocks: int = 1,
        small_kernel: int = 3,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if (
            channels < 1
            or kernel < 3
            or kernel % 2 != 1
            or small_kernel % 2 != 1
            or small_kernel > kernel
        ):
            raise ValueError("Invalid RepLK adapter dimensions/kernels")
        if blocks != 1 or drop_path != 0:
            raise ValueError("Expected one deterministic depthwise block")
        self.base = base
        self.replk = LightRepLKDepthwise(channels, kernel, small_kernel)
        self.residual_scale = nn.Parameter(torch.zeros(channels))
        self.channels = int(channels)
        self.kernel = int(kernel)
        self.small_kernel = int(small_kernel)

    def forward(self, x):
        b = self.base(x)
        delta = self.replk(b) - b
        scale = torch.tanh(self.residual_scale).to(dtype=b.dtype).view(1, -1, 1, 1)
        return b + scale * delta

    def merge_replk_kernels(self):
        """Eval-only reparameterization of the added RepLK branch, not base."""
        if self.training:
            raise RuntimeError("Call eval before kernel merging")
        merged = 0
        for module in list(self.replk.modules()):
            if hasattr(module, "merge_kernel"):
                module.merge_kernel()
                merged += 1
        return merged


class BoundedConcatAFF(nn.Module):
    """Apply local and global channel gates before concatenation."""

    def __init__(self, cx: int, cy: int, *, reduction: int = 4, bound: float = 0.25):
        super().__init__()
        if min(cx, cy, reduction) < 1 or not (math.isfinite(bound) and 0 < bound < 1):
            raise ValueError("Invalid AFF dimensions/bound")
        self.cx, self.cy = int(cx), int(cy)
        self.bound = float(bound)
        total = cx + cy
        hidden = max(4, total // reduction)
        self.local_att = nn.Sequential(
            nn.Conv2d(total, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, total, 1, bias=True),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, total, 1, bias=True),
        )
        for branch in (self.local_att, self.global_att):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def attention_gains(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("AFF expects two input feature tensors")
        x, y = inputs
        if (
            x.ndim != 4
            or y.ndim != 4
            or x.shape[0] != y.shape[0]
            or x.shape[2:] != y.shape[2:]
        ):
            raise ValueError("AFF expects BCHW tensors with matching B,H,W")
        if x.shape[1] != self.cx or y.shape[1] != self.cy:
            raise ValueError("AFF input channel identity mismatch")
        xy = torch.cat((x, y), dim=1)
        gains = 1.0 + self.bound * torch.tanh(self.local_att(xy) + self.global_att(xy))
        return gains.split((self.cx, self.cy), dim=1)

    def forward(self, inputs):
        gx, gy = self.attention_gains(inputs)
        return torch.cat((inputs[0] * gx, inputs[1] * gy), dim=1)


class LargeSmallDepthwise(nn.Module):
    """A large depthwise kernel and a separately normalized local kernel."""

    def __init__(self, width, kernel):
        super().__init__()
        self.large = self._branch(width, kernel)
        self.small = self._branch(width, 3)

    @staticmethod
    def _branch(width, kernel):
        return nn.Sequential(
            nn.Conv2d(
                width, width, kernel, padding=kernel // 2, groups=width, bias=False
            ),
            nn.BatchNorm2d(width, eps=0.001, momentum=0.03),
        )

    def forward(self, features):
        if hasattr(self, "merged"):
            return self.merged(features)
        return self.large(features) + self.small(features)

    @torch.no_grad()
    def merge(self):
        """Fold both branches into one equivalent eval-mode depthwise convolution."""
        if self.training:
            raise ValueError("Kernel merging requires eval mode")
        if hasattr(self, "merged"):
            return
        large = torch.nn.utils.fusion.fuse_conv_bn_eval(*self.large)
        small = torch.nn.utils.fusion.fuse_conv_bn_eval(*self.small)
        pad = (large.kernel_size[0] - small.kernel_size[0]) // 2
        large.weight.add_(F.pad(small.weight, (pad, pad, pad, pad)))
        large.bias.add_(small.bias)
        self.merged = large
        del self.large, self.small


class RepLKResidual(nn.Module):
    """Keep A2C2f's full output and learn an additive projected context correction."""

    def __init__(self, native, channels, width, kernel):
        super().__init__()
        self.native = native
        self.reduce = nn.Conv2d(channels, width, 1, bias=False)
        self.context = LargeSmallDepthwise(width, kernel)
        self.expand = nn.Conv2d(width, channels, 1, bias=False)
        nn.init.zeros_(self.expand.weight)

    def forward(self, inputs):
        base = self.native(inputs)
        delta = self.context(self.reduce(base))
        return base + self.expand(F.silu(delta))


class BalancedConcatAFF(nn.Module):
    """Fuse local and global context while preserving mean channel gain."""

    def __init__(self, channels, local_width, global_width, local_stride):
        super().__init__()
        self.local_stride = local_stride
        self.pool = (
            nn.AvgPool2d(local_stride, local_stride, ceil_mode=True)
            if local_stride > 1
            else nn.Identity()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_gate = self._gate(channels, local_width)
        self.global_gate = self._gate(channels, global_width)

    @staticmethod
    def _gate(channels, width):
        layers = nn.Sequential(
            nn.Conv2d(channels, width, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(width, channels, 1, bias=True),
        )
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return layers

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        local = self.local_gate(self.pool(x))
        if self.local_stride > 1:
            local = F.interpolate(local, size=x.shape[-2:], mode="nearest")
        logits = local + self.global_gate(self.global_pool(x))
        modulation = logits.tanh()
        gain = 1.0 + 0.10 * (modulation - modulation.mean(dim=1, keepdim=True))
        return x * gain


def _direction_stencils(groups):
    axes = (
        [(1, i) for i in range(3)],
        [(i, 1) for i in range(3)],
        [(i, i) for i in range(3)],
        [(i, 2 - i) for i in range(3)],
    )
    kernels = torch.full((len(axes), 1, 3, 3), -1.0 / 6)
    for kernel, axis in zip(kernels, axes):
        for row, column in axis:
            kernel[0, row, column] = 1.0 / 3
    conv = nn.Conv2d(
        groups,
        groups * len(axes),
        3,
        padding=1,
        groups=groups,
        bias=False,
        padding_mode="replicate",
    )
    del conv.weight
    conv.register_buffer("weight", kernels.repeat(groups, 1, 1, 1))
    return conv


class DirectionalConcatAFF(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.local_stride = native.local_stride
        self.pool = native.pool
        self.global_pool = native.global_pool
        self.local_gate = native.local_gate
        self.global_gate = native.global_gate
        self.direction_groups = 8
        self.stencils = _direction_stencils(self.direction_groups)
        self.direction_mix = nn.Conv2d(
            4 * self.direction_groups,
            self.direction_groups,
            1,
            groups=self.direction_groups,
            bias=False,
        )
        nn.init.zeros_(self.direction_mix.weight)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        local = self.local_gate(self.pool(x))
        local = F.interpolate(local, size=x.shape[-2:], mode="nearest")
        context = local + self.global_gate(self.global_pool(x))
        grouped = x.unflatten(1, (self.direction_groups, -1))
        response = self.stencils(grouped.mean(dim=2)).abs()
        correction = self.direction_mix(response).unsqueeze(2)
        logits = (
            context.unflatten(1, (self.direction_groups, -1)) + correction
        ).flatten(1, 2)
        modulation = logits.tanh()
        gain = 1.0 + 0.10 * (modulation - modulation.mean(dim=1, keepdim=True))
        return x * gain


def configure_stages(model, stages):
    if any(kind == "ProjectedRepLK" for _, kind, _ in stages):
        seed = torch.default_generator.initial_seed()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed((seed + 1) % (2**64))
            _install_stages(model, [stage for stage in stages if stage[1] == "ProjectedRepLK"])
            torch.default_generator.manual_seed((seed + 2) % (2**64))
            _install_stages(model, [stage for stage in stages if stage[1] != "ProjectedRepLK"])
        return
    _install_stages(model, stages)


def _install_stages(model, stages):
    for index, kind, args in stages:
        original = model.model[index]
        if kind == "GatedRepLK":
            block = NativeRepLKAdapter(original, original.cv2.conv.out_channels, *args)
        elif kind == "ProjectedRepLK":
            block = RepLKResidual(original, original.cv2.conv.out_channels, *args)
        elif kind == "BoundedAFF":
            first, second, reduction, bound = args
            block = BoundedConcatAFF(first, second, reduction=reduction, bound=bound)
        elif kind in {"BalancedAFF", "DirectionalAFF"}:
            block = BalancedConcatAFF(*args)
            if kind == "DirectionalAFF":
                block = DirectionalConcatAFF(block)
        else:
            raise ValueError(f"Unknown CAF stage {kind!r}")
        if kind in {"GatedRepLK", "BoundedAFF"}:
            branch = block.replk if kind == "GatedRepLK" else block
            for layer in branch.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eps, layer.momentum = 0.001, 0.03
        block.i, block.f = original.i, original.f
        block.np = sum(p.numel() for p in block.parameters())
        block.type = type(block).__name__
        model.model[index] = block

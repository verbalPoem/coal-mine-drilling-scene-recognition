import torch
from torch import nn
import torch.nn.functional as F
import e2cnn.nn as enn
import mmrotate.models.utils.enn as U
from mmrotate.models.backbones.re_resnet import ReResNet
from mmrotate.models.necks.re_fpn import ReFPN
from mmrotate.models.builder import ROTATED_BACKBONES


@torch.no_grad()
def _restore_conv_cache(module, state, prefix, *args):
    """Match the derived e2cnn buffers to the current train/eval mode."""
    kernel_key = prefix + 'filter'
    bias_key = prefix + 'expanded_bias'
    if module._buffers.get('filter') is None:
        state.pop(kernel_key, None)
        state.pop(bias_key, None)
        return
    if kernel_key not in state and prefix + 'weights' in state:
        coefficients = state[prefix + 'weights'].to(module.weights)
        state[kernel_key] = module.basisexpansion(coefficients).reshape_as(module.filter)
    if module.bias is not None and bias_key not in state and prefix + 'bias' in state:
        expansion = state.get(prefix + 'bias_expansion', module.bias_expansion)
        state[bias_key] = expansion.to(module.bias) @ state[prefix + 'bias'].to(module.bias)


class PointwiseSwish(nn.Module):

    def forward(self, x):
        return enn.GeometricTensor(x.tensor * x.tensor.sigmoid(), x.type)


class ReCBAM(nn.Module):

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.in_type = U.build_enn_divide_feature(channels)
        hidden = max(8, ((channels // reduction + 7) // 8) * 8)
        self.mlp = nn.Sequential(U.ennConv(channels, hidden, 1),
                                 U.ennReLU(hidden),
                                 U.ennConv(hidden, channels, 1))
        self.sa_type = enn.FieldType(U.gspace, [U.gspace.trivial_repr] * 2)
        self.sa = enn.R2Conv(self.sa_type,
                             enn.FieldType(U.gspace, [U.gspace.trivial_repr]),
                             spatial_kernel,
                             padding=spatial_kernel // 2,
                             bias=False,
                             frequencies_cutoff=lambda r: 3 * r)

    def forward(self, x):
        assert x.type == self.in_type
        avg = enn.GeometricTensor(x.tensor.mean((2, 3), keepdim=True), x.type)
        maximum = enn.GeometricTensor(x.tensor.amax((2, 3), keepdim=True),
                                      x.type)
        channel = (self.mlp(avg).tensor + self.mlp(maximum).tensor).sigmoid()
        z = x.tensor * channel
        spatial_input = torch.cat(
            [z.mean(1, keepdim=True),
             z.amax(1, keepdim=True)], 1)
        spatial = self.sa(enn.GeometricTensor(spatial_input,
                                              self.sa_type)).tensor.sigmoid()
        return enn.GeometricTensor(z * spatial, x.type)


class ReSiASPP(nn.Module):

    def __init__(self, in_channels=256, branch_channels=64):
        super().__init__()
        self.in_type = U.build_enn_divide_feature(in_channels)
        self.out_type = U.build_enn_divide_feature(branch_channels * 5)

        def branch(kernel, dilation=1):
            return nn.Sequential(
                U.ennConv(in_channels,
                          branch_channels,
                          kernel,
                          padding=dilation if kernel == 3 else 0,
                          dilation=dilation),
                U.build_enn_norm_layer(branch_channels)[1], PointwiseSwish())

        self.local = nn.ModuleList(
            [branch(1), branch(3, 6),
             branch(3, 12),
             branch(3, 18)])
        self.global_branch = branch(1)

    def forward(self, x):
        assert x.type == self.in_type
        outputs = [m(x).tensor for m in self.local]
        pooled = enn.GeometricTensor(x.tensor.mean((2, 3), keepdim=True),
                                     x.type)
        global_output = self.global_branch(pooled).tensor
        outputs.append(
            F.interpolate(global_output,
                          size=x.tensor.shape[-2:],
                          mode='nearest'))
        return enn.GeometricTensor(torch.cat(outputs, 1), self.out_type)


class AttentionBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = ReResNet(depth=50,
                             num_stages=4,
                             out_indices=(0, 1, 2, 3),
                             frozen_stages=1,
                             style='pytorch',
                             pretrained=None)
        self.attention = nn.ModuleDict({'1': ReCBAM(512), '2': ReCBAM(1024)})

    def forward(self, image):
        b = self.base
        x = enn.GeometricTensor(image, b.in_type)
        x = b.maxpool(b.relu(b.norm1(b.conv1(x))))
        outputs = []
        for i, layer_name in enumerate(b.res_layers):
            x = getattr(b, layer_name)(x)
            if str(i) in self.attention:
                x = self.attention[str(i)](x)
            outputs.append(x)
        return tuple(outputs)


class CoarseFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_type = U.build_enn_divide_feature(128)
        self.conv = U.ennConv(128, 8, 1)

    def forward(self, previous, current):
        assert len(previous) == len(current) == 2
        outputs = []
        for old, new in zip(previous, current):
            gate = self.conv(enn.GeometricTensor(new,
                                                 self.in_type)).tensor.mean(
                                                     (2, 3),
                                                     keepdim=True).sigmoid()
            n, c, h, w = new.shape
            assert c == 128
            gate = gate.reshape(n, 1, 8, 1, 1)
            a, b = new.reshape(n, 16, 8, h, w), old.reshape(n, 16, 8, h, w)
            outputs.append((gate * a + (1 - gate) * b).reshape_as(new))
        return tuple(outputs)


@ROTATED_BACKBONES.register_module()
class RecurrentFeaturePyramid(nn.Module):

    def __init__(self, norm_eval=True, init_cfg=None, pretrained=None):
        super().__init__()
        assert init_cfg is None, 'No implicit pretrained load'
        self.norm_eval = bool(norm_eval)
        self.pretrained = pretrained
        self.backbone = AttentionBackbone()
        self.neck = ReFPN(in_channels=[256, 512, 1024, 2048],
                          out_channels=128,
                          num_outs=5)
        self.aspp = nn.ModuleList([ReSiASPP(128, 32), ReSiASPP(128, 32)])
        self.feedback = nn.ModuleList(
            [U.ennConv(160, 1024, 1),
             U.ennConv(160, 2048, 1)])
        self.fusion = CoarseFusion()
        for module in self.modules():
            if isinstance(module, enn.R2Conv):
                module._register_load_state_dict_pre_hook(_restore_conv_cache, with_module=True)

    def init_weights(self):
        if self.pretrained is not None:
            checkpoint = torch.load(self.pretrained, map_location="cpu")
            state = checkpoint.get("state_dict", checkpoint)
            state = {k.removeprefix("module."): v for k, v in state.items()}
            base = self.backbone.base
            for name, param in base.named_parameters():
                if name not in state or state[name].shape != param.shape:
                    raise ValueError(
                        f"Incompatible backbone checkpoint tensor: {name}")
            base.load_state_dict(state, strict=False)
        self.neck.init_weights()

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, image):
        features = self.backbone(image)
        original = self.neck(features)
        ptype = U.build_enn_divide_feature(128)
        corrections = tuple(
            proj(aspp(enn.GeometricTensor(original[i + 2], ptype)))
            for i, (aspp, proj) in enumerate(zip(self.aspp, self.feedback)))
        assert corrections[0].tensor.shape == features[2].tensor.shape
        assert corrections[1].tensor.shape == features[3].tensor.shape
        c4 = self.backbone.base.layer3[-1](features[2] + corrections[0])
        c4 = self.backbone.attention['2'](c4)
        c5 = self.backbone.base.layer4[-1](features[3] + corrections[1])
        lat4 = self.neck.lateral_convs[2](c4)
        lat5 = self.neck.lateral_convs[3](c5)
        lat4 = lat4 + self.neck.up_samples[3](lat5)
        p4 = self.neck.fpn_convs[2](lat4).tensor
        p5 = self.neck.fpn_convs[3](lat5).tensor
        p4, p5 = self.fusion(original[2:4], (p4, p5))
        p6 = self.neck.max_pools[0](enn.GeometricTensor(p5, ptype)).tensor
        return (original[0], original[1], p4, p5, p6)

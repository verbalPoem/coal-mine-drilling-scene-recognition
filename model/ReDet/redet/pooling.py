# Copyright (c) OpenMMLab. All rights reserved.
# CPU adaptation of MMCV RiRoIAlignRotated, Apache-2.0.
import math

import torch
from mmcv.ops import RiRoIAlignRotated
from mmrotate.models.builder import ROTATED_ROI_EXTRACTORS
from mmrotate.models.roi_heads.roi_extractors.rotate_single_level_roi_extractor import RotatedSingleRoIExtractor


def _bilinear(plane, y, x):
    """plane CxHxW; exact boundary rule in common_cuda_helper.hpp."""
    _, h, w = plane.shape
    invalid = (y < -1) | (y > h) | (x < -1) | (x > w)
    # Coordinates <=0 clamp to 0; the upper edge [H-1,H] clamps to H-1.
    # Values outside [-1,H] / [-1,W] are zero, not grid_sample border padding.
    yc = y.clamp(min=0, max=h - 1)
    xc = x.clamp(min=0, max=w - 1)
    yl, xl = yc.to(torch.long), xc.to(torch.long)
    yh, xh = (yl + 1).clamp(max=h - 1), (xl + 1).clamp(max=w - 1)
    ly, lx = yc - yl, xc - xl
    hy, hx = 1 - ly, 1 - lx
    value = (plane[:, yl, xl] * (hy * hx) +
             plane[:, yl, xh] * (hy * lx) +
             plane[:, yh, xl] * (ly * hx) +
             plane[:, yh, xh] * (ly * lx))
    return value.masked_fill(invalid.unsqueeze(0), 0)


def riroi_align_rotated_cpu(input, rois, output_size, spatial_scale=1.0,
                           sampling_ratio=0, num_orientations=8,
                           clockwise=False):
    """NCHW input, Rx6 [batch,cx,cy,w,h,angle_rad] -> RxCxPHxPW.

    Float32/64 CPU only. Feature gradients are propagated; RoIs are detached to
    match MMCV's native backward contract. Adaptive sampling_ratio=0 is supported.
    """
    if input.device.type != 'cpu' or rois.device.type != 'cpu':
        raise ValueError('Expected CPU tensors')
    if input.ndim != 4 or rois.ndim != 2 or rois.shape[1] != 6:
        raise ValueError('Expected NCHW and Rx6 tensors')
    if input.dtype not in (torch.float32, torch.float64):
        raise TypeError('Only float32 and float64 are supported')
    if not torch.isfinite(rois).all():
        raise ValueError('Non-finite RoI coordinates')
    if input.shape[1] % num_orientations:
        raise ValueError('Channel count must be divisible by num_orientations')
    if isinstance(output_size, int):
        ph, pw = output_size, output_size
    else:
        ph, pw = map(int, output_size)
    if ph < 1 or pw < 1 or sampling_ratio < 0:
        raise ValueError('Invalid pooling size/sampling ratio')
    rois = rois.detach().to(dtype=input.dtype)
    nc = input.shape[1]
    groups = nc // num_orientations
    po = torch.arange(num_orientations)
    pc = torch.arange(groups)[:, None] * num_orientations
    outs = []
    for roi in rois:
        batch = int(roi[0].item())
        if batch < 0 or batch >= input.shape[0]:
            raise IndexError('Invalid RoI batch index')
        cw, ch = roi[1] * spatial_scale, roi[2] * spatial_scale
        width = (roi[3] * spatial_scale).clamp(min=1.)
        height = (roi[4] * spatial_scale).clamp(min=1.)
        theta = roi[5]
        bh, bw = height / ph, width / pw
        indf = theta * num_orientations / (2 * math.pi)
        ind = int(torch.floor(indf).item())
        lv, rv = indf - ind, 1 - (indf - ind)
        oi = (po - ind) % num_orientations
        oj = (oi + 1) % num_orientations
        gh = int(sampling_ratio) if sampling_ratio > 0 else int(torch.ceil(height / ph).item())
        gw = int(sampling_ratio) if sampling_ratio > 0 else int(torch.ceil(width / pw).item())
        angle = -theta if clockwise else theta
        ct, st = torch.cos(angle), torch.sin(angle)
        yy = (-height / 2 + torch.arange(ph, dtype=input.dtype)[:, None] * bh +
              (torch.arange(gh, dtype=input.dtype)[None, :] + .5) * bh / gh)
        xx = (-width / 2 + torch.arange(pw, dtype=input.dtype)[:, None] * bw +
              (torch.arange(gw, dtype=input.dtype)[None, :] + .5) * bw / gw)
        # Shape PH,PW,GH,GW, with sample order iy then ix as in CUDA.
        yy = yy[:, None, :, None]
        xx = xx[None, :, None, :]
        y = yy * ct - xx * st + ch
        x = yy * st + xx * ct + cw
        values = _bilinear(input[batch], y, x)
        left = values[(pc + oi).reshape(-1)]
        right = values[(pc + oj).reshape(-1)]
        mixed = rv * left + lv * right
        # Sequential sample accumulation matches the pinned kernel's order.
        out = input.new_zeros((nc, ph, pw))
        for iy in range(gh):
            for ix in range(gw):
                out = out + mixed[:, :, :, iy, ix]
        outs.append(out / max(gh * gw, 1))
    return (torch.stack(outs) if outs else
            input.new_zeros((0, nc, ph, pw)) + input.sum() * 0)


class _PoolCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, rois, size, scale, samples, orientations, clockwise):
        ctx.save_for_backward(features, rois)
        ctx.options = (size, scale, samples, orientations, clockwise)
        return riroi_align_rotated_cpu(features, rois, *ctx.options)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        features, rois = ctx.saved_tensors
        with torch.enable_grad():
            x = features.detach().requires_grad_(True)
            output = riroi_align_rotated_cpu(x, rois, *ctx.options)
            grad, = torch.autograd.grad(output, x, grad_output)
        return grad, None, None, None, None, None, None


class OrientedPooling(RiRoIAlignRotated):
    def forward(self, features, rois):
        if features.device.type != "cpu":
            return super().forward(features, rois)
        return _PoolCPU.apply(
            features, rois, self.out_size, self.spatial_scale,
            self.num_samples, self.num_orientations, self.clockwise
        )


@ROTATED_ROI_EXTRACTORS.register_module()
class EquivariantRoIExtractor(RotatedSingleRoIExtractor):
    def build_roi_layers(self, layer_cfg, featmap_strides):
        options = layer_cfg.copy()
        if options.pop("type") != "RiRoIAlignRotated":
            return super().build_roi_layers(layer_cfg, featmap_strides)
        return torch.nn.ModuleList(
            [OrientedPooling(spatial_scale=1 / stride, **options)
             for stride in featmap_strides]
        )

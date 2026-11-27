# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from MMRotate, Apache License 2.0.

import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox2roi
from mmrotate.core import obb2xyxy
from mmrotate.models.builder import ROTATED_HEADS
from mmrotate.models.roi_heads.roi_trans_roi_head import RoITransRoIHead
from mmrotate.models.roi_heads.bbox_heads.convfc_rbbox_head import (
    RotatedShared2FCBBoxHead, )


@ROTATED_HEADS.register_module()
class RotatedRefinementHead(RotatedShared2FCBBoxHead):

    def __init__(self, *args, input_box_type='rbox', **kwargs):
        if input_box_type not in ('hbox', 'rbox'):
            raise ValueError('input_box_type must be hbox or rbox')
        self.input_box_type = input_box_type
        super().__init__(*args, **kwargs)

    @force_fp32(apply_to=('rois', 'bbox_pred'))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        input_dim = 4 if self.input_box_type == 'hbox' else 5
        if rois.ndim != 2 or rois.shape[1] not in (input_dim, input_dim + 1):
            raise ValueError('Unexpected RoI geometry for ' +
                             self.input_box_type)
        with_batch = rois.shape[1] == input_dim + 1
        if bbox_pred.shape[0] != rois.shape[0] or label.shape[0] != rois.shape[
                0]:
            raise ValueError('RoI, label and delta row counts differ')
        expected = 5 if self.reg_class_agnostic else self.num_classes * 5
        if bbox_pred.ndim != 2 or bbox_pred.shape[1] != expected:
            raise ValueError(
                'Rotated regression requires five deltas per class')
        if rois.shape[0] == 0:
            return rois.new_empty((0, 6 if with_batch else 5))
        if not self.reg_class_agnostic:
            if bool(((label < 0) | (label >= self.num_classes)).any()):
                raise ValueError('Refinement requires foreground class labels')
            gather = label.long()[:, None] * 5 + torch.arange(
                5, device=label.device)[None]
            bbox_pred = torch.gather(bbox_pred, 1, gather)
        boxes = rois[:, 1:] if with_batch else rois
        refined = self.bbox_coder.decode(boxes,
                                         bbox_pred,
                                         max_shape=img_meta['img_shape'])
        return torch.cat(
            (rois[:, :1], refined), dim=1) if with_batch else refined

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if rois.shape[0] == 0:
            if cfg is not None:
                return rois.new_empty((0, 6)), rois.new_empty((0, ),
                                                              dtype=torch.long)
            width = bbox_pred.shape[1] if bbox_pred is not None else 5
            scores = cls_score.softmax(
                dim=-1) if cls_score is not None else None
            return rois.new_empty((0, width)), scores
        return super().get_bboxes(rois,
                                  cls_score,
                                  bbox_pred,
                                  img_shape,
                                  scale_factor,
                                  rescale=rescale,
                                  cfg=cfg)


@ROTATED_HEADS.register_module()
class ContextCascadeRoIHead(RoITransRoIHead):

    def __init__(self,
                 *args,
                 context_scales=(1.0, 1.2, 1.5),
                 context_weights=(1.0, 0.6, 0.3),
                 **kwargs):
        if len(context_scales) != len(context_weights) or not context_scales:
            raise ValueError('Context scale and weight counts must match')
        if any(float(s) <= 0 for s in context_scales):
            raise ValueError('Context scales must be positive')
        self.context_scales = tuple(float(s) for s in context_scales)
        self.context_weights = tuple(float(w) for w in context_weights)
        super().__init__(*args, **kwargs)
        if self.num_stages != 4 or self.version != 'le90':
            raise ValueError('Expected four le90 stages')
        for stage, (extractor, head) in enumerate(
                zip(self.bbox_roi_extractor, self.bbox_head)):
            if extractor.num_inputs < 2:
                raise ValueError(
                    'Multi-level extraction is required for native context rescaling'
                )
            if head.in_channels != extractor.out_channels * len(
                    self.context_scales):
                raise ValueError(
                    'Head channels must equal concatenated context channels')
            expected = 'hbox' if stage == 0 else 'rbox'
            if head.input_box_type != expected:
                raise ValueError(
                    'Wrong native input box geometry for cascade stage')

    def _bbox_forward(self, stage, x, rois):
        extractor = self.bbox_roi_extractor[stage]
        views = [
            extractor(x[:extractor.num_inputs], rois, roi_scale_factor=scale) *
            weight
            for scale, weight in zip(self.context_scales, self.context_weights)
        ]
        bbox_feats = torch.cat(views, dim=1)
        cls_score, bbox_pred = self.bbox_head[stage](bbox_feats)
        return dict(cls_score=cls_score,
                    bbox_pred=bbox_pred,
                    bbox_feats=bbox_feats)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

        if self.train_cfg is None:
            raise RuntimeError('Training losses require a train_cfg')
        if gt_masks is not None:
            raise NotImplementedError('Mask targets are unsupported')
        num_imgs = len(img_metas)
        if not (len(proposal_list) == len(gt_bboxes) == len(gt_labels) ==
                num_imgs):
            raise ValueError('Image, proposal and GT batch lengths differ')
        ignores = [
            None
        ] * num_imgs if gt_bboxes_ignore is None else gt_bboxes_ignore
        if len(ignores) != num_imgs:
            raise ValueError('Ignored GT batch length differs')
        losses = {}
        for stage in range(self.num_stages):
            self.current_stage = stage
            train_cfg = self.train_cfg[stage]
            sampling_results = []
            for image in range(num_imgs):
                stage_gt = obb2xyxy(
                    gt_bboxes[image],
                    self.version) if stage == 0 else gt_bboxes[image]
                ignored = ignores[image]
                if stage == 0 and ignored is not None:
                    ignored = obb2xyxy(ignored, self.version)
                assigned = self.bbox_assigner[stage].assign(
                    proposal_list[image], stage_gt, ignored, gt_labels[image])
                sampled = self.bbox_sampler[stage].sample(
                    assigned,
                    proposal_list[image],
                    stage_gt,
                    gt_labels[image],
                    feats=[feature[image][None] for feature in x])
                sampled.pos_gt_bboxes = (
                    gt_bboxes[image][sampled.pos_assigned_gt_inds, :] if
                    gt_bboxes[image].numel() else gt_bboxes[image].new_zeros(
                        (0, 5)))
                sampling_results.append(sampled)
            results = self._bbox_forward_train(stage, x, sampling_results,
                                               gt_bboxes, gt_labels, train_cfg)
            for name, value in results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (value *
                                              self.stage_loss_weights[stage]
                                              if 'loss' in name else value)
            if stage < self.num_stages - 1:
                with torch.no_grad():
                    scores = results['cls_score']
                    head = self.bbox_head[stage]
                    if head.custom_activation:
                        scores = head.loss_cls.get_activation(scores)
                    labels = results['bbox_targets'][0]
                    labels = torch.where(labels == head.num_classes,
                                         scores[:, :-1].argmax(1), labels)
                    proposal_list = head.refine_bboxes(
                        results['rois'], labels, results['bbox_pred'],
                        [sampled.pos_is_gt
                         for sampled in sampling_results], img_metas)
        return losses

    def forward_dummy(self, x, proposals):

        rois = bbox2roi([proposals])
        outs = ()
        shape = (x[0].shape[-2] * 4, x[0].shape[-1] * 4, 3)
        for stage in range(self.num_stages):
            results = self._bbox_forward(stage, x, rois)
            outs += (results['cls_score'], results['bbox_pred'])
            if stage < self.num_stages - 1:
                labels = results['cls_score'][:, :-1].argmax(1)
                rois = self.bbox_head[stage].regress_by_class(
                    rois, labels, results['bbox_pred'], dict(img_shape=shape))
        return outs

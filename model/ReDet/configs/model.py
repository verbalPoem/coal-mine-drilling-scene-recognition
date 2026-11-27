from copy import deepcopy


def detector_config():
    angle_version = 'le90'
    ordinary = dict(type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign',
                                   output_size=7,
                                   sampling_ratio=2,
                                   aligned=True,
                                   use_torchvision=False),
                    out_channels=128,
                    featmap_strides=[4, 8, 16, 32])
    rotated = dict(type='EquivariantRoIExtractor',
                   roi_layer=dict(type='RiRoIAlignRotated',
                                  out_size=7,
                                  num_samples=2,
                                  num_orientations=8,
                                  clockwise=True),
                   out_channels=128,
                   featmap_strides=[4, 8, 16, 32])
    heads = []
    stages = []
    for stage in range(4):
        coder = dict(type='DeltaXYWHAHBBoxCoder'
                     if stage == 0 else 'DeltaXYWHAOBBoxCoder',
                     angle_range=angle_version,
                     norm_factor=2 if stage == 0 else None,
                     edge_swap=True,
                     target_means=[0., 0., 0., 0., 0.],
                     target_stds=[.1, .1, .2, .2, 1.]
                     if stage == 0 else [.05, .05, .1, .1, .5])
        if stage:
            coder['proj_xy'] = True
        heads.append(
            dict(type='RotatedRefinementHead',
                 input_box_type='hbox' if stage == 0 else 'rbox',
                 in_channels=384,
                 fc_out_channels=416,
                 roi_feat_size=7,
                 num_classes=6,
                 reg_class_agnostic=(stage == 0),
                 bbox_coder=coder,
                 loss_cls=dict(type='CrossEntropyLoss',
                               use_sigmoid=False,
                               loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                loss_weight=1.0)))
        stages.append(
            dict(assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=.5,
                neg_iou_thr=.5,
                min_pos_iou=.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D' if stage ==
                                    0 else 'RBboxOverlaps2D')),
                 sampler=dict(
                     type='RandomSampler' if stage == 0 else 'RRandomSampler',
                     num=512,
                     pos_fraction=.25,
                     neg_pos_ub=-1,
                     add_gt_as_proposals=True),
                 pos_weight=-1,
                 debug=False))
    proposal = dict(nms_pre=2000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=.7),
                    min_bbox_size=0)
    return dict(
        type='ReDet',
        backbone=dict(type='RecurrentFeaturePyramid', norm_eval=True),
        neck=None,
        rpn_head=dict(type='RotatedRPNHead',
                      in_channels=128,
                      feat_channels=128,
                      version=angle_version,
                      anchor_generator=dict(type='AnchorGenerator',
                                            scales=[8],
                                            ratios=[.5, 1., 2.],
                                            strides=[4, 8, 16, 32, 64]),
                      bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                      target_means=[0., 0., 0., 0.],
                                      target_stds=[1., 1., 1., 1.]),
                      loss_cls=dict(type='CrossEntropyLoss',
                                    use_sigmoid=True,
                                    loss_weight=1.0),
                      loss_bbox=dict(type='SmoothL1Loss',
                                     beta=1. / 9.,
                                     loss_weight=1.0)),
        roi_head=dict(type='ContextCascadeRoIHead',
                      version=angle_version,
                      num_stages=4,
                      stage_loss_weights=[1., 1., 1., 1.],
                      context_scales=(1., 1.2, 1.5),
                      context_weights=(1., .6, .3),
                      bbox_roi_extractor=[ordinary] +
                      [deepcopy(rotated) for _ in range(3)],
                      bbox_head=heads),
        train_cfg=dict(rpn=dict(assigner=dict(type='MaxIoUAssigner',
                                              pos_iou_thr=.7,
                                              neg_iou_thr=.3,
                                              min_pos_iou=.3,
                                              match_low_quality=True,
                                              ignore_iof_thr=-1),
                                sampler=dict(type='RandomSampler',
                                             num=256,
                                             pos_fraction=.5,
                                             neg_pos_ub=-1,
                                             add_gt_as_proposals=False),
                                allowed_border=0,
                                pos_weight=-1,
                                debug=False),
                       rpn_proposal=deepcopy(proposal),
                       rcnn=stages),
        test_cfg=dict(rpn=deepcopy(proposal),
                      rcnn=dict(nms_pre=2000,
                                min_bbox_size=0,
                                score_thr=.001,
                                nms=dict(iou_thr=.1),
                                max_per_img=300)))


model = detector_config()
del detector_config, deepcopy

model["backbone"]["pretrained"] = None
custom_imports = dict(imports=["redet"], allow_failed_imports=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RResize", img_scale=(640, 640)),
    dict(type="RRandomFlip", flip_ratio=[0.25, 0.25, 0.25],
         direction=["horizontal", "vertical", "diagonal"], version="le90"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=(640, 640)),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="MultiScaleFlipAug", img_scale=(640, 640), flip=False, transforms=[
        dict(type="RResize"),
        dict(type="Normalize", **img_norm_cfg),
        dict(type="Pad", size=(640, 640)),
        dict(type="DefaultFormatBundle"),
        dict(type="Collect", keys=["img"]),
    ]),
]
optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1/3,
                 step=[4, 5])
runner = dict(type="EpochBasedRunner", max_epochs=6)

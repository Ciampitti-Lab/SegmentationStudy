default_scope = 'mmdet'

classes = ('spike',)
data_root = '/content/drive/MyDrive/Repos/new/datasets/spike_segm/'

model = dict(
    type='CascadeRCNN',

    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),

    neck=[
        dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
        dict(type='BFP', in_channels=256, num_levels=5, refine_level=2, refine_type='non_local')
    ],

    rpn_head=dict(
        type='RPNHead',
        in_channels=256, feat_channels=256,
        anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0],
                              strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.],
                        target_stds=[1., 1., 1., 1.]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),

    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3, stage_loss_weights=[1.0, 0.5, 0.25],

        bbox_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='sum',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256, featmap_strides=[4, 8, 16, 32]
        ),

        bbox_head=[
            dict(
                type='SABLHead',
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                num_classes=len(classes),
                bbox_coder=dict(
                    type='BucketingBBoxCoder',
                    num_buckets=14,
                    scale_factor=1.0
                ),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox_cls=dict(type='CrossEntropyLoss', use_sigmoid=True,  loss_weight=1.0),
                loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1.0),
            ),
            dict(
                type='SABLHead',
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                num_classes=len(classes),
                bbox_coder=dict(
                    type='BucketingBBoxCoder',
                    num_buckets=14,
                    scale_factor=1.0
                ),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox_cls=dict(type='CrossEntropyLoss', use_sigmoid=True,  loss_weight=1.0),
                loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1.0),
            ),
            dict(
                type='SABLHead',
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                num_classes=len(classes),
                bbox_coder=dict(
                    type='BucketingBBoxCoder',
                    num_buckets=14,
                    scale_factor=1.0
                ),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox_cls=dict(type='CrossEntropyLoss', use_sigmoid=True,  loss_weight=1.0),
                loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1.0),
            ),
        ],

        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256, featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(type='FCNMaskHead', num_convs=4, in_channels=256,
                       conv_out_channels=256, num_classes=len(classes)),
    ),

    # ---- MODEL-LEVEL train/test cfg ----
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5, neg_iou_thr=0.5,
                    min_pos_iou=0.5, match_low_quality=False
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True
                ),
                mask_size=28
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6, neg_iou_thr=0.6,
                    min_pos_iou=0.6, match_low_quality=False
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True
                ),
                mask_size=28
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7, neg_iou_thr=0.7,
                    min_pos_iou=0.7, match_low_quality=False
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True
                ),
                mask_size=28
            )
        ]
    ),

    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type='soft_nms', iou_threshold=0.5), max_per_img=100, mask_thr_binary=0.5)
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False))

test_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False))

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_val.json',
                     metric=['bbox', 'segm'])
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_test.json',
                      metric=['bbox', 'segm'])

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[100, 140], gamma=0.5, by_epoch=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=160, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50)
)

env_cfg = dict(cudnn_benchmark=False)
visualizer = dict(type='DetLocalVisualizer')
work_dir = 'work_dirs/wheatspikenet_v3'

_base_ = ['wheatspikenet_v3_fallback.py']

model = dict(
    # Προσθήκη training config για Cascade Mask R-CNN
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3,
                match_low_quality=True),
            sampler=dict(
                type='RandomSampler', num=256, pos_fraction=0.5,
                neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1, pos_weight=-1, debug=False),
        rpn_proposal=dict(
            nms_pre=2000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(  # stage 1
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5,
                    match_low_quality=False),
                sampler=dict(
                    type='RandomSampler', num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=28, pos_weight=-1, debug=False),
            dict(  # stage 2
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6, neg_iou_thr=0.6, min_pos_iou=0.6,
                    match_low_quality=False),
                sampler=dict(
                    type='RandomSampler', num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=28, pos_weight=-1, debug=False),
            dict(  # stage 3
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7, neg_iou_thr=0.7, min_pos_iou=0.7,
                    match_low_quality=False),
                sampler=dict(
                    type='RandomSampler', num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=28, pos_weight=-1, debug=False),
        ],
    ),
)

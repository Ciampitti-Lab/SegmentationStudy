"""
WheatSpikeNet (Cascade R-CNN base) with MMDetection 3.x.
"""

from typing import Dict, Any
import torch

from mmengine import ConfigDict
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS


# -------------------------- helpers -------------------------- #
def _to_cfgdict(x):
    """Recursively convert dict/list/tuple to ConfigDict for attribute access."""
    if isinstance(x, dict):
        return ConfigDict({k: _to_cfgdict(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_to_cfgdict(v) for v in x)
    return x


def _model_cfg(
    num_classes: int = 1,
    keep_dcn: bool = False,
    pretrained_backbone: bool = False,
) -> Dict[str, Any]:
    """Return a plain-Python config (will be turned into ConfigDict later)."""

    # ----- Backbone (ResNet50) -----
    backbone = dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=(
            dict(type="Pretrained", checkpoint="torchvision://resnet50")
            if pretrained_backbone
            else None
        ),
    )
    if keep_dcn:
        backbone.update(
            dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True),
        )

    # ----- Model dict -----
    cfg = dict(
        type="CascadeRCNN",
        data_preprocessor=dict(
            type="DetDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
        ),
        backbone=backbone,
        neck=dict(
            type="FPN",
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
        ),
        rpn_head=dict(
            type="RPNHead",
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator",
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
        ),
        roi_head=dict(
            type="CascadeRoIHead",
            num_stages=3,
            stage_loss_weights=[1.0, 0.5, 0.25],
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
            ),
            bbox_head=[
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.1, 0.1, 0.2, 0.2],
                    ),
                    reg_class_agnostic=False,
                ),
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.05, 0.05, 0.1, 0.1],
                    ),
                    reg_class_agnostic=False,
                ),
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.033, 0.033, 0.067, 0.067],
                    ),
                    reg_class_agnostic=False,
                ),
            ],
        ),
        # ----- Legacy-style train/test cfg at top-level -----
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=[
                dict(  # stage 1
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                ),
                dict(  # stage 2
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                ),
                dict(  # stage 3
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                ),
            ],
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type="nms", iou_threshold=0.5),
                max_per_img=100,
            ),
        ),
    )

    return cfg


def wheatspikenet(
    device: str = "cpu",
    num_classes: int = 1,
    keep_dcn: bool = False,
    pretrained_backbone: bool = False,
):
    """Build and return the model in eval() mode, moved to the requested device."""
    register_all_modules()

    cfg = _model_cfg(
        num_classes=num_classes,
        keep_dcn=keep_dcn,
        pretrained_backbone=pretrained_backbone,
    )

    # Tranform into ConfigDict for attribute-style access from mmdet
    cfg = _to_cfgdict(cfg)

    model = MODELS.build(cfg)
    # init_weights
    if hasattr(model, "init_weights"):
        model.init_weights()

    device = device.lower()
    use_cuda = (device == "cuda") and torch.cuda.is_available()
    model.to("cuda" if use_cuda else "cpu").eval()
    return model

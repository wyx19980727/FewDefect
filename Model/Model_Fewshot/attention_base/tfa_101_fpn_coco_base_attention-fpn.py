_base_ = [
    '../_base_/datasets/fine_tune_based/base_neu_coco.py',
    '../_base_/schedules/schedule.py',
    '../_base_/models/self-attention/faster_rcnn_r50_caffe_attention-fpn.py',
    '../_base_/default_runtime.py'
]
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'))

evaluation = dict(interval=400, metric='bbox',save_best='bbox_mAP',classwise=True,iou_thrs = [0.5])

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6800, 7600])
runner = dict(type='IterBasedRunner', max_iters=10000)

# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=3)))

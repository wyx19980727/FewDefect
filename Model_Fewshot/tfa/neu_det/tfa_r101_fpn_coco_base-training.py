_base_ = [
    '../../_base_/datasets/fine_tune_based/base_neu_coco.py',
    '../../_base_/schedules/schedule.py',
    '../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../_base_/default_runtime.py'
]
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'))
lr_config = dict(warmup_iters=1000, step=[85000, 100000])
runner = dict(max_iters=110000)
# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=3)))

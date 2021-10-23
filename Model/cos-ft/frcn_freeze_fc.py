_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn_fc.py",
    "../_base_/datasets/coco_detection.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py"
]

# optimizer
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[17, 19])
runner = dict(type='EpochBasedRunner', max_epochs=24)
work_dir = "/home/dlsuncheng/Work_dir/Steel_Defect/20211012/FRCN_cos-ft/"

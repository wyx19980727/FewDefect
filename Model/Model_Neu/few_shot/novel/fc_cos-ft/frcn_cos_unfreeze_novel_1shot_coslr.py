_base_ = [
    "../../_base_/models/faster_rcnn_r50_fpn_novel_fc_cos.py",
    "../../_base_/datasets/coco_detection_novel_1shot.py",
    "../../_base_/schedules/schedule_1x.py",
    "../../_base_/default_runtime.py"
]

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

# optimizer = dict(type='Adam', lr=0.0025, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))


# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    min_lr_ratio=0.001)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

runner = dict(type='EpochBasedRunner', max_epochs=10)
load_from = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Weights/weights_removed.pth"
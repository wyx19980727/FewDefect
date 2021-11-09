_base_ = [
    "../../_base_/models/faster_rcnn_r50_fpn_novel_fc_cos.py",
    "../../_base_/datasets/coco_detection_novel_30shot.py",
    "../../_base_/schedules/schedule_1x.py",
    "../../_base_/default_runtime.py"
]

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))


# learning policy
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.001,
    step=[17, 19])

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

runner = dict(type='EpochBasedRunner', max_epochs=24)
work_dir = "/home/dlsuncheng/Work_dir/FsMMdet/20211101/FRCN_fc_cos-ft/novel_unfreeze_30shot"
load_from = "/home/dlsuncheng/FSOD/FsMMdet/Weights/model_reset_remove.pth"
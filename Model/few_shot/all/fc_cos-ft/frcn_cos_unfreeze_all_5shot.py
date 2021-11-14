_base_ = [
    "../../../_base_/models/faster_rcnn_r50_fpn_all_fc_cos.py",
    "../../../_base_/datasets/all/coco_detection_all_5shot.py",
    "../../../_base_/schedules/schedule_1x.py",
    "../../../_base_/default_runtime.py"
]

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))


# learning policy
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    step=[17, 19])

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

runner = dict(type='EpochBasedRunner', max_epochs=24)

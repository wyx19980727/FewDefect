# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotCocoDataset
data_root = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Datasets/NEU_DET/"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotNEUDataset',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root+'/COCO_Annotation/Seed1/trainval.json')
        ],
        img_prefix=data_root+"/JPEGImages",
        pipeline=train_pipeline,
        classes=None),
    val=dict(
        type='FewShotNEUDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root+'/COCO_Annotation/Seed1/test.json')
        ],
        img_prefix=data_root+"/JPEGImages",
        pipeline=test_pipeline,
        classes=None),
    test=dict(
        type='FewShotNEUDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root+'/COCO_Annotation/Seed1/test.json')
        ],
        img_prefix=data_root+"/JPEGImages",
        pipeline=test_pipeline,
        test_mode=True,
        classes=None))
evaluation = dict(interval=200, metric='bbox',save_best='bbox_mAP',classwise=True,iou_thrs = [0.5])

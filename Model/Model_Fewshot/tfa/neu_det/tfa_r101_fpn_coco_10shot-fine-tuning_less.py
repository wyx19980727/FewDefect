_base_ = [
    '../../_base_/datasets/fine_tune_based/few_shot_neu_coco.py',
    '../../_base_/schedules/schedule.py', '../tfa_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        type='FewShotNEUDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_10SHOT')],
        num_novel_shots=10,
        num_base_shots=10,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(interval=400,class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'],metric='bbox',save_best='bbox_mAP',classwise=True,iou_thrs = [0.5])

checkpoint_config = dict(interval=80000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=10, step=[7800])
runner = dict(max_iters=8000)
model = dict(
    frozen_parameters=['backbone', 'neck'],
    roi_head=dict(bbox_head=dict(num_classes=6))
    )# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
load_from = ("/home/dlsuncheng/Projects/FSOD/FsMMdet/Weights/base_model_random_init_bbox_head.pth")

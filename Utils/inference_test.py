# #%%
# from mmcv import Config
# from mmdet import datasets
# from mmdet.models import build_detector
# from mmdet.datasets import build_dataset
# from mmcv.runner import load_checkpoint
# import numpy as np
# import time
# import torch

# timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

# teacher_config_path = '/home/dlsuncheng/Projects/FSOD/FsMMdet/Model/Model_Neu/full_data/frcn_all.py'
# teacher_config = Config.fromfile(teacher_config_path)
# teacher_model = build_detector(teacher_config["model"])

# # datasets = build_dataset(teacher_config.data.train,rank=None,work_dir = None,timestamp = timestamp)

# teacher_ckpt_path = "/home/dlsuncheng/Work_dir/Steel_Defect/20211021/FRCN_all/epoch_23.pth"
# load_checkpoint(teacher_model,teacher_ckpt_path,map_location="cpu")

# img = torch.zeros([1 ,3, 200, 200])
# # img_metas = dict(img = img,
# #                  img_fields=["img"],
# #                  img_shape = img.shape,
# #                  ori_shape = img.shape)

# img_metas = None

# teacher_x = teacher_model.extract_feat(img)
# out_teacher = teacher_model.rpn_head(teacher_x)
# results = teacher_model.roi_head.simple_test(teacher_x,out_teacher,img_metas)
# print(1)
# #%%
# from mmcv import Config
# from mmdet import datasets
# from mmdet.datasets import build_dataset

# timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
# teacher_config = Config.fromfile(teacher_config_path)

# teacher_config_path = '/home/dlsuncheng/Projects/FSOD/FsMMdet/Model/Model_Neu/full_data/frcn_all.py'
# datasets = build_dataset(teacher_config.data.train)

#%%
### model direct inference
import torch
from mmdet.apis import init_detector,inference_detector

teacher_config_path = '/home/dlsuncheng/Projects/FSOD/FsMMdet/Model/Model_Neu/full_data/frcn_all.py'
teacher_ckpt_path = "/home/dlsuncheng/Work_dir/FsMMdet/20211102/FRCN_all/zero_dce/best_bbox_mAP.pth"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Datasets/NEU_DET/JPEGImages/crazing_4.jpg"
model = init_detector(teacher_config_path, teacher_ckpt_path, device="cuda:0")
# test a single image
result = inference_detector(model,img)
print(1)
# %%

#%%
#### check mmfewshot dataloader from config file
from mmcv import Config
from mmcv.utils import config
from mmfewshot.detection.datasets import build_dataset,build_dataloader
import time 

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

base_config_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_base-training.py"
novel_config_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_10shot-fine-tuning.py"

# base_config = Config.fromfile(base_config_path)
novel_config = Config.fromfile(novel_config_path)
# for cfg in [base_config,novel_config]:
datasets = build_dataset(
        novel_config.data.train,
        rank=None,
        work_dir=None,
        timestamp=timestamp)

# print(datasets)
# %%
### print complete config file
import argparse
import warnings

from mmcv import Config, DictAction

base_config_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_base-training.py"
novel_config_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_10shot-fine-tuning.py"
ld_config = "/home/dlsuncheng/Packages/MMdet/configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py"
frcn_config = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model/Model_Neu/full_data/frcn_all.py"
aa_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_10shot-fine-tuning_less.py"
attention_fpn = "/home/user/sun_chen/Work_dir/AttentionFPN/20211218/attention-fpn/tfa_101_fpn_coco_base_attention-fpn.py"
# base_config = Config.fromfile(base_config_path)
# novel_config = Config.fromfile(novel_config_path)
def print_cfg(cfg_path):
    cfg = Config.fromfile(cfg_path)

    # # import modules from string list.
    # if cfg.get('custom_imports', None):
    #     from mmcv.utils import import_modules_from_strings
    #     import_modules_from_strings(**cfg['custom_imports'])
    print(f'Config:\n{cfg.pretty_text}')
cfg = Config.fromfile(attention_fpn)
print(type(cfg.model.neck.self_attention["layer"]))

# print_cfg(attention_fpn)
# %%
### check checkpoint
import torch
import os

random_init_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Weights/base_model_random_init_bbox_head.pth"
base_best_path = "/home/dlsuncheng/Work_dir/FsMMdet/20211201/base_train_10000iter/best_bbox_mAP.pth"

def show_weight_param(path):
    print("####################################")
    torch_pth = torch.load(path)
    for param_name in torch_pth["state_dict"].keys():
        if "roi_head" in param_name:
            print(param_name)
            print(torch_pth["state_dict"][param_name].shape)
    print("####################################")

show_weight_param(base_best_path)
show_weight_param(random_init_path)

# %%
from inspect import FrameInfo
import random
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os
from tqdm import tqdm
import numpy as np
import json
import cv2
def Read_Json(json_path):
    with open(json_path,"r") as load_f:
        json_file = json.load(load_f)
    return json_file

img = np.ones((128,728,3), dtype="uint8")
img.fill(255)
draw = Image.fromarray(img)

anno_path = "/home/dlsuncheng/Dataset/Steel_Defect/Annotation/Annotation.json"
json_file = Read_Json(anno_path)
annos = json_file["annotations"]

for anno in annos[1000:1050]:
    x, y, w, h = anno['bbox']
    print(x,y,w,h)
    bbox = (int(x), int(y), int(x+w), int(y+h))
    ImageDraw.Draw(draw).rectangle(bbox,outline='blue', width=1)
draw.save("a.png")

# %%
### model inference
student_module = "neck.fpn_convs.3.conv"
new_nodule = 'student_' + student_module.replace('.','_')
# %%
from mmcv.cnn.bricks.non_local import NonLocal2d
from mmcv.cnn.bricks.context_block import ContextBlock
from mmcv.cnn.bricks.generalized_attention import GeneralizedAttention
a = "NonLocal2d"
print(eval(a)(in_channels=10))
# %%
test = [3,4,5,6]
try_list = [7,8,9,10,11,12]
for i,num in enumerate(test):
    print(num)
    print()
    try_list[len(try_list)-i-1] = num*try_list[len(test)-i-1]
    print(try_list)
# %%

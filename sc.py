#%%
#### check mmfewshot dataloader from config file
from mmcv import Config
from mmfewshot.detection.datasets import build_dataset,build_dataloader
import time 

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

base_config_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_base-training.py"
novel_config_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_10shot-fine-tuning.py"

# base_config = Config.fromfile(base_config_path)
novel_config = Config.fromfile(novel_config_path)
# for cfg in [base_config,novel_config]:
datasets = build_dataset(
        novel_config.data.train,
        rank=None,
        work_dir=None,
        timestamp=timestamp)

# print(datasets)



# # import json
# def Read_Json(json_path):
#     with open(json_path,"r") as load_f:
#         json_file = json.load(load_f)
#     return json_file
# base_anno = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Datasets/NEU_DET/COCO_Annotation/Seed1/trainval.json"
# base_json = Read_Json(base_anno)
# print(len(base_json["annotations"]))

# %%
import argparse
import warnings

from mmcv import Config, DictAction

base_config_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_base-training.py"
novel_config_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_10shot-fine-tuning.py"

base_config = Config.fromfile(base_config_path)
novel_config = Config.fromfile(novel_config_path)
def print_cfg(cfg_path):
    cfg = Config.fromfile(cfg_path)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    print(f'Config:\n{cfg.pretty_text}')
print_cfg(base_config_path)
print_cfg(novel_config_path)
# %%

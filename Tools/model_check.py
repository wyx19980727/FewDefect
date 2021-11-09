import json
import pandas as pd
from collections import Counter
import numpy as np
from mmdet.datasets import build_dataset
from mmcv import Config
import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",default="",type=str,help="model config path")
    parser.add_argument("--model_component")

def Read_Json(path):
    with open(path,"r") as load_f:
        json_file = json.load(load_f)
    return json_file
    
# json_path = './Datasets/NEU_DET/COCO_Annotation/base_novel/seed1/30shot_all_seed1.json'
# json_file = Read_Json(json_path)
# imgs = json_file["images"]
# annos = json_file["annotations"]

if __name__ == "__main__":
    config_path = "/home/dlsuncheng/FSOD/FsMMdet/Model/full_data/frcn_cos_unfreeze_all_30shot.py"
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.val)

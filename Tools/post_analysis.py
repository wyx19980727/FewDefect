### confusion matrix 
### PR curve
### visualization


import json
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import argparse
def Read_Json(path):
    with open(path,"r") as load_f:
        json_file = json.load(load_f)
    return json_file

def args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--result_file",type=str,default="",
                        help="path for the result json file")
    parser.add_argument("--gt_file",type=str,default = "",
                        help="filepath for ground truth anno file")
    parser.add_argument("--save_dir",type=str,default="/home/dlsuncheng/FSOD/FsMMdet/Weights/",
                        help="path for the pth file")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    result_path = args.result_file
    
    test_anno_path = args.gt_file

    result_json = Read_Json(result_path)
    test_json = Read_Json(test_anno_path)

    img_info = pd.DataFrame(columns=["image_id","gt_category","pred_category"])
    gt_image_id_list = [img_info["image_id"] for img_info in test_json["annotations"]]
    gt_image_id = list(set(gt_image_id_list))
    gt_image_id.sort()
    img_info["image_id"] = gt_image_id

    ## 误分类分析
    ### 同一张图片内的类别应该相同
    ### image_id:[category_ids]
    pred_category = []
    gt_category = []
    for img_id in gt_image_id:
        pred_cat_id_list = np.array([img["category_id"] for img in result_json if img["image_id"]==img_id])
        gt_cat_id_list = np.array([anno["category_id"] for anno in test_json["annotations"] if anno["image_id"]==img_id])
        gt_cat_id = gt_cat_id_list[0]*np.ones(pred_cat_id_list.shape)
        pred_category.extend(pred_cat_id_list)
        gt_category.extend(gt_cat_id)

    con=confusion_matrix(gt_category,pred_category)
    # sn.set()

    # plt.rcParams['font.sans-serif']=['SimHei']
    # fig = plt.figure(figsize=(10,10))
    # sn.heatmap(con,annot = True,linewidths=1)
    # plt.savefig("./cls_heat.png")
    # plt.show()
#%%
import json
import pickle
from collections import OrderedDict

import numpy as np
from mmdet.core import eval_map, eval_recalls

pkl_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/results.pkl"
json_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Datasets/NEU_DET/COCO_Annotation/test.json"
def read_json(path):
    with open(path,"r") as load_f:
        json_file = json.load(load_f)
    return json_file
def read_pkl(path):
    with open(path,"rb") as load_f:
        pkl_file = pickle.load(load_f)
    return pkl_file
gt_file = read_json(json_path)
pred_file = read_pkl(pkl_path)
#%%
annotations = []
image_id_ist = [img["id"] for img in gt_file["images"]]
for id in image_id_ist:
    box_anno = [anno["bbox"]for anno in gt_file["annotations"] if anno["image_id"]==id ]
    box_class = [anno["category_id"]for anno in gt_file["annotations"] if anno["image_id"]==id ]   
    box_anno = np.array(box_anno, ndmin=2) - 1
    box_class = np.array(box_class) 
    img_anno_info = dict(bboxes=box_anno,labels=box_class)
    annotations.append(img_anno_info)
#%%
### voc eval

results = pred_file
metric='recall'
logger=None
proposal_nums=(100, 300, 1000)
iou_thr=0.5
scale_ranges=None

eval_results = OrderedDict()
iou_thrs = [0.5]

if metric == 'mAP':
    mean_aps = []
    for iou_thr in iou_thrs:
        mean_ap, _ = eval_map(
            results,
            annotations,
            scale_ranges=None,
            iou_thr=iou_thr,
            dataset="sc",
            logger=logger,
            use_legacy_coordinate=True)
        mean_aps.append(mean_ap)
        eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
    eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
elif metric == 'recall':
    gt_bboxes = [ann['bboxes'] for ann in annotations]
    recalls = eval_recalls(
        gt_bboxes,
        results,
        proposal_nums,
        iou_thrs,
        logger=logger,
        use_legacy_coordinate=True)
    for i, num in enumerate(proposal_nums):
        for j, iou_thr in enumerate(iou_thrs):
            eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
    if recalls.shape[1] > 1:
        ar = recalls.mean(axis=1)
        for i, num in enumerate(proposal_nums):
            eval_results[f'AR@{num}'] = ar[i]

# %%
from mmdet.datasets import VOCDataset, coco
anno_file = "/crazing_1.xml"
pipeline = None
img_dir  = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Datasets/NEU_DET/"
anno_dir = "/Annotations"
voc = VOCDataset(ann_file=anno_file,pipeline=pipeline,img_subdir = img_dir,ann_subdir = anno_dir)

# %%
### fast inference model
from mmdet.apis import init_detector,inference_detector
import PIL
import mmcv
import numpy as np
cfg_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model/full_data/frcn_all.py"
checkpoint_path = "/home/dlsuncheng/Work_dir/Steel_Defect/20211021/FRCN_all/epoch_23.pth"
img_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Datasets/NEU_DET/JPEGImages/crazing_3.jpg"

image = mmcv.imread(img_path).astype(np.uint8)
model = init_detector(cfg_path,checkpoint_path)

result = inference_detector(model,image)
# %%
from mmcv import Config
from mmdet.datasets import build_dataset
config = Config.fromfile(cfg_path)
coco_data = build_dataset(config.data.train)
# %%
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()
# if classwise:  # Compute per-category AP
#     # Compute per-category AP
#     # from https://github.com/facebookresearch/detectron2/
#     precisions = cocoEval.eval['precision']
#     assert len(self.cat_ids) == precisions.shape[2]
    
#     results_per_category = []
#     for idx, catId in enumerate(self.cat_ids):
#         # area range index 0: all area ranges
#         # max dets index -1: typically 100 per image
#         nm = self.coco.loadCats(catId)[0]
#         precision = precisions[:, :, idx, 0, -1]
#         precision = precision[precision > -1]
#         if precision.size:
#             ap = np.mean(precision)
#         else:
#             ap = float('nan')
#         results_per_category.append(
#             (f'{nm["name"]}', f'{float(ap):0.3f}'))

#     num_columns = min(6, len(results_per_category) * 2)
#     results_flatten = list(
#         itertools.chain(*results_per_category))
#     headers = ['category', 'AP'] * (num_columns // 2)
#     results_2d = itertools.zip_longest(*[
#         results_flatten[i::num_columns]
#         for i in range(num_columns)
#     ])
#     table_data = [headers]
#     table_data += [result for result in results_2d]
#     table = AsciiTable(table_data)
#     print_log('\n' + table.table, logger=logger)
#%%
### evalutate 的format是什么样？
### python 断点调试
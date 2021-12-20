import torch
import mmcv
from mmdet.models import build_detector
from mmdet.datasets import build_dataset


with torch.no_grad():
    teacher_x = self.teacher_model.extract_feat(img)
    out_teacher = self.teacher_model.rpn_head(teacher_x)
    results_teacher = self.teacher_model.roi_head(teacher_x,out_teacher,img_metas,rescale=False)
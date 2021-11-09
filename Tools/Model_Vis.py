import torchvision
import torch
import hiddenlayer as h
from mmdet.apis.inference import *
from mmdet.models import build_detector

import torch.nn as nn
import mmcv
from torchviz import make_dot

config = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Model/few_shot/fc_cos-ft/frcn_cos_unfreeze_novel_30shot.py"
# detector = init_detector(config)
# detector.forward_dummy
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.config = mmcv.Config.fromfile(config)
        self.model= init_detector(self.config)

    def forward(self,x):
        result = self.model.forward_dummy(x)
        # result = self.model(x)
        return result
input = torch.zeros([1 ,3, 200, 200])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input = input.to(device)
input.to(device)
MyConvNet = ConvNet()
MyConvNet.model.to(device)
y_rpn,y = MyConvNet(input)

dot = make_dot(y, params=dict(MyConvNet.model.named_parameters()))
dot.format = 'png'
dot.render('torchviz-sample')
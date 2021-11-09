import torch.nn as nn
import torch
  
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead


@HEADS.register_module()
class CosineSimHead(BBoxHead):
    def __init__(
        self, 
        init_cfg=dict(
            type='Normal',
            override=[
                dict(type='Normal', name='fc_cls', std=0.01),
                dict(type='Normal', name='fc_reg', std=0.001),]),
        *args,
        **kwargs
    ):
    
        """
                                                   /-> cosine simmilarity based cls -> cls
        roi features -> shared convs -> shared fcs
                                                   \-> reg fcs -> reg

        """
        super(CosineSimHead, self).__init__(*args, init_cfg=init_cfg, **kwargs)

        self.input_size = self.in_channels*self.roi_feat_area 
        
        ### define model
        if self.with_cls:
            
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.input_size,
                out_features=cls_channels)

        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.input_size,
                out_features=out_dim_reg)

        self.scale = 20

    def forward(self, x):

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)

        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = torch.norm(self.fc_cls.weight.data,
                            p=2, dim=1).unsqueeze(1).expand_as(self.fc_cls.weight.data)

        self.fc_cls.weight.data = self.fc_cls.weight.data.div(temp_norm + 1e-5)

        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.fc_reg(x)
        return scores, proposal_deltas
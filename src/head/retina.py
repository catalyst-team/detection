from typing import Tuple

import torch
import torch.nn as nn

from .core import HeadSpec
from utils import ConvModule, normal_init, bias_init_with_prob


class RetinaHead(HeadSpec):

    def __init__(
        self,
        stacked_convs=4,
        **kwargs
    ):
        self.stacked_convs = stacked_convs
        super(RetinaHead, self).__init__(**kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.num_classes,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4,
            3,
            padding=1)

    def _init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(
        self,
        single_level_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_feat = single_level_features
        reg_feat = single_level_features
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

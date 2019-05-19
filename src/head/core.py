from typing import List, Tuple
from abc import abstractmethod, ABC

import torch
import torch.nn as nn

from utils.initialization import normal_init
from utils.misc import multi_apply


class HeadSpec(ABC, nn.Module):

    def __init__(
        self,
        num_classes,
        in_channels,
        num_anchors,
        feat_channels=256,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_anchors = num_anchors

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.num_classes, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def _init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(
        self,
        single_level_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_score = self.conv_cls(single_level_features)
        bbox_pred = self.conv_reg(single_level_features)
        return cls_score, bbox_pred

    def forward(
        self,
        multi_level_features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        output = multi_apply(self.forward_single, multi_level_features)
        return output

    def set_requires_grad(self, requires_grad):
        for m in self.modules():
            for param in m.parameters():
                param.requires_grad = bool(requires_grad)

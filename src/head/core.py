from typing import List, Tuple
from abc import abstractmethod, ABC

import torch
import torch.nn as nn

from utils.initialization import normal_init


class HeadSpec(ABC, nn.Module):

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        anchor_scales=[8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        anchor_base_sizes=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(anchor_strides) \
            if anchor_base_sizes is None \
            else anchor_base_sizes
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

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

    @abstractmethod
    def forward(
        self,
        multi_level_features: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        output = list(map(self.forward_single, multi_level_features))
        return output

    def set_requires_grad(self, requires_grad):
        for m in self.modules():
            for param in m.parameters():
                param.requires_grad = bool(requires_grad)

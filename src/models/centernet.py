from typing import Dict, Callable

import torch
import torch.nn as nn


class CenterNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_fn: Callable,
        embedding_dim: int = 128,
        model_params: dict = None,
        backbone_key: str = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        model_params = model_params or {}
        self.backbone = model_fn(**model_params)
        self.backbone_key = backbone_key

        self.head_heatmap = nn.Conv2d(
            embedding_dim, self.num_classes,
            kernel_size=(3, 3), padding=1, bias=True
        )
        self.head_heatmap.bias.data.fill_(-4.)  # MUST FIX BIAS ELSE START LOSS OVER 9000
        self.head_width_height = nn.Conv2d(embedding_dim, 2, kernel_size=(3, 3), padding=1, bias=True)
        self.head_offset_regularizer = nn.Conv2d(embedding_dim, 2, kernel_size=(3, 3), padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        value = self.backbone(x)
        if self.backbone_key is not None:
            value = value[self.backbone_key]

        features = torch.relu_(value)
        value = {
            "hm": self.head_heatmap(features),
            "wh": self.head_width_height(features),
            "reg": self.head_offset_regularizer(features),
        }
        return value

    def predict(self, x: torch.Tensor):
        """
        Method to trace
        """
        value = self.forward(x)
        return value["hm"], value["wh"], value["reg"]


__all__ = ["CenterNet"]

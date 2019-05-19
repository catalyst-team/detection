from typing import List, Tuple
import torch

from .core import DetectorSpec
from src.registry import BACKBONES, NECKS, HEADS, BBOXERS


class SingleStageDetector(DetectorSpec):
    def __init__(
        self,
        backbone_params,
        neck_params=None,
        head_params=None,
        bboxer_params=None,
    ):
        super().__init__()
        self.backbone = BACKBONES.get_from_params(**backbone_params)
        if neck_params is not None:
            self.neck = NECKS.get_from_params(**neck_params)
        self.bboxer = BBOXERS.get_from_params(**bboxer_params)
        self.bbox_head = HEADS.get_from_params(
            **head_params,
            num_classes=self.bboxer.num_classes,
            num_anchors=self.bboxer.num_anchors
        )

    def _extract_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _extract_multi_level_predictions(
        self,
        image: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        image = self._extract_feature_maps(image)
        # List[Tuple(cls_score, bbox_pred), ]
        # -> Tuple(List[cls_score], List[bbox_pred])
        multi_level_cls_score, multi_level_bbox_pred = self.bbox_head(image)
        return multi_level_cls_score, multi_level_bbox_pred

    def forward_train(self, image, **kwargs):
        multi_level_cls_score, multi_level_bbox_pred = \
            self._extract_multi_level_predictions(image)
        losses_cls, losses_reg = self.bboxer.compute_loss(
            multi_level_cls_score, multi_level_bbox_pred, **kwargs
        )

        return multi_level_cls_score, multi_level_bbox_pred, \
               losses_cls, losses_reg

    def forward_infer(self, image, **kwargs):
        multi_level_cls_score, multi_level_bbox_pred = \
            self._extract_multi_level_predictions(image)
        predictions = self.bboxer.get_bboxes(
            multi_level_cls_score, multi_level_bbox_pred, **kwargs
        )
        return multi_level_cls_score, multi_level_bbox_pred, predictions

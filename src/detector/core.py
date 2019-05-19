from typing import List
from abc import abstractmethod, ABC

import torch
import torch.nn as nn


class DetectorSpec(ABC, nn.Module):

    # @property
    # def with_neck(self):
    #     return hasattr(self, "neck") and self.neck is not None

    # @property
    # def with_bbox(self):
    #     return hasattr(self, "bbox_head") and self.bbox_head is not None

    # @property
    # def with_mask(self):
    #     return hasattr(self, "mask_head") and self.mask_head is not None

    @abstractmethod
    def _extract_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def forward_train(self, x, **kwargs):
        pass

    @abstractmethod
    def forward_infer(self, x, **kwargs):
        pass

    def forward(self, image, inference=False, **kwargs):
        if not inference:
            return self.forward_train(image, **kwargs)
        else:
            return self.forward_infer(image, **kwargs)

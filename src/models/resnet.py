from catalyst.contrib.models import segmentation

from .centernet import CenterNet


class ResnetCenterNet(CenterNet):
    def __init__(
        self,
        num_classes: int,
        down_ratio: int = 1,
        embedding_dim: int = 128,
        arch: str = "ResnetFPNUnet",
        backbone_params: dict = None,
    ):

        model_fn = segmentation.__dict__[arch]
        backbone_params = backbone_params or {}
        model_params = {"num_classes": embedding_dim, **backbone_params}
        super().__init__(
            num_classes=num_classes,
            model_fn=model_fn,
            down_ratio=down_ratio,
            embedding_dim=embedding_dim,
            model_params=model_params
        )

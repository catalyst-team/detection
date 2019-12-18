from catalyst.contrib.models import segmentation

from .centernet import CenterNet


class ResnetCenterNet(CenterNet):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        arch: str = "ResnetFPNUnet",
        backbone_params: dict = None,
    ):

        model_fn = segmentation.__dict__[arch]
        backbone_params = backbone_params or {}
        model_params = {"num_classes": embedding_dim, **backbone_params}
        super().__init__(
            num_classes,
            model_fn,
            embedding_dim,
            model_params=model_params
        )

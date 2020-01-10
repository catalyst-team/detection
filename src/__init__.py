# flake8: noqa
# from .runner import Runner
from catalyst.dl import SupervisedRunner as Runner
from catalyst.dl import registry

from .experiment import Experiment

from .callbacks import DecoderCallback, MeanAPCallback
from .losses import CenterNetDetectionLoss, \
    RegL1Loss, MSEIndLoss, BCEIndLoss, FocalIndLoss
from . import models


registry.Criterion(CenterNetDetectionLoss)
registry.Criterion(RegL1Loss)
registry.Criterion(MSEIndLoss)
registry.Criterion(BCEIndLoss)
registry.Criterion(FocalIndLoss)

registry.Callback(DecoderCallback)
registry.Callback(MeanAPCallback)

registry.MODELS.add_from_module(models)

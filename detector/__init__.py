# flake8: noqa
from .runner import Runner
from .experiment import Experiment

from .callbacks import DecoderCallback, MeanAPCallback #, CriterionDebugCallback
from .losses import CenterNetDetectionLoss, \
    RegL1Loss, MSEIndLoss, BCEIndLoss, FocalIndLoss
from . import models

from catalyst.dl import registry

registry.Criterion(CenterNetDetectionLoss)
registry.Criterion(RegL1Loss)
registry.Criterion(MSEIndLoss)
registry.Criterion(BCEIndLoss)
registry.Criterion(FocalIndLoss)

#registry.Callback(CriterionDebugCallback)
registry.Callback(DecoderCallback)
registry.Callback(MeanAPCallback)

registry.MODELS.add_from_module(models)

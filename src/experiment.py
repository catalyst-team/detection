import datetime
import os
from collections import OrderedDict
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import safitty
import torch
from catalyst.data.collate_fn import FilteringCollateFn
from catalyst.data.sampler import MiniEpochSampler
from catalyst.dl import ConfigExperiment

from .dataset import DetectionDataset
from .transforms import train_transform, valid_transform, infer_transform

torch.multiprocessing.set_sharing_strategy("file_system")


class Experiment(ConfigExperiment):
    def get_datasets(
            self,
            stage: str,
            **kwargs,
    ):
        def process_kwargs_by_default_values(parameter, default_parameter):
            if parameter not in kwargs:
                if default_parameter not in kwargs:
                    raise ValueError('You must specify \"{}\" or default value(\"{}\") in config'
                                     .format(parameter, default_parameter))
                else:
                    kwargs[parameter] = kwargs[default_parameter]

        process_kwargs_by_default_values('train_annotation_file', 'annotation_file')
        process_kwargs_by_default_values('valid_annotation_file', 'annotaiton_file')
        process_kwargs_by_default_values('train_images_dir', 'images_dir')
        process_kwargs_by_default_values('valid_images_dir', 'images_dir')

        train_dataset = DetectionDataset(annotation_file=kwargs['train_annotation_file'],
                                         images_dir=kwargs['train_images_dir'],
                                         down_ratio=kwargs['down_ratio'],
                                         max_objects=kwargs['max_objs'],
                                         num_categories=kwargs['num_categories'],
                                         image_size=kwargs['image_size'],
                                         transform=None
                                         )

        # TODO TRAIN IS NOW EQUAL TO VAL
        valid_dataset = DetectionDataset(annotation_file=kwargs['valid_annotation_file'],
                                         images_dir=kwargs['valid_images_dir'],
                                         down_ratio=kwargs['down_ratio'],
                                         max_objects=kwargs['max_objs'],
                                         num_categories=kwargs['num_categories'],
                                         image_size=kwargs['image_size'],
                                         transform=None
                                         )

        return {
            'train': {
                'dataset': train_dataset,
                'collate_fn': FilteringCollateFn('bboxes', 'labels')
            },
            'valid': {
                'dataset': valid_dataset,
                'collate_fn': FilteringCollateFn('bboxes', 'labels')
            },
        }

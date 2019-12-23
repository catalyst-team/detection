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
        train_dataset = DetectionDataset(annotation_file=kwargs['annotation_file'],
                                         images_dir=kwargs['images_dir'],
                                         down_ratio=kwargs['down_ratio'],
                                         max_objects=kwargs['max_objs'],
                                         num_categories=kwargs['num_categories'],
                                         image_size=kwargs['image_size'],
                                         transform=None
                                         )

        # TODO TRAIN IS NOW EQUAL TO VAL
        valid_dataset = DetectionDataset(annotation_file=kwargs['annotation_file'],
                                         images_dir=kwargs['images_dir'],
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

    '''
    @staticmethod
    def get_transforms(
        *,
        stage: str = None,
        mode: str = None,
        image_size: Tuple[int, int],
    ):
        height, width = image_size

        if mode == "train":
            transforms = train_transform(height)
        elif mode == "valid":
            transforms = valid_transform(height)
        else:
            return infer_transform(height)
        return transforms
    
    def get_datasets(
        self,
        stage: str,
        classes,
        dataset_root,
        image_size,
        n_jobs: str,
        image_folders: dict = None,
        down_ratio: float = 1.0,
        max_objs: int = 128,
        sampler_params: dict = None,
        train_samples: str = None,
        valid_samples: str = None,
    ):
        datasets = OrderedDict()

        class2id = dict(zip(classes, range(len(classes))))

        dataset_root = Path(dataset_root)
        data_dir = dataset_root

        train_samples_path = dataset_root / train_samples
        valid_samples_path = dataset_root / valid_samples

        load = partial(
            _load_data, n_jobs=n_jobs,
            data_dir=data_dir, class2label=class2id, image_folders=image_folders
        )

        train_filepaths, train_bboxes, train_labels = load(samples_path=train_samples_path)

        valid_filepaths, valid_bboxes, valid_labels = load(samples_path=valid_samples_path)

        num_classes = len(classes)
        if not stage.startswith("infer"):
            collate_fn = FilteringCollateFn("bboxes", "labels", "filename")
            train_set = DetectionDataset(
                num_classes,
                down_ratio,
                max_objs,
                train_filepaths, train_bboxes, train_labels,
                image_size=image_size,
                transform=self.get_transforms(
                    mode="train",
                    image_size=image_size
                ),
            )

            valid_set = DetectionDataset(
                num_classes,
                down_ratio,
                max_objs,
                valid_filepaths, valid_bboxes, valid_labels,
                image_size=image_size,
                transform=self.get_transforms(
                    mode="valid",
                    image_size=image_size
                ),
            )

            datasets["train"] = {
                "dataset": train_set,
                "collate_fn": collate_fn
            }
            tqdm.write(f"\nTrain dataset len: {len(train_set)}")

            datasets["valid"] = {
                "dataset": valid_set,
                "collate_fn": collate_fn
            }
            tqdm.write(f"\nValid dataset len: {len(valid_set)}")

            if sampler_params is not None:
                safitty.set(
                    datasets, "train", "sampler",
                    value=MiniEpochSampler(
                        data_len=len(train_set), **sampler_params
                    )
                )
        else:
            collate_fn = FilteringCollateFn("bboxes", "labels", "filename", "orig_image")
            infer_set = DetectionDataset(
                num_classes,
                down_ratio,
                max_objs,
                valid_filepaths + train_filepaths,
                valid_bboxes + train_bboxes,
                valid_labels + train_labels,
                transform=self.get_transforms(
                    mode="valid",
                    image_size=image_size
                ),
            )
            datasets["infer"] = {
                "dataset": infer_set,
                "collate_fn": collate_fn
            }

        return datasets
    '''

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
from tqdm import tqdm

from .dataset import DetectionDataset
from .transforms import train_transform, valid_transform, infer_transform

torch.multiprocessing.set_sharing_strategy("file_system")


def run_one(data):
    sample, data_dir, class2label, image_folders = data
    if sample is None:
        return None

    filepath = data_dir / sample["name"]

    objs = sample["objs"]
    bbox = []
    label = []
    for i, obj in enumerate(objs):
        coords = [
            obj["x_min"],
            obj["y_min"],
            obj["x_max"],
            obj["y_max"]
        ]

        bbox.append(coords)
        label.append(int(class2label[obj["class"]]))

    bbox = np.array(bbox)

    return filepath, bbox, label

def _load_data(
    n_jobs: int,
    data_dir: Path,
    samples_path: Path,
    class2label: Dict[str, int],
    image_folders
):
    filepaths = []
    bboxes = []
    labels = []

    samples = safitty.load(samples_path)

    length = len(samples)

    sequence = zip(
        samples, repeat(data_dir), repeat(class2label), repeat(image_folders)
    )
    with Pool(n_jobs) as p:
        samples_ = list(
            tqdm(
                p.imap(run_one, sequence),
                total=length, desc=f"{samples_path}"
            )
        )

    for (f, b, l) in samples_:
        filepaths.append(f)
        bboxes.append(b)
        labels.append(l)

    tqdm.write(f"{samples_path}: {len(filepaths), len(bboxes), len(labels)}")
    return filepaths, bboxes, labels, ages


class Experiment(ConfigExperiment):
    def _get_logdir(self, config: Dict) -> str:
        timestamp = datetime.datetime.utcnow().strftime("%y%m%d.%H%M%S.%f")
        data_params = safitty.get(config, "stages", "data_params")
        mini_epoch_len = safitty.get(data_params, "sampler_params", "mini_epoch_len", default="none")

        dataset = safitty.get(data_params, "dataset_root", apply=Path).stem

        model = safitty.get(config, "model_params", "backbone_params", "arch")

        num_epochs = safitty.get(config, "shared", "num_epochs")
        batch_size = safitty.get(config, "shared", "batch_size")

        image_size = safitty.get(
            config, "shared", "image_size",
            transform=lambda x: "x".join([str(i) for i in x])
        )
        result = f"{timestamp}" \
            f"-model-{model}" \
            f"-epochs-{num_epochs}" \
            f"-bs-{batch_size}" \
            f"-imgsize-{image_size}" \
            f"-mini_epoch-{mini_epoch_len}" \
            f"-dataset-{dataset}"

        return result

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

import warnings

import torch
from catalyst.data.collate_fn import FilteringCollateFn
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
        process_kwargs_by_default_values('valid_annotation_file', 'annotation_file')
        process_kwargs_by_default_values('train_images_dir', 'images_dir')
        process_kwargs_by_default_values('valid_images_dir', 'images_dir')

        if kwargs['train_annotation_file'] == kwargs['valid_annotation_file']:
            warnings.warn("Valid is now equal to train, is it expected?", RuntimeWarning)

        train_dataset = DetectionDataset(annotation_file=kwargs['train_annotation_file'],
                                         images_dir=kwargs['train_images_dir'],
                                         down_ratio=kwargs['down_ratio'],
                                         max_objects=kwargs['max_objs'],
                                         num_classes=kwargs['num_classes'],
                                         image_size=kwargs['image_size'],
                                         transform=train_transform(kwargs['image_size'][0])
                                         )

        valid_dataset = DetectionDataset(annotation_file=kwargs['valid_annotation_file'],
                                         images_dir=kwargs['valid_images_dir'],
                                         down_ratio=kwargs['down_ratio'],
                                         max_objects=kwargs['max_objs'],
                                         num_classes=kwargs['num_classes'],
                                         image_size=kwargs['image_size'],
                                         transform=valid_transform(kwargs['image_size'][0])
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

from typing import Dict, Optional, Any, Tuple

import cv2
import math
import torch
from torch.utils.data import Dataset
import numpy as np

from .coco import DetectionMSCOCODataset
from catalyst import utils

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def get_affine_transform(
        center,
        scale,
        rot,
        output_size,
        shift=np.array([0, 0], dtype=np.float32),
        inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(point: np.array, transform_matrix: np.array) -> np.array:
    new_pt = np.array([point[0], point[1], 1.], dtype=np.float32).T
    new_pt = np.dot(transform_matrix, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class DetectionDataset(Dataset):
    def __init__(self,
                 annotation_file: str,
                 images_dir: str,
                 down_ratio: int,
                 max_objects: int,
                 num_classes: Optional[int] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 transform: Optional[Any] = None,
                 **kwargs
                 ):
        super(DetectionDataset, self).__init__()

        self._annotations_dataset = DetectionMSCOCODataset(annotation_file, images_dir)

        self._num_classes = num_classes
        if self._num_classes is None:
            self._num_classes = self._annotations_dataset.get_num_classes()

        self._down_ratio = down_ratio
        self._max_objects = max_objects

        assert image_size[0] == image_size[1], "Only square image are now supported"
        self.image_size = image_size[0]
        self.transform = transform

    def __len__(self) -> int:
        return len(self._annotations_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        annotation = self._annotations_dataset[idx]
        image_name = annotation['image_name']
        detections = annotation['detections']

        image = utils.imread(image_name)
        x_scale, y_scale = self.image_size / image.shape[1], self.image_size / image.shape[0]

        image = cv2.resize(image, (self.image_size, self.image_size), cv2.INTER_LINEAR)

        detections = [
            {
                'category_id': detection['category_id'],
                'category_name': detection['category_name'],
                'bbox': detection['bbox'].copy()
            } for detection in detections
        ]

        for detection in detections:
            detection['bbox'][0::2] *= x_scale
            detection['bbox'][1::2] *= y_scale

        bboxes = []
        labels = []
        for detection in detections:
            median_x = (detection['bbox'][0] + detection['bbox'][2]) // 2
            median_y = (detection['bbox'][1] + detection['bbox'][3]) // 2

            # CenterNet are VERY bad when center of detected objects not in the images
            # Let's delete this bboxes
            if not (0 <= median_x <= image.shape[1]) or not (0 <= median_y <= image.shape[0]):
                continue

            detection['bbox'][0::2] = np.clip(detection['bbox'][0::2], 0, image.shape[1])
            detection['bbox'][1::2] = np.clip(detection['bbox'][1::2], 0, image.shape[0])

            bboxes.append(detection['bbox'])
            labels.append(detection['category_id'])

        bboxes = np.array(bboxes)
        labels = np.array(labels)

        if self.transform is not None:
            result = self.transform(
                image=image,
                bboxes=bboxes,
                labels=labels,
            )
        else:
            result = dict(
                image=image,
                bboxes=bboxes,
                labels=labels,
            )

        image = result["image"].astype(np.uint8)
        bboxes = result["bboxes"]
        labels = result["labels"]

        input_height, input_width = image.shape[0], image.shape[1]

        # Normalization
        input = (image.astype(np.float32) / 255.) * 2. - 1.
        input = input.transpose(2, 0, 1)

        output_height = input_height // self._down_ratio
        output_width = input_width // self._down_ratio
        # trans_output = get_affine_transform(center, scale, 0, [output_width, output_height])

        heatmap = np.zeros((self._num_classes, output_height, output_width), dtype=np.float32)
        width_height = np.zeros((self._max_objects, 2), dtype=np.float32)

        reg = np.zeros((self._max_objects, 2), dtype=np.float32)
        ind = np.zeros(self._max_objects, dtype=np.int64)
        reg_mask = np.zeros(self._max_objects, dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian

        new_bboxes = []
        num_objs = min(len(bboxes), self._max_objects)
        for i in range(num_objs):
            bbox = np.array(bboxes[i], dtype=np.float32) / self._down_ratio
            class_id = labels[i]

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_width - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_height - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            new_bboxes.append(bbox)

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                _center = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    dtype=np.float32
                )
                _center_int = _center.astype(np.int32)
                draw_gaussian(heatmap[class_id], _center_int, radius)
                width_height[i] = 1. * w, 1. * h
                ind[i] = _center_int[1] * output_width + _center_int[0]
                reg[i] = _center - _center_int
                reg_mask[i] = 1

        result = {
            "filename": image_name,
            "input": torch.from_numpy(input),
            "hm": torch.from_numpy(heatmap),
            "reg_mask": torch.from_numpy(reg_mask),
            "ind": torch.from_numpy(ind),
            "wh": torch.from_numpy(width_height),
            "reg": torch.from_numpy(reg),
            "bboxes": np.array(bboxes),
            "labels": np.array(labels),
        }

        return result

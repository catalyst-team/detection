from typing import List

import cv2
import math
import torch
from torch.utils.data import Dataset
import numpy as np

import albumentations as A
from .transforms import pre_transform, pre_transform_flickr, augmentations, BBOX_PARAMS
from catalyst import utils

import safitty

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


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
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
    def __init__(
        self,
        num_classes: int,
        down_ratio: float,
        max_objs: int,
        filepaths: List[str],
        bboxes,
        labels,
        image_size,
        transform=None,
    ):
        super(DetectionDataset, self).__init__()
        self.num_classes = num_classes
        self.down_ratio = down_ratio
        self.max_objs = max_objs
        self.filepaths, self.bboxes, self.labels = filepaths, bboxes, labels
        self.image_size = image_size[0]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filename = self.filepaths[idx]
        image = utils.imread(filename)
        
        if self.transform is not None:
            result = self.transform(
                image=image,
                bboxes=self.bboxes[idx],
                labels=self.labels[idx],
            )
        else:
            result = dict(
                image=image,
                bboxes=self.bboxes[idx],
                labels=self.labels[idx],
            )

        image = result["image"].astype(np.uint8)
        bboxes = result["bboxes"])
        labels = result["labels"]]

        input_height, input_width = image.shape[0], image.shape[1]
        center = np.array(
            [input_width / 2.0, input_height / 2.0],
            dtype=np.float32
        )
        scale = np.array([input_width, input_height], dtype=np.float32)

        # recenter and rescale image
        trans_input = get_affine_transform(center, scale, 0, [input_width, input_height])
        input = cv2.warpAffine(
            image,
            trans_input,
            (input_width, input_height),
            flags=cv2.INTER_LINEAR
        )
        # Normalization
        input = (input.astype(np.float32) / 255.) * 2. - 1.
        input = input.transpose(2, 0, 1)

        output_height = input_height // self.down_ratio
        output_width = input_width // self.down_ratio
        trans_output = get_affine_transform(center, scale, 0, [output_width, output_height])

        heatmap = np.zeros((self.num_classes, output_height, output_width), dtype=np.float32)
        weight_height = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros(self.max_objs, dtype=np.int64)
        reg_mask = np.zeros(self.max_objs, dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian

        new_bboxes = []
        num_objs = min(len(self.bboxes[idx]), self.max_objs)
        for i in range(num_objs):
            bbox = np.array(bboxes[i], dtype=np.float32)
            class_id = labels[i]
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
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
                weight_height[i] = 1. * w, 1. * h
                ind[i] = _center_int[1] * output_width + _center_int[0]
                reg[i] = _center - _center_int
                reg_mask[i] = 1

        _res_ages = (np.array(res_ages, dtype=np.float32) / 100.0).clip(0.0, 1.0)
        result = {
            "filename": filename,
            "input": torch.from_numpy(input),
            "hm": heatmap,
            "reg_mask": reg_mask,
            "ind": ind,
            "wh": weight_height,
            "reg": reg,
            "bboxes": torch.from_numpy(np.array(new_bboxes)),
            "labels": torch.from_numpy(np.array(labels)),
        }

        return result

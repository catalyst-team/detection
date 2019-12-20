import os
import json
import numpy as np
import pickle
from typing import Any

from pycocotools.coco import COCO
from torch.utils.data import Dataset


class DetectionMSCOCODataset(Dataset):
    def __init__(self, annotation_file: str, image_dir: str):

        self._annotation_file = annotation_file
        self._image_dir = image_dir
        self._cache_file = self._annotation_file + ".cache"

        self._coco = COCO(self._annotation_file)

        self._img_ids = self._coco.getImgIds()
        self._cat_ids = self._coco.getCatIds()
        self._ann_ids = self._coco.getAnnIds()

        self._data = "coco"
        self._classes = {
            ind: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._load_data()
        self._db_inds = np.arange(len(self._image_names))

        self._load_coco_data()

    def _load_data(self):
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_names], f)
            print("Cache file created")
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_names = pickle.load(f)

    def _load_coco_data(self):
        with open(self._annotation_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
            }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):

        self._image_names = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in self._img_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_name) in enumerate(zip(self._img_ids, self._image_names)):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
            categories = []

            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bbox[[2, 3]] += bbox[[0, 1]]
                    bboxes.append(bbox)

                    categories.append(category)

            self._detections[image_name] = [{
                'bbox': bbox.astype(np.float32),
                'category_id': category,
                'category_name': self.class_name(category)
            } for bbox, category in zip(bboxes, categories)]

    def __getitem__(self, ind: int) -> Any:
        image_name = self._image_names[ind]

        return {
            'image_name': os.path.join(self._image_dir, image_name),
            'detections': self._detections[image_name]
        }

    def __len__(self) -> int:
        return len(self._img_ids)

    def get_num_categories(self) -> int:
        return len(self._cat_ids)

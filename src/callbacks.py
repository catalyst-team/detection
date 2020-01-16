from typing import List, Dict

import numpy as np
from catalyst.dl import Callback, RunnerState, CallbackOrder, CriterionCallback
from catalyst.utils import detach

from .losses.ctdet_loss import decode_centernet_predictions
from .metrics import class_agnostic_mean_ap, calculate_map, construct_mAP_list_from_bboxes


class DecoderCallback(Callback):
    def __init__(self, down_ratio: int = 1, max_objs: int = 80):
        super().__init__(order=CallbackOrder.Metric - 1)
        self.down_ratio = down_ratio
        self.max_objs = max_objs

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("valid"):
            detections = decode_centernet_predictions(
                state.output["hm"],
                state.output["wh"],
                state.output["reg"],
                K=self.max_objs
            )
            detections = detach(detections).reshape(
                (detections.shape[0], -1, detections.shape[2])
            )
            detections[:, :, :4] *= self.down_ratio

            bboxes = detections[:, :, :4].astype(int)
            scores = detections[:, :, 4]
            labels = detections[:, :, 5].astype(int)

            result = dict(
                bboxes=bboxes,
                labels=labels,
                scores=scores,
            )
            state.output.update(result)


class MeanAPCallback(Callback):
    def __init__(
            self,
            num_classes: int = None,
            prefix: str = "mAP",
            bboxes_key: str = "bboxes",
            scores_key: str = "scores",
            labels_key: str = "labels",
            iou_threshold: float = 0.9
    ):
        super().__init__(order=CallbackOrder.Metric)
        self.prefix = prefix
        self.classes = list(range(num_classes))
        self.mean_mAP = []

        self.bboxes_key = bboxes_key
        self.scores_key = scores_key
        self.labels_key = labels_key
        # List (dictionary value) contains of pairs of correct/not correct bboxes and model confidence by class
        self.classes_predictions: Dict[str, List[(bool, float)]] = {c: [] for c in range(num_classes)}
        self.iou_threshold = iou_threshold

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("valid"):
            bboxes = state.output[self.bboxes_key]
            scores = state.output[self.scores_key]
            labels = state.output[self.labels_key]

            gt_bboxes = [
                np.array(item_bboxes.detach().cpu())
                for item_bboxes in state.input[self.bboxes_key]]
            gt_labels = [
                np.array(item_label.detach().cpu())
                for item_label in state.input[self.labels_key]
            ]

            for i, _class in enumerate(self.classes):
                predict_bboxes_batch = []
                predict_scores_batch = []

                target_bboxes_batch = []
                for batch_elem in zip(bboxes, scores, labels, gt_bboxes, gt_labels):
                    bboxes_, scores_, labels_, gt_bboxes_, gt_labels_ = batch_elem

                    bboxes_ = bboxes_[scores_ > 0]
                    labels_ = labels_[scores_ > 0]
                    scores_ = scores_[scores_ > 0]

                    mask = (labels_ == i)
                    predict_bboxes_batch.append(bboxes_[mask])
                    predict_scores_batch.append(scores_[mask])

                    gt_mask = gt_labels_ == i
                    target_bboxes_batch.append(gt_bboxes_[gt_mask])

                if len(predict_bboxes_batch) != 0:
                    per_box_correctness = [
                        construct_mAP_list_from_bboxes(img_pred_bboxes.reshape(-1, 4), img_scores,
                                                       img_gt_bboxes.reshape(-1, 4), self.iou_threshold)
                        for img_pred_bboxes, img_scores, img_gt_bboxes
                        in zip(predict_bboxes_batch, predict_scores_batch, target_bboxes_batch)
                    ]
                    for answers in per_box_correctness:
                        self.classes_predictions[_class].extend(answers)

            mean_value = class_agnostic_mean_ap(bboxes, scores, gt_bboxes)
            self.mean_mAP.append(mean_value)

    def on_loader_end(self, state: RunnerState):
        if state.loader_name.startswith("valid"):
            all_predictions = []
            for class_name, predictions in self.classes_predictions.items():
                # metric_name = f"{self.prefix}/{class_name}"
                # mAP = calculate_map(predictions)
                # state.metrics.epoch_values[state.loader_name][metric_name] = mAP
                all_predictions.extend(predictions)

            # mean_AP = calculate_map(all_predictions)
            # state.metrics.epoch_values[state.loader_name][f'{self.prefix}/_mean'] = mean_AP

            ap_with_false_negatives = calculate_map(all_predictions, use_false_negatives=True)
            state.metrics.epoch_values[state.loader_name][f'{self.prefix}/_mean_with_fn'] = ap_with_false_negatives

            # old mAP
            # state.metrics.epoch_values[state.loader_name][f'{self.prefix}/_mean_old'] = np.mean(self.mean_mAP)
            self.mean_mAP = []
            self.classes_predictions: Dict[str, List[(bool, float)]] = {c: [] for c in self.classes}


__all__ = ["DecoderCallback", "MeanAPCallback"]

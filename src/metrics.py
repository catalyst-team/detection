from typing import Tuple, List

import numpy as np
from sklearn.metrics import average_precision_score


def construct_mAP_list_from_bboxes(predicted_bboxes, scores, gt_bboxes, iou_threshold=.9) -> List[Tuple[bool, float]]:
    """
    :param predicted_bboxes: np.array of shape (n, 4) with predictions
    :param scores: np.array of shape (n,) with model confidences
    :param gt_bboxes: np.array of shape(n, 4) with ground truth bboxes
    :param iou_threshold:
    :return:
    """
    ious_matrix = bbox_iou(predicted_bboxes, gt_bboxes)
    result = _construct_list_for_map(ious_matrix, scores, iou_thresh=iou_threshold)
    return result


def _construct_list_for_map(ious_matrix, scores, iou_thresh=.9) -> List[Tuple[bool, float]]:
    """
    :param ious_matrix: np.array of shape (n, m) with ious between predicted and ground-truth objects
    :param scores: np.array of shape (n) with model confidences for objects
    :param iou_thresh:
    :return:
    """
    ious_thresholded = ious_matrix > iou_thresh
    correct_bboxes = np.where(ious_thresholded.sum(axis=1).astype(bool))[0]
    incorrect_bboxes = np.where(~ious_thresholded.sum(axis=1).astype(bool))[0]
    fn_bboxes = np.where(ious_thresholded.sum(axis=0) == 0)[0]

    result = []
    result.extend([(True, scores[i]) for i in correct_bboxes])
    result.extend([(False, scores[i]) for i in incorrect_bboxes])
    result.extend([(True, 0) for _ in fn_bboxes])
    return result


def calculate_map(predictions: List[Tuple[bool, float]], use_false_negatives: bool = False) -> float:
    """
    Calculates average precision metric for list of predictions with confidences
    :param predictions:  List of Tuples containing all predicted bboxes with scores and corrent/incorrect flag
    :param use_false_negatives: Flag to use false negatives in metric
    :return: average precision
    """
    predictions = np.array(predictions)
    if not use_false_negatives:
        predictions = predictions[predictions[:, 1] > 0]
    true_labels = predictions[:, 0].astype(int)
    scores = predictions[:, 1]

    # Corner cases, critical for sklearn realisation
    if len(predictions) == 0:
        return 0
    if len(predictions) == 1:
        return 1

    result: float = average_precision_score(true_labels, scores)
    return result


def bbox_iou(predicted, target) -> np.ndarray:
    p_xmin, p_ymin, p_xmax, p_ymax = np.hsplit(predicted, 4)
    t_xmin, t_ymin, t_xmax, t_ymax = np.hsplit(target, 4)

    int_xmin = np.maximum(p_xmin, t_xmin.T)
    int_xmax = np.minimum(p_xmax, t_xmax.T)
    int_ymin = np.maximum(p_ymin, t_ymin.T)
    int_ymax = np.minimum(p_ymax, t_ymax.T)

    int_area = np.maximum(int_ymax - int_ymin, 0) \
               * np.maximum(int_xmax - int_xmin, 0)

    un_xmin = np.minimum(p_xmin, t_xmin.T)
    un_xmax = np.maximum(p_xmax, t_xmax.T)
    un_ymin = np.minimum(p_ymin, t_ymin.T)
    un_ymax = np.maximum(p_ymax, t_ymax.T)

    un_area = np.maximum(un_ymax - un_ymin, 0) \
              * np.maximum(un_xmax - un_xmin, 0)

    return int_area / un_area


def image_stats(pred_bboxes, scores, gt_bboxes, thresholds, iou_threshold=.5):
    ious = bbox_iou(pred_bboxes, gt_bboxes)

    true_positives, false_positives = \
        image_positives_stats(ious, scores, thresholds, iou_threshold)

    false_negatives = image_false_negatives(ious, scores, thresholds,
                                            iou_threshold=iou_threshold)

    stats = np.hstack((true_positives, false_positives, false_negatives))

    return stats


def image_positives_stats(
        ious: np.ndarray,
        scores,
        thresholds,
        iou_threshold
) -> Tuple[np.ndarray, np.ndarray]:
    pred_bbox_max_iou = np.max(ious, axis=1, initial=0)

    potential_tp = pred_bbox_max_iou >= iou_threshold
    potential_fp = ~potential_tp

    mask: np.ndarray = thresholds[:, np.newaxis] <= scores[np.newaxis, :]
    true_positives = mask.compress(potential_tp, axis=1).sum(axis=1)
    false_positives = mask.compress(potential_fp, axis=1).sum(axis=1)

    return true_positives, false_positives


def image_false_negatives(
        ious: np.ndarray,
        scores,
        thresholds,
        iou_threshold
):
    n_pred, n_gt = ious.shape

    if n_gt == 0:
        return np.zeros(thresholds.shape)

    if len(thresholds) == 0 or n_pred == 0:
        return np.full(thresholds.shape, n_gt)

    gt_max_iou_idx = ious.argmax(axis=0)

    always_fn = \
        ious[gt_max_iou_idx, np.arange(len(gt_max_iou_idx))] < iou_threshold

    gt_bbox_max_iou_bbox_score = \
        scores.take(gt_max_iou_idx.compress(~always_fn))
    fn = (thresholds[:, np.newaxis]
          > gt_bbox_max_iou_bbox_score[np.newaxis, :]).sum(axis=1)

    return always_fn.sum() + fn


def class_agnostic_mean_ap(
        pred_bboxes, pred_bbox_score, gt_bboxes,
        sort_scores=True, iou_threshold=0.9
):
    if len(pred_bboxes):
        pass

    thresholds = np.concatenate(pred_bbox_score)
    if sort_scores:
        thresholds = np.sort(thresholds)[::-1]

    per_item_stats = [
        image_stats(img_pred_bboxes.reshape(-1, 4), img_scores,
                    img_gt_bboxes.reshape(-1, 4), thresholds, iou_threshold)
        for img_pred_bboxes, img_scores, img_gt_bboxes
        in zip(pred_bboxes, pred_bbox_score, gt_bboxes)
    ]

    tp, fp, fn = np.hsplit(np.sum(per_item_stats, axis=0), 3)

    all_real_positives = tp + fn
    all_real_positives[all_real_positives == 0] = 1

    recall = tp / all_real_positives

    all_pred_positives = tp + fp
    all_pred_positives[all_pred_positives == 0] = 1

    precision = tp / all_pred_positives

    precisions = []
    for recall_threshold in np.linspace(0, 1, 11):
        precisions.append(
            np.max(precision[recall <= recall_threshold], initial=0))

    mAP = np.mean(precisions) if len(precisions) > 0 else 0

    return mAP

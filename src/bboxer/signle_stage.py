from typing import List, Dict, Tuple
import numpy as np
import torch

from utils.anchors import AnchorGenerator, get_anchor_targets
from utils.criterion import weighted_sigmoid_focal_loss, weighted_smoothl1
from utils.transforms import delta2bbox
from utils.bbox_nms import multiclass_nms
from utils.misc import multi_apply


class SingleStageBboxer:
    def __init__(
        self,
        num_classes,
        anchor_scales=[8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        anchor_base_sizes=None,
        target_means=(.0, .0, .0, .0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
        assigner_params: Dict = None,
        allowed_border: int = -1,
        pos_weight: int = -1,
        unmap_outputs: bool = True,
    ):
        assert assigner_params is not None
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(anchor_strides) \
            if anchor_base_sizes is None \
            else anchor_base_sizes

        self.target_means = target_means
        self.target_stds = target_stds

        self.anchor_generators: List[AnchorGenerator] = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self.num_classes = num_classes

        self.assigner_params = assigner_params
        self.allowed_border = allowed_border
        self.pos_weight = pos_weight
        self.unmap_outputs = unmap_outputs

    def get_anchors(
        self,
        featmap_sizes : List[Tuple[int, int]],
        img_metas: List[Dict]
    ) -> Tuple[List[List], List[List]]:
        """
        Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        # @TODO: check for or *
        anchor_list = [multi_level_anchors] * num_imgs

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta["pad_shape"]
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        labels: torch.Tensor,
        label_weights: torch.Tensor,
        bbox_targets: torch.Tensor,
        bbox_weights: torch.Tensor,
        num_total_samples: int,
        gamma: float = 2.0,
        alpha: float = 0.25,
        smoothl1_beta: float = 1.0,
    ):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.num_classes)
        loss_cls = weighted_sigmoid_focal_loss(
            cls_score,
            labels,
            label_weights,
            gamma=gamma,
            alpha=alpha,
            avg_factor=num_total_samples
        )

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=smoothl1_beta,
            avg_factor=num_total_samples
        )

        return loss_cls, loss_reg

    def compute_loss(
        self,
        multi_level_cls_score: List[torch.Tensor],
        multi_level_bbox_pred: List[torch.Tensor],
        gt_bboxes,
        gt_labels,
        img_metas,
        gt_bboxes_ignore=None
    ):
        featmap_sizes = [
            featmap.shape[-2:]
            for featmap in multi_level_cls_score
        ]
        assert len(featmap_sizes) == len(self.anchor_generators)

        multi_level_anchors, multo_level_valid_flags = \
            self.get_anchors(featmap_sizes, img_metas)

        cls_reg_targets = get_anchor_targets(
            multi_level_anchors,
            multo_level_valid_flags,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            target_means=self.target_means,
            target_stds=self.target_stds,
            assigner_params=self.assigner_params,
            allowed_border=self.allowed_border,
            pos_weight=self.pos_weight,
        )
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = num_total_pos

        losses_cls, losses_reg = multi_apply(
            self.loss_single,
            multi_level_cls_score,
            multi_level_bbox_pred,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        return losses_cls, losses_reg

    def get_bboxes_single(
        self,
        cls_scores,
        bbox_preds,
        mlvl_anchors,
        img_shape,
        scale_factor,
        cfg,
        rescale=False
    ):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(
                cls_scores, bbox_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.num_classes)

            scores = cls_score.sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(
                anchors,
                bbox_pred,
                self.target_means,
                self.target_stds,
                img_shape
            )
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)

        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img
        )
        return det_bboxes, det_labels

    def get_bboxes(
        self,
        multi_level_cls_score,
        multi_level_bbox_pred,
        img_metas,
        cfg,
        rescale=False
    ):
        assert len(multi_level_cls_score) == len(multi_level_bbox_pred)
        num_levels = len(multi_level_cls_score)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                multi_level_cls_score[i].size()[-2:],
                self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                multi_level_cls_score[i][img_id].detach()
                for i in range(num_levels)
            ]
            bbox_pred_list = [
                multi_level_bbox_pred[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            proposals = self.get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                mlvl_anchors,
                img_shape,
                scale_factor,
                cfg,
                rescale
            )
            result_list.append(proposals)
        return result_list

import torch
import torch.nn as nn
import torch.nn.functional as F


def _neg_loss(outputs: torch.Tensor, targets: torch.Tensor):
    """
     Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory

    Arguments:
        outputs (torch.Tensor): BATCH x C x H x W
        targets (torch.Tensor): BATCH x C x H x W
    """
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4)

    loss = 0

    pos_loss = torch.log(outputs) * torch.pow(1 - outputs, 2) * pos_inds
    neg_loss = torch.log(1 - outputs) * torch.pow(outputs, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_centernet_predictions(
    heat, wh, reg=None, K=100
):
    with torch.no_grad():
        batch, cat, height, width = heat.size()
        # mask = reg_mask.unsqueeze(2).expand_as(pred).float()

        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = _nms(heat)

        scores, inds, clses, ys, xs = _topk(heat, K=K)
        if reg is not None:
            reg = _tranpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = _tranpose_and_gather_feat(wh, inds).view(batch, K, 2)

        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, outputs, targets):
        return self.neg_loss(outputs, targets)


class RegL1Loss(nn.Module):
    def __init__(
        self,
        key: str = "",
        mask_key: str = "reg_mask",
        ind_key: str = "ind",
        debug: bool = False
    ):
        super(RegL1Loss, self).__init__()
        self.key = key
        self.mask_key = mask_key
        self.ind_key = ind_key
        self.debug = debug

    #def forward(self, outputs, targets):
    #    result = self._forward(
    #        outputs[self.key], targets[self.mask_key],
    #        targets[self.ind_key], targets[self.key]
    #    )
    #    return result

    def forward(self, outputs_key, targets_mask_key, targets_ind_key, targets_key):
        result = self._forward(
            outputs_key, targets_mask_key, targets_ind_key, targets_key
        )
        return result

    def _forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()

        if self.debug:
            import ipdb; ipdb.set_trace()
        loss = F.l1_loss(pred * mask, target * mask)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CenterNetDetectionLoss(nn.Module):
    def __init__(self):
        super(CenterNetDetectionLoss, self).__init__()
        self.focal = FocalLoss()

    def forward(self, outputs, targets):
        loss = self.focal(_sigmoid(outputs), targets)
        return loss


class MSEIndLoss(nn.Module):
    def __init__(
        self,
        key: str,
        mask_key: str = "reg_mask",
        ind_key: str = "ind",
        debug: bool = False,
        reduction: str = "mean"
    ):
        super(MSEIndLoss, self).__init__()
        self.key = key
        self.mask_key = mask_key
        self.ind_key = ind_key
        self.debug = debug

        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, outputs, targets):
        result = self._forward(
            outputs[self.key], targets[self.mask_key],
            targets[self.ind_key], targets[self.key]
        )

        return result

    def _forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        _mask = mask.unsqueeze(2).expand_as(pred).float()

        if self.debug:
            import ipdb; ipdb.set_trace()
        loss = self.loss(_sigmoid(pred) * _mask, target.unsqueeze(2) * _mask)
        # loss = loss / (_mask.sum() + 1e-4)
        return loss


class BCEIndLoss(nn.Module):
    def __init__(
        self,
        key: str,
        mask_key: str = "reg_mask",
        ind_key: str = "ind",
        debug: bool = False
    ):
        super(BCEIndLoss, self).__init__()
        self.key = key
        self.mask_key = mask_key
        self.ind_key = ind_key
        self.loss = nn.BCELoss()
        self.debug = debug

    def forward(self, outputs, targets):
        result = self._forward(
            outputs[self.key], targets[self.mask_key],
            targets[self.ind_key], targets[self.key]
        )

        return result

    def _forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        _mask = mask.unsqueeze(2).expand_as(pred).float()
        if self.debug:
            import ipdb; ipdb.set_trace()

        loss = self.loss(_sigmoid(pred) * _mask, target * _mask)
        # loss = loss / (mask.sum() + 1e-4)
        return loss


class FocalIndLoss(nn.Module):
    def __init__(
        self,
        key: str,
        mask_key: str = "reg_mask",
        ind_key: str = "ind",
        debug: bool = False
    ):
        super(FocalIndLoss, self).__init__()
        self.key = key
        self.mask_key = mask_key
        self.ind_key = ind_key
        self.loss = FocalLoss()
        self.debug = debug

    def forward(self, outputs, targets):
        result = self._forward(
            outputs[self.key], targets[self.mask_key],
            targets[self.ind_key], targets[self.key]
        )

        return result

    def _forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        _mask = mask.unsqueeze(2).expand_as(pred).float()
        if self.debug:
            import ipdb; ipdb.set_trace()

        loss = self.loss(_sigmoid(pred) * _mask, target * _mask)
        # loss = loss / (mask.sum() + 1e-4)
        return loss

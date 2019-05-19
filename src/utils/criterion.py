import torch
import torch.nn.functional as F

from src.ops import sigmoid_focal_loss


def weighted_sigmoid_focal_loss(
    logits,
    targets,
    weights,
    gamma=2.0,
    alpha=0.25,
    avg_factor=None,
    num_classes=None
):
    if avg_factor is None:
        avg_factor = torch.sum(weights > 0).float().item() / num_classes + 1e-6
    return torch.sum(
        sigmoid_focal_loss(logits, targets, gamma, alpha, "none")
        * weights.view(-1, 1))[None] / avg_factor


def smooth_l1_loss(pred, target, beta=1.0, reduction="mean"):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta,
        0.5 * diff * diff / beta,
        diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction="none")
    return torch.sum(loss * weight)[None] / avg_factor

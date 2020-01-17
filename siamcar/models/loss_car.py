"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch import nn
import numpy as np

from siamcar.utils.iou_loss import IOULoss
from .loss_cls import select_cross_entropy_loss

INF = 100000000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets, num = self.compute_targets_for_locations(
            points, labels, gt_bbox
        )
        return labels, reg_targets, num

    def compute_targets_for_locations(self, locations, labels, gt_bbox):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        # area = (bboxes[:,2] - bboxes[:,0] + 1) * (bboxes[:,3] - bboxes[:,1] + 1)
        labels = labels.view(625,-1)

        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        s1 = reg_targets_per_im[:, :, 0] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        pos = np.where(is_in_boxes == 1)
        num = pos[0].shape[0]
        labels[pos] = 1

        return labels.permute(1,0).contiguous(), reg_targets_per_im.permute(1,0,2).contiguous(),num

    def select(self, position, keep_num=16):
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        label_cls, reg_targets, num = self.prepare_targets(locations, labels, reg_targets)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)

        if pos_inds.numel() > 0:

            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator

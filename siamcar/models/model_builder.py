# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamcar.core.config import cfg
from siamcar.models.loss_car import make_fcos_loss_evaluator
from siamcar.models.head import CARHead
from siamcar.models.neck import get_neck
from siamcar.utils.car_utils import compute_locations
from siamcar.core.xcorr import xcorr_depthwise

from siamcar.models.backbone import resnet50


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = resnet50(**cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build fcos head
        self.rpn_head = CARHead(cfg, **cfg.CAR.KWARGS)

        #build response map
        self.xcorr_depthwise = xcorr_depthwise

        #build loss
        self.loss_evaluator = make_fcos_loss_evaluator(cfg)

        # build activate function
        self.down = nn.ConvTranspose2d(256*3, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0], self.zf[0])
        for i in range(len(xf) - 1):
            feature_new = self.xcorr_depthwise(xf[i + 1], self.zf[i + 1])
            features = torch.cat([features, feature_new], 1)
        features = self.down(features)

        cls, loc, cen = self.rpn_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen,
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        features = self.xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf)-1):
            feature_new = self.xcorr_depthwise(xf[i+1], zf[i+1])
            features = torch.cat([features, feature_new], 1)
        features = self.down(features)
        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, 8)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cls_loss + 1.5*loc_loss + cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss

        return outputs

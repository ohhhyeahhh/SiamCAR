# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamrpn_r50_l234_8gpu_dwxcorr"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Testing options
# ------------------------------------------------------------------------ #
__C.TEST = CN()

__C.TEST.NUM_CLASSES = 2

__C.TEST.NUM_CONVS = 4

__C.TEST.PRIOR_PROB = 0.01

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.CAR = CN()

# RPN type
__C.CAR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamCARTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# UPSIZE
__C.TRACK.UPSIZE = 193

# Score size
__C.TRACK.SCORE_SIZE = 25

# if add hanming
__C.TRACK.hanming = True

# if use average bbox
__C.TRACK.ave_reg = True

# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #

__C.HP_SEARCH = CN()

__C.HP_SEARCH.OTB = [0.15, 0.1, 0.4]

__C.HP_SEARCH.GOT = [0.6, 0.04, 0.1]

__C.HP_SEARCH.UAV123 = [0.39, 0.04, 0.37]

__C.HP_SEARCH.VOT2019 = [0.41, 0.04, 0.3]

__C.HP_SEARCH.LaSOT = [0.33, 0.04, 0.40]

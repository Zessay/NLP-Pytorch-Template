#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: __init__.py.py
@time: 2019/11/27 19:31
@description: 
'''

from snlp.losses.cb_focal_loss import CBFocalLoss
from snlp.losses.focal_loss import FocalLoss
from snlp.losses.rank_cross_entropy_loss import RankCrossEntropyLoss
from snlp.losses.rank_hinge_loss import RankHingeLoss
from snlp.losses.label_smoothing_loss import LabelSmoothLoss


__all__ = ["CBFocalLoss", "FocalLoss", "RankCrossEntropyLoss",
           "RankHingeLoss", "LabelSmoothLoss"]
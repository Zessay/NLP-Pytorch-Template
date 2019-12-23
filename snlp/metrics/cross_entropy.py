#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: cross_entropy.py
@time: 2019/11/27 21:40
@description: 计算cross_entropy
'''

import numpy as np
from snlp.base import BaseMetric
from snlp.tools.common import one_hot

class CrossEntropy(BaseMetric):
    ALIAS = ['cross_entropy', 'ce']

    def __init__(self):
        """Construct"""

    def __repr__(self) -> str:
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, eps: float=1e-12) -> float:
        """
        Calculate cross entropy.
        :param y_true: 一维ndarray
        :param y_pred: 二维ndarray，表示概率
        :return:
        """
        y_pred = np.clip(y_pred, eps, 1. - eps)
        y_true = one_hot(y_true, num_classes=y_pred.shape[1])

        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_pred.shape[0]
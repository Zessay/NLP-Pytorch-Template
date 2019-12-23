#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: average_precision.py
@time: 2019/11/27 21:31
@description: 平均Precision@k
'''

import numpy as np
from snlp.base import BaseMetric
from snlp.metrics.precision import Precision

class AveragePrecision(BaseMetric):
    ALIAS = ['average_precision', 'ap']

    def __init__(self, threshold: float=0.):
        self._threshold = threshold

    def __repr__(self) -> str:
        return f"{self.ALIAS[0]}({self._threshold})"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision_metrics = [Precision(k+1) for k in range(y_pred.shape[1])]
        out = [metric(y_true, y_pred) for metric in precision_metrics]
        if not out:
            return 0.
        return np.mean(out).item()
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: mean_reciprocal_rank.py
@time: 2019/11/27 21:47
@description: 平均倒数排序，越接近1越好
'''
import numpy as np
from snlp.base import BaseMetric
from snlp.tools.common import sort_and_couple


class MeanReciprocalRank(BaseMetric):
    ALIAS = ['mean_reciprocal_rank', 'mrr']

    def __init__(self, threshold: float=0.):
        self._threshold = threshold

    def __repr__(self) -> str:
        return f'{self.ALIAS[0]}({self._threshold})'

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        coupled_pair = sort_and_couple(y_true, y_pred)
        for idx, (label, pred) in enumerate(coupled_pair):
            if label > self._threshold:
                return 1. / (idx + 1)
        return 0.
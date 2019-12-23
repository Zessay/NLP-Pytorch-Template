#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: precision.py
@time: 2019/11/27 21:16
@description: Precision@k
'''

import numpy as np
from snlp.base import BaseMetric
from snlp.tools.common import sort_and_couple

class Precision(BaseMetric):
    ALIAS = ['precision']

    def __init__(self, k: int=1, threshold: float=0.):
        self._k = k
        self._threshold = threshold


    def __repr__(self) -> str:
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Precision@k
        :param y_true: 一维ndarray，表示一个样例
        :param y_pred: 一维ndarray，表示一个样例的概率值
        :return:
        """
        if self._k <= 0:
            raise ValueError(f"k must be greater than 0. {self._k} received. ")

        coupled_pair = sort_and_couple(y_true, y_pred)
        precision = 0.0
        for idx, (label, score) in enumerate(coupled_pair):
            if idx >= self._k:
                break
            if label > self._threshold:
                precision += 1.
        return precision / self._k


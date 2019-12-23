#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: accuracy.py
@time: 2019/11/27 21:05
@description: 准确率
'''

import numpy as np
from snlp.base import BaseMetric

class Accuracy(BaseMetric):
    # 必须要定义ALIAS，便于打印日志的时候使用
    ALIAS = ['accuracy', 'acc']
    def __init__(self):
        """Construct"""

    def __repr__(self) -> str:
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy.
        :param y_true: 一维ndarray
        :param y_pred: 二维ndarray，表示预测的概率
        :return:
        """
        y_pred = np.argmax(y_pred, axis=1)
        return np.sum(y_pred == y_true) / float(y_true.size)
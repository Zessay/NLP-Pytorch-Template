#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: base_metric.py
@time: 2019/11/21 20:36
@description: 
'''
import abc
import numpy as np

class BaseMetric(abc.ABC):
    """Metric base class"""
    @abc.abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Call to compute the metric
        :param y_true:
        :param y_pred:
        :return:
        """

    @abc.abstractmethod
    def __repr__(self):
        """
        String representation of the metric
        """

    def __eq__(self, other):
        return (type(self) is type(other)) and (vars(self) == vars(other))

    def __hash__(self):
        return str(self).__hash__()
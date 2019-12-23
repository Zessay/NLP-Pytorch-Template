#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: ranking.py
@time: 2019/12/3 20:28
@description: 排序任务
'''
from snlp.base.base_task import BaseTask

class Ranking(BaseTask):
    TYPE = 'ranking'

    @classmethod
    def list_available_losses(cls) -> list:
        """:return: a list of available losses."""
        return ['mse']

    @classmethod
    def list_available_metrics(cls) -> list:
        """:return: a list of available metrics."""
        return ['map']

    @property
    def output_shape(self) -> tuple:
        """:return: output shape of a single sample of the task."""
        return 1,

    @property
    def output_dtype(self):
        """:return: target data type, expect `float` as output."""
        return float

    def __str__(self):
        """:return: Task name as string."""
        return 'Ranking Task'


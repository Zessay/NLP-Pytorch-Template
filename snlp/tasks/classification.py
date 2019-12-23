#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: classification.py
@time: 2019/12/3 20:25
@description: 分类任务
'''

from snlp.base.base_task import BaseTask

class Classification(BaseTask):
    TYPE = 'classification'

    def __init__(self, num_classes: int = 2, **kwargs):
        """Classification task."""
        super().__init__(**kwargs)
        if not isinstance(num_classes, int):
            raise TypeError("Number of classes must be an integer.")
        if num_classes < 2:
            raise ValueError("Number of classes can't be smaller than 2")
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        """:return: number of classes to classify."""
        return self._num_classes

    @classmethod
    def list_available_losses(cls) -> list:
        """:return: a list of available losses."""
        return ['cross_entropy']

    @classmethod
    def list_available_metrics(cls) -> list:
        """:return: a list of available metrics."""
        return ['acc']

    @property
    def output_shape(self) -> tuple:
        """:return: output shape of a single sample of the task."""
        return self._num_classes,

    @property
    def output_dtype(self):
        """:return: target data type, expect `int` as output."""
        return int

    def __str__(self):
        """:return: Task name as string."""
        return f'Classification Task with {self._num_classes} classes'

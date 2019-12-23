#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: base_task.py
@time: 2019/12/3 19:54
@description: Task的基类
'''
import typing
import abc

import torch.nn as nn

from snlp.base import base_metric
from snlp.tools.parse import parse_metric, parse_loss

class BaseTask(abc.ABC):
    TYPE = 'base'
    def __init__(self, losses=None, metrics=None):
        self._losses = self._convert(losses, parse_loss)
        self._metrics = self._convert(metrics, parse_metric)
        self._assure_losses()
        self._assure_metrics()

    def _convert(self, identifiers, parse):
        if not identifiers:
            identifiers = []
        elif not isinstance(identifiers, list):
            identifiers = [identifiers]

        return [
            parse(identifier)
            for identifier in identifiers
        ]

    def _assure_losses(self):
        if not self._losses:
            first_available = self.list_available_losses()[0]
            self._losses = self._convert(first_available, parse_loss)

    def _assure_metrics(self):
        if not self._metrics:
            first_available = self.list_available_metrics()[0]
            self._metrics = self._convert(first_available, parse_metric)

    @property
    def losses(self):
        return self._losses

    @property
    def metrics(self):
        return self._metrics

    @losses.setter
    def losses(self,
               new_losses: typing.Union[
                   typing.List[str],
                   typing.List[nn.Module],
                   str,
                   nn.Module]):
        self._losses = self._convert(new_losses, parse_loss)

    @metrics.setter
    def metrics(
        self,
        new_metrics: typing.Union[
            typing.List[str],
            typing.List[base_metric.BaseMetric],
            str,
            base_metric.BaseMetric
        ]
    ):
        self._metrics = self._convert(new_metrics, parse_metric)

    @classmethod
    @abc.abstractmethod
    def list_available_losses(cls) -> list:
        """:return: a list of available losses."""

    @classmethod
    @abc.abstractmethod
    def list_available_metrics(cls) -> list:
        """:return: a list of available metrics."""

    @property
    @abc.abstractmethod
    def output_shape(self) -> tuple:
        """:return: output shape of a single sample of the task."""

    @property
    @abc.abstractmethod
    def output_dtype(self):
        """:return: output data type for specific task."""

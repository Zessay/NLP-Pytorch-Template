#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: base_stateunit.py
@time: 2019/11/21 21:16
@description: 需要事先进行fit的处理单元
'''
import abc
import typing

from .base_unit import Unit

class StatefulUnit(Unit, metaclass=abc.ABCMeta):
    """
    Unit with inner state.
    """

    def __init__(self):
        self._context = {}

    @property
    def context(self):
        return self._context

    @abc.abstractmethod
    def fit(self, input_: typing.Any):
        """Abstract base method, need to be implemented in subclass."""
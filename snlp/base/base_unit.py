#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: base_unit.py
@time: 2019/11/21 21:13
@description: 各种处理单元的基类
'''

import abc
import typing

class Unit(metaclass=abc.ABCMeta):
    """Process unit do not need prior state (i.e. do not need fit)"""
    @abc.abstractmethod
    def transform(self, input_: typing.Any):
        """Abstract base method, need to be implemented in subclass. """
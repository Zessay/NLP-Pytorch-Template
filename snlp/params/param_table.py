#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: param_table.py
@time: 2019/12/3 11:30
@description: 保存一个模型所有参数的类
'''

import typing
import pandas as pd
import collections.abc

from snlp.params.param import Param
from snlp.params import hyper_spaces

class ParamTable(object):
    def __init__(self):
        self._params = {}

    def add(self, param: Param):
        if not isinstance(param, Param):
            raise TypeError("Only accepts a Param instance.")
        if param.name in self._params:
            msg = f"Parameter named {param.name} already exists.\n" \
                f"To re-assign parameter {param.name} value, " \
                f"use `params[\"{param.name}\"] = value` instead."
            raise ValueError(msg)
        self._params[param.name] = param

    def get(self, key) -> Param:
        return self._params[key]

    def set(self, key, param: Param):
        if not isinstance(param, Param):
            raise ValueError("Only accepts a Param instance.")
        self._params[key] = param

    @property
    def hyper_space(self) -> dict:
        full_space = {}
        for param in self:
            if param.hyper_space is not None:
                param_space = param.hyper_space
                if isinstance(param_space, hyper_spaces.HyperoptProxy):
                    param_space = param_space.convert(param.name)
                full_space[param.name] = param_space
        return full_space

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data={
                'Name': [p.name for p in self],
                'Description': [p.desc for p in self],
                'Value': [p.value for p in self],
                'Hyer-Space': [p.hyper_space for p in self]
            }, columns=['Name', 'Description', 'Value', 'Hyper-Space']
        )
        return df

    def __getitem__(self, key: str) -> typing.Any:
        return self._params[key].value

    def __setitem__(self, key: str, value: typing.Any):
        self._params[key].value = value

    def __str__(self):
        """:return: Pretty formatted parameter table."""
        return '\n'.join(param.name.ljust(30) + str(param.value)
                         for param in self._params.values())

    def __iter__(self) -> typing.Iterator:
        """:return: A iterator that iterates over all parameter instances."""
        yield from self._params.values()

    def completed(self, exclude: typing.Optional[list]=None) -> bool:
        """
        Check if all params are filled.
        :param exclude:
        :return:
        """
        return all(param for param in self if param.name not in exclude)

    def __contains__(self, item):
        return item in self._params

    def update(self, other: dict):
        """
        Update `self`.

        Update `self` with key/value pairs from other, overwriting
        existing keys. Notice that this does not add new keys to `self`.

        This method is usually used by models to obtain useful information
        from a preprocessors's context.

        :param other:
        :return:
        """
        for key in other:
            if key in self:
                self[key] = other[key]

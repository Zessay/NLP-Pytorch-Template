#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: param.py
@time: 2019/12/3 11:12
@description: 定义保存参数的类
'''
import inspect
import numbers
import typing

import hyperopt.pyll
from snlp.params import hyper_spaces

SpaceType = typing.Union[hyperopt.pyll.Apply, hyper_spaces.HyperoptProxy]

class Param(object):
    """
    Param Class.
    :param name: 参数的名称
    :param value: 参数的值，如果为None，表示没有填充
    :param hyper_space: 参数空间，如果不为None，那么这个参数会在参数优化时进行优化
    :param validator: 对参数进行检查的函数，或者值
    :param desc:
    """
    def __init__(self,
                 name: str,
                 value: typing.Any=None,
                 hyper_space: typing.Optional[SpaceType] = None,
                 validator: typing.Optional[
                     typing.Callable[[typing.Any], bool]]=None,
                 desc: typing.Optional[str]=None):
        self._name = name
        self._desc = desc
        self._value = None
        self._hyper_space=None
        self._validator=None
        self._pre_assignment_hook = None

        self.validator = validator
        self.hyper_space = hyper_space

        if value is not None:
            self.value = value

    @property
    def name(self) -> str:
        """:return: Name of the parameter."""
        return self._name

    @property
    def value(self) -> typing.Any:
        """:return: Value of the parameter."""
        return self._value

    @value.setter
    def value(self, new_value: typing.Any):
        """
        Set the value of parameter to `new_value`.

        Notice that this setter validates `new_value` before assignment. As
        a result, if the validaiton fails, the value of the parameter is not
        changed.

        :param new_value: New value of the parameter to set.
        """
        if self._pre_assignment_hook:
            new_value = self._pre_assignment_hook(new_value)
        self._validate(new_value)
        self._value = new_value
        if not self._pre_assignment_hook:
            self._infer_pre_assignment_hook()

    @property
    def hyper_space(self) -> SpaceType:
        """:return: Hyper space of the parameter."""
        return self._hyper_space

    @hyper_space.setter
    def hyper_space(self, new_space: SpaceType):
        """:param new_space: New space of the parameter to set."""
        self._hyper_space = new_space

    @property
    def validator(self) -> typing.Callable[[typing.Any], bool]:
        """:return: Validator of the parameter."""
        return self._validator

    @validator.setter
    def validator(self, new_validator: typing.Callable[[typing.Any], bool]):
        """:param new_validator: New space of the parameter to set."""
        if new_validator and not callable(new_validator):
            raise TypeError("Validator must be a callable or None.")
        self._validator = new_validator

    @property
    def desc(self) -> str:
        """:return: Parameter description."""
        return self._desc

    @desc.setter
    def desc(self, value: str):
        """:param value: New description of the parameter."""
        self._desc = value

    def _infer_pre_assignment_hook(self):
        if isinstance(self._value, numbers.Number):
            self._pre_assignment_hook = lambda x: type(self._value)(x)

    def _validate(self, value):
        if self._validator:
            valid = self._validator(value)
            if not valid:
                error_msg = "Validator not satifised.\n"
                error_msg += "The validator's definition is as follows:\n"
                error_msg += inspect.getsource(self._validator).strip()
                raise ValueError(error_msg)

    def __bool__(self):
        """:return: `False` when the value is `None`, `True` otherwise."""
        return self._value is not None

    def set_default(self, val, verbose=1):
        """
        Set default value, has no effect if already has a value.

        :param val: Default value to set.
        :param verbose: Verbosity.
        """
        if self._value is None:
            self.value = val
            if verbose:
                print(f"Parameter \"{self._name}\" set to {val}.")

    def reset(self):
        self._value = None
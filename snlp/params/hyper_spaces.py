#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: hyper_spaces.py
@time: 2019/12/3 10:46
@description: 超参搜索
'''

import typing
import numbers
import hyperopt
import hyperopt.pyll.base

class HyperoptProxy(object):
    """
    Hyperopt proxy class.

    See `hyperopt`'s documentation for more details:
    https://github.com/hyperopt/hyperopt/wiki/FMin

    为什么重新封装？
    - 原始的hyperopt需要label参数进行实例化，这个label之后会用作采样的原始参数空间的参照；
    - 我们定义的超参选择只会在Param类中使用，只有当label和Param中的name匹配时，才会进行采样；
    - 因此，这里的封装是为了隐层label参数，它只会和Param中的name参数绑定。

    Examples::
        >>> from hyperopt.pyll.stochastic import sample

    Basic Usage:
        >>> model = snlp.models.DenseBaseline()
        >>> sample(model.params.hyper_space)  # doctest: +SKIP
         {'mlp_num_layers': 1.0, 'mlp_num_units': 274.0}

    Arithmetic Operations:
        >>> new_space = 2 ** snlp.params.hyper_spaces.quniform(2, 6)
        >>> model.params.get('mlp_num_layers').hyper_space = new_space
        >>> sample(model.params.hyper_space)  # doctest: +SKIP
        {'mlp_num_layers': 8.0, 'mlp_num_units': 292.0}
    """
    def __init__(self,
                 hyperopt_func: typing.Callable[..., hyperopt.pyll.Apply],
                 **kwargs
                 ):
        self._func = hyperopt_func
        self._kwargs = kwargs

    def convert(self, name: str) -> hyperopt.pyll.Apply:
        """
        Attach name as `hyperopt.hp`'s label.
        :param name:
        :return: a `hyperopt` ready search space.
        """
        return self._func(name, **self._kwargs)

    def __add__(self, other):
        """__add__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x + y)

    def __radd__(self, other):
        """__radd__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x + y)

    def __sub__(self, other):
        """__sub__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x - y)

    def __rsub__(self, other):
        """__rsub__."""
        return _wrap_as_composite_func(self, other, lambda x, y: y - x)

    def __mul__(self, other):
        """__mul__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x * y)

    def __rmul__(self, other):
        """__rmul__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x * y)

    def __truediv__(self, other):
        """__truediv__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        """__rtruediv__."""
        return _wrap_as_composite_func(self, other, lambda x, y: y / x)

    def __floordiv__(self, other):
        """__floordiv__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x // y)

    def __rfloordiv__(self, other):
        """__rfloordiv__."""
        return _wrap_as_composite_func(self, other, lambda x, y: y // x)

    def __pow__(self, other):
        """__pow__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x ** y)

    def __rpow__(self, other):
        """__rpow__."""
        return _wrap_as_composite_func(self, other, lambda x, y: y ** x)

    def __neg__(self):
        """__neg__."""
        return _wrap_as_composite_func(self, None, lambda x, _: -x)

def _wrap_as_composite_func(self, other, func):
    def _wrapper(name, **kwargs):
        return func(self._func(name, **kwargs), other)

    return HyperoptProxy(_wrapper, **self._kwargs)

# ---------------- 从给定的list中随机选择 ---------------------------

class choice(HyperoptProxy):
    """func: `hyperopt.hp.choice` proxy."""
    def __init__(self, options: list):
        super().__init__(hyperopt_func=hyperopt.hp.choice, options=options)
        self._options = options
    def __str__(self):
        return f"choice n {self._options}"

# ---------------------- 指定步长的均匀采样 --------------------------------

class quniform(HyperoptProxy):
    """:func:`hyperopt.hp.quniform` proxy."""

    def __init__(
        self,
        low: numbers.Number,
        high: numbers.Number,
        q: numbers.Number = 1
    ):
        """
        :func:`hyperopt.hp.quniform` proxy.

        If using with integer values, then `high` is exclusive.

        :param low: lower bound of the space
        :param high: upper bound of the space
        :param q: similar to the `step` in the python built-in `range`
        """
        super().__init__(hyperopt_func=hyperopt.hp.quniform,
                         low=low,
                         high=high, q=q)
        self._low = low
        self._high = high
        self._q = q

    def __str__(self):
        """:return: `str` representation of the hyper space."""
        return f'quantitative uniform distribution in  ' \
               f'[{self._low}, {self._high}), with a step size of {self._q}'


# ---------------------------- 不指定步长的均匀采样 ----------------------------

class uniform(HyperoptProxy):
    """:func:`hyperopt.hp.uniform` proxy."""

    def __init__(
        self,
        low: numbers.Number,
        high: numbers.Number
    ):
        """
        :func:`hyperopt.hp.uniform` proxy.

        :param low: lower bound of the space
        :param high: upper bound of the space
        """
        super().__init__(hyperopt_func=hyperopt.hp.uniform, low=low, high=high)
        self._low = low
        self._high = high

    def __str__(self):
        """:return: `str` representation of the hyper space."""
        return f'uniform distribution in  [{self._low}, {self._high})'

# ---------------------------- 用于查看参数空间中的值 -------------------------

def sample(space):
    return hyperopt.pyll.stochastic.sample(space)
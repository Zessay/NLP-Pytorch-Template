#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: early_stopping.py
@time: 2019/11/27 19:08
@description: 提前停止
'''

import typing

class EarlyStopping:
    """
    EarlyStopping stops training if no improvement after a given patience.
    """
    def __init__(self,
                 patience: typing.Optional[int]=None,
                 key: typing.Any=None):
        """
        Early stopping Constructor
        :param patience: int型，表示能够忍受metric没有改变的eval的次数
        :param key: 表示关注的metric的名称
        """
        self._patience = patience
        self._key = key
        self._best_so_far = 0
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = False
        self._early_stop = False

    def state_dict(self) -> typing.Dict[str, typing.Any]:
        """A Trainer can use this to serialize the state."""
        return {
            'patience': self._patience,
            'best_so_far': self._best_so_far,
            'is_best_so_far': self._is_best_so_far,
            'epochs_with_no_improvement': self._epochs_with_no_improvement
        }

    def load_state_dict(self,
                        state_dict: typing.Dict[str, typing.Any]):
        self._patience = state_dict['patience']
        self._is_best_so_far = state_dict['is_best_so_far']
        self._best_so_far = state_dict['best_so_far']
        self._epochs_with_no_improvement = state_dict['epochs_with_no_improvement']

    def update(self, result: list):
        score = result[self._key]
        if score > self._best_so_far:
            self._best_so_far = score
            self._is_best_so_far = True
            self._epochs_with_no_improvement = 0
        else:
            self._is_best_so_far = False
            self._epochs_with_no_improvement += 1

    @property
    def best_so_far(self) -> bool:
        return self._best_so_far

    @property
    def is_best_so_far(self) -> bool:
        return self._is_best_so_far

    @property
    def should_stop_early(self) -> bool:
        if not self._patience:
            return False
        else:
            return self._epochs_with_no_improvement >= self._patience
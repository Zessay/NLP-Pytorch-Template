#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: frequency_filter.py
@time: 2019/11/22 16:05
@description: 基于频率对文本中的单词进行过滤，fit输入是二维列表，transform输入一维列表
'''
import collections
import typing
import numpy as np

from snlp.base import StatefulUnit

class FrequencyFilter(StatefulUnit):
    """Frequency filter unit"""
    def __init__(self, low: float=0, high: float=float('inf'),
                 mode: str='df'):
        super().__init__()
        self._low = low
        self._high = high
        self._mode = mode

    def fit(self, list_of_tokens: typing.List[typing.List[str]]):
        """Use to train data to get context"""
        valid_terms = set()
        if self._mode == 'tf':
            stats = self._tf(list_of_tokens)
        elif self._mode == 'df':
            stats = self._df(list_of_tokens)
        elif self._mode == 'idf':
            stats = self._idf(list_of_tokens)
        else:
            raise ValueError(f"{self._mode} if not defined. "
                             f"Mode must be one of 'tf', 'df', and 'idf'. ")

        ## 根据上面的结果得到有效的tokens
        for k, v in stats.items():
            if self._low <= v < self._high:
                valid_terms.add(k)
        self._context[self._mode] = valid_terms

    def transform(self, input_: list) -> list:
        """Transform a list of tokens"""
        valid_terms = self._context[self._mode]
        result = list(filter(lambda token: token in valid_terms, input_))
        ## 如果过滤之后为空，则不过滤了
        if len(result) <= 0:
            result = input_
        return result

    @classmethod
    def _tf(cls, list_of_tokens: list) -> dict:
        stats = collections.Counter()
        for sentence in list_of_tokens:
            stats.update(sentence)
        return stats

    @classmethod
    def _df(cls, list_of_tokens: list) -> dict:
        stats = collections.Counter()
        for sentence in list_of_tokens:
            stats.update(set(sentence))
        return stats

    @classmethod
    def _idf(cls, list_of_tokens: list) -> dict:
        num_docs = len(list_of_tokens)
        stats = cls._df(list_of_tokens)
        for key, val in stats.most_common():
            stats[key] = np.log((1 + num_docs) / (1 + val)) + 1
        return stats
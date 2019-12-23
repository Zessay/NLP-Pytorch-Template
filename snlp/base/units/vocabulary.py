#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: vocabulary.py
@time: 2019/11/26 16:22
@description: 基于输入构建词表，fit的输入是list，transform的输入是list
'''

from snlp.base import StatefulUnit
import snlp.tools.constants as constants

class TermIndex(dict):
    """Map term to index"""
    def __missing__(self, key):
        """Map out-of-vocabulary terms to index 1."""
        return 1



class Vocabulary(StatefulUnit):
    def __init__(self, pad_value: str=constants.PAD_WORD, oov_value: str=constants.UNK_WORD):
        super().__init__()
        self._pad = pad_value
        self._oov = oov_value
        self._context['term_index'] = TermIndex()
        self._context['index_term'] = dict()

    def fit(self, tokens: list):
        # 先将Pad和OOV添加到词表中
        self._context['term_index'][self._pad] = 0
        self._context['term_index'][self._oov] = 1
        self._context['index_term'][0] = self._pad
        self._context['index_term'][1] = self._oov
        terms = set(tokens)
        for index, term in enumerate(terms):
            self._context['term_index'][term] = index + 2
            self._context['index_term'][index + 2] = term

    def transform(self, input_: list):
        """Transform a list of tokens to corresponding indices"""
        if type(input_[0]) == list:
            return [[self._context['term_index'][token] for token in uttr] for uttr in input_]
        else:
            return [self._context['term_index'][token] for token in input_]

#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: ngram_letter.py
@time: 2019/11/26 16:48
@description: 将单词转化为ngram形式，输入为list
'''
from snlp.base import Unit

class NgramLetter(Unit):
    def __init__(self, ngram: int=3, reduce_dim: bool=True, keep_source: bool=False):
        """
        初始化函数
        :param ngram: int型，表示n-gram中n的值
        :param reduce_dim: 表示是否将每个词的ngram用单独的list保存
        :param keep_source: 表示是否保留原始单词
        """
        self._ngram = ngram
        self._reduce_dim = reduce_dim
        self._keep_source = keep_source

    def transform(self, input_: list) -> list:
        n_letters = []
        if len(input_) == 0:
            token_ngram = []
            if self._reduce_dim:
                n_letters.extend(token_ngram)
            else:
                n_letters.append(token_ngram)
        else:
            for token in input_:
                new_token = '#' + token + '#'
                token_ngram = []
                while len(new_token) >= self._ngram:
                    token_ngram.append(new_token[:self._ngram])
                    new_token = new_token[1:]
                if self._keep_source:
                    token_ngram = [token] + token_ngram
                if self._reduce_dim:
                    n_letters.extend(token_ngram)
                else:
                    n_letters.append(token_ngram)
        return n_letters
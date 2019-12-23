#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: ngram.py
@time: 2019/12/3 17:27
@description: 将输入单词转化为ngram的形式
'''
import numpy as np
import snlp.tools.constants as constants
import snlp.base.units as units
from snlp.base import BaseCallback, BasePreprocessor

def _build_word_ngram_map(
        ngram_process_unit: units.NgramLetter,
        ngram_vocab_unit: units.Vocabulary,
        index_term: dict,
        mode: str='index'
) -> dict:
    """
    Generate the word to ngram vector mapping.
    :param ngram_process_unit: The fitted :class:`NgramLetter` object.
    :param ngram_vocab_unit: The fitted :class:`Vocabulary` object.
    :param index_term: The word index to term mapping dict.
    :param mode: It should be one of 'index', 'onehot', 'sum' or 'aggregate'.
    :return:
    """
    word_to_ngram = {}
    ngram_size = len(ngram_vocab_unit.context['index_term'])
    for idx, word in index_term.items():
        if idx == 0:
            continue
        elif idx == 1: # OOV
            word_ngram = [1]
        else:
            ## 将一个word转换成ngram列表
            ngrams = ngram_process_unit.transform([word])
            ## 将一个ngram列表转化为对应的索引
            word_ngram = ngram_vocab_unit.transform(ngrams)
        num_ngrams=  len(word_ngram)
        if mode == 'index':
            word_to_ngram[idx] = word_ngram
        elif mode == 'onehot':
            ## 得到一个单词对应的二维列表
            onehot = np.zeros((num_ngrams, ngram_size))
            onehot[np.arange(num_ngrams), word_ngram] = 1
            word_to_ngram[idx] = onehot
        elif mode == 'sum' or mode == 'aggregate':
            ## 得到一个单词在每个ngram索引上的数量和，一维列表
            onehot = np.zeros((num_ngrams, ngram_size))
            onehot[np.arange(num_ngrams), word_ngram] = 1
            sum_vector = np.sum(onehot, axis=0)
            word_to_ngram[idx] = sum_vector
        else:
            raise ValueError(f"mode error, it should be one of `index`, "
                             f"`onehot`, `sum` or `aggregate`.")
    return word_to_ngram

class DoubleNgram(BaseCallback):
    """
    Genrate the character n-gram for data.
    """
    def __init__(self,
                 preprocessor: BasePreprocessor,
                 mode: str='index'):
        self._mode = mode
        self._word_to_ngram = _build_word_ngram_map(
            preprocessor.context['ngram_process_unit'],
            preprocessor.context['ngram_vocab_unit'],
            preprocessor.context['vocab_unit'].context['index_term'],
            mode
        )
    def on_batch(self, x, y):
        """Insert `ngram_left` and `ngram_right` to `x`. """
        batch_size = len(x[constants.TEXT_LEFT])
        x[constants.NGRAM_LEFT] = [[] for _ in range(batch_size)]
        x[constants.NGRAM_RIGHT] = [[] for _ in range(batch_size)]
        for idx, row in enumerate(x[constants.TEXT_LEFT]):
            for term in row:
                x[constants.NGRAM_LEFT][idx].append(self._word_to_ngram[term])
        for idx, row in enumerate(x[constants.TEXT_RIGHT]):
            for term in row:
                x[constants.NGRAM_RIGHT][idx].append(self._word_to_ngram[term])
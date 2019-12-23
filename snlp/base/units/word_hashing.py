#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: word_hashing.py
@time: 2019/11/26 16:36
@description: 根据term_index对单词进行hash映射，输入是list型（可能是一维或者二维）
'''

import collections
import numpy as np
from snlp.base import Unit

class WordHashing(Unit):
    """
    The input of this class should be a list of word sub-letter list extracted from one document. The
    output is the word-hashing representation of this document.

    :class: `NgramLetterUnit` and :class: `VocabularyUnit` are two essential
    prerequisite of :class: `WordHashingUnit`

    Examples:
       >>> letters = [['#te', 'tes','est', 'st#'], ['oov']]
       >>> word_hashing = WordHashing(
       ...     term_index={
       ...      '_PAD': 0, 'OOV': 1, 'st#': 2, '#te': 3, 'est': 4, 'tes': 5
       ...      })
       >>> hashing = word_hashing.transform(letters)
       >>> hashing[0]
       [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
       >>> hashing[1]
       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    """
    def __init__(self, term_index: dict):
        self._term_index = term_index

    def transform(self, input_: list) -> list:
        if any([isinstance(elem, list) for elem in input_]):
            hashing = np.zeros((len(input_), len(self._term_index)))
            for idx, word in enumerate(input_):
                ## 对应的值是出现的频率
                counted_letters = collections.Counter(word)
                for key, value in counted_letters.items():
                    letter_id = self._term_index.get(key, 1)
                    hashing[idx, letter_id] = value
        else:
            hashing = np.zeros(len(self._term_index))
            counted_letters = collections.Counter(input_)
            for key, value in counted_letters.items():
                letter_id = self._term_index.get(key, 1)
                hashing[letter_id] = value

        return hashing.tolist()
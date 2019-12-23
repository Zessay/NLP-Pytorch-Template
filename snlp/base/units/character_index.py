#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: character_index.py
@time: 2019/11/21 21:37
@description: 构建基于字符的索引，输入是二维的list
'''
from snlp.base import Unit

class CharacterIndex(Unit):
    """
    Examples:
        >>> input_ = [['#', 'a', '#'],['#', 'o', 'n', 'e', '#']]
        >>> character_index = CharacterIndex(
        ...     char_index={
        ...      '<PAD>': 0, '<OOV>': 1, 'a': 2, 'n': 3, 'e':4, '#':5})
        >>> index = character_index.transform(input_)
        >>> index
        [[5, 2, 5], [5, 1, 3, 4, 5]]
    """
    def __init__(self, char_index: dict):
        """
        :param char_index: 表示字符到索引的字典
        """
        self._char_index = char_index

    def transform(self, input_: list) -> list:
        idx = []
        for i in range(len(input_)):
            current = [
                self._char_index.get(input_[i][j], 1)
                for j in range(len(input_[i]))]
            idx.append(current)
        return idx
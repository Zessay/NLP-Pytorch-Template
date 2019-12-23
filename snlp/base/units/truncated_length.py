#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: truncated_length.py
@time: 2019/11/26 16:14
@description: 对超出长度的句子进行截断，输入是list
'''
from snlp.base import Unit

class TruncatedLength(Unit):
    """Process unit to truncate the text that exceeds the set length."""
    def __init__(self, text_length: int, truncate_mode: str='post'):
        """
        初始化函数
        :param text_length: int型，表示截断后文本的长度
        :param truncate_mode: str型，`pre`表示从前面截断，`post`表示从后面截断
        """
        self._text_length = text_length
        self._truncate_mode = truncate_mode

    def transform(self, input_: list) -> list:
        """Truncate the text that exceeds the specified maximum length."""
        if len(input_) <= self._text_length:
            truncated_tokens = input_
        else:
            if self._truncate_mode == "pre":
                truncated_tokens = input_[-self._text_length:]
            elif self._truncate_mode == 'post':
                truncated_tokens = input_[:self._text_length]
            else:
                raise ValueError(f'{self._truncate_mode} is a invalid truncate mode.')
        return truncated_tokens

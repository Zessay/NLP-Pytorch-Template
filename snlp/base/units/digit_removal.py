#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: digit_removal.py
@time: 2019/11/21 21:28
@description: 移除数字，输入是list型
'''

from snlp.base import Unit
from snlp.tools.common import is_number

class DigitRemoval(Unit):
    """
    Remove digits from list of tokens
    """
    def transform(self, input_: list) -> list:
        return [token for token in input_ if not is_number(token)]

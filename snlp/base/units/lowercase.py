#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: lowercase.py
@time: 2019/11/22 16:21
@description: 将字母小写，输入是list
'''
from snlp.base import Unit


class Lowercase(Unit):
    """Process unit for text lower case."""

    def transform(self, input_: list) -> list:
        return [token.lower() for token in input_]
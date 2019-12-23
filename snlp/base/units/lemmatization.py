#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: lemmatization.py
@time: 2019/11/22 16:23
@description: 对动词的词形进行转换，输入是list
'''

import nltk

from snlp.base import Unit


class Lemmatization(Unit):
    """Process unit for token lemmatization."""

    def transform(self, input_: list) -> list:
        lemmatizer = nltk.WordNetLemmatizer()
        return [lemmatizer.lemmatize(token, pos='v') for token in input_]

#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: en_tokenize.py
@time: 2019/11/26 16:11
@description: 用于英文分词，输入是str
'''
import nltk
from snlp.base import Unit

class Tokenize(Unit):
    """Process unit for english text tokenization"""
    def transform(self, input_: str) -> list:
        return nltk.word_tokenize(input_)

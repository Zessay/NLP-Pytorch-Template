#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: cn_tokenize.py
@time: 2019/11/21 21:26
@description: 中文的分词或者分字，注意这里的输入是str型
'''
import jieba
from snlp.base import Unit
from snlp.tools.common import is_chinese_char

class CNTokenize(Unit):
    """Process unit for text tokenization."""

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """

        return jieba.lcut(input_)

class CNCharTokenize(Unit):
    """基于字符进行分隔"""
    def transform(self, input_: str) -> list:
        result = ""
        for ch in input_:
            if is_chinese_char(ch):
                result += ' ' + ch + ' '
            else:
                result += ch

        return result.split()
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: stemming.py
@time: 2019/11/22 16:24
@description: 对英文单词进行词干化
'''
import nltk

from snlp.base import Unit

class Stemming(Unit):
    """Process unit for token stemming"""
    def __init__(self, stemmer='porter'):
        """stemmer可选`porter`或者`lancaster`"""
        self.stemmer = stemmer

    def transform(self, input_: list) -> list:
        if self.stemmer == 'porter':
            porter_stemmer = nltk.stem.PorterStemmer()
            return [porter_stemmer.stem(token) for token in input_]
        elif self.stemmer == 'lancaster' or self.stemmer == 'krovetz':
            lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
            return [lancaster_stemmer.stem(token) for token in input_]
        else:
            raise ValueError(
                'Not supported supported stemmer type: {}'.format(
                    self.stemmer))



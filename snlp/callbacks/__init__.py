#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: __init__.py.py
@time: 2019/11/20 20:32
@description: 
'''
from snlp.callbacks.lambda_callback import LambdaCallback
from snlp.callbacks.ngram import DoubleNgram
from snlp.callbacks.padding import SingleBasicPadding, DoubleBasicPadding, MultiQAPadding
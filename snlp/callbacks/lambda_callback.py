#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: lambda_callback.py
@time: 2019/12/3 16:56
@description: 自定义Lambda表达式用于callback
'''

from snlp.base import BaseCallback

class LambdaCallback(BaseCallback):
    def __init__(self,
                 on_batch=None):
        self._on_batch = on_batch

    def on_batch(self, x, y):
        if self._on_batch:
            self.on_batch(x, y)

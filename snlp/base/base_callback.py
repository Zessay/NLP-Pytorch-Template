#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: base_callback.py
@time: 2019/11/20 20:34
@description: 
'''
import abc

class BaseCallback(abc.ABC):
    """
    The base callback class
    """
    def on_batch(self, x, y):
        """
        callback during iter batch.
        """

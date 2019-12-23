#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: average_meter.py
@time: 2019/11/26 21:38
@description: 定义一个计算平均值的类
'''

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0.
        self._avg = 0.
        self._sum = 0.
        self._count = 0.

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count

    @property
    def avg(self):
        return self._avg
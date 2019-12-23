#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: timer.py
@time: 2019/11/26 21:37
@description: 定义一个计时器类
'''
import time

class Timer(object):
    """Computes elapsed time."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset timer."""
        self.running = True
        self.total = 0
        self.start = time.time()

    def resume(self):
        """Resume."""
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        """Stop."""
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    @property
    def time(self):
        """Return time."""
        if self.running:
            return self.total + time.time() - self.start
        return self.total
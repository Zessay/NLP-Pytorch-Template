#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: basic_dataset.py
@time: 2019/12/17 11:17
@description: 注意所有的Dataset的index_pool必须是二维的
'''

import numpy as np

from snlp.base import BaseDataset

class BasicDataset(BaseDataset):
    def __init__(self, x: list, y: list, callbacks=None):
        super().__init__(callbacks=callbacks)
        self.x = x
        self.y = y
        self.sample()  # 先获取候选的索引池（index pool）

    def get_index_pool(self):
        '''
        index_pool用来保存每一次索引返回的list
        :return:
        '''
        # 默认为x的长度，这里要保证是二维的，便于统一，即[[0], [1], [2],...]
        index_pool = np.expand_dims(range(len(self.x)), axis=1).tolist()
        return index_pool

    def sort(self):
        '''
        按照x中数据的长度进行排序
        '''
        old_index_pool = self._index_pool
        lengths = [len(item) for item in self.x]
        sort_index = np.argsort(lengths)
        self._index_pool = [old_index_pool[index] for index in sort_index]

    def __getitem__(self, item: int):
        x, y = self.x[item], self.y[item]
        self._handle_callback_on_batch(x, y)
        return x, y
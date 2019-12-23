#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: base_dataset.py
@time: 2019/11/19 20:46
@description: 注意所有的Dataset的index_pool必须是二维的
'''

import abc
import numpy as np
from torch.utils import data

class BaseDataset(data.Dataset, abc.ABC):
    def __init__(self, callbacks=None):
        self._callbacks = callbacks
        self._index_pool = None

    def get_index_pool(self):
        '''
        index_pool用来保存每一次索引返回的list,
        需要根据情况自己实现
        :return:
        '''
        return []

    def sample(self):
        '''
        表示从原始的数据中采样，来获取index池
        '''
        self._index_pool = self.get_index_pool()

    def shuffle(self):
        '''
        将索引池打乱
        '''
        np.random.shuffle(self._index_pool)

    @abc.abstractmethod
    def sort(self):
        '''
        按照x中数据的长度进行排序
        '''

    @property
    def index_pool(self):
        return self._index_pool

    def __len__(self) -> int:
        return len(self._index_pool)

    @abc.abstractmethod
    def __getitem__(self, item: int):
        """实现取元素的代码"""


    def _handle_callback_on_batch(self, x, y):
        if self._callbacks is not None:
            for callback in self._callbacks:
                callback.on_batch(x, y)


#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: sampler.py
@time: 2019/11/20 21:06
@description: 这里定义的每一个Sampler，每次迭代返回的是一个list；
              所以对应的index_pool中的每一个元素也是一个list，这需要在定义Dataset的时候注意；
              于是DataLoader只能接受一个一维的list作为索引，所以在BatchSampler中需要将上面
              得到的每一个list合并，直到满足batch_size的大小。
'''
import math
from torch.utils.data import Sampler, Dataset

class SequentialSampler(Sampler):
    """
    Sample elements sequentially
    """
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __iter__(self):
        return iter(self._dataset.index_pool)

    def __len__(self):
        return len(self._dataset)

class SortedSampler(Sampler):
    """
    Sample elements according to length
    """
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __iter__(self):
        self._dataset.sort()
        return iter(self._dataset.index_pool)

    def __len__(self):
        return len(self._dataset)

# -------------------------------------------------------------------

class RandomSampler(Sampler):
    """
    Sample elements randomly
    """
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __iter__(self):
        self._dataset.shuffle()
        return iter(self._dataset.index_pool)

    def __len__(self):
        return len(self._dataset)

# -------------------------------------------------------------------

class BatchSampler(Sampler):
    """
    Wraps another sampler to yield th indices of a batch
    """
    def __init__(self,
                 sampler: Sampler,
                 batch_size: int = 32):
        self._sampler = sampler
        self._batch_size = batch_size

    def __iter__(self):
        """Get the indices of a batch"""
        batch = []
        for idx in self._sampler:
            batch.append(idx)
            if len(batch) == self._batch_size:
                batch = sum(batch, [])
                yield batch
                batch = []
        if len(batch) > 0:
            batch = sum(batch, [])
            yield batch

    def __len__(self):
        """Get the total number of batch"""
        return math.ceil(len(self._sampler) / self._batch_size)
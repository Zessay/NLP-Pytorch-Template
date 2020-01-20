#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: pair_dataset.py
@time: 2019/12/17 11:08
@description: 包含负样本的Dataset，注意所有的Dataset的index_pool必须是二维的
'''
import typing
import numpy as np
import pandas as pd

from snlp.base import BaseDataset
import snlp.tools.constants as constants

class PairDataset(BaseDataset):
    def __init__(self, df: pd.DataFrame,
                 num_neg: int,
                 callbacks: typing.Optional[list]=None):
        super().__init__(callbacks=callbacks)
        self.df = df
        self._columns = list(df.columns)
        if constants.LABEL in self._columns:
            self._columns.remove(constants.LABEL)
        self._num_neg = num_neg
        self.sample()

    def get_index_pool(self):
        index_pool = []
        step_size = self._num_neg + 1
        num_instances = int(self.df.shape[0] / step_size)
        for i in range(num_instances):
            ## 获取每一个样例对应的索引的上下边界
            lower = i * step_size
            upper = (i + 1) * step_size
            ## 获取对应的索引并添加到pool中
            indices = list(range(lower, upper))
            if indices:
                index_pool.append(indices)
        return index_pool

    def sort(self):
        """按照每一组元素右边长度排序"""
        old_index_pool = self._index_pool
        max_instance_right_length = []
        for row in old_index_pool:
            instance = self.df.loc[row, :][constants.SORT_LEN].values
            max_instance_right_length.append(max(instance))
        sort_index = np.argsort(max_instance_right_length)
        self._index_pool = [old_index_pool[index] for index in sort_index]

    def __getitem__(self, item: int):
        item_df = self.df.iloc[item]
        # 这里只是取一个元素，所以不需要使用.values
        x, y = item_df[self._columns].to_dict(), item_df[constants.LABEL]
        self._handle_callback_on_batch(x, y)
        return x, y

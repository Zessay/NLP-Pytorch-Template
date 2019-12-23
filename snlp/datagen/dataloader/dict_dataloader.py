#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: dict_dataloader.py
@time: 2019/11/20 20:37
@description: 
'''
import math
import typing
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import snlp.tools.constants as constants
from snlp.base.base_callback import BaseCallback
from snlp.datagen.sampler import SequentialSampler, SortedSampler, RandomSampler, BatchSampler

class DictDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 64,
                 stage='train',
                 device: typing.Union[torch.device, int, list, None] = None,
                 shuffle: bool = False,
                 sort: bool = True,
                 num_workers: int = 0,
                 timeout: int = 0,
                 collate_fn: typing.Optional[typing.Callable]=None,
                 callback: BaseCallback = None,
                 pin_memory: bool = False,
                 worker_init_fn=None):
        if stage not in ('train', 'dev', 'test'):
            raise ValueError(f"{stage} is not a valid stage type."
                             f"Must be one of `train`, `dev`, `test`.")

        if shuffle and sort:
            raise ValueError(f"parameters `shuffle` and `sort` conflict, "
                             f"should not both be `True`.")
        if isinstance(device, list) and len(device):
            device = device[0]
        elif not (isinstance(device, torch.device) or isinstance(device, int)):
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self._dataset = dataset
        self._shuffle = shuffle
        self._sort = sort
        self._batch_size = batch_size

        self._pin_memory = pin_memory
        self._timeout = timeout
        self._num_workers = num_workers
        self._worker_init_fn = worker_init_fn
        self._collate_fn = collate_fn or mz_collate
        self._device = device
        self._stage = stage
        self._callback = callback

        self._dataloader = None

    def __len__(self) -> int:
        """Get the total number of batches."""
        return math.ceil(len(self._dataset) / self._batch_size)

    def init_epoch(self):
        if not self._shuffle and not self._sort:
            sampler = SequentialSampler(self._dataset)
        elif not self._shuffle and self._sort:
            sampler = SortedSampler(self._dataset)
        elif self._shuffle and not self._sort:
            sampler = RandomSampler(self._dataset)

        batch_sampler = BatchSampler(sampler, self._batch_size)

        self._dataloader = DataLoader(
            self._dataset,
            collate_fn=self._collate_fn,
            batch_sampler=batch_sampler,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            timeout=self._timeout,
            worker_init_fn=self._worker_init_fn
        )

    def __iter__(self):
        self.init_epoch()
        for batch_data in self._dataloader:
            # 这里x是dict类型，而y是ndarray类型
            x, y = batch_data
            self._handle_callbacks_on_batch(x, y)

            batch_x = {}
            for key, value in x.items():
                if key == constants.ID_LEFT or key == constants.ID_RIGHT:
                    continue
                if value is None or type(value[0]) == str:
                    continue
                try:
                    batch_x[key] = torch.tensor(
                        value.tolist(),
                        device=self._device, pin_memory=self._pin_memory)
                except:
                    pass

            # 这里返回的batch_x是字典型的
            if self._stage == 'test':
                yield batch_x, None
            else:
                if y.dtype == int: # 对应task='classification'
                    batch_y = torch.tensor(
                        np.array(y).squeeze(), dtype=torch.long,
                        device=self._device, pin_memory=self._pin_memory)
                else: # 对应task='ranking'或者'regression'
                    batch_y = torch.tensor(
                        np.array(y).squeeze(), dtype=torch.float,
                        device=self._device, pin_memory=self._pin_memory)
                yield batch_x, batch_y

    def _handle_callbacks_on_batch(self, x, y):
        if self._callback is not None:
            self._callback.on_batch(x, y)

    @property
    def label(self) -> np.ndarray:
        # 转化为一维列表
        indices = sum(self._dataset.index_pool[:], [])
        _, y = self._dataset[indices]
        return y.values.squeeze() if y is not None else None


def mz_collate(batch):
    """将一个batch中的所有字段存放到一个字典中，字段的最外围是batch size"""
    batch_x = collections.defaultdict(list)
    batch_y = []

    for x, y in batch:
        for key in x.keys():
            batch_x[key].append(x[key])
        if y is not None:
            batch_y.append(y)

    for key in batch_x.keys():
        batch_x[key] = np.array(batch_x[key])

    if len(batch_y) == 0:
        batch_y = None
    else:
        batch_y = np.array(batch_y)
    return batch_x, batch_y

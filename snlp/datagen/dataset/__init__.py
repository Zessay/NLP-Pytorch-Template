#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: __init__.py.py
@time: 2019/12/17 11:17
@description: 
'''

from snlp.datagen.dataset.basic_dataset import BasicDataset
from snlp.datagen.dataset.pair_dataset import PairDataset

__all__ = ["BasicDataset", "PairDataset"]
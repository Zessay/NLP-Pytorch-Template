#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: build_unit.py
@time: 2019/11/26 17:08
@description: 根据DataFrame构建单词到索引的映射表
'''
import typing
import functools
import pandas as pd
from tqdm import tqdm
from snlp.base import StatefulUnit, Unit
from snlp.base.units import Vocabulary

def build_unit_from_df(unit: StatefulUnit,
                       data: pd.DataFrame,
                       columns: list,
                       flatten: bool=True,
                       verbose: int=1):

    corpus = []
    if flatten:
        for col in columns:
            corpus.extend(sum(data[col].values.tolist(), [])) # 将二维列表展开
    else:
        for col in columns:
            corpus.extend(data[col].values.tolist())

    if verbose:
        description = 'Building ' + unit.__class__.__name__ + \
                      ' from a dataframe.'
        # 在这里显示进度条信息
        corpus = tqdm(corpus, desc=description)
    unit.fit(corpus)
    return unit

def build_vocab_unit(data: pd.DataFrame, columns: list, verbose: int=1) -> Vocabulary:
    return build_unit_from_df(
        unit=Vocabulary(),
        data=data,
        columns=columns,
        flatten=True,
        verbose=verbose
    )


def chain_transform(units: typing.List[Unit]) -> typing.Callable:
    """
    Compose unit transformations into a single function.
    """
    @functools.wraps(chain_transform)
    def wrapper(arg):
        for unit in units:
            arg = unit.transform(arg)
        return arg

    unit_names = ' => '.join(unit.__class__.__name__ for unit in units)
    wrapper.__name__ += ' of ' + unit_names
    return wrapper

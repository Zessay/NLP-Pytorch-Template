#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: naive_preprocessor.py
@time: 2019/11/26 19:04
@description: 最简单的预处理
'''
import pandas as pd
from tqdm import tqdm

from snlp.base import BasePreprocessor, units
from snlp.preprocessors import apply_on_df_columns
from snlp.tools.build_unit import build_vocab_unit, chain_transform


tqdm.pandas()

class NaivePreprocessor(BasePreprocessor):
    """Define Naive preprocessors"""

    def fit(self, data: pd.DataFrame, columns: list,  verbose: int=1):
        func = chain_transform(self._default_units())
        # 应用所有的是转换
        data = apply_on_df_columns(data, columns, func, verbose=verbose)
        vocab_unit = build_vocab_unit(data, columns=columns, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit
        return self

    def transform(self, data: pd.DataFrame, columns: list,
                  verbose: int=1) -> pd.DataFrame:
        """
        Apply transformation on data, create truncated length, representation.
        """
        units_ = self._default_units()
        units_.append(self._context['vocab_unit'])
        units_.append(
            units.TruncatedLength(text_length=30, truncate_mode='post')
        )
        func = chain_transform(units_)
        data = apply_on_df_columns(data, columns, func, verbose=verbose)
        for col in columns:
            data[col+'_len'] = data[col].apply(len)
            empty_id = data[data[col+'_len'] == 0].index.tolist()
            data.drop(index=empty_id, axis=0, inplace=True)
        data.dropna(axis=0, inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data
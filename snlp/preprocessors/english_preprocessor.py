#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: english_preprocessor.py
@time: 2019/11/26 19:43
@description: 常规的英文预处理工具
'''
from tqdm import tqdm
import typing
import pandas as pd

from snlp.base import BasePreprocessor, units
from snlp.preprocessors import apply_on_df_columns
from snlp.tools.build_unit import build_vocab_unit, build_unit_from_df, chain_transform

tqdm.pandas()

class ENPreprocessor(BasePreprocessor):
    def __init__(self,
                 truncated_mode: str='post',
                 truncated_length: int = None,
                 filter_mode: str='df',
                 filter_low_freq: float=1,
                 filter_high_freq: float=float('inf'),
                 remove_stop_words: bool=False,
                 ngram_size: typing.Optional[int]=None):
        super().__init__()
        ## 基于词频或者逆文档频率对单词进行过滤
        self._filter_unit = units.FrequencyFilter(
            low=filter_low_freq,
            high=filter_high_freq,
            mode=filter_mode
        )

        self._units = self._default_units()
        if remove_stop_words:
            self._units.append(units.StopRemoval())

        self._truncated_mode = truncated_mode
        self._truncated_length = truncated_length
        if self._truncated_length:
            self._truncated_unit = units.TruncatedLength(
                self._truncated_length, self._truncated_mode
            )

        self._ngram_size = ngram_size
        if ngram_size:
            self._context['ngram_process_unit'] = units.NgramLetter(
                ngram=ngram_size, reduce_dim=True
            )

    def fit(self, data: pd.DataFrame, columns: list, filter_cols: typing.Optional[list]=None, verbose: int=1):
        """
        Fit pre-processing context for transformation.
        """
        data = data.copy()
        ## 指定需要过滤的列，默认只对最后一列进行过滤
        self.filter_cols = filter_cols or [columns[-1]]

        ## 对每一列的语句进行转换
        self._trans_func = chain_transform(self._units)
        ### 应用所有的转换单元
        data = apply_on_df_columns(data, columns, self._trans_func, verbose=verbose)
        ## 需要先将每一行转换成list型，再定义过滤器，这里只针对最后一列进行过滤
        fitted_filter_unit = build_unit_from_df(self._filter_unit,
                                                data,
                                                self.filter_cols,
                                                flatten=False,
                                                verbose=verbose)
        self._context['filter_unit'] = fitted_filter_unit
        data = apply_on_df_columns(data, self.filter_cols, fitted_filter_unit.transform, verbose=verbose)

        vocab_unit = build_vocab_unit(data, columns, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit

        ## 获取词表大小
        vocab_size = len(vocab_unit.context['term_index'])
        # 保存单词到索引的映射
        self._context['term_index'] = vocab_unit['term_index']
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size

        if self._ngram_size:
            data = apply_on_df_columns(data, columns, self._context['ngram_process_unit'].transform,
                                       verbose=verbose)
            ngram_unit = build_vocab_unit(data, columns, verbose=verbose)
            self._context['ngram_vocab_unit'] = ngram_unit
            self._context['ngram_vocab_size'] = len(ngram_unit.context['term_index'])
        return self

    def transform(self, data: pd.DataFrame, columns: list,  verbose: int=1) -> pd.DataFrame:
        data = data.copy()
        ## 应用所有的转换单元
        data = apply_on_df_columns(data, columns, self._trans_func, verbose=verbose)
        ## 应用单词过滤器
        data = apply_on_df_columns(data, self.filter_cols, self._context['filter_unit'].transform, verbose=verbose)
        ## 将单词转换成索引
        data = apply_on_df_columns(data, columns, self._context['vocab_unit'].transform, verbose=verbose)
        ## 是否应用截断
        if self._truncated_length:
            data = apply_on_df_columns(data, columns, self._truncated_unit.transform, verbose=verbose)
        ## 添加表示长度的字段
        for col in columns:
            data[col+'_len'] = data[col].apply(len)
            empty_id = data[data[col+'_len'] == 0].index.tolist()
            data.drop(index=empty_id, axis=0, inplace=True)
        data.dropna(axis=0, inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data
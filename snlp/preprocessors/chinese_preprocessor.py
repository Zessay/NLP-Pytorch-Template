#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: chinese_preprocessor.py
@time: 2019/11/26 20:32
@description: 中文的常规预处理类
'''
from tqdm import tqdm
import typing
import logging
import pandas as pd

from snlp.base import BasePreprocessor, units
from snlp.preprocessors import apply_on_df_columns
from snlp.tools.build_unit import build_vocab_unit, build_unit_from_df, chain_transform
from albert_pytorch.model import tokenization_albert, tokenization_bert

tqdm.pandas()

logger = logging.getLogger("Preprocessor")

class CNPreprocessor(BasePreprocessor):
    def __init__(self,
                 tokenize_mode: str='word',
                 truncated_mode: str='post',
                 truncated_length: int=None,
                 filter_mode: str='df',
                 filter_low_freq: float=1,
                 filter_high_freq: float=float('inf'),
                 stopwords: typing.Optional[list]=None,
                 remove_punc: bool=False,
                 lowercase: bool=False):
        """
        Preprocessor for Chinese.
        :param tokenize_mode: 'word'表示分词，'char'表示分字
        """
        super().__init__()
        self._units = []
        self._truncated_mode = truncated_mode
        self._truncated_length = truncated_length
        if self._truncated_length:
            self._truncated_unit = units.TruncatedLength(
                self._truncated_length, self._truncated_mode
            )
        self._filter_low_freq = filter_low_freq
        if filter_low_freq > 1:
            self._filter_unit = units.FrequencyFilter(
                low=filter_low_freq,
                high=filter_high_freq,
                mode=filter_mode
            )
        ## 定义分词的方式
        if tokenize_mode == 'word':
            self._tokenize_unit = units.CNTokenize()
        elif tokenize_mode == 'char':
            self._tokenize_unit = units.CNCharTokenize()
        else:
            raise ValueError(f"This tokenize mode {tokenize_mode} is not defined.")
        self._units.append(self._tokenize_unit)
        ## 是否去除停止词
        if stopwords is not None:
            self._units.append(units.CNStopRemoval(stopwords))
        ## 是否去除标点
        if remove_punc:
            self._units.append(units.CNPuncRemoval())
        ## 是否将英文字母小写
        if lowercase:
            self._units.append(units.Lowercase())

    def fit(self, data: pd.DataFrame, columns: list, filter_cols: typing.Optional[list]=None, verbose: int=1):
        logger.info("** 训练预处理器 **")
        data = data.copy()
        ## 对每一列的语句进行转换
        self._trans_func = chain_transform(self._units)
        ### 应用所有的转换单元
        data = apply_on_df_columns(data, columns, self._trans_func, verbose=verbose)

        if self._filter_low_freq > 1:
            # 基于词频或者逆文档频率过滤单词
            ## 指定需要过滤的列，默认只对最后一列进行过滤
            self.filter_cols = filter_cols or [columns[-1]]
            fitted_filter_unit = build_unit_from_df(self._filter_unit,
                                                    data,
                                                    self.filter_cols,
                                                    flatten=False,
                                                    verbose=verbose)
            self._context['filter_unit'] = fitted_filter_unit
            data = apply_on_df_columns(data, self.filter_cols, fitted_filter_unit.transform, verbose=verbose)

        ## 构建词表
        logger.info("** 构建词表 **")
        vocab_unit = build_vocab_unit(data, columns, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit
        vocab_size = len(vocab_unit.context['term_index'])
        self._context['term_index'] = vocab_unit.context['term_index']
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size
        logger.info(f"** 词表大小为 {vocab_size} **")
        logger.info("** 预处理器训练完成 **")
        return self

    def transform(self, data: pd.DataFrame, columns: list, drop_threshold: int=1, drop=True, verbose: int=1) -> pd.DataFrame:
        logger.info("** 对数据进行transform **")
        data = data.copy()
        for col in columns:
            data[col+'_len'] = data[col].apply(len)
            if drop:
                empty_id = data[data[col+'_len'] < drop_threshold].index.tolist()
                data.drop(index=empty_id, axis=0, inplace=True)
        data.dropna(axis=0, inplace=True)
        data.reset_index(drop=True, inplace=True)

        ## 应用所有的转换单元
        data = apply_on_df_columns(data, columns, self._trans_func, verbose=verbose)
        if self._filter_low_freq > 1:
            ## 应用单词过滤器
            data = apply_on_df_columns(data, self.filter_cols, self._context['filter_unit'].transform, verbose=verbose)

        ## 将单词转换成索引
        data = apply_on_df_columns(data, columns, self._context['vocab_unit'].transform, verbose=verbose)

        ## 是否应用截断
        if self._truncated_length:
            data = apply_on_df_columns(data, columns, self._truncated_unit.transform, verbose=verbose)
        # 长度应该是单词序列的长度，这有可能是词的，也有可能是字的
        for col in columns:
            data[col+'_len'] = data[col].apply(len)
            if drop:
                empty_id = data[data[col+'_len'] < 1].index.tolist()
                data.drop(index=empty_id, axis=0, inplace=True)
        data.dropna(axis=0, inplace=True)
        data.reset_index(drop=True, inplace=True)
        ## 添加表示长度的字段
        logger.info("** transform结束 **")
        return data

# -------------------------------------------------------------------------------------------

class CNPreprocessorForMultiQA(CNPreprocessor):
    def transform(self, data: pd.DataFrame, uttr_col: str="utterances", resp_col: str="response",
                  drop_threshold: int=3, drop=True, verbose: int=1) -> pd.DataFrame:
        def split_and_cut(uttrs):
            """将所有的utterances转化为分词之后的二维list"""
            # 得到所有utterances组成的list
            uttrs_list = uttrs.split("\t")
            for i in range(len(uttrs_list)):
                uttrs_list[i] = self._trans_func(uttrs_list[i])
            return uttrs_list

        logger.info("** 对数据进行transform **")

        # 只针对resp和最后一个utterance计算长度
        data[uttr_col+'_len'] = data[uttr_col].apply(lambda s: len(s.split('\t')[-1]))
        data[resp_col+'_len'] = data[resp_col].apply(len)
        if drop:
            empty_id = data[data[uttr_col+'_len'] < drop_threshold].index.tolist() + data[data[resp_col+'_len'] < drop_threshold].index.tolist()
            data.drop(index=empty_id, axis=0, inplace=True)
        # 丢弃缺失值
        data.dropna(axis=0, subset=[uttr_col, resp_col], inplace=True)
        data.reset_index(drop=True, inplace=True)

        ## 分别对utterances和response进行分词
        data = apply_on_df_columns(data, [uttr_col], split_and_cut, verbose=verbose)
        data = apply_on_df_columns(data, [resp_col], self._trans_func, verbose=verbose)
        # 使用单词过滤器
        if self._filter_low_freq > 1:
            data = apply_on_df_columns(data, self.filter_cols, self._context['filter_unit'].transform, verbose=verbose)
        # 将单词转换成索引
        data = apply_on_df_columns(data, [uttr_col, resp_col], self._context['vocab_unit'].transform, verbose=verbose)

        # 如果指定使用阶段，则只对resp进行截断
        if self._truncated_length:
            data = apply_on_df_columns(data, [resp_col], self._truncated_unit.transform, verbose=verbose)

        # 计算转换成id之后的长度，可能是分词的，也有可能是分字的
        data[uttr_col+'_len'] = data[uttr_col].apply(lambda s: len(s[-1]))
        data[resp_col+'_len'] = data[resp_col].apply(len)
        if drop:
            empty_id = data[data[uttr_col+'_len'] < 1].index.tolist() + data[data[resp_col+'_len'] < 1].index.tolist()
            data.drop(index=empty_id, axis=0, inplace=True)
        # 丢弃缺失值
        data.dropna(axis=0, subset=[uttr_col, resp_col], inplace=True)
        data.reset_index(drop=True, inplace=True)

        logger.info("** transform结束 **")
        return data


class CNAlbertPreprocessorForMultiQA(BasePreprocessor):
    def __init__(self, vocab_file, uttr_len, resp_len):
        ## 使用encode_plus会截断但是不会填充
        self.tokenizer = tokenization_bert.BertTokenizer(vocab_file=vocab_file).encode_plus
        self.uttr_len = uttr_len
        self.resp_len = resp_len

    def fit(self):
        pass

    def transform(self, data: pd.DataFrame,
                  uttr_col: str="utterances",
                  resp_col: str="response",
                  drop: bool=True,
                  drop_threshold: int=3,
                  verbose: int=1) -> pd.DataFrame:


        def split_and_cut(uttrs):
            uttrs_list = uttrs.split("\t")
            for i in range(len(uttrs_list)):
                uttrs_list[i] = self.tokenizer(uttrs_list[i], add_special_tokens=True,
                                                           max_length=self.uttr_len)['input_ids']
            return uttrs_list

        logger.info("** 对数据进行transform **")

        # 只针对resp和最后一个utterance计算长度
        data[uttr_col+'_len'] = data[uttr_col].apply(lambda s: len(s.split('\t')[-1]))
        data[resp_col+'_len'] = data[resp_col].apply(len)
        if drop:
            empty_id = data[data[uttr_col+'_len'] < drop_threshold].index.tolist() + data[data[resp_col+'_len'] < drop_threshold].index.tolist()
            data.drop(index=empty_id, axis=0, inplace=True)
        # 丢弃缺失值
        data.dropna(axis=0, subset=[uttr_col, resp_col], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # 使用Albert的结果进行分词
        data[uttr_col] = data[uttr_col].apply(split_and_cut)
        data[resp_col] = data[resp_col].apply(lambda s: self.tokenizer(s,
                                                                       add_special_tokens=True,
                                                                       max_length=self.resp_len)['input_ids'])

        # 长度应该是单词序列的长度，这有可能是词的，也有可能是字的
        # 计算转换成id之后的长度，可能是分词的，也有可能是分字的
        data[uttr_col+'_len'] = data[uttr_col].apply(lambda s: len(s[-1]))
        data[resp_col+'_len'] = data[resp_col].apply(len)
        if drop:
            empty_id = data[data[uttr_col+'_len'] < 1].index.tolist() + data[data[resp_col+'_len'] < 1].index.tolist()
            data.drop(index=empty_id, axis=0, inplace=True)
        # 丢弃缺失值
        data.dropna(axis=0, subset=[uttr_col, resp_col], inplace=True)
        data.reset_index(drop=True, inplace=True)

        logger.info("** transform结束 **")
        return data
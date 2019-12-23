#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: base_preprocessor.py
@time: 2019/11/21 20:47
@description: 用于文本预处理的一些类和方法
'''
import abc
import pandas as pd
import typing
from pathlib import Path
import pickle
from snlp.base import units


class BasePreprocessor(metaclass=abc.ABCMeta):
    """
    A preprocessors should be used in two steps.
    - `fit`: the collect useful information
    - `transform`: to use the information that saved when fit
    """
    DATA_FILENAME = "preprocessors.pkl"
    def __init__(self):
        self._context = {}

    @property
    def context(self):
        return self._context

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame, columns: list,  verbose: int=1) -> 'BasePreprocessor':
        """This method is expected to return itself as a callable object"""

    @abc.abstractmethod
    def transform(self, data: pd.DataFrame, columns: list,  verbose: int=1) -> pd.DataFrame:
        """Transform input data to expected manner"""

    def fit_transform(self, data: pd.DataFrame, columns: list, verbose: int=1):
        """Call fit-transform"""
        return self.fit(data, columns, verbose=verbose).transform(data, columns, verbose=verbose)

    def save(self, dirpath: typing.Union[str, Path]):
        dirpath = Path(dirpath)
        data_file_path = dirpath.joinpath(self.DATA_FILENAME)
        if not dirpath.exists():
            dirpath.mkdir(parents=True)
        pickle.dump(self, open(data_file_path, mode='wb'))

    @classmethod
    def _default_units(cls) -> list:
        ## 先分词，再小写，最后去标点
        return [
            units.Tokenize(),
            units.Lowercase(),
            units.CNPuncRemoval()
        ]

def load_preprocessor(dirpath: typing.Union[str, Path]) -> 'BasePreprocessor':
    """Load the fitted context"""
    dirpath = Path(dirpath)
    data_file_path = dirpath.joinpath(BasePreprocessor.DATA_FILENAME)
    return pickle.load(open(data_file_path, mode='rb'))
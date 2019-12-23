#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: __init__.py.py
@time: 2019/11/21 21:10
@description: 
'''

from tqdm import tqdm

tqdm.pandas()

def apply_on_df_columns(data, columns, func, verbose=1):
    for col in columns:
        if verbose:
            tqdm.pandas(desc="Processing " + col + " with " + func.__name__)
            data[col] = data[col].progress_apply(func)
        else:
            data[col] = data[col].apply(func)
    return data

from snlp.preprocessors.naive_preprocessor import NaivePreprocessor
from snlp.preprocessors.english_preprocessor import ENPreprocessor
from snlp.preprocessors.chinese_preprocessor import CNPreprocessor


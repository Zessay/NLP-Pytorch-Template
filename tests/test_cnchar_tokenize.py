#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_cnchar_tokenize.py
@time: 2020/1/2 14:07
@description: 
'''
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.getcwd()))

from snlp.base.units import CNCharTokenize

cnchar_token = CNCharTokenize()

print(cnchar_token.transform("你好，你叫什么名字啊，[robot_name]"))
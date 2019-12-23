#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: constants.py
@time: 2019/12/4 11:40
@description: 定义一些常量
'''
import os

# ----------------- log文件相关配置 -------------------
PACKAGE_NAME = os.path.basename(os.path.dirname(os.getcwd()))
LOG_FILE = "./train.log" # log保存的路径

# ----------------- DataFrame字段配置 -----------------
## 标签字段的名称
LABEL = 'label'

######## ------- 对于单个文本的字段名 -------
TEXT = 'text'
TEXT_LEN = 'text_len'
# ****** 以上是必须包含的字段，下面是可选字段 ********
NGRAM = 'ngram'

######## -------- 两个文本的字段名 ----------
TEXT_LEFT = 'text_left'
TEXT_RIGHT = 'text_right'
LEFT_LEN = 'text_left_len' # 必须是原列名加上`_len`
RIGHT_LEN = 'text_right_len'
# ****** 以上是必须包含的字段，下面是可选字段 *******
ID_LEFT = 'id_left'
ID_RIGHT = 'id_right'
NGRAM_LEFT = 'ngram_left'
NGRAM_RIGHT = 'ngram_right'

######## -------- 用于多轮QA的字段名 ----------
UTTRS = 'utterances'
RESP = 'response'
UTTRS_LEN = 'utterances_len'
RESP_LEN = 'response_len'

## 用于排序比较文本长度的字段
SORT_LEN = 'response_len'

# ----------------- 填充词汇相关配置 -------------------

PAD = 0  # 必须为0
UNK = 1  # 必须为1
BOS = 2
EOS = 3

PAD_WORD = "<pad>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"
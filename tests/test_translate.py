#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_translate.py
@time: 2020/3/27 10:51
@description: 
'''
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
import numpy as np
from pathlib import Path

from snlp.tools.translation import Google, Youdao

basename = "/home/speech/data"
file = "quora_duplicate_questions.tsv"

df = pd.read_csv(Path(basename) / file, sep="\t")
tmp = df[:10]
trans = Google()

tmp['question1'] = tmp['question1'].apply(trans.run)
tmp['question2'] = tmp['question2'].apply(trans.run)

tmp.to_csv(Path(basename) / "result_trans.csv", index=False)
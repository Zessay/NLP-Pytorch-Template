#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: process_for_es.py
@time: 2020/2/24 18:40
@description: 
'''
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import json
import copy
from pathlib import Path
from snlp.tools.common import is_chinese_char, is_chinese_punc

data_name = "loan"
basename = "/home/speech/data/test_data"
date = "0225"
# from_file = "business_dialogue.txt"
# to_file = "business_single_turn.json"
from_file = f"{data_name}_dialogue_{date}.txt"
to_file = f"{data_name}_single_turn_{date}.json"


template = {"id": 0,
           "utterance": "",
           "responses": [
               {
               "emotion": "",
               "sentiment": "",
               "topic": "",
               "timeline": {},
               "character": "",
               "reply": "",
               "language_style": ""
               }
           ],
           "app_id": f"{data_name}_dialogue"
           }

uttrs = []
resps = []
flag = 0
with open(Path(basename) / from_file, 'r', encoding="utf8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        j = i
        if not line or flag:
            if not line:
                flag += 1
            j = j - flag

        if line:
            if j & 1 == 0:
                uttrs.append(line)
            else:
                resps.append(line)


for i, uttr in enumerate(uttrs):
    result = ""
    for ch in uttr:
        if is_chinese_punc(ch) or is_chinese_char(ch):
            result += f" {ch} "
        else:
            result += ch
    result = result.strip().replace("  ", " ")
    uttrs[i] = result



dialogues = []
for i, (uttr, resp) in enumerate(zip(uttrs, resps)):
    d = copy.deepcopy(template)
    d["id"] = i
    d["utterance"] = uttr
    d["responses"][0]["reply"] = resp
    dialogues.append(d)

with open(Path(basename) / to_file, 'w', encoding="utf8") as f:
    json.dump(dialogues, f, ensure_ascii=False, indent=4)
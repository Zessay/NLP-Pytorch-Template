#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: clean_manager_log.py
@time: 2020/2/19 17:30
@description: 用于提取dialogue manager的日志数据中的对话结果
'''

import re
import collections
import pandas as pd
import numpy as np
import json
import demjson
from pathlib import Path

threshold = 500
basename = "/home/speech/data/log_data"
log_file = "manager_29.log"
to_file = "log_dialogue.txt"
keys = ["user_id", "robot_id", "utterance", "responses", 'timestamp']

# 获取数据
with open(Path(basename) / log_file, 'r', encoding='utf8') as f:
    text = f.read()
    results = re.findall(r"@@ Robot:\s(\{.*?\})\s", text)
# 转换成json格式
for i in range(len(results)):
    try:
        results[i] = demjson.decode(results[i])
    except Exception as e:
        print(results[i])
        print(e)
# 转换成pandas需要的格式
dicts = collections.defaultdict(list)
for item in results:
    for key in keys:
        try:
            dicts[key].append(item[key])
        except Exception as e:
            if key == "responses" and "response" in item:
                dicts[key].append(item["response"])
            else:
                dicts[key].append(np.nan)

df = pd.DataFrame(dicts)
df.dropna(subset=['user_id'], inplace=True)
df['responses'] = df['responses'].apply(lambda s: eval(str(s))[0])
df = df[df['user_id'] != 'User_0']
df.drop_duplicates(inplace=True)
df = df.reset_index(drop=True)

# 分组按照timestamp排序
result_df = pd.DataFrame()
for key, group in df.groupby(by=['user_id'], as_index=False, sort=False):
    group = group.sort_values(by=['timestamp'])
    f_ts = group['timestamp'].values.astype(int)
    l_ts = np.append(f_ts[1:], f_ts[-1])
    sub = l_ts - f_ts
    group['sub'] = [0] + sub.tolist()[:-1]
    result_df = pd.concat([result_df, group], axis=0, sort=False, ignore_index=True)

result_df.drop_duplicates(inplace=True)
result_df['flag'] = result_df['sub'] >= threshold
result_df['flag'] = result_df['flag'].astype(int)

with open(Path(basename) / to_file, 'w', encoding="utf8") as to_f:
    for key, group in result_df.groupby(by=['user_id'], as_index=False, sort=False):
        for index, row in group.iterrows():
            if row.flag:
                to_f.write("\n")
            to_f.write(row.utterance+'\n')
            to_f.write(row.responses+'\n')
        to_f.write("\n")
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
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from snlp.tools.common import is_chinese_char, is_chinese_punc
from snlp.tools.log import logger


data_name = "loan"
basename = "/home/speech/data/style_data"
files = ["style_data.csv"]

to_file = "clean_buy0331.json"

# date = "0225"
csv = True
data_type = "pair"
# from_file = "business_dialogue.txt"
# to_file = "business_single_turn.json"
# from_file = f"{data_name}_dialogue_{date}.txt"
# to_file = f"{data_name}_single_turn_{date}.json"
# app_id= f"{data_name}_dialogue"
app_id = "clean_buy"

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
           "app_id": app_id
           }

def get_uttr_resps_oneline_single(files):
    uttrs = []
    resps = []
    flag = 0
    for file in files:
        logger.info(f"正在处理文件 {file}")
        with open(Path(basename) / file, 'r', encoding="utf8") as f:
            for i, line in tqdm(enumerate(f)):
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
    return uttrs, resps

def get_uttr_resp_oneline_pair(files):
    uttrs, resps = [], []
    for file in files:
        logger.info(f"正在处理文件 {file}")
        with open(Path(basename) / file, 'r', encoding="utf8") as f:
            for line in tqdm(f):
                line = line.strip()
                if line:
                    line_list = line.split("\t")
                    uttrs.append(line_list[0].strip().replace(" ", ""))
                    resps.append(line_list[-1].strip().replace(" ", ""))
    return uttrs, resps

def get_uttr_resp_csv(files, ucol="query", acol="answer"):
    data = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(Path(basename) / file)
        data = pd.concat([data, tmp], axis=0, sort=False, ignore_index=True)

    data = data[[ucol, acol]]
    data.dropna(axis=0, inplace=True)
    data[ucol] = data[ucol].apply(lambda s: s.strip().replace(" ", ""))
    data[acol] = data[acol].apply(lambda s: s.strip().replace(" ", ""))

    uttrs = data[ucol].values.tolist()
    resps = data[acol].values.tolist()

    return uttrs, resps


def main():
    if csv:
        uttrs, resps = get_uttr_resp_csv(files)
    else:
        if data_type == "single":
            uttrs, resps = get_uttr_resps_oneline_single(files)
        else:
            uttrs, resps = get_uttr_resp_oneline_pair(files)

    logger.info("处理uttrance")
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
    for i, (uttr, resp) in tqdm(enumerate(zip(uttrs, resps))):
        d = copy.deepcopy(template)
        d["id"] = i
        d["utterance"] = uttr
        d["responses"][0]["reply"] = resp
        dialogues.append(d)

    with open(Path(basename) / to_file, 'w', encoding="utf8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
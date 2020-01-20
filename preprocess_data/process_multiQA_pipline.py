#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: process_multiQA_pipline.py
@time: 2019/12/25 12:22
@description: 
'''
import json
import os
import re
import argparse
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import shutil

log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                               datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger("Process-MultiQA")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger.handlers = [console_handler]

def seed_everything(seed=2020):
    os.environ['PYTHONASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def read_and_get_QAlist_from_json(config):
    """根据标准的多轮对话json文件生成"""
    with open(Path(config.from_path) / config.from_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    seq_num = []
    question = []
    answer = []
    num = -1

    for dialogue in data:
        utterance = dialogue['utterance']
        reply = dialogue['responses'][0]['reply']

        uttrs = re.split(r"\[U=\d\]", utterance.replace(" ", ""))[1:]
        if len(uttrs) <= 1:
            num += 1

        q = "\t".join(uttrs)
        a = reply

        seq_num.append(f"D_{config.to_file_name}{num}")
        question.append(q.strip())
        answer.append(a)
    return seq_num, question, answer

def read_and_get_QAlist_from_txt(config):
    """一行一句话，每一个dialogue用换行符分隔"""
    data = []
    with open(Path(config.from_path)/ config.from_file, 'r', encoding='utf8') as f:
        dialogue = []
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                dialogue.append(line)
            else:
                data.append(dialogue)
                dialogue = []
    seq_num, question, answer = [], [], []

    for index, dialogue in enumerate(data):
        q = ""
        for i in range(0, len(dialogue), 2):
            q += dialogue[i] + "\t"
            if (i + 1) >= len(dialogue):
                continue
            a = dialogue[i + 1]
            seq_num.append(f"D_{config.to_file_name}{index}")
            question.append(q.strip())
            answer.append(a)
    return seq_num, question, answer

def process_and_save(seq_num, question, answer):
    # seq_num相同表示是同一组对话
    # questions表示是当前response的历史utterances
    # answer表示当前回复
    pos_data = pd.DataFrame({"D_num": seq_num, "utterances": question, "response": answer})
    ## 得到最后一个utterance以及之前的utterances
    pos_data['prev_uttrs'] = pos_data['utterances'].apply(lambda s: "\t".join(s.split("\t")[:-1]))
    pos_data['last'] = pos_data['utterances'].apply(lambda s: s.split("\t")[-1])
    ## 获取包含当前轮总共的轮数
    pos_data['turns'] = pos_data['utterances'].apply(lambda s: len(s.split("\t")))

    pos_data = pos_data[['D_num', 'turns', 'utterances', 'prev_uttrs', 'last', 'response']]
    # 返回处理好的数据
    return pos_data

def sample_multiQA_neg_random_turn(pos_data, start, end, responses, length, prefix):
    """
    从任意轮的回复中选择一次response作为neg回复
    :param pos_data:
    :return:
    """
    neg_data = pos_data[start:end].reset_index(drop=True).copy()
    to_file = prefix + f"_{start}_{end}" + '.csv'
    path = "./data"
    if not Path(path).exists():
        Path(path).mkdir(parents=True)
    t_file = Path(path) / to_file
    pid = os.getpid()
    logger.info(f"{mp.current_process()} 已启动...")

    for i, record in tqdm(neg_data.iterrows()):
        pos = record.response
        index = np.random.randint(0, length)
        low = max(0, index-10)
        high = min(length, index+10)
        candidates = responses[low:high]
        if pos in candidates:
            candidates.remove(pos)
        neg_index = np.random.randint(len(candidates))
        neg = candidates[neg_index]

        if (i+1) % 5000 == 0:
            logger.info(f"{pid}\t已经处理了 {i+1} 条数据...")
        record.response = neg
        neg_data.loc[i] = record
    neg_data.to_csv(t_file, index=False)

def sample_multiQA_neg_same_turn(pos_data, config):
    """
    从当前轮次的对话中选择一句（非pos的）作为neg回复
    :param pos_data:
    :return:
    """
    neg_data = pos_data.copy()
    to_file = Path(config.to_path) / f"multi_{config.to_file_name}_neg_same.csv"

    for i, (tag, group) in tqdm(enumerate(neg_data.groupby(by=['D_num'],
                                                    as_index=False,
                                                    sort=False))):
        # 将所有的回复和之前的问题合并
        candidates = group['response'].values.tolist() + group.iloc[-1].utterances.split("\t")
        for j, record in group.iterrows():
            pos = record.response
            candidates.remove(pos)
            if len(candidates) <= 0:
                neg = "嗯嗯，好吧！"
            else:
                neg = np.random.choice(candidates)

            record.response = neg
            candidates += [pos]
            group.loc[j, :] = record
        if len(group.shape) == 1:
            group = group.to_frame()
        if i == 0:
            group.to_csv(to_file, mode="w", encoding="utf8", header=True, index=False)
        else:
            group.to_csv(to_file, mode="a", encoding="utf8", header=False, index=False)
        if (i+1) % 1000 == 0:
            logger.info(f"已经处理了 {i+1} 组数据 ...")
    neg_data = pd.read_csv(to_file)
    os.remove(to_file)
    return neg_data

def gen_multiQA_train_val(p_data, n_data, config):
    """
    生成训练集和验证集数据
    :param p_data:
    :param n_data:
    :return:
    """
    p_data['label'] = 1
    n_data['label'] = 0
    # 将索引打乱
    p_all_index = p_data.index.tolist()
    n_all_index = n_data.index.tolist()

    p_rand_index = np.random.permutation(p_all_index)
    n_rand_index = np.random.permutation(n_all_index)

    # 定义训练集的长度
    p_train_len = int(len(p_rand_index) * 0.8)
    n_train_len = int(len(n_rand_index) * 0.8)

    p_train = p_data.iloc[p_rand_index[:p_train_len], :].reset_index(drop=True)
    p_valid = p_data.iloc[p_rand_index[p_train_len:], :].reset_index(drop=True)

    n_train = n_data.iloc[n_rand_index[:n_train_len], :].reset_index(drop=True)
    n_valid = n_data.iloc[n_rand_index[n_train_len:], :].reset_index(drop=True)

    multi_train = pd.concat([p_train, n_train], axis=0, sort=False, ignore_index=True)
    multi_valid = pd.concat([p_valid, n_valid], axis=0, sort=False, ignore_index=True)

    train_index = np.random.permutation(range(multi_train.shape[0]))
    valid_index = np.random.permutation(range(multi_valid.shape[0]))

    multi_train = multi_train.iloc[train_index, :]
    multi_valid = multi_valid.iloc[valid_index, :]

    multi_train.to_csv(Path(config.to_path) / f"multi_{config.to_file_name}_train.csv", index=False)
    multi_valid.to_csv(Path(config.to_path) / f"multi_{config.to_file_name}_val.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument("--from_path", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--from_file", default=None, type=str, required=True,
                        help="The input file, should be .json preprocessed or .data.format.")
    parser.add_argument("--to_path", default=None, type=str, required=True,
                        help="The save path of train and val file.")
    parser.add_argument("--to_file_name", default=None, type=str, required=True,
                        help="The final file name, like 50w, persona, or pass, don't need the full name.")
    parser.add_argument("--sample", default='random', type=str, choices=['random', 'same'],
                        help="Sample the neg in the random turn or the same turn.")
    parser.add_argument("--n_jobs", default=5, type=int,
                        help="The number of jobs to use when random sample.")
    parser.add_argument("--seed", default=2020, type=int,
                        help="The sample seed.")

    config = parser.parse_args()
    seed_everything(config.seed)

    logger.info(f"读取文件 {config.from_file}")
    # 首先读取数据
    if config.from_file.endswith(".json"):
        seq_num, question, answer = read_and_get_QAlist_from_json(config)
    else:
        seq_num, question, answer = read_and_get_QAlist_from_txt(config)

    # 简单处理得到正样本
    logger.info("对文件进行处理")
    p_data = process_and_save(seq_num, question, answer)

    logger.info(f"在 {config.sample} turn 中进行采样")
    if config.sample == "random":
        # 随机采样

        responses = p_data['response'].values.tolist()
        length = p_data.shape[0]
        prefix = f"multi_{config.to_file_name}_neg_random"
        results = []
        pool = mp.Pool()
        for p in range(config.n_jobs):
            start = int(p / config.n_jobs * length)
            end = int((p+1) / config.n_jobs * length)
            result = pool.apply_async(sample_multiQA_neg_random_turn,
                                      args=(p_data,start,end,responses,length,prefix,))
            results.append(result)
        pool.close()
        pool.join()
        if all([res.ready() for res in results]):
            logger.info("负采样完成")
            files = os.listdir("./data")
            n_data = pd.DataFrame()
            for file in files:
                tmp = pd.read_csv(Path("./data") / file)
                n_data = pd.concat([n_data, tmp], axis=0, sort=False, ignore_index=True)
            assert n_data.shape == p_data.shape
            # 删除中间生成的数据
            shutil.rmtree("./data")
    elif config.sample == "same":
        n_data = sample_multiQA_neg_same_turn(p_data, config)
    else:
        raise ValueError(f"The {config.sample} is invalid, only `random` and `same` allow.")

    # 得到最终的数据并保存
    logger.info(f"将数据保存为训练集和验证集，路径为 {config.to_path}")
    gen_multiQA_train_val(p_data, n_data, config)


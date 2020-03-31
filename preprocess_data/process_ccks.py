#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: process_ccks.py
@time: 2020/3/19 13:43
@description: 
'''
import json
import copy
import pandas as pd

def entity_clear(entity):
    '''
    将一些特殊字符替换
    :param entity: 一个实体名字
    :return: 替换后的实体
    '''
    pun = {'，': ',',
           '·': '•',
           '：': ':',
           '！': '!',
           }
    for key, value in pun.items():
        if key in entity:
            entity=entity.replace(key, value)
    return entity

def add_new_alias(kb_data, records):
    """
    将训练数据中不能链接到实体库的mention，统计出现的次数，添加到对应的实体别名中
    如 'bilibili': {'b站', '哔哩哔哩', '哔哩哔哩弹幕视频网'}
    kb_data: list型，每一条记录表示一个record，表示知识库中的一个实体信息，是dict型
    records：list型，保存训练集中的每一条记录，每一个元素是一个dict
    :return:
    """
    # id_entity保存id到entity的映射，entity_id保存entity到id的映射
    entity_id, id_entity = {}, {}
    # 获取知识库中，实体到id的映射和id到实体的映射
    for en_info in kb_data:
        ## 获取实体的名称
        subject = en_info['subject']
        subject_id = en_info['subject_id']
        # alias = set()
        # ## 将所有和该实体等价的实体都添加到alias中
        # for a in en_info['alias']:
        #     alias.add(a)
        #     alias.add(a.lower())
        # alias.add(subject)
        # alias.add(subject.lower())
        # alias.add(entity_clear(subject))

        ## 保存id到entity的映射
        id_entity[subject_id] = subject
        # ## 保存entity到id的映射
        # for a in alias:
        #     if a in entity_id:
        #         entity_id[a].add(subject_id)
        #     else:
        #         entity_id[a] = set()
        #         entity_id[a].add(subject_id)

    # 用来保存实体以及没有出现过的别名出现的次数
    entity_alias_num = {}
    for record in records:
        ## 获取训练集语句中所有的实体
        mention_data = record['mention_data']
        ## 目的是将训练集中的实体添加到实体库的alias中
        for men in mention_data:
            mention = men['mention']
            kb_id = men['kb_id']
            ## kb_id == "NIL"表示未识别实体
            if kb_id != "NIL":
                ## 如果训练集中对应的实体不等于实体库的实体名
                ## 则要考虑是否将该实体添加到alias中去
                en_in_kb = id_entity[kb_id]
                if en_in_kb != mention:
                    if mention not in en_in_kb:
                        if en_in_kb in entity_alias_num:
                            entity_alias_num[en_in_kb]['count'] += 1
                            if mention in entity_alias_num[en_in_kb]:
                                entity_alias_num[en_in_kb][mention] += 1
                            else:
                                entity_alias_num[en_in_kb][mention] = 1
                        else:
                            entity_alias_num[en_in_kb] = {}
                            entity_alias_num[en_in_kb]['count'] = 1
                            entity_alias_num[en_in_kb][mention] = 1

    ## 保存实体库中实体可能对应的别名
    entity_alias = {}
    for en in entity_alias_num:
        total_num = entity_alias_num[en]['count']
        if total_num > 4:
            entity_alias[en] = set()
            for alias in entity_alias_num[en]:
                if alias == 'count':
                    continue
                alias_num = entity_alias_num[en][alias]
                if alias_num > 3:
                    entity_alias[en].add(alias)
            if len(entity_alias[en]) == 0:
                entity_alias.pop(en)
    return entity_alias

def get_len(text_lens, max_len=510, min_len=30):
    """
    按照比例对文本进行截断，输入的text_lens每一个都表示实体predicate+object相连的长度
    :param text_lens: list型，entity的每一个predicate+object的长度
    :param max_len: int型，所有文本的最长长度
    :param min_len: int型，所有文本的最短长度
    :return:
    """
    new_len = len(text_lens) * [min_len]
    sum_len = sum(text_lens)
    del_len = sum_len - max_len
    del_index = []
    for i, l in enumerate(text_lens):
        if l > min_len:
            del_index.append(i)
        else:
            new_len[i] = l
    del_sum = sum([text_lens[i]-min_len for i in del_index])
    for i in del_index:
        new_len[i] = text_lens[i] - int(((text_lens[i] - min_len) / del_sum)*del_len) - 1

    return new_len

def get_text(en_data, max_len=510, min_len=30):
    """
    根据data字段数据生成entity的描述文本，将predicate和object相连，将超过长度的部分进行截断
    :param en_data: list型，表示entity对应的所有描述数据
    :param max_len: 所有predicate+object的最大长度
    :param min_len: 超过长度的predicate+object的最小长度
    :return:
    """
    texts = []
    text = ""
    for data in en_data:
        texts.append(data['predicate'] + ":" + data['object'] + "，")
    text_lens = []
    for t in texts:
        text_lens.append(len(t))
    if sum(text_lens) < max_len:
        for t in texts:
            text = text + t
    else:
        new_text_lens = get_len(text_lens, max_len=max_len, min_len=min_len)
        for t, l in zip(texts, new_text_lens):
            text = text + t[:l]
    return text[:max_len]

def del_bookname(entity_name):
    '''
    删除书名号
    :param entity_name: 实体名字
    :return: 删除后的实体名字
    '''
    if entity_name.startswith(u'《') and entity_name.endswith(u'》'):
        entity_name = entity_name[1:-1]
    return entity_name

def process_kb(kb_data, records):
    """
    对kb进行处理，得到一些数据并保存
    :param kb_data: list型，每一条表示kb中entity的信息
    :param records: list型，每一条表示train数据的信息
    :return:
    """
    ## 根据训练集的情况，决定为知识库中的哪些实体添加alias
    new_entity_alias = add_new_alias(kb_data, records)
    id_text = {}  # 表示id到description的映射
    id_entity, entity_id = {}, {}   # id到entity和entity到id的映射
    type_index = {}                 # type对应的index的映射
    type_index['NAN'] = 0
    type_i = 0
    id_type = {}                    # id到type的映射

    for record in records:
        ## 获取实体以及对应的id
        subject = record['subject']
        subject_id = record['subject_id']
        ## 添加实体的等价描述
        alias = set()
        for a in record['alias']:
            alias.add(a)
            alias.add(a.lower())
        alias.add(subject.lower())
        alias.add(entity_clear(subject))
        if subject in new_entity_alias:
            alias = alias | new_entity_alias[subject]
        alias.add(subject)
        en_data = record['data']
        en_type = record['type']
        ## 保存实体类型对应的索引
        for t in en_type:
            if t not in type_index:
                type_index[t] = type_i
                type_i += 1

        ## 保存实体到id的映射
        for n in alias:
            n = del_bookname(n)
            if n in entity_id:
                entity_id[n].append(subject_id)
            else:
                entity_id[n] = []
                entity_id[n].append(subject_id)
        ## 保存实体id对应的类型
        id_type[subject_id] = en_type
        text = get_text(en_data)
        id_text[subject_id] = text
        id_entity[subject_id] = subject_id

    pd.to_pickle(entity_id, "entity_id.pkl")  # 保存实体到id的映射
    pd.to_pickle(id_entity, "id_entity.pkl")  # 保存id到实体的映射
    pd.to_pickle(type_index, 'type_index.pkl')  # 保存类型到数值索引的映射
    pd.to_pickle(id_type, 'id_type.pkl')    # 保存id到类型的映射
    pd.to_pickle(id_text, 'id_text.pkl')    # 保存id到文本的映射

if __name__ == "__main__":
    basename = "/home/speech/data/ccks2019_el"
    train_file = "train.json"
    kb_file = "kb_data"
    valid_file = "develop.json"

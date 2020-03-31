#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: trie.py
@time: 2020/3/18 17:21
@description: 
'''
import json
import pandas as pd

class Trie(object):
    def __init__(self):
        self.root = {}
        self.end = -1

    def insert(self, word):
        """向Trie树中插入单词"""
        curNode = self.root
        for c in word:
            if c not in curNode:
                curNode[c] = {}
            curNode = curNode[c]
        curNode[self.end] = True

    def search(self, word):
        """判断单词是否在文本中存在"""
        flag, curNode = self.startWith(word)
        if flag and (self.end in curNode):
            return True
        return False

    def startWith(self, prefix):
        """查找是否存在以prefix为前缀的实体"""
        curNode = self.root
        for c in prefix:
            if c not in curNode:
                return False, curNode
            curNode = curNode[c]
        return True, curNode

    def search_entity(self, text):
        """正向最大实体搜索，从文本中查找实体"""
        text = text.lower()
        text_len = len(text)
        entitys = []
        i = 0
        while (i < text_len):
            e = i + 1
            ## 如果存在对应前缀的实体
            flag, curNode = self.startWith(text[i:e])
            if flag:
                # 获取对应的前缀
                en = text[i:e]
                ## 最长匹配
                while (e <= text_len):
                    inner_flag, inner_curNode = self.startWith(text[i:e])
                    if inner_flag:
                        en = text[i:e]
                        curNode = inner_curNode
                        e += 1
                    else:
                        break
                if self.end in curNode:
                    entitys.append((en, i))
                    i = e - 1
                else:
                    i += 1
            else:
                i += 1
        return entitys

def build_Trie(entities, min_len=2):
    trie_obj = Trie()
    for en in entities:
        if len(en) >= min_len:
            trie_obj.insert(en)
    return trie_obj

def get_split_entity(entities, records, del_en):
    """
    最大匹配会出现漏掉部分实体，只匹配长实体。为了处理这种情况，统计他们出现的次数，并根据出现次数决定这类实体怎么处理。
    1. 仅保留最长实体
    2. 分开，保留短实体
    3. 都保留
    entities: list型，表示知识库中所有的实体
    records： list型，表示所有的训练数据，每一条是一个dict
    del_en：表示要删除的实体，处理badcase
    :return:
    """
    def entity_split():
        # 根据实体构建Trie树
        trie_obj = build_Trie(entities)
        entity_count = {}
        for i, record in enumerate(records):
            ## 表示原始的文本
            text = record['text']
            ## 表示文本中所有的实体数据
            mention_data = record['mention_data']
            ## 获取原始文本中和知识库中所有匹配的实体
            match_list = trie_obj.search_entity(text)
            for men in mention_data:
                ### 对于每一个实体获取在原文本中的起始位置和结束位置（短实体）
                start = int(men['offset'])
                end = start + len(men['mention'])
                for en in match_list:
                    ### 获取实体库中匹配实体的起始位置和结束位置（长实体）
                    b = en[1]
                    e = b + len(en[0])
                    ### 如果当前实体属于实体库中实例的子串
                    if start >= b and end <= e:
                        ### 原始实体为短实体，
                        if (e - b) != (end - start):
                            ### 如果训练集中的实体长度大于1
                            if len(men['mention']) > 1:
                                ### 计算长实体出现的次数
                                if en[0] in entity_count:
                                    entity_count[en[0]]['count'] += 1
                                    ### 计算短实体出现次数
                                    if men['mention'] in entity_count[en[0]]['en']:
                                        entity_count[en[0]]['en'][men['mention']] += 1
                                    else:
                                        entity_count[en[0]]['en'][men['mention']] = 1
                                else:
                                    entity_count[en[0]] = {}
                                    entity_count[en[0]]['count'] = 1
                                    entity_count[en[0]]['en'] = {}
                                    entity_count[en[0]]['en'][men['mention']] = 1

        # 返回长实体以及对应的短实体出现的次数
        return entity_count

    def get_sub_entity(count, en_dict):
        """
        :param count: 表示长实体出现的次数
        :param en_dict: 表示短实体字典以及对应出现的次数
        :return:
        """
        sub_entity = set()
        for en in en_dict:
            ## 如果短实体出现次数超过一定阈值
            if en_dict[en] / count > 0.42 and en_dict[en] > 3:
                sub_entity.add(en)
        return sub_entity

    # 得到长实体和短实体的计数字典
    en_dict = entity_split()
    entity_full = dict()
    # 计算训练集中所有实体出现的次数
    for i, record in enumerate(records):
        mention_data = record['mention_data']
        for men in mention_data:
            ## 如果作为长实体出现过
            if men['mention'] in en_dict:
                if men['mention'] in entity_full:
                    entity_full[men['mention']] += 1
                else:
                    entity_full[men['mention']] = 1
    split_dict = dict()
    # 对于每一个长实体（存在于知识库中）
    for en in en_dict:
        if en in del_en:
            continue
        if en_dict[en]['count'] < 10:
            continue

        if en in entity_full:
            if entity_full[en] / en_dict[en]['count'] > 1.5:
                continue
            ## 如果短实体出现的次数和超过长实体的3倍，则添加短实体
            if en_dict[en]['count'] / entity_full[en] > 3:
                sub_entity = get_sub_entity(en_dict[en]['count'], en_dict[en]['en'])
                ## 向分割实体中添加子实体（不添加长实体）
                if len(sub_entity) > 0:
                    split_dict[en] = sub_entity
                continue
            sub_entity = get_sub_entity(en_dict[en]['count'], en_dict[en]['en'])
            ### 同时添加长实体
            sub_entity.add(en)
            if len(sub_entity) > 0:
                split_dict[en] = sub_entity
        else:
            ## 如果没有在训练集中出现过
            sub_entity = get_sub_entity(en_dict[en]['count'], en_dict[en]['en'])
            ### 只添加子实体
            if len(sub_entity) > 0:
                split_dict[en] = sub_entity

    ## 返回每个长实体以及对应的可能的变换实体
    return split_dict

def func_prob(x):
    return -0.066*x + 1

def entity_link_prob(entities, records):
    """
    :param kb_en: dict型，键是实体，值是列表，表示对应的id信息
    :param records: 表示训练集的记录
    :return:
    """
    entity_dict = {}
    for en in entities:
        entity_dict[en] = {'link_num': 0.1, 'match_num': 0.1}
    trie_obj = Trie()
    for en in entities:
        if (len(en) > 1):
            trie_obj.insert(en)

    for i, record in enumerate(records):
        text = record['text']
        mention_data = record['mention_data']
        ## 搜索语句中存在的实体
        match_list = trie_obj.search_entity(text)
        for men in mention_data:
            ## 计算在训练集中出现的次数
            if men['mention'] in entity_dict:
                entity_dict[men['mention']]['link_num'] += 1
        for en in match_list:
            if en[0] in entity_dict:
                entity_dict[en[0]]['match_num'] += 1

    entity_num = {}
    for en in entity_dict:
        entity_num[en] = entity_dict[en]['match_num'] / entity_dict[en]['link_num']
    en_sorted = sorted(entity_num.items(), key=lambda item: item[1], reverse=True)
    entity_prob = {}
    for item in en_sorted[:15000]:
        if entity_dict[item[0]]['match_num'] > 15:
            ## 计算匹配到链接的转化率
            entity_prob[item[0]] = func_prob(item[1])
    return entity_prob




#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: utils_ner.py
@time: 2020/3/4 17:38
@description: 
'''

import csv
import json
import copy
import torch
from albert_pytorch.model.tokenization_bert import BertTokenizer

class CNerTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append("[UNK]")
        return _tokens

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quote_char=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quote_char=quote_char)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


    @classmethod
    def _read_text(cls, input_file):
        lines = []
        with open(input_file, 'r', encoding="utf8") as f:
            words = []
            labels = []
            for line in f:
                ## 表示一句话的结束或者文本的开始
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    line = line.strip()
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        with open(input_file, 'r', encoding="utf8") as f:
            for line in f:
                line = json.loads(line.strip())
                # 获取文本以及对应的标签
                text = line['text']
                label_entities = line.get('label', None)  # 获取对应的实体ground truth
                words = list(text)
                labels = ["O"] * len(words)
                ## 打标签
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert "".join(words[start_index:end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = "S-" + key
                                else:
                                    labels[start_index] = "B-" + key
                                    labels[start_index+1:end_index+1] = ["I-"+key] * (len(sub_name)-1)
                lines.append({"words": words, "labels": labels})
        return lines

# -------------------------------------------------------------------------------------

class InputExample(object):
    def __init__(self, guid, text_a, labels):
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

# ---------------------------------------------------------------------------------

def get_entity_bios(seq, id2label):
    """
    根据序列标注的结果获取实体
    :param seq: list类型
    :param id2label: dict型，表示索引到标签值的映射
    :return: list型，每一个元素是一个元组，(entity_type, start_index, end_index)
    """
    chunks = []
    chunk = [-1, -1, -1]
    for idx, tag in enumerate(seq):
        if not isinstance(tag, str):
            # 将对应的类别id转换为标签
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk[1] = idx
            chunk[2] = idx
            chunk[0] = tag.split("-")[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = idx
            chunk[0] = tag.split("-")[1]
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = tag.split("-")[1]
            # 必须和B的类型是相同的
            if _type == chunk[0]:
                chunk[2] = idx
            if idx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq, id2label):
    """
    根据标签序列获取实体对应的索引
    :param seq:
    :param id2label:
    :return:
    """
    chunks = []
    chunk = [-1, -1, -1]
    for idx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = idx
            chunk[2] = idx
            chunk[0] = tag.split('-')[1]
            if idx == len(seq)-1:
                chunks.append(chunk)
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = idx

            if idx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq, id2label, markup="bios"):
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)

def bert_extract_item(start_logits, end_logits):
    """
    用于文本摘要或者问答提取文章片段
    :param start_logits:
    :param end_logits:
    :return:
    """
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                # [0]表示对应的标记，[1]表示起始位置，[2]表示结束位置
                S.append((s_l, i, i+j))
                break
    return S

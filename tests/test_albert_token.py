#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_albert_token.py
@time: 2020/1/7 16:18
@description: 测试albert的分词
'''
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.getcwd()))
import torch

from albert_pytorch.model.modeling_albert_bright import AlbertModel, AlbertConfig
from albert_pytorch.model import tokenization_albert, tokenization_bert

config_file = "/home/speech/models/albert_tiny_pytorch_489k/config.json"
vocab_file = "/home/speech/models/albert_tiny_pytorch_489k/vocab.txt"
model_path = "/home/speech/models/albert_tiny_pytorch_489k/"

# tokenizer = tokenization_albert.FullTokenizer(vocab_file=vocab_file)
tokenizer = tokenization_bert.BertTokenizer(vocab_file=vocab_file)
sentence = "5000+28等于多少"


input_ids = tokenizer.encode_plus(sentence)
print(input_ids)
# input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
# print(input_ids)
# length = len(input_ids)
# input_ids.extend([0, 0])
# print("input ids: ", len(input_ids))
# input_mask = [1] * length + [0, 0]
# albert_model = AlbertModel.from_pretrained(model_path)
#
# outputs = albert_model(torch.tensor(input_ids).view(1, -1), attention_mask=torch.tensor(input_mask).view(1, -1))
#
# print(outputs[0].size())
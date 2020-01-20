#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_bert_sim.py
@time: 2020/1/9 14:45
@description: 
'''
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.getcwd()))
from pathlib import Path
import torch
import time

from albert_pytorch.model.modeling_albert_bright import AlbertModel, AlbertConfig
from albert_pytorch.model.tokenization_bert import BertTokenizer

from snlp.tools.vector_similarity import cosine_similarity

albert_path = "/home/speech/models/albert_tiny_pytorch_489k"
vocab_file = "vocab.txt"
config_file = "config.json"

text_1 = "咱两谁最漂亮"
text_2 = "咱俩谁最漂亮"
text_3 = "你好"

tokenizer = BertTokenizer.from_pretrained(Path(albert_path) / vocab_file)
config = AlbertConfig.from_pretrained(Path(albert_path) / config_file)
model = AlbertModel.from_pretrained(Path(albert_path), config=config)

start = time.time()
input_ids_1 = tokenizer.encode_plus(text_1, add_special_tokens=True)['input_ids']
input_ids_2 = tokenizer.encode_plus(text_2, add_special_tokens=True)['input_ids']
input_ids_3 = tokenizer.encode_plus(text_3, add_special_tokens=True)['input_ids']
input_ids_3.extend([0,0,0,0])
# print(input_ids_1)

result = model(torch.tensor([input_ids_1, input_ids_2, input_ids_3]))[1]

result = result.detach().cpu().numpy()

print("1和2：", cosine_similarity(result[0], result[1]))
print("1和3：", cosine_similarity(result[0], result[2]))
print("共计用时：", (time.time()-start)*1000, " ms")
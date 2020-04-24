#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_crf.py
@time: 2020/4/17 16:18
@description: 
'''
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))

import time
import torch
from snlp.modules.crf import CRF

tagset_size = 6
tag_dict = {"X": 0, "[CLS]": 1, "B-LOC":2, "I-LOC": 3, "O": 4, "[SEP]": 5}

id2label = {}
for key, value in tag_dict.items():
    id2label[value] = key


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CRF(tagset_size, tag_dict, device, is_bert=True)

logits = torch.randn(2, 6, 6).to(device)
tag_list = torch.tensor([[1, 4, 4, 2, 3, 5], [1, 4, 4, 4, 5, 0]]).to(device)
lengths = torch.tensor([6, 5]).to(device)

print(model.calculate_loss(logits, tag_list, lengths=torch.tensor([6, 5]).to(device)))

start = time.time()
print(model._obtain_labels(logits, id2label, lengths))
print(f"耗时: {(time.time() - start) *1000} ms")
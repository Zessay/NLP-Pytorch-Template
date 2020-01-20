#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_model.py
@time: 2020/1/3 9:57
@description: 
'''
import sys
import os
import time
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.getcwd()))

from snlp.embedding import load_from_file
# 将数据封装成Dataset和DataLoader
from snlp.datagen.dataset.pair_dataset import PairDataset
from snlp.callbacks.padding import MultiQAPadding
from snlp.datagen.dataloader.dict_dataloader import DictDataLoader
from snlp.models.retrieval.dam import DAM
from snlp.models.retrieval.msn import MSN
from snlp.models.retrieval.imn import IMN
from snlp import tasks, metrics, losses
from snlp.preprocessors.chinese_preprocessor import CNPreprocessorForMultiQA

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# 定义任务类型
start = time.time()
cls_task = tasks.Classification(num_classes=2, losses = nn.CrossEntropyLoss())
cls_task.metrics = ['accuracy']


fixed_length_uttr = 20
fixed_length_resp = 20
fixed_length_turn = 5

name = 'dam'
batch_size = 16

print(f"测试 {name.upper()} 模型")

# 定义所有的模型
MODELS = {'dam': DAM(uttr_len=fixed_length_uttr, resp_len=fixed_length_resp, turns=fixed_length_turn),
          'msn': MSN(uttr_len=fixed_length_uttr, resp_len=fixed_length_resp, turns=fixed_length_turn),
          'imn': IMN(uttr_len=fixed_length_uttr, resp_len=fixed_length_resp, turns=fixed_length_turn)}

# 对数据进行预处理
print("正在读取数据....")
file = "../sample_data/multi_qa.csv"
data = pd.read_csv(file)

# ------------------ 01 测试预处理类 ------------------------

# 对数据进行预处理
print("对数据进行预处理....")
preprocessor = CNPreprocessorForMultiQA(stopwords=['\t'])
preprocessor = preprocessor.fit(data, columns=['utterances', 'response'])
data = preprocessor.transform(data)

data = data[['D_num', 'turns', 'utterances', 'response', 'utterances_len', 'response_len']]
data['label'] = 1

# ----------------- 02 测试词向量类 --------------------------

basename = "/home/speech/models"
# 构建词向量矩阵
print("加载词向量矩阵....")
word_embedding = load_from_file(Path(basename) / "500000-small.txt")
embedding_matrix = word_embedding.build_matrix(preprocessor.context['term_index'])

# ---------------- 03 测试Dataset, Padding以及DataLoader ----------------

print("对数据进行封装....")
dataset = PairDataset(data, num_neg=0)
padding = MultiQAPadding(fixed_length_uttr=fixed_length_uttr,
                         fixed_length_resp=fixed_length_resp,
                         fixed_length_turn=fixed_length_turn)
dataloader = DictDataLoader(dataset, batch_size=batch_size, turns=fixed_length_turn, shuffle=False, sort=False, callback=padding)

# ---------------- 04 定义模型并前向传播 -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"当前正在测试的模型 {name.upper()}")
print("定义模型和参数...")
model = MODELS[name]
params = model.get_default_params()
params['task'] = cls_task
params['embedding'] = embedding_matrix
model.params = params
model.build()
model = model.to(device)
model = model.float()

for i, batch in enumerate(dataloader):
    if i > 0:
        break
    print(batch[0]['response'][0])
    a = model(batch[0])
    print(a)

print("总计用时： ", time.time()-start, 's')
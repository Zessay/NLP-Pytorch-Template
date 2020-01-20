#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_albert_match_model.py
@time: 2020/1/7 20:21
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

from albert_pytorch.model.modeling_albert_bright import AlbertConfig

# 将数据封装成Dataset和DataLoader
from snlp.datagen.dataset.pair_dataset import PairDataset
from snlp.callbacks.padding import MultiQAPadding
from snlp.datagen.dataloader.dict_dataloader import DictDataLoader
from snlp.models.retrieval.albert_msn import AlbertMSN
from snlp.models.retrieval.albert_imn import AlbertIMN
from snlp import tasks, metrics, losses
from snlp.preprocessors.chinese_preprocessor import CNAlbertPreprocessorForMultiQA

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
# 定义任务类型
start = time.time()
cls_task = tasks.Classification(num_classes=2, losses = nn.CrossEntropyLoss())
cls_task.metrics = ['accuracy']


fixed_length_uttr = 40
fixed_length_resp = 40
fixed_length_turn = 5

name = 'imn'
batch_size = 16

# 定义一些路径信息
albert_path = "/home/speech/models/albert_tiny_pytorch_489k"
vocab_file = "vocab.txt"
config_file = "config.json"



print(f"测试 {name.upper()} 模型")

# 定义所有模型
MODELS = {
    'msn' : AlbertMSN,
    'imn': AlbertIMN
}

# 对数据进行预处理
file = "../sample_data/multi_qa.csv"
data = pd.read_csv(file)

## --------------------- 01 测试预处理类 --------------------------

print("对数据进行预处理...")
preprocessor = CNAlbertPreprocessorForMultiQA(Path(albert_path) / vocab_file,
                                              uttr_len=fixed_length_uttr,
                                              resp_len=fixed_length_resp)
data = preprocessor.transform(data)

data = data[['D_num', 'turns', 'utterances', 'response', 'utterances_len', 'response_len']]
data['label'] = 1

## --------------------- 02 封装数据 --------------------------
dataset = PairDataset(data, num_neg=0)
padding = MultiQAPadding(fixed_length_uttr=fixed_length_uttr,
                         fixed_length_resp=fixed_length_resp,
                         fixed_length_turn=fixed_length_turn)
dataloader = DictDataLoader(dataset, batch_size=batch_size, turns=fixed_length_turn, shuffle=False, sort=False, callback=padding)

## -------------------- 03 定义模型并前向传播 -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前正在测试的模型 {name.upper()}")
print("定义模型和参数...")

config = AlbertConfig.from_pretrained(Path(albert_path) / config_file)
model = MODELS[name](uttr_len=fixed_length_uttr, resp_len=fixed_length_resp,
                     turns=fixed_length_turn, config=config, model_path=albert_path)
params = model.get_default_params()
params['task'] = cls_task
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

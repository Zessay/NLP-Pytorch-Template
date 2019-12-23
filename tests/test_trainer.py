#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_trainer.py
@time: 2019/12/23 15:58
@description: 测试Trainer类
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

from snlp import tasks
from snlp.embedding import load_from_file
# 将数据封装成Dataset和DataLoader
from snlp.datagen.dataset.pair_dataset import PairDataset
from snlp.callbacks.padding import MultiQAPadding
from snlp.datagen.dataloader.dict_dataloader import DictDataLoader
from snlp.models.retrieval.dam import DAM
from snlp.preprocessors.chinese_preprocessor import CNPreprocessorForMultiQA
from snlp.optimizer import RAdam
from snlp.trainers import Trainer
from snlp.tools.log import logger

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


start = time.time()
cls_task = tasks.Classification(num_classes=2, losses = nn.CrossEntropyLoss())
cls_task.metrics = ['accuracy']

# 对数据进行预处理
file = "../sample_data/multi_qa.csv"
logger.info("读取数据 %s" % os.path.basename(file))
data = pd.read_csv(file)
data['label'] = 1

# 对数据进行预处理
## 合并训练预处理器是一种leaky
logger.info("使用Preprocessor处理数据")
preprocessor = CNPreprocessorForMultiQA(stopwords=['\t'])
preprocessor = preprocessor.fit(data, columns=['utterances', 'response'])
data = preprocessor.transform(data)
data = data[['D_num', 'turns', 'utterances', 'response', 'utterances_len', 'response_len', 'label']]
data['label'] =  data['label'].astype(int)

# 划分训练集和测试集
train = data[:90]
valid = data[90:]

# 加载预训练词向量
basename = "/home/speech/models"
# 构建词向量矩阵
logger.info("读取词向量文件")
word_embedding = load_from_file(Path(basename) / "500000-small.txt")
embedding_matrix = word_embedding.build_matrix(preprocessor.context['term_index'])

# 对训练集和验证集进行封装
logger.info("使用Dataset和DataLoader对数据进行封装")
train_dataset = PairDataset(train, num_neg=0)
valid_dataset = PairDataset(valid, num_neg=0)
padding = MultiQAPadding(fixed_length_uttr=20, fixed_length_resp=20, fixed_length_turn=4)

train_dataloader = DictDataLoader(train_dataset, batch_size=16,
                                  shuffle=False,
                                  sort=False, callback=padding)
valid_dataloader = DictDataLoader(valid_dataset, batch_size=16,
                                  shuffle=False,
                                  sort=False,
                                  callback=padding)


# 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("定义模型和参数")
model = DAM()
params = model.get_default_params()
params['task'] = cls_task
params['embedding'] = embedding_matrix
model.params = params
model.build()
model = model.float()

optimizer = RAdam(model.parameters())

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=train_dataloader,
    validloader=valid_dataloader,
    epochs=2
)

logger.info("开始训练模型")
trainer.run()
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_predictor.py
@time: 2020/1/6 14:53
@description: 
'''
import sys
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.getcwd()))

from snlp.base import load_preprocessor
from snlp.embedding import load_from_file
# 将数据封装成Dataset和DataLoader
from snlp.datagen.dataset.pair_dataset import PairDataset
from snlp.callbacks.padding import MultiQAPadding
from snlp.datagen.dataloader.dict_dataloader import DictDataLoader
from snlp.models.retrieval.dam import DAM
from snlp.models.retrieval.imn import IMN
from snlp.models.retrieval.msn import MSN
from snlp import tasks, metrics, losses
import snlp.tools.constants as constants
from snlp.preprocessors.chinese_preprocessor import CNPreprocessorForMultiQA
from snlp.trainers.predictor import Predictor

os.environ['CUDA_VISIBLE_DEVICES'] = "5, 6, 7"

# 定义任务类型
start = time.time()
cls_task = tasks.Classification(num_classes=2, losses = nn.CrossEntropyLoss())
cls_task.metrics = ['accuracy']


fixed_length_uttr = 20
fixed_length_resp = 20
fixed_length_turn = 5

# 定义所有的模型
MODELS = {'dam': DAM(uttr_len=fixed_length_uttr, resp_len=fixed_length_resp, turns=fixed_length_turn),
          'msn': MSN(uttr_len=fixed_length_uttr, resp_len=fixed_length_resp, turns=fixed_length_turn),
          'imn': IMN(uttr_len=fixed_length_uttr, resp_len=fixed_length_resp, turns=fixed_length_turn)}

name = 'msn'
precision = 764
batch_size = 64
NUMBERS = 200

# path相关参数
preprocessor_path = "/home/speech/projects/pycharm_projects/Pytorch_Templates/train_models/save"
model_dir = "/home/speech/projects/pycharm_projects/Pytorch_Templates/models"
checkpoint = f"{name}_model{precision}.pt"
data_dir = "/home/speech/data/multi_clean"
test_file = "multi_val_multi_turns.csv"
embedding_basename = "/home/speech/models"
embedding_file = "500000-small.txt"
to_csv = Path("./save") / f"test_{name}.csv"

print(f"测试 {name.upper()} 模型")

# 定义预处理数据的方法
def process_data(test_file, preprocessor=None, remove_placeholder=False):
    data = pd.read_csv(test_file)
    data = data[:NUMBERS]
    data.dropna(axis=0, subset=[constants.UTTRS, constants.RESP], inplace=True)
    # 去除人设数据的占位符
    if remove_placeholder:
        print("去除占位符的无效标记")
        placeholder = ['[', ']', '_']
        columns = [constants.UTTRS, constants.LAST, constants.RESP]
        for col in columns:
            data[col] = data[col].apply(lambda s: ''.join([c for c in s if c not in placeholder]))

    df = data.copy()
    # 加载预处理器
    if preprocessor is None:
        if os.path.exists(preprocessor_path):
            preprocessor = load_preprocessor(preprocessor_path)
        else:
            raise ValueError(f"Don't exist the preprocessor path {preprocessor_path}.")

    data = preprocessor.transform(data,
                                  uttr_col=constants.UTTRS,
                                  resp_col=constants.RESP,
                                  drop=False)

    use_cols = ['D_num', 'turns', 'utterances', 'response', 'utterances_len', 'response_len', 'label']
    data = data[use_cols]

    data[constants.LABEL] = data[constants.LABEL].astype(int)
    return df, data, preprocessor

# 获取处理好的dataloader
def get_dataloader(data):
    dataset = PairDataset(data, num_neg=0)
    padding = MultiQAPadding(fixed_length_uttr=fixed_length_uttr, fixed_length_resp=fixed_length_resp,
                             fixed_length_turn=fixed_length_turn)
    dataloader = DictDataLoader(dataset, batch_size=batch_size,
                                turns=fixed_length_turn,
                                stage='dev',
                                shuffle=False,
                                sort=False,
                                callback=padding)
    return dataloader


# 对数据进行预处理
print("正在读取数据....")
test_file = Path(data_dir) / test_file
df, data, preprocessor = process_data(test_file)

# 读取词向量文件
print("加载词向量")
word_embedding = load_from_file(Path(embedding_basename) / embedding_file)
embedding_matrix = word_embedding.build_matrix(preprocessor.context['term_index'])

dataloader = get_dataloader(data)

model = MODELS[name]
params = model.get_default_params()
params['task'] = cls_task
params['embedding'] = embedding_matrix
model.params = params
model.build()
model = model.float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictor = Predictor(model, save_dir=model_dir, checkpoint=checkpoint)

pred_labels = []
for i, batch in enumerate(dataloader):
    start = time.time()
    prob = predictor.predict(batch[0])
    l = np.argmax(prob, axis=1)
    if i == 0:
        print("一次inference的时间： ", (time.time()-start) * 1000, ' ms')
    pred_labels.extend(l.tolist())

print(df.label.values[:10])
print(len(pred_labels), pred_labels[:10])
print(Counter(pred_labels))
df[f'{name}_label'] = pred_labels

df.to_csv(to_csv, index=False)
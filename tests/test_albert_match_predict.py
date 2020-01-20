#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: test_albert_match_predict.py
@time: 2020/1/13 16:25
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
import snlp.tools.constants as constants
from snlp.preprocessors.chinese_preprocessor import CNAlbertPreprocessorForMultiQA
from snlp.trainers.predictor import Predictor
from collections import Counter
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# 定义任务类型
start = time.time()
cls_task = tasks.Classification(num_classes=2, losses=nn.CrossEntropyLoss())
cls_task.metrics = ['accuracy']

fixed_length_uttr = 40
fixed_length_resp = 40
fixed_length_turn = 5

name = "imn"
batch_size = 64

MODELS = {'imn': AlbertIMN}

albert_path = "/home/speech/models/albert_tiny_pytorch_489k"
vocab_file = "vocab.txt"
config_file = "config.json"
model_dir = "/home/speech/projects/pycharm_projects/Pytorch_Templates/models"
checkpoint = "aimn_model843.pt"

data_dir = "/home/speech/data/multi_clean"
test_file = "multi_val_multi_turns.csv"

to_csv = Path("./save") / f"test_albert_{name}.csv"

NUMBERS = 10000

print(f"测试 {name.upper()} 模型")

def process_data(test_file, preprocessor=None):
    if preprocessor is None:
        preprocessor = CNAlbertPreprocessorForMultiQA(
            Path(albert_path) / vocab_file,
            uttr_len=fixed_length_uttr,
            resp_len=fixed_length_resp
        )

    print("读取数据")
    data = pd.read_csv(test_file)[:NUMBERS]
    data.dropna(axis=0, subset=[constants.UTTRS, constants.RESP], inplace=True)

    df = data.copy()

    print("使用预处理器对数据进行处理")
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
dataloader = get_dataloader(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("定义模型和参数")
config = AlbertConfig.from_pretrained(Path(albert_path) / config_file)
model = MODELS[name](uttr_len=fixed_length_uttr, resp_len=fixed_length_resp,
                         turns=fixed_length_turn, config=config, model_path=albert_path)
params = model.get_default_params()
params['task'] = cls_task
model.params = params
model.build()
model = model.to(device)
model = model.float()

predictor = Predictor(model, save_dir=model_dir, checkpoint=checkpoint)

pred_labels = []
for i, batch in enumerate(dataloader):
    start = time.time()
    prob = predictor.predict(batch[0])
    l = np.argmax(prob, axis=1)
    if i == 0:
        print("一次inference的时间： ", (time.time()-start) * 1000, ' ms')
    pred_labels.extend(l.tolist())

pred_labels = np.array(pred_labels)
print(df.label.values[:10])
print(len(pred_labels), pred_labels[:10])
true_labels = df.label.values
acc_nums = sum((pred_labels == true_labels))
print("准确率： ", acc_nums / len(pred_labels))

print(Counter(pred_labels))
df[f'{name}_label'] = pred_labels

df.to_csv(to_csv, index=False)
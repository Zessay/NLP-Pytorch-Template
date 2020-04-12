#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: run_albert_multiqa.py
@time: 2020/1/7 22:04
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
from snlp.tools.log import logger
from snlp.preprocessors.chinese_preprocessor import CNAlbertPreprocessorForMultiQA
from snlp.tools.common import seed_everything
from snlp.optimizer import RAdam, AdamW
from snlp.schedule import get_linear_schedule_with_warmup
from snlp.trainers import Trainer
from snlp.losses import LabelSmoothLoss

os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"

# 定义任务类型
start = time.time()
cls_task = tasks.Classification(num_classes=2, losses=[nn.CrossEntropyLoss(), LabelSmoothLoss(label_smoothing=0.2)])
cls_task.metrics = ['accuracy']

fixed_length_uttr = 40
# fixed_length_uttr = 20
fixed_length_resp = 20
fixed_length_turn = 3

name = 'imn'
batch_size = 64
seed = 2020
# lr = 5e-4
# bert_lr = 2e-5
lr = 5e-5
bert_lr = 1e-5
epochs = 15
# l2_reg = 1e-3
l2_reg = 0.0
freeze_bert = False
# weight_decay = 1e-4
weight_decay = 0.0

# 目录相关
albert_path = "/home/speech/models/albert_tiny_pytorch_489k"
vocab_file = "vocab.txt"
config_file = "config.json"
# data_dir = "/home/speech/data/multi_clean"
# train_file = "multi_train_multi_turns.csv"
# valid_file = "multi_val_multi_turns.csv"
data_dir = "/home/speech/data/multi_0324"
# train_file = "multi_turn_train.csv"
# valid_file = "multi_turn_val.csv"
train_file = "multi_persona_train.csv"
valid_file = "multi_persona_val.csv"
checkpoint_file = "/home/speech/ss_models/aimn_uru_8207.pt"
# checkpoint_file = None
data_type = "uru"  # 表示utterance部分是将Utterance和Response进行拼接的，uru或者uur
add_specical_tokens = True  # 表示Bert分词时是否添加[CLS]和[SEP]两个占位符




# 定义所有模型
MODELS = {
    'msn' : AlbertMSN,
    'imn' : AlbertIMN
}

# 数据预处理
def preprocess_train_and_val(train_file, valid_file):
    logger.info("读取数据")
    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)
    train_data.dropna(axis=0, subset=[constants.UTTRS, constants.RESP], inplace=True)
    valid_data.dropna(axis=0, subset=[constants.UTTRS, constants.RESP], inplace=True)
    preprocessor = CNAlbertPreprocessorForMultiQA(Path(albert_path) / vocab_file,
                                                  uttr_len=fixed_length_uttr,
                                                  resp_len=fixed_length_resp,
                                                  add_special_tokens=add_specical_tokens)
    logger.info(f"训练集数据量为：{train_data.shape[0]} | 验证集数据量为：{valid_data.shape[0]}")

    logger.info("使用Preprocessor处理数据")
    train_data = preprocessor.transform(train_data,
                                        uttr_col=constants.UTTRS,
                                        resp_col=constants.RESP)
    valid_data = preprocessor.transform(valid_data,
                                        uttr_col=constants.UTTRS,
                                        resp_col=constants.RESP)
    use_cols = ['D_num', 'turns', 'utterances', 'response', 'utterances_len', 'response_len', 'label']
    train_data = train_data[use_cols]
    valid_data = valid_data[use_cols]

    label_type = int
    train_data[constants.LABEL] = train_data[constants.LABEL].astype(label_type)
    valid_data[constants.LABEL] = valid_data[constants.LABEL].astype(label_type)
    return train_data, valid_data


def get_dataloader(train_data, valid_data):
    train_dataset = PairDataset(train_data, num_neg=0)
    valid_dataset = PairDataset(valid_data, num_neg=0)
    padding = MultiQAPadding(fixed_length_uttr=fixed_length_uttr, fixed_length_resp=fixed_length_resp,
                             fixed_length_turn=fixed_length_turn, data_type=data_type)

    train_dataloader = DictDataLoader(train_dataset, batch_size=batch_size,
                                      turns=fixed_length_turn,
                                      stage='train',
                                      shuffle=True,
                                      sort=False, callback=padding)
    valid_dataloader = DictDataLoader(valid_dataset, batch_size=batch_size,
                                      turns=fixed_length_turn,
                                      stage='dev',
                                      shuffle=False,
                                      sort=False,
                                      callback=padding)

    for i, (x, y) in enumerate(train_dataloader):
        # 打印Utterance的形状
        logger.info(f"The shape of utternace is {x[constants.UTTRS].shape}")
        if i == 0:
            break
    return train_dataloader, valid_dataloader

if __name__ == "__main__":

    logger.info(f"----------------------- 训练 {name.upper()}: {time.ctime()} --------------------------")
    seed_everything(seed)

    logger.info(f"使用参数为 —— \n L2_REG: {l2_reg} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | "
                f"Uttr_Len: {fixed_length_uttr} | Resp_Len: {fixed_length_resp} | Turn: {fixed_length_turn} | Type: {data_type.upper()}")
    train_processed_file = train_file[0:-4] + "_albert_processed.csv"
    valid_processed_file = valid_file[0:-4] + "_albert_processed.csv"
    try:
        logger.info("加载预处理好的数据")
        from ast import literal_eval

        train_data = pd.read_csv(Path(data_dir) / train_processed_file)
        valid_data = pd.read_csv(Path(data_dir) / valid_processed_file)
        ## 将这两列转换回list类型
        logger.info(f"将{constants.UTTRS}和{constants.RESP}转换成list")
        for col in [constants.UTTRS, constants.RESP]:
            train_data[col] = train_data[col].apply(literal_eval)
            valid_data[col] = valid_data[col].apply(literal_eval)
        logger.info("加载完成")
    except:
        train_data, valid_data = preprocess_train_and_val(
            Path(data_dir) / train_file,
            Path(data_dir) / valid_file
        )
        train_data.to_csv(Path(data_dir) / train_processed_file, index=False)
        valid_data.to_csv(Path(data_dir) / valid_processed_file, index=False)
        logger.info("保存了预处理好的数据")

    logger.info(f"处理之后——训练集数据量为：{train_data.shape[0]} | 验证集数据量为：{valid_data.shape[0]}")
    train_dataloader, valid_dataloader = get_dataloader(train_data, valid_data)

    # ------------------------------------
    # 定义模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("定义模型和参数")
    config = AlbertConfig.from_pretrained(Path(albert_path) / config_file)
    model = MODELS[name](uttr_len=fixed_length_uttr, resp_len=fixed_length_resp,
                         turns=fixed_length_turn, config=config, model_path=albert_path,
                         freeze_bert=freeze_bert, data_type=data_type)
    params = model.get_default_params()
    params['task'] = cls_task
    model.params = params
    model.build()
    model = model.to(device)
    model = model.float()

    # albert_params = list(model.albert.named_parameters())
    # print(albert_params)
    ## 使用分层学习率
    no_decay = ['bias', 'LayerNorm.weight']

    if freeze_bert:
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optimizer_grouped_params = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay, 'lr': lr},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': lr}
        ]
    else:
        bert_params = list(model.albert.named_parameters())
        bert_names = [n for n, p in bert_params]
        other_params = [(n, p) for n, p in model.named_parameters() if "albert" not in n]
        other_names = [n for n, p in other_params]

        optimizer_grouped_params = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay, 'lr': bert_lr},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': bert_lr},
            {'params': [p for n, p in other_params if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay, 'lr': lr},
            {'params': [p for n, p in other_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': lr}
        ]

    num_training_steps = (train_data.shape[0] // batch_size + 1) * epochs
    # num_training_steps = 100000

    # optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer = RAdam(params=optimizer_grouped_params)
    step_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps // epochs,
                                                     num_training_steps=num_training_steps)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=train_dataloader,
        validloader=valid_dataloader,
        epochs=epochs,
        l2_reg=l2_reg,
        step_scheduler=step_scheduler,
        save_dir=f"albert_{name}_{data_type}",
        checkpoint=checkpoint_file
    )

    logger.info("开始训练模型")
    trainer.run()

    logger.info("----------------------------------------------------------------------")
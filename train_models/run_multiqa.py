#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: run_multiqa.py
@time: 2020/1/3 10:29
@description: 
'''
import sys
import os
import shutil
import time
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.getcwd()))

from snlp import tasks
from snlp.base import load_preprocessor
from snlp.embedding import load_from_file
# 将数据封装成Dataset和DataLoader
from snlp.datagen.dataset.pair_dataset import PairDataset
from snlp.callbacks.padding import MultiQAPadding
from snlp.datagen.dataloader.dict_dataloader import DictDataLoader
from snlp.models.retrieval.dam import DAM
from snlp.models.retrieval.msn import MSN
from snlp.models.retrieval.imn import IMN
from snlp.preprocessors.chinese_preprocessor import CNPreprocessorForMultiQA
from snlp.trainers import Trainer
from snlp.tools.log import logger
from snlp.losses import FocalLoss
from snlp.tools.common import seed_everything
from snlp.tools.parse import parse_optimizer
import snlp.tools.constants as constants
from snlp.schedule import get_linear_schedule_with_warmup
from snlp.optimizer import RAdam


MODELS = {'dam': DAM,
          'msn': MSN,
          'imn': IMN}

def preprocess_train_and_val(train_file, valid_file, args, remove_placeholder=False):
    logger.info("读取数据")
    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)
    train_data.dropna(axis=0, subset=[constants.UTTRS, constants.RESP], inplace=True)
    valid_data.dropna(axis=0, subset=[constants.UTTRS, constants.RESP], inplace=True)
    # 去除人设数据的占位符
    if remove_placeholder:
        logger.info("去除占位符的无效标记")
        placeholder = ['[', ']', '_']
        columns = [constants.UTTRS, constants.LAST, constants.RESP]
        for col in columns:
            train_data[col] = train_data[col].apply(lambda s: ''.join([c for c in s if c not in placeholder]))
            valid_data[col] = train_data[col].apply(lambda s: ''.join([c for c in s if c not in placeholder]))
    logger.info(f"训练集数据量为：{train_data.shape[0]} | 验证集数据量为：{valid_data.shape[0]}")
    load_fail = True
    # -----------------------------------

    logger.info("使用Preprocessor处理数据")
    # 是否加载之前训练好的预处理工具
    if args.is_load_preprocessor:
        try:
            preprocessor = load_preprocessor(args.load_preprocessor_path)
            logger.info(f"成功从 {args.load_preprocessor_path} 加载了预处理器")
            load_fail = False
        except:
            load_fail = True
    if load_fail:
        logger.info("基于现有数据训练预处理器")
        preprocessor = CNPreprocessorForMultiQA()
        preprocessor = preprocessor.fit(train_data, columns=[constants.LAST, constants.RESP])
        preprocessor.save(args.load_preprocessor_path)

    train_data = preprocessor.transform(train_data,
                                        uttr_col=constants.UTTRS,
                                        resp_col=constants.RESP)
    valid_data = preprocessor.transform(valid_data,
                                        uttr_col=constants.UTTRS,
                                        resp_col=constants.RESP)
    use_cols = ['D_num', 'turns', 'utterances', 'response', 'utterances_len', 'response_len', 'label']
    train_data = train_data[use_cols]
    valid_data = valid_data[use_cols]

    logger.info(f"处理之后——训练集数据量为：{train_data.shape[0]} | 验证集数据量为：{valid_data.shape[0]}")

    return train_data, valid_data, preprocessor

def get_dataloader(train_data, valid_data, args):
    logger.info("使用Dataset和DataLoader对数据进行封装")
    train_dataset = PairDataset(train_data, num_neg=0)
    valid_dataset = PairDataset(valid_data, num_neg=0)
    padding = MultiQAPadding(fixed_length_uttr=args.fixed_length_uttr, fixed_length_resp=args.fixed_length_resp,
                             fixed_length_turn=args.fixed_length_turn)

    train_dataloader = DictDataLoader(train_dataset, batch_size=args.batch_size,
                                      turns=args.fixed_length_turn,
                                      stage='train',
                                      shuffle=True,
                                      sort=False, callback=padding)
    valid_dataloader = DictDataLoader(valid_dataset, batch_size=args.batch_size,
                                      turns=args.fixed_length_turn,
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
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--model_name', default=None, type=str, required=True,
                        help="The model name.")
    parser.add_argument('--task_type', default='classification', type=str,
                        help="The type of task, choose from `classification` or `ranking`.")
    parser.add_argument('--fixed_length_uttr', default=20, type=int,
                        help="The max sequence length of each utterance.")
    parser.add_argument('--fixed_length_resp', default=20, type=int,
                        help="The max sequence length of the response.")
    parser.add_argument('--fixed_length_turn', default=4, type=int,
                        help="The max length of turn.")

    parser.add_argument('--l2_reg', default=0.0, type=float,
                        help="The coefficient of the L2 Regularize.")
    parser.add_argument('--epochs', default=30, type=int,
                        help="The epochs of training phase.")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="The batch size of train and valid.")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="The learning rate of the training phase.")
    parser.add_argument('--seed', default=42, type=int,
                        help="The seed of the random function.")

    parser.add_argument('--optimizer', default='radam', type=str,
                        help="The optimizer of training phase.")
    parser.add_argument('--data_dir', default="/home/speech/data/multi_clean", type=str,
                        help="The base path of the data.")
    parser.add_argument('--train_file', default="multi_train_multi_turns.csv", type=str,
                        help="The name of the train file.")
    parser.add_argument('--valid_file', default="multi_val_multi_turns.csv", type=str,
                        help="The name of the valid file.")
    parser.add_argument('--embedding_basename', default="/home/speech/models", type=str,
                        help="The base name of the embedding file.")
    parser.add_argument('--embedding_file', default="500000-small.txt", type=str,
                        help="The name of embedding file.")
    parser.add_argument('--load_preprocessor_path', default="./save", type=str,
                        help="The path of the preprocessor.")

    parser.add_argument('--is_load_preprocessor', action='store_true',
                        help="Whether to load preprocessor.")
    parser.add_argument('--remove_preprocessor', action='store_true',
                        help="Whether to remove the exist preprocessor.")

    # 一些配置参数
    parser.add_argument('--cuda_num', default='0,1,2,3', type=str,
                        help="The number of the used CUDA.")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num

    logger.info(f"----------------------- 训练 {args.model_name.upper()}: {time.ctime()} --------------------------")
    seed_everything(args.seed)

    # 记录参数
    logger.info(f"使用参数为 —— \n L2_REG: {args.l2_reg} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | "
                f"Uttr_Len: {args.fixed_length_uttr} | Resp_Len: {args.fixed_length_resp} | Turn: {args.fixed_length_turn}")

    # 是否删除之前训练好的预处理器
    ## 删除文件夹以及其中的文件
    if args.remove_preprocessor:
        try:
            shutil.rmtree(args.load_preprocessor_path)
        except Exception as e:
            logger.error("preprocessor doesn't exist.")

    # -----------------------------------------------------------

    if args.task_type == "classification":
        task = tasks.Classification(num_classes=2, losses=[nn.CrossEntropyLoss()])
        task.metrics = ['accuracy']
    elif args.task_type == "ranking":
        task = tasks.Ranking(losses=[nn.BCEWithLogitsLoss()])
        task.metrics = ['accuracy']
    else:
        raise ValueError("Only `classification` and `ranking` task allowed.")


    train_processed_file = args.train_file[0:-4] + "_processed.csv"
    valid_processed_file = args.valid_file[0:-4] + "_processed.csv"
    try:
        logger.info("加载预处理好的数据")
        from ast import literal_eval

        train_data = pd.read_csv(Path(args.data_dir) / train_processed_file)
        valid_data = pd.read_csv(Path(args.data_dir) / valid_processed_file)
        # 将这两列转换会list类型
        logger.info(f"将{constants.UTTRS}和{constants.RESP}转换成list")
        for col in [constants.UTTRS, constants.RESP]:
            train_data[col] = train_data[col].apply(literal_eval)
            valid_data[col] = valid_data[col].apply(literal_eval)
        preprocessor = load_preprocessor(args.load_preprocessor_path)
        logger.info("加载完成")
    except:
        train_data, valid_data, preprocessor = preprocess_train_and_val(Path(args.data_dir) / args.train_file,
                                                                        Path(args.data_dir) / args.valid_file,
                                                                        args)
        train_data.to_csv(Path(args.data_dir) / train_processed_file, index=False)
        valid_data.to_csv(Path(args.data_dir) / valid_processed_file, index=False)
        logger.info("保存了预处理好的数据")

    label_type = float
    if args.task_type == "classification":
        label_type = int
    train_data[constants.LABEL] = train_data[constants.LABEL].astype(label_type)
    valid_data[constants.LABEL] = valid_data[constants.LABEL].astype(label_type)

    logger.info("读取词向量文件")
    word_embeding = load_from_file(Path(args.embedding_basename) / args.embedding_file)
    embedding_matrix = word_embeding.build_matrix(preprocessor.context['term_index'])

    train_dataloader, valid_dataloader = get_dataloader(train_data, valid_data, args)

    # ------------------------------------
    # 定义模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("定义模型和参数")
    model = MODELS[args.model_name](uttr_len=args.fixed_length_uttr,
                                    resp_len=args.fixed_length_resp,
                                    turns=args.fixed_length_turn)
    params = model.get_default_params()
    params['task'] = task
    params['embedding'] = embedding_matrix
    model.params = params
    model.build()
    model = model.float()

    num_training_steps = (train_data.shape[0] // args.batch_size + 1 ) * args.epochs

    if args.optimizer == 'lookahead':
        optimizer = parse_optimizer(args.optimizer)(RAdam(model.parameters(), lr=args.lr), device=device)
    else:
        optimizer = parse_optimizer(args.optimizer)(model.parameters(), lr=args.lr)
    step_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps // args.epochs,
                                                     num_training_steps=num_training_steps)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=train_dataloader,
        validloader=valid_dataloader,
        epochs=args.epochs,
        l2_reg=args.l2_reg,
        step_scheduler=step_scheduler,
        save_dir=args.model_name,
        # checkpoint=Path(args.model_name) / "model.pt"
    )

    logger.info("开始训练模型")
    trainer.run()

    logger.info("----------------------------------------------------------------------")
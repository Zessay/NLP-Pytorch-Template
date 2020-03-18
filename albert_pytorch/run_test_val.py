#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: run_test_val.py
@time: 2020/3/9 11:13
@description:
'''
import os
import sys
import json
import re
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.getcwd()))
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from albert_pytorch.model.modeling_albert_bright import AlbertForSequenceClassification, AlbertConfig
from albert_pytorch.model import tokenization_albert
from albert_pytorch.processors.utils import InputFeatures
from albert_pytorch.processors.glue import _truncate_seq_pair
from snlp.tools.log import logger
from preprocess_data.common import replace_kv


# 模型地址
model_path = "/home/speech/models/style_model"
# 数据所在文件夹
basename = "/home/speech/data/es_data"
## 文件名
files = ["personality.data_single_0317.json"]
## 文件格式说明
# - json格式：即可以上传到es上的json格式文件，需包含utterance, responses等字段
# - 其他格式：每一行是一个pair，query和answer中间用\t分隔

to_file_prefix = "style"  # 保存文件的前缀
save_to_json = True  # 是否要保存为json格式，只有原始文件是json格式时有效


labels = ["无", "高冷女神", "热情活泼", "浪漫文艺", "温柔知性"]

MODELS = {
    "albert": (AlbertConfig, AlbertForSequenceClassification, tokenization_albert.FullTokenizer)
}

def init_model(model_path, model_type, device):
    config_class, model_class, tokenizer_class = MODELS[model_type]
    args = torch.load(Path(model_path) / "training_args.bin")
    config = config_class.from_pretrained(Path(model_path) / "config.json")
    tokenizer = tokenizer_class(Path(model_path) / "vocab.txt", do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(model_path, config=config)
    model.to(device)

    return model, tokenizer, config

def _convert_examples_to_features(text_lefts, text_rights=None, tokenizer=None,
                                  max_length=64, pad_token=0, pad_token_segment_id=0,
                                  mask_padding_with_zero=True):
    features = []
    for i in range(len(text_lefts)):
        tokens_a = tokenizer.tokenize(text_lefts[i])
        if text_rights:
            tokens_b = tokenizer.tokenize(text_rights[i])
            _truncate_seq_pair(tokens_a, tokens_b, max_length-3)
        else:
            tokens_a = tokens_a[:max_length-2]
        tokens, token_type_ids = [], []
        tokens.append("[CLS]")
        token_type_ids.append(0)

        # 添加text_a
        tokens += tokens_a
        token_type_ids += len(tokens_a) * [0]
        tokens.append("[SEP]")
        token_type_ids.append(0)

        # 添加text_b
        if text_rights:
            tokens += tokens_b
            token_type_ids += len(tokens_b) * [1]
            tokens.append("[SEP]")
            token_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 计算mask
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        padding_len = max_length - input_len
        # 对输入和mask进行padding
        input_ids = input_ids + ([pad_token] * padding_len)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_len)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_len)

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          input_len=input_len,
                          label=None)
        )

    return features

def load_dataset(text_lefts, text_rights=None,
                 tokenizer=None, max_length=64,
                 pad_token=0, pad_token_segment_id=0,
                 mask_padding_with_zero=True):
    if isinstance(text_lefts, str):
        text_lefts = [text_lefts] * len(text_rights)
    elif isinstance(text_lefts, list):
        text_lefts = text_lefts
    else:
        raise TypeError("Only `str` ir `list` type recognized.")
    features = _convert_examples_to_features(text_lefts, text_rights,
                                             tokenizer=tokenizer,
                                             max_length=max_length,
                                             pad_token=pad_token,
                                             pad_token_segment_id=pad_token_segment_id,
                                             mask_padding_with_zero=mask_padding_with_zero)
    #将Tensor进行转换并构建dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids  for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    return dataset

def predict(dataset, model, device, batch_size=64):
    batch_size = len(dataset) if len(dataset) < batch_size else batch_size
    pred_dataloader = DataLoader(dataset, batch_size=batch_size)
    preds = None
    for step, batch in enumerate(pred_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            outputs = model(**inputs)
            logits = F.softmax(outputs[0], dim=-1)
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    if 'cuda' in device.type:
        torch.cuda.empty_cache()
    # preds = preds[:, 1]
    preds = np.argmax(preds, axis=-1)
    return preds

def albert_predict(text_a, text_b, tokenizer, model, device):
    # 对数据进行封装
    dataset = load_dataset(text_a, text_b, tokenizer=tokenizer)
    probs = predict(dataset, model=model, device=device)
    return probs

def save_json(basename, json_files, tokenizer, model, device):
    for file in json_files:
        logger.info(f"正在处理文件 {file} ...")
        with open(Path(basename) / file, 'r', encoding="utf8") as f:
            records = json.load(f)
        for i, record in tqdm(enumerate(records)):
            uttr = record['utterance']
            uttr_list = re.split(r"\[U=\d+\]", uttr)
            uttr = uttr_list[-1]
            uttr = replace_kv(uttr).replace(" ", "")
            responses = record['responses']
            for j, reply in enumerate(responses):
                ans = reply['reply']
                ans = replace_kv(ans)
                pred = albert_predict(text_a=[uttr], text_b=[ans], tokenizer=tokenizer, model=model, device=device)
                lan_style = labels[pred[0]]
                ## 保存语言风格
                reply['language_style'] = lan_style
                responses[j] = reply
            records[i]['responses'] = responses

        save_file = to_file_prefix + "_" + file
        logger.info(f"处理完成并保存在 {save_file}")
        with open(save_file, 'w', encoding='utf8') as to_f:
            json.dump(records, to_f, ensure_ascii=False, indent=4)


def get_data_json(basename, json_files):
    questions = []
    answers = []
    for file in json_files:
        with open(Path(basename) / file, 'r', encoding="utf8") as f:
            records = json.load(f)
        for record in records:
            uttr = record['utterance']
            uttr_list = re.split(r"\[U=\d+\]", uttr)
            uttr = uttr_list[-1]
            uttr = replace_kv(uttr).replace(" ", "")
            responses = record['responses']
            for reply in responses:
                ans = reply['reply']
                ans = replace_kv(ans)
                questions.append(uttr)
                answers.append(ans)
    assert len(questions) == len(answers), "QA长度不匹配"
    return questions, answers

def get_data_txt(basename, files):
    questions = []
    answers = []
    for file in files:
        with open(Path(basename) / file, 'r', encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    line_list= line.split("\t")
                    q, a = line_list[0].strip(), line_list[-1].strip()
                    q, a = replace_kv(q.replace(" ", "")), replace_kv(a.replace(" ", ""))
                    questions.append(q)
                    answers.append(a)
    assert len(questions) == len(answers), "QA长度不匹配"
    return questions, answers


def process_and_save_json(model, tokenizer, device):
    save_json(basename, files, tokenizer, model=model, device=device)

def process_and_save_txt(model, tokenizer, device):
    logger.info("准备数据")
    if files[0].endswith("json"):
        text_lefts, text_rights = get_data_json(basename, files)
    else:
        text_lefts, text_rights = get_data_txt(basename, files)
    logger.info("数据预处理完成")
    logger.info("开始预测...")
    preds = albert_predict(text_lefts, text_b=text_rights, tokenizer=tokenizer, model=model, device=device)
    #print(Counter(preds))
    logger.info("保存文件")
    with open(f"{to_file_prefix}_result.txt", 'w', encoding="utf8") as f:
        for q, a, l in zip(*(text_lefts, text_rights, preds)):
            f.write(labels[l]+"\t")
            f.write(q+"\t"+a)
            f.write("\n")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("初始化模型")
    model, tokenizer, config = init_model(model_path, model_type="albert", device=device)
    if files[0].endswith("json") and save_to_json:
        process_and_save_json(model, tokenizer, device)
    else:
        process_and_save_txt(model, tokenizer, device)
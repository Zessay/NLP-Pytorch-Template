# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-05
import typing
import torch
import numpy as np
from pathlib import Path

from albert_pytorch.model.modeling_albert_bright import AlbertForSequenceClassificationLS
from albert_pytorch.model import tokenization_albert
from albert_pytorch.processors.glue import _truncate_seq_pair


class SingleQAPredictor(object):
    def __init__(self,
                 model_path: str,
                 vocab_file: str = "vocab.txt",
                 start_token: str = "[CLS]",
                 sep_token: str = "[SEP]",
                 max_seq_length: int = 64,
                 do_lower_case: bool = True):
        super().__init__()
        # 初始化分词器和模型
        self.tokenizer = tokenization_albert.FullTokenizer(vocab_file=Path(model_path) / vocab_file,
                                                           do_lower_case=do_lower_case)

        self.model = AlbertForSequenceClassificationLS.from_pretrained(model_path)
        self.model.eval()
        self.max_seq_length = max_seq_length
        self.start_token = start_token
        self.sep_token = sep_token

    def get_processed_input(self, utterance: str, responses: list):
        # 获取一一匹配的样本
        utterances = [utterance] * len(responses)
        # 用来保存当前输入的最大长度
        cur_max_length = 0
        all_input_ids, all_attention_mask, all_token_type_ids = [], [], []
        for query, answer in zip(utterances, responses):
            tokens_query = self.tokenizer.tokenize(query)
            tokens_answer = self.tokenizer.tokenize(answer)

            # 根据最大长度进行截断
            # -3表示一个[CLS]和两个[SEP]
            _truncate_seq_pair(tokens_query, tokens_answer, self.max_seq_length - 3)

            tokens = []
            token_type_ids = []
            # 添加cls标志
            tokens.append(self.start_token)
            token_type_ids.append(0)
            # 添加query
            for token in tokens_query:
                tokens.append(token)
                token_type_ids.append(0)
            # 添加sep标志
            tokens.append(self.sep_token)
            token_type_ids.append(0)
            # 添加answer
            for token in tokens_answer:
                tokens.append(token)
                token_type_ids.append(1)
            tokens.append(self.sep_token)
            token_type_ids.append(1)

            # 将输入的token转化成id
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # 添加mask向量
            cur_length = len(input_ids)
            attention_mask = [1] * cur_length
            # 保存最大长度
            if cur_length > cur_max_length:
                cur_max_length = cur_length

            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)
            all_attention_mask.append(attention_mask)

        # 根据当前最大长度进行padding
        for i, (input_ids, token_type_ids, attention_mask) in enumerate(
                zip(all_input_ids, all_token_type_ids, all_attention_mask)):
            while len(input_ids) < cur_max_length:
                input_ids.append(0)
                token_type_ids.append(0)
                attention_mask.append(0)

            all_input_ids[i] = input_ids
            all_token_type_ids[i] = token_type_ids
            all_attention_mask[i] = attention_mask

        model_input = {"input_ids": torch.tensor(all_input_ids, dtype=torch.long),
                       "token_type_ids": torch.tensor(all_token_type_ids, dtype=torch.long),
                       "attention_mask": torch.tensor(all_attention_mask, dtype=torch.long)}

        return model_input

    def predict(self, utterance: str, responses: typing.List[str]) -> np.ndarray:
        """
        用于预测结果的方法
        :param utterance: str型，表示当前轮的query
        :param responses: list型，表示候选的response
        :return:
        """
        # 获取模型输入
        model_input = self.get_processed_input(utterance, responses)
        # logits, (hidden_states), (attentions)
        with torch.no_grad():
            outputs = self.model(**model_input)
        logits = outputs[0]
        # 属于正样本的概率
        predictions = torch.softmax(logits, dim=-1)[:, 1]
        predictions = predictions.detach().cpu().numpy()

        return predictions


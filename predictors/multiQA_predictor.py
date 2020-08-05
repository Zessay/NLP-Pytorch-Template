# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-04

import typing
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np

from albert_pytorch.model.modeling_albert_bright import AlbertConfig
from snlp import tasks
from snlp.datagen.dataset.pair_dataset import PairDataset
from snlp.callbacks.padding import MultiQAPadding
from snlp.datagen.dataloader.dict_dataloader import DictDataLoader
from snlp.models.retrieval.albert_imn import AlbertIMN
from snlp.preprocessors.chinese_preprocessor import CNAlbertPreprocessorForMultiQA
from snlp.trainers.predictor import Predictor
import snlp.tools.constants as constants

class MultiQAPredictor(object):
    def __init__(self,
                 model_path: str,
                 model_file: str,
                 vocab_file: str = "vocab.txt",
                 config_file: str = "config.json",
                 device: typing.Union[torch.device, int, str] = "cpu",
                 turns: int = 3, uttr_len: int = 20, resp_len: int = 20,
                 data_type: str = "uru", add_special_tokens: bool = True):
        """
        :param model_path: str型，模型的路径，需要包含训练好的模型，预训练模型pytorch_model.bin以及对应的词表和config文件
        :param model_file: str型，训练好的模型的文件名
        :param vocab_file: str型，词表的文件名
        :param config_file: str型，预训练模型的配置文件
        """
        super().__init__()
        self.preprocessor = CNAlbertPreprocessorForMultiQA(Path(model_path) / vocab_file,
                                                      uttr_len=uttr_len,
                                                      resp_len=resp_len,
                                                      add_special_tokens=add_special_tokens)

        # 初始化模型
        config = AlbertConfig.from_pretrained(Path(model_path) / config_file)
        model = AlbertIMN(uttr_len, resp_len, turns, config, model_path,
                          data_type=data_type)
        cls_task = tasks.Classification(num_classes=2, losses=[nn.CrossEntropyLoss()])
        cls_task.metrics = ['accuracy']
        params = model.get_default_params()
        params['task'] = cls_task
        model.params = params
        model.build()
        model = model.float()

        # 加载模型
        self.predictor = Predictor(model, save_dir=model_path, checkpoint=model_file, device=device)

        self.device = self.predictor._device
        self.turns = turns
        self.uttr_len = uttr_len
        self.resp_len = resp_len
        self.data_type = data_type

    def get_dataloader(self, utterance: str, responses: list):
        data = pd.DataFrame()
        turns = len(utterance.split("\t"))
        data[constants.RESP] = responses
        data[constants.UTTRS] = utterance
        data[constants.TURNS] = turns
        data[constants.LABEL] = 0
        data = self.preprocessor.transform(data, drop=False)

        dataset = PairDataset(data, num_neg=0)
        padding = MultiQAPadding(self.uttr_len, self.resp_len,
                                 self.turns, data_type=self.data_type)
        dataloader = DictDataLoader(dataset, batch_size=len(dataset),
                                    turns=self.turns, stage="test",
                                    device=self.device,
                                    shuffle=False, sort=False,
                                    callback=padding)
        return dataloader

    def predict(self, utterance: str, responses: typing.List[str]) -> np.ndarray:
        """
        用于预测结果的方法
        :param utterance: str型，表示包含当前轮的context，中间用\t分割（不超过3轮）
        :param responses: list型，表示候选的response
        :return:
        """
        dataloader = self.get_dataloader(utterance, responses)
        predictions = np.array([])
        for batch in dataloader:
            pred = self.predictor.predict(batch[0], softmax=True)
            predictions = np.append(predictions, pred[:, 1])
        return predictions





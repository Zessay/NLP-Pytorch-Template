#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: predictor.py
@time: 2019/12/4 9:42
@description: 根据checkpoint进行还原并预测
'''
import typing
import torch
from typing import Any
from pathlib import Path

from snlp.base.base_model import BaseModel
from snlp.tools.common import parse_device


class Predictor:
    def __init__(self,
                 model: BaseModel,
                 save_dir: typing.Union[str, Path]=None,
                 checkpoint: typing.Union[str, Path]=None,
                 device: typing.Union[torch.device, int, str]=None):
        self.model = model
        self._parse_device(device)
        self._load_model(save_dir, checkpoint)


    def _parse_device(self, device: typing.Union[torch.device, int, str]):
        self._device = parse_device(device)

    def _load_model(self,
                   save_dir: typing.Union[str, Path],
                   checkpoint: typing.Union[str, Path]):
        if not Path(save_dir).exists():
            raise FileNotFoundError(f"{save_dir} don't exists.")

        if checkpoint:
            file_path = Path(save_dir).joinpath(checkpoint)
            if file_path.exists():
                self.restore_model(file_path)

    def restore_model(self, checkpoint: typing.Union[str, Path]):
        state = torch.load(checkpoint, map_location=self._device)
        self.model.load_state_dict(state)
        self.model.to(self._device)

    def predict(self, inputs: typing.Dict[str, Any], softmax: bool=True):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs).detach().cpu()
        if softmax:
            outputs = torch.softmax(outputs, dim=-1)
        outputs = outputs.numpy()
        return outputs

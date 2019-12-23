#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: dssm.py
@time: 2019/12/3 17:12
@description: DSSM模型，适用于文本匹配，Adhoc检索
'''

import torch.nn.functional as F

from snlp.params.param_table import ParamTable
from snlp.params.param import Param
from snlp.base.base_model import BaseModel

class DSSM(BaseModel):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_default_params(cls) -> ParamTable:
        params = super().get_default_params(with_multi_layer_perceptron=True)
        params.add(Param(name='vocab_size', value=419,
                         desc="Size of vocabulary."))
        return params

    @classmethod
    def get_default_padding_callback(cls):
        return None

    def build(self):
        """DSSM use Siamese architecture"""
        self.mlp_left = self._make_perceptron_layer(
            self._params['vocab_size']
        )
        self.mlp_right = self._make_perceptron_layer(
            self._params['vocab_size']
        )
        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        input_left, input_right = inputs['ngram_left'], inputs['ngram_right']
        input_left = self.mlp_left(input_left)
        input_right = self.mlp_right(input_right)

        x = F.cosine_similarity(input_left, input_right)
        out = self.out(x.unsqueeze(dim=1))
        return out
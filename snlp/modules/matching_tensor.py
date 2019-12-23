#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: matching_tensor.py
@time: 2019/12/2 11:00
@description: 计算两个张量基于匹配矩阵的匹配结果
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingTensor(nn.Module):
    """
    Module that captures the basic interactions between two tensors.

    :param matching_dim: int型，匹配矩阵的维度，一般就是词向量的维度
    :param channels: int型，结果的通道数
    :param normalize: bool型，是否对输入的词向量进行归一化
    :param init_diag: bool型，是否初始化匹配矩阵的对角值
    """
    def __init__(self,
                 matching_dim: int,
                 channels: int=4,
                 normalize: bool=True,
                 init_diag: bool=True):
        super().__init__()
        self._matching_dim = matching_dim
        self._channels = channels
        self._normalize = normalize
        self._init_diag = init_diag

        self.interaction_matrix = torch.empty(
            self._channels, self._matching_dim, self._matching_dim
        )

        if self._init_diag:
            self.interaction_matrix.uniform_(-0.05, 0.05)
            for channel_index in range(self._channels):
                self.interaction_matrix[channel_index].fill_diagonal_(0.1)
            self.interaction_matrix = nn.Parameter(self.interaction_matrix)
        else:
            self.interaction_matrix = nn.Parameter(self.interaction_matrix.uniform_())

    def forward(self, x, y):
        if self._normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        # output = [b, c, l, r]
        output = torch.einsum(
            'bld,cde,bre->bclr',
            x, self.interaction_matrix, y
        )
        return output
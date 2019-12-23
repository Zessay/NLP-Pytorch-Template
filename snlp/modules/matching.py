#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: matching.py
@time: 2019/12/2 11:17
@description: 计算两个张量的匹配结果
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Matching(nn.Module):
    """
    Module that computes a matching matrix between samples in two tensors.
    """
    def __init__(self,
                 normlize: bool=False,
                 matching_type: str='dot'):
        super().__init__()
        self._normalize = normlize
        self._validate_matching_type(matching_type)
        self._matching_type = matching_type

    @classmethod
    def _validate_matching_type(cls, matching_type):
        valid_matching_type = ['dot', 'exact', 'mul', 'plus', 'minus', 'concat']
        if matching_type not in valid_matching_type:
            raise ValueError(f"{matching_type} is not a valid matching type, "
                             f"{valid_matching_type} expected.")


    def forward(self, x, y):
        """
        对输入计算交互结果
        :param x: shape, [B, L, D]
        :param y: shape, [B, R, D]
        :return:
        """
        length_left = x.shape[1]
        length_right = y.shape[1]
        if self._matching_type == 'dot':
            if self._normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)
            return torch.einsum('bld,brd->blr', x, y)
        elif self._matching_type == 'exact':
            ## 这里的输入是[B, L]和[B, R]
            ## 表示L中每个位置单词在R中出现的次数，以及R中每个位置的单词在L中出现的次数
            x = x.unsqueeze(dim=2).repeat(1, 1, length_right)
            y = y.unsqueeze(dim=1).repeat(1, length_left, 1)
            matching_matrix = (x == y)
            x = torch.sum(matching_matrix, dim=2, dtype=torch.float)
            y = torch.sum(matching_matrix, dim=1, dtype=torch.float)
            return x, y
        else:
            x = x.unsqueeze(dim=2).repeat(1, 1, length_right, 1)
            y = y.unsqueeze(dim=1).repeat(1, length_left, 1, 1)
            if self._matching_type == 'mul':
                return x * y
            elif self._matching_type == 'plus':
                return x + y
            elif self._matching_type == 'minus':
                return x - y
            elif self._matching_type == 'concat':
                return torch.cat((x, y), dim=3)




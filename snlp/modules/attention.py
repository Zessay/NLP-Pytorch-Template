#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: attention.py
@time: 2019/12/3 9:33
@description: 注意力机制
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention Module.
    """
    def __init__(self, input_size: int=100):
        super().__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, x, x_mask):
        """
        前向传播
        :param x: [B, L, D]
        :param x_mask: [B, L]，需要mask的位置为1
        :return:
        """
        x = self.linear(x).squeeze(dim=-1)
        # 对x_mask中的值为1的位置进行mask
        x = x.masked_fill(x_mask, -float('inf'))
        score = F.softmax(x, dim=-1)
        return score



class BidirectionalAttention(nn.Module):
    """Computing the soft attention between two sequence."""
    def __init__(self):
        super().__init__()


    def forward(self, v1, v2, v1_mask, v2_mask):
        # [B, L, D] -> [B, L, L]
        similarity_matrix = v1.bmm(v2.transpose(2, 1).contiguous())

        ## v1对v2的attention以及v2对v1的attention，输出都是[B, L, L]
        v1_v2_attn = F.softmax(
            similarity_matrix.masked_fill(
                v2_mask.unsqueeze(1), -float('inf')), dim=2)
        v2_v1_attn = F.softmax(
            similarity_matrix.masked_fill(
                v1_mask.unsqueeze(2), -float('inf')), dim=1)

        attended_v1 = v1_v2_attn.bmm(v2)
        attended_v2 = v2_v1_attn.transpose(1, 2).bmm(v1)

        attended_v1.masked_fill_(v1_mask.unsqueeze(2), 0)
        attended_v2.masked_fill_(v2_mask.unsqueeze(2), 0)

        return attended_v1, attended_v2

class MatchModule(nn.Module):
    """
    Computing the match representation for Match LSTM.
    """
    def __init__(self, hidden_size, dropout_rate=0):
        super().__init__()
        self.v2_proj = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size*4, hidden_size*2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, v1, v2, v2_mask):
        proj_v2 = self.v2_proj(v2)
        ## 计算相似度匹配矩阵 [B, L1, L2]
        similarity_matrix = v1.bmm(proj_v2.transpose(2, 1).contiguous())
        ## 计算v1对v2经过mask的attention score，[B, L, L]
        v1_v2_attn = F.softmax(
            similarity_matrix.masked_fill(
                v2_mask.unsqueeze(1).bool(), -float('inf')), dim=2)
        ## 得到各个位置attention之后的值，[B, L, D]
        v2_wsum = v1_v2_attn.bmm(v2)
        fusion = torch.cat([v1, v2_wsum, v1-v2_wsum, v1*v2_wsum], dim=2)
        match = self.dropout(F.relu(self.proj(fusion)))
        return match
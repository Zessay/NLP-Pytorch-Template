#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: semantic_composite.py
@time: 2019/12/2 15:47
@description: 用于DIIN的语义组合
'''
import torch
import torch.nn as nn

class SemanticComposite(nn.Module):
    """
    SemanticComposite Module.

    Apply a self-attention layer and a semantic composite fuse gate to compute the
    encoding result of one tensor.
    """
    def __init__(self,
                 in_features,
                 dropout_rate: float=0.0):
        super().__init__()
        self.att_linear = nn.Linear(3 * in_features, 1, False)
        self.z_gate = nn.Linear(2 * in_features, in_features, True)
        self.r_gate = nn.Linear(2 * in_features, in_features, True)
        self.f_gate = nn.Linear(2 * in_features, in_features, True)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Forward.
        :param x: [B, L, D]
        :return:
        """
        seq_length = x.shape[1]
        x_1 = x.unsqueeze(dim=2).repeat(1, 1, seq_length, 1)
        x_2 = x.unsqueeze(dim=1).repeat(1, seq_length, 1, 1)
        x_concat = torch.cat([x_1, x_2, x_1*x_2], dim=-1)

        # Self-Attention Layer
        ## [B, L, L, 3 * in_features]
        x_concat = self.dropout(x_concat)
        ## [B, L, L]
        attn_matrix = self.att_linear(x_concat).squeeze(dim=-1)
        attn_weight = torch.softmax(attn_matrix, dim=2)
        # 对不同位置的单词进行attention
        attn = torch.bmm(attn_weight, x)

        # 语义组合
        x_attn_concat = self.dropout(torch.cat([x, attn], dim=-1))
        z = torch.tanh(self.z_gate(x_attn_concat))
        r = torch.sigmoid(self.r_gate(x_attn_concat))
        f = torch.sigmoid(self.f_gate(x_attn_concat))
        encoding = r * x + f * z

        return encoding
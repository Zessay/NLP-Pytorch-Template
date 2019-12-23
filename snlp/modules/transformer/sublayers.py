#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: sublayers.py
@time: 2019/12/17 16:32
@description: 定义Transformer中的一些子模块
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """
    定义基于点积的Attention类.
    :param temperature: 表示向量的维度，即v的向量维度
    :param attn_dropout:
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q: 表示query矩阵，维度为[B, N, Lq, Dk]，其中N表示head的数量
        :param k: key矩阵，维度同上
        :param v: value矩阵，维度同上
        :param mask: mask矩阵，实际上是对key的mask，需要mask的为0，维度为[B, 1, 1, Lk]
        :return:
        """
        # [B, N, Lq, Lq]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        # output的维度为[B, N, Lq, Dk]
        output = torch.matmul(attn, v)

        # [B, N, Lq, Dk]和[B, N, Lq, Lq]
        return output, attn

# ------------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    '''
    Pre-LayerNorm的多头注意力.
    :param n_head: int型，表示head的数量
    :param d_model: 表示embedding的维度
    :param d_k: 表示每一个head中的k的维度，q的维度必须和k的维度相同
    :param d_v: 表示每一个head中的v的维度
    :param dropout: float型
    '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        :param q: 维度为 [B, lq, d_model]
        :param k: 维度为 [B, Lk, d_model]
        :param v: 维度为 [B, Lv, d_model]
        :param mask: 需要mask的位置为0，维度为 [B, 1, Lk]或者[B, Lq, Lk]
        :return:
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # [B, Lq, N, d_k]
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # [B, N, Lq, d_k]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # [B, 1, 1, Lk]
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        # [B, N, Lq, d_k] 和 [B, N, Lq, Lk]
        q, attn = self.attention(q, k, v, mask=mask)

        # [B, Lq, d_model]
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # [B, Lq, d_model]
        q = self.dropout(self.fc(q))
        q += residual
        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' Pre-LayerrNorm的FFN '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: 相当于是多头注意力的输出，维度为[B, Lq, d_model]
        :return:
        """
        residual = x
        x = self.layer_norm(x)

        # 经过两层线性层，输出为[B, Lq, d_model]
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        # [B, Lq, d_model]
        x += residual
        return x

# ------------------------------------------------------------------------------
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: mask.py
@time: 2019/12/18 16:32
@description: 
'''
import torch
import numpy as np

import snlp.tools.constants as Constants

def get_non_pad_mask(seq):
    """
    获取对pad进行mask的矩阵，将要mask的位置设置为0
    :param seq: 输入的序列，[B, L]
    :return:
    """
    assert seq.dim() == 2
    # 不等于PAD的位置为1，PAD对应的位置为0，扩展为3维
    # [B, L, 1]
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    """
    mask掉key中属于pad的部分，将要mask的位置设置为1
    :param seq_k: 维度为 [B, Lk]
    :param seq_q: 维度为 [B, Lq]
    :return:
    """
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    # [B, Lq, Lk]
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)

    return padding_mask

def get_subsequent_mask(seq):
    """
    用于解码自注意力过程，mask掉当前词语后面的词，设置mask的位置为1
    :param seq: [B, L]
    :return:
    """
    sz_b, len_s = seq.size()
    ## 上三角矩阵
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8),
                                 diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)
    return subsequent_mask

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    获取位置编码
    :param n_position: int型，表示序列的总长度
    :param d_hid: int型，表示位置向量的维度
    :param padding_idx: int型，表示padding对应的索引
    :return:
    """
    def cal_angle(position, hid_idx):
        """
        计算每个位置每一维对应的角度值
        :param position: int型，表示计算的位置
        :param hid_idx: int型，表示计算的维度
        :return:
        """
        return position / np.power(10000, 2*(hid_idx//2)/d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    # 将padding位置的编码设置为0
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: layers.py
@time: 2019/12/18 11:04
@description: 定义Transformer中的编码器层和解码器层
'''
import torch.nn as nn

from snlp.modules.transformer.sublayers import MultiHeadAttention, PositionwiseFeedForward

# -----------------------------------
class CrossAttentionLayer(nn.Module):
    """
    一层编码器层
    :param d_model: int型，词向量的维度
    :param d_inner: int型，内部隐层的维度
    :param n_head: int型，head的数量
    :param d_k: int型，每个head中向量的维度
    :param d_v: int型，每个head中value向量的维度
    :param dropout:
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, cross_attn_mask=None):
        """
        :param query, key, value: 编码器的输入，维度为[B, L, d_model]
        :param cross_attn_mask: 自注意力的输入，主要忽略key中的pad，维度为[B, 1, L]
        :return:
        """
        # [B, L, d_model]和[B, L, L]
        ## 这里已经在self-attention的时候将pad部分忽略过了，所以可以不用再乘一次non_pad_mask
        enc_output, enc_slf_attn = self.slf_attn(
            query, key, value, mask=cross_attn_mask)
        # [B, L, d_model]
        enc_output = self.pos_ffn(enc_output)
        enc_output = self.layer_norm(enc_output)

        return enc_output, enc_slf_attn

# ------------------------------------

class EncoderLayer(nn.Module):
    """
    一层编码器层
    :param d_model: int型，词向量的维度
    :param d_inner: int型，内部隐层的维度
    :param n_head: int型，head的数量
    :param d_k: int型，每个head中向量的维度
    :param d_v: int型，每个head中value向量的维度
    :param dropout:
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        """
        :param enc_input: 编码器的输入，维度为[B, L, d_model]
        :param slf_attn_mask: 自注意力的输入，主要忽略key中的pad，维度为[B, 1, L]
        :return:
        """
        # [B, L, d_model]和[B, L, L]
        ## 这里已经在self-attention的时候将pad部分忽略过了，所以可以不用再乘一次non_pad_mask
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # [B, L, d_model]
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """
    一层解码器
    :param d_model: int型，向量的维度
    :param d_inner: int型，前向传播时隐层的维度
    :param n_head: head的数量
    :param d_k: 每一个head的query和key维度
    :param d_v: 每一个head中value的维度
    :param dropout:
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        """
        :param dec_input: 表示解码器的输入，维度为[B, Ld, d_model]
        :param enc_output: 表示编码器的输出，维度为[B, Le, d_model]
        :param slf_attn_mask: 表示解码器自编码部分的attention，每个词只能看到当前位置以及当前位置之前的词，mask位置对应为0，维度为[B, Ld, Ld]
        :param dec_enc_attn_mask: 表示交互时attention部分的mask，需要mask的部分为0，维度为[B, 1, Le]
        :return:
        """
        # [B, Ld, d_model]和[B, Ld, Ld]
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        # [B, Ld, d_model]和[B, Ld, Le]
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        # [B, Ld, d_model]
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

# ------------------------------------------------------------------------------------------------------

#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: models.py
@time: 2019/12/18 14:15
@description: 定义编码器和解码器，以及Transformer框架
'''
import numpy as np
import torch
import torch.nn as nn
import snlp.tools.constants as Constants
from snlp.modules.transformer.layers import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    """
    获取对pad进行mask的矩阵，pad对应的位置为0
    :param seq: [B, L]
    :param pad_idx: pad的索引
    :return: [B, 1, L]
    """
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """
    对当前位置之后进行mask的矩阵，需要mask的部分对应的值为0
    :param seq: [B, L]
    :return: [B, L, L]
    """
    sz_b, len_s = seq.size()
    # [B, L, L]
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    """
    获取position embedding
    :param d_hid: 表示position embedding的维数
    :param n_position: 表示position的位置数
    """
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # 提前注册到缓冲区中
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        ## 由于索引为0的位置本来就是全0向量，所以不需要特殊设置
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        # 在最开始增加一维，表示batch的占位符，用于和word_embedding相加
        ## 维度为 [1, n_position, d_hid]
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        :param x: 表示word embedding，维度为[B, L, d_vec]
        :return: 返回和position embedding相加的结果 [B, L, d_vec]
        """
        ## 这里需要detach表示这个矩阵不需要反向传播
        return x + self.pos_table[:, :x.size(1)].clone().detach()

# ---------------------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    定义带有自注意力的多层编码器模块
    :param n_src_vocab: int型，词表的大小
    :param d_word_vec: int型，表示词向量的维度
    :param n_layers: int型，表示Encoder的层数
    :param n_head: int型，表示head的数量
    :param d_k: int型，表示每一个head中key的维度
    :param d_v: int型，表示每一个head中value的维度
    :param d_model: int型，表示输入序列的维度，通常和d_word_vec相同
    :param d_inner: int型，表示内部隐层的维度
    :param pad_idx: int型，表示pad的索引
    :param dropout: float型
    :param n_position: int型，表示序列最大的长度
    :param word_embedding: ndarray或者torch.tensor类型，表示传入的embedding矩阵
    :param is_word_embedding: 是否需要添加word embedding
    :param is_pos_embedding: 是否需要加上pos embedding
    :param is_layer_norm： 输出之前是否需要LayerNorm
    """
    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200,
            word_embedding=None, freeze=False,
            is_word_embedding=True, is_pos_embedding=True, is_layer_norm=True):

        super().__init__()
        if is_word_embedding:
            if word_embedding is not None:
                self.src_word_emb = nn.Embedding.from_pretrained(word_embedding, freeze=freeze)
            else:
                self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        ## 定义词向量和位置向量相加的类
        if is_pos_embedding:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        if is_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # 是否需要进行embedding
        self.is_word_embedding = is_word_embedding
        self.is_pos_embedding = is_pos_embedding
        self.is_layer_norm = is_layer_norm

    def forward(self, src_seq, src_mask, return_attns=False):
        """
        :param src_seq: 编码器的输入，维度为[B, L]
        :param src_mask: 编码器自注意力的mask，维度为[B, 1, L]或者[B, L, L]
        :param return_attns: bool型，是否返回每一层自注意力的值
        :return:
        """

        enc_slf_attn_list = []
        # 可能是[B, L]或者[B, L, d_model]
        enc_output = src_seq

        # [B, L, d_model]
        if self.is_word_embedding:
            enc_output = self.src_word_emb(enc_output)
        if self.is_pos_embedding:
            enc_output = self.position_enc(enc_output)
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            # [B, L, d_model]和[B, L, L]
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # 经过所有编码器之后经过一次LayerNorm
        # [B, L, d_model]
        if self.is_layer_norm:
            enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

# ------------------------------------------------------------------------------------

class StackedEncoder(nn.Module):
    """
    定义带有自注意力的多层编码器模块，将所有曾的编码结果全部返回
    :param n_layers: int型，表示Encoder的层数
    :param n_head: int型，表示head的数量
    :param d_k: int型，表示每一个head中key的维度
    :param d_v: int型，表示每一个head中value的维度
    :param d_model: int型，表示输入序列的维度，通常和d_word_vec相同
    :param d_inner: int型，表示内部隐层的维度
    :param pad_idx: int型，表示pad的索引
    :param dropout: float型
    :param n_position: int型，表示序列最大的长度
    :param word_embedding: ndarray或者torch.tensor类型，表示传入的embedding矩阵
    """

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=200, add_position=True):

        super().__init__()
        ## 定义词向量和位置向量相加的类
        if add_position:
            self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.add_pos = add_position


    def forward(self, src_seq, src_mask, return_attns=False):
        """
        :param src_seq: 编码器的输入，维度为[B, L, d_model]
        :param src_mask: 编码器自注意力的mask，维度为[B, 1, L]或者[B, L, L]
        :param return_attns: bool型，是否返回每一层自注意力的值
        :return:
        """
        enc_output_list = []
        enc_slf_attn_list = []

        # non_pad_mask = src_mask.squeeze(1).unsqueeze(-1)
        # [B, L, d_model]
        if self.add_pos:
            enc_output = self.dropout(self.position_enc(src_seq))
        else:
            enc_output = self.dropout(src_seq)

        for enc_layer in self.layer_stack:
            # [B, L, d_model]和[B, L, L]
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            # 所有输出经过一次LayerNorm再添加进去
            enc_output_list += [self.layer_norm(enc_output)]  # 将每一层的结果按顺序添加进去
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # [NL, B, L, d_model]，表示enc_output_list的维度，这里NL表示编码器的层数
        if return_attns:
            return enc_output_list, enc_slf_attn_list
        return enc_output_list,


# -------------------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    定义解码器层
    : 所有参数的含义和解码器一致
    """
    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1,
            word_embedding=None, freeze=False):

        super().__init__()

        if word_embedding is not None:
            self.tgt_word_emb = nn.Embedding.from_pretrained(word_embedding, freeze=freeze)
        else:
            self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        """
        :param trg_seq: 解码器的输入，维度为 [B, Ld]
        :param trg_mask: 解码器的mask，维度为 [B, Ld, Ld]
        :param enc_output: 编码器的输出，维度为 [B, Le, d_model]
        :param src_mask: 编码器的mask，维度为 [B, 1, Le]
        :param return_attns:
        :return:
        """

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # 解码器的输入 [B, Ld, d_model]
        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))

        # 逐步经过解码器中的每一层
        for dec_layer in self.layer_stack:
            # [B, Ld, d_model], [B, Ld, Ld], [B, Ld, Le]
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        # [B, Ld, d_model]
        dec_output = self.layer_norm(dec_output)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

# --------------------------------------------------------------------------------

class Transformer(nn.Module):
    """定义seq2seq的Transformer模型"""
    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            src_word_embedding=None,
            tgt_word_embedding=None,
            src_freeze=False,
            tgt_freeze=False):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout,
            word_embedding=src_word_embedding,
            freeze=src_freeze)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout,
            word_embedding=tgt_word_embedding,
            freeze=tgt_freeze)

        # weight的shape为[n_trg_word, d_model]，相当于转置
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # decoder的输入和输出共享Embedding矩阵
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            # 编码器的输入和解码器的输入共享Embedding矩阵
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        """
        :param src_seq: 维度为[B, Le]
        :param trg_seq: 维度为[B, Ld]
        :return:
        """
        # [B, 1, Le]，编码器的注意力mask
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        # [B, 1, Ld]，解码器的注意力mask
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        # [B, Le, d_model]
        enc_output, *_ = self.encoder(src_seq, src_mask)
        # [B, Ld, d_model]
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        # [B, Ld, vocab_size]，使用x_logit_scale进行放缩
        seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale

        # [B*Ld, vocab_size]
        return seq_logit.view(-1, seq_logit.size(2))

# ----------------------------------------------------------------------------------

#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: character_embedding.py
@time: 2019/12/2 9:33
@description: 字符编码
'''
import torch
import torch.nn as nn

class CharEmbedding(nn.Module):
    """Char Embedding Layer"""
    def __init__(self,
                 char_embedding_input_dim: int=100,
                 char_embedding_output_dim: int=8,
                 char_conv_filters: list=[100,100,100],
                 char_conv_kernel_size: list=[3,4,5],
                 pretrain_embeddings: torch.Tensor=None,
                 freeze: bool=True):
        super().__init__()
        self.char_embedding = nn.Embedding(
            num_embeddings=char_embedding_input_dim,
            embedding_dim=char_embedding_output_dim
        )
        if pretrain_embeddings is not None:
            self.char_embedding.from_pretrained(pretrain_embeddings,
                                                freeze=freeze)
        convs = []
        for filter_num, kernel_size in zip(char_conv_filters, char_conv_kernel_size):
            conv = nn.Conv1d(
                in_channels=char_embedding_output_dim,
                out_channels=filter_num,
                kernel_size=kernel_size
            )
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        """
        对输入进行charEmbedding
        :param x: shape [B, seq_len, word_len]
        :return:
        """
        embed_x = self.char_embedding(x)
        batch_size, seq_len, word_len, embed_dim = embed_x.shape
        # conv1d的输入是[N, C, L]，即第二维是channels，最后一维是length
        embed_x = embed_x.contiguous().view(-1, word_len, embed_dim).transpose(1, 2)

        embeds = []
        for conv in self.convs:
            embed = conv(embed_x)
            embed = torch.max(embed, dim=-1)[0]  # 对length维度进行最大池化
            embeds.append(embed)
        embed_x = torch.cat(embeds, dim=-1)  # 按照最后一个维度进行拼接
        embed_x = embed_x.view(batch_size, seq_len, -1)
        return embed_x


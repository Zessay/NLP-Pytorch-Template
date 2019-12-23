#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: rnn_dropout.py
@time: 2019/12/2 10:56
@description: 对输入的单词进行随机Drop
'''

import torch.nn as nn
import torch.nn.functional as F


class RNNDropout(nn.Dropout):
    """Dropout for RNN."""

    def forward(self, sequences_batch):
        """Masking whole hidden vector for tokens."""
        # B: batch size
        # L: sequence length
        # D: hidden size

        # sequence_batch: BxLxD
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = F.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch

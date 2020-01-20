#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: stacked_brnn.py
@time: 2019/12/2 21:47
@description: 多层双向循环神经网络
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedBRNN(nn.Module):
    """
    Stacked Bi-directional RNNs.

    和Pytorch标准的BiLSTM不同，这个类可以选择拼接所有隐层的输出。
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout_rate=0,
                 dropout_output=False,
                 rnn_type=nn.LSTM,
                 stack_layers=False):
        super().__init__()
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.stack_layers = stack_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2*hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask=None):
        """Encode either padded or non-padded sequences."""
        # 输入x的维度是 [B, L, D], x_mask的维度为[B, L, 1]
        output = self._forward_unpadded(x, x_mask)
        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        # [B, L, D] -> [L, B, D]
        x = x.transpose(0, 1)
        x_mask = x_mask.transpose(0, 1)

        # 对所有层进行编码
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # 对隐层进行dropout
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            rnn_output = self.rnns[i](rnn_input)[0]
            rnn_output = rnn_output * x_mask
            outputs.append(rnn_output)

        if self.stack_layers:
            output = torch.stack(outputs[1:], dim=2)
        else:
            output = outputs[-1]
        # [B, L, 2*hid]或者[B, L, NL, 2*hid]
        output = output.transpose(0, 1)

        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

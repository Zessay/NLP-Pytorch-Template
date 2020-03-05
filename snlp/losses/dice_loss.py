#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: dice_loss.py
@time: 2020/3/4 11:39
@description: 
'''
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Useful in dealing with unbalanced data.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        """
        :param input: [N, C]
        :param target: [N, ]
        :return:
        """
        prob = torch.softmax(input, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        dsc_i = 1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss
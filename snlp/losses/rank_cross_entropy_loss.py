#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: rank_cross_entropy_loss.py
@time: 2019/11/27 19:50
@description: 用于排序任务的交叉熵损失函数，即listwise
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class RankCrossEntropyLoss(nn.Module):
    """ Creates a criterion that measures rank cross entropy loss. """
    __constants__ = ['num_neg']
    def __init__(self, num_neg: int=1):
        super().__init__()
        self._num_neg = num_neg

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculate rank cross entropy loss.
        :param y_pred: 二维的，但是最后一层神经元只有1个，这是概率值
        :param y_true: 二维的，表示归一化之后的排序值
        :return:
        """
        ## 获取正例以及对应的正样本标签
        logits = y_pred[::(self.num_neg + 1), :]
        labels = y_true[::(self.num_neg + 1), :]
        ## 对于每一个负样本
        for neg_idx in range(self.num_neg):
            ## 获取负样本的概率以及对应的标签值
            neg_logits = y_pred[(neg_idx+1)::(self.num_neg+1), :]
            neg_labels = y_true[(neg_idx+1)::(self.num_neg+1), :]
            logits = torch.cat((logits, neg_logits), dim=-1)
            labels = torch.cat((labels, neg_labels), dim=-1)

        return -torch.mean(
            torch.sum(labels * torch.log(F.softmax(logits, dim=-1)+torch.finfo(float).eps),
                      dim=-1)
        )


    @property
    def num_neg(self):
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        self._num_neg = value
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: rank_hinge_loss.py
@time: 2019/11/27 19:31
@description: 用于pairwise的loss，用于排序任务
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class RankHingeLoss(nn.Module):
    """
    Creates a criterion that measures rank hinge loss.

    Given inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and vice-versa
    for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        loss_{x, y} = max(0, -y * (x1 - x2) + margin)
    """

    __constants__ = ['num_neg', 'margin', 'reduction']

    def __init__(self, num_neg: int=1, margin: float=1., reduction: str='mean'):
        """
        Constructor
        :param num_neg: int型，表示每个正样本对应的负例的数量
        :param margin: int型，表示正样本和负样本预测结果的期望差距
        :param reduction: 可选 'none' | 'mean' | 'sum'，分别表示不适用reduction，平均以及求和
        """
        super().__init__()
        self._num_neg = num_neg
        self._margin = margin
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor=None):
        """
        Calculate rank hinge loss
        :param y_pred: 预测结果，二维的，但是只有一个值，也就是最后一维只有一个神经元，这是概率值
        :param y_true: 真实标签值
        :return:
        """
        y_pos = y_pred[::(self.num_neg+1), :]
        y_neg = []
        for neg_idx in range(self.num_neg):
            neg = y_pred[(neg_idx+1)::(self.num_neg+1), :]
            y_neg.append(neg)
        y_neg = torch.cat(y_neg, dim=-1)
        ## 对所有的负例概率求平均
        y_neg = torch.mean(y_neg, dim=-1, keepdim=True)
        y_true = torch.ones_like(y_pos)
        return F.margin_ranking_loss(
            y_pos, y_neg, y_true,
            margin=self.margin,
            reduction=self.reduction
        )

    @property
    def num_neg(self):
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        self._num_neg = value

    @property
    def margin(self):
        return self._margin

    @margin.setter
    def margin(self, value):
        self._margin = value
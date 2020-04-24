#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: dice_loss.py
@time: 2020/3/4 11:39
@description: 论文 https://arxiv.org/pdf/1911.02855.pdf
'''
import torch
import torch.nn as nn

class DSCLoss(nn.Module):
    """
    Useful in dealing with unbalanced data.
    """
    def __init__(self, reduction="mean", ignore_index=-100):
        super(DSCLoss, self).__init__()

        self._reduction = reduction
        self._ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        计算公式为 1 - [(2*(1-p)*p*y) / (1-p)*p+y]
        :param y_pred: [N, C]
        :param y_true: [N, ]
        :return:
        """
        prob = torch.softmax(y_pred.view(-1, y_pred.size(-1)), dim=-1)
        prob = torch.gather(prob, dim=1, index=y_true.unsqueeze(1))
        loss = 1 - (2 * (1 - prob) * prob) / ((1 - prob) * prob + 1)
        # 计算有效的样本数
        valid = (y_true != self._ignore_index).unsqueeze(1)
        num_labels = valid.sum()
        loss = loss * valid

        if self._reduction == "none":
            loss = loss
        elif self._reduction == "mean":
            loss = loss.sum() / num_labels
        elif self._reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(f"{self._reduction} is not allow, only permit `none` `mean` and `sum`. ")
        return loss

# ---------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(DiceLoss, self).__init__()

        self._ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        计算公式 1 - [(2 * p * y) / (p^2 + y^2)]
        :param y_pred: [N, C]
        :param y_true: [N, ]
        :return:
        """

        prob = torch.softmax(y_pred.view(-1, y_pred.size(-1)), dim=-1)
        # 获取标签对应的概率
        prob = prob.gather(dim=1, index=y_true.unsqueeze(1))
        # 计算每一个对应的损失
        loss = 1 - (2 * prob) / (torch.pow(prob, 2) + 1)

        # 计算有效的样本数
        valid = (y_true != self._ignore_index).unsqueeze(1)
        num_labels = valid.sum()
        loss = loss * valid

        if self._reduction == "none":
            loss = loss
        elif self._reduction == "mean":
            loss = loss.sum() / num_labels
        elif self._reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(f"{self._reduction} is not allow, only permit `none` `mean` and `sum`.")
        return loss

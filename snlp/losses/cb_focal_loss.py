#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: cb_focal_loss.py
@time: 2019/12/4 16:31
@description: Class Balanced Focal Loss： https://arxiv.org/pdf/1901.05555.pdf
'''
import typing
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBFocalLoss(nn.Module):
    """Class Balanced Focal Loss"""
    def __init__(self,
                 class_num: list,
                 beta: float=0.99,
                 gamma: float=2.0,
                 reduction: str="mean"):
        """
        初始化函数
        :param class_num: list类型，表示每种类别对应的样本数，从0开始
        :param beta: 表示有效样本数占总样本数的比例，一般选择[0.999, 0.99, 0.9]
        :param gamma: 表示平衡指数
        :param reduction: 可选`mean`和`sum`
        """
        super(CBFocalLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.gamma = gamma
        self._nums = torch.zeros(len(class_num)).to(device)
        self.sub_beta = torch.zeros(len(class_num)).to(device)
        for i in range(len(class_num)):
            self._nums[i] = class_num[i]
            self.sub_beta[i] = (1 - self.beta) / (1 - math.pow(self.beta, class_num[i]))

        self._reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        计算Loss的过程
        :param y_pred: shape为[B, C]
        :param y_true: shape为[B, ]
        :return:
        """
        y_pred = y_pred.view(-1, y_pred.size(-1))
        # 计算每个类别的概率和对数概率
        y_prob = F.softmax(y_pred, dim=1)
        y_logprob = torch.log(y_prob)
        ## 获取每个位置对应的概率和对数概率
        y_prob = y_prob.gather(1, y_true.view((-1, 1)))
        y_logprob = y_logprob.gather(1, y_true.view(-1, 1))
        ## 获取每个位置对应的分母
        self.sub_beta = self.sub_beta.gather(0, y_true.view(-1))
        ## 按元素相乘
        loss = - (torch.pow((1 - y_prob), self.gamma) * y_logprob)
        loss = torch.mul(self.sub_beta, loss.t())

        if self._reduction == "mean":
            loss = loss.mean()
        elif self._reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(f"{self._reduction} is not allow, only permit `mean` and `sum`. ")
        return loss
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: circle_loss.py
@time: 2020/4/20 15:39
@description: https://github.com/TinyZeaMays/CircleLoss
预测时，随机选择每个类别的一个样例，计算测试集和当前类别相似的概率，概率最大的即为对应的类别
'''
import torch
import torch.nn as nn

def convert_label_to_similarity(y_norm: torch.Tensor, y_true: torch.Tensor):
    """
    转化为相似度
    :param y_norm: [B, n_dim]，dim表示特征的维度
    :param y_true: [B, ]，表示样例的ground truth
    :return:
    """
    # 计算输出特征的相似度矩阵
    similarity_matrix = y_norm @ y_norm.transpose(1, 0)
    # 计算相同标签的矩阵
    label_matrix = (y_true.unsqueeze(1) == y_true.unsqueeze(0))

    # 得到相同标签对应的上三角矩阵，不包含自身
    positive_matrix = label_matrix.triu(diagonal=1)
    # 得到不同标签对应的上三角矩阵
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    # 得到标签相同的相似度值
    positive_matrix = positive_matrix.view(-1)
    # 得到标签不同的相似度值
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float=0.25, gamma:float=256):
        """
        :param m: 表示松弛因子，实验中设置为0.25时效果最佳
        :param gamma: 表示放缩因子，一般为80, 128, 256, 512
        """
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, y_norm: torch.Tensor, y_true: torch.Tensor):
        """
        计算pair之后的损失
        :param y_norm: [N, n_dim]，归一化之后的特征
        :param y_true: [N, ]，表示标签的ground truth
        :return:
        """
        sp, sn = convert_label_to_similarity(y_norm, y_true)

        ap = torch.clamp_min(-sp.detach()+1+self.m, min=0.)
        an = torch.clamp_min(sn.detach()+self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

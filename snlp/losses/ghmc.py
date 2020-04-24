#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: ghmc.py
@time: 2020/4/20 11:25
@description: 论文 https://arxiv.org/pdf/1811.05181.pdf
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GHMC(nn.Module):
    def __init__(self,
                 bins: int=10,
                 reduction: str="mean",
                 ignore_index=-100):
        """
        计算改进版的FocalLoss
        :param bins: 表示将概率范围进行划分
        :param reduction: str型，表示loss最终累计的方式
        :param ignore_index: 表示忽略的index
        """
        super(GHMC, self).__init__()
        self.bins = bins
        # 表示将概率区间分段，统计不同段的样本数量
        self.histrogram = torch.arange(bins + 1).float() / bins

        self._reduction = reduction
        self._ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        损失计算
        :param y_pred: 预测输出，shape为[B, C]
        :param y_true: 实际类型，shape为[B, ]
        :return:
        """
        y_pred = y_pred.view(-1, y_pred.size(-1))
        # 计算softmax
        y_prob = F.softmax(y_pred, dim=1)
        # 获取每个类别对应的位置的概率
        y_prob = y_prob.gather(1, y_true.view(-1, 1))
        # 用来保存每个样本对应的权重
        weights = torch.zeros_like(y_prob)
        # 计算模长的梯度，每个位置对应的概率越大，表示越易分
        g = 1 - y_prob
        # 丢弃无效标签
        valid = (y_true != self._ignore_index).unsqueeze(1)
        num_labels = valid.sum()  # 计算有效的标签数
        n = 0  # 表示有限的bins的数量
        # 计算落在每个bins的数量
        for i in range(self.bins):
            ## 找到位于指定概率区间的对应的索引
            inds = (g >= self.histrogram[i]) & (g < self.histrogram[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                weights[inds] = num_labels / num_in_bin
                n += 1

        if n > 0:
            weights = (weights / n).to(dtype=y_prob.dtype, device=y_prob.device)

        # 计算loss
        y_logprob = torch.log(y_prob)
        y_logprob.masked_fill_((y_true == self._ignore_index).unsqueeze(1), 0)

        loss = -(weights * y_logprob)

        if self._reduction == "none":
            loss = loss
        elif self._reduction == "mean":
            loss = loss.sum() / num_labels
        elif self._reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(f"{self._reduction} is not allow, only permit `none` `mean` and `sum`. ")
        return loss

if __name__ == "__main__":
    ghmc = GHMC(ignore_index=0)
    y_pred = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
    y_true = torch.tensor([1, 0])
    print(ghmc(y_pred, y_true))
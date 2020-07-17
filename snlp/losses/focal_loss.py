#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: focal_loss.py
@time: 2019/12/4 16:03
@description: Focal Loss的实现，用于分类任务
'''
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int=2,
                 alpha: typing.Union[float, list]=0.5,
                 gamma: int=2,
                 reduction: str="mean",
                 ignore_index=-100):
        """
        FocalLoss损失函数：-alpha*(1-pred_i)^2 * log(pred_i)
        :param num_classes: 表示类别数量
        :param alpha: 类别权重，当alpha为list时，表示各个类别的权重；当alpha为常数是，类别权重为[alpha, 1-alpha, 1-alpha, ...]
        :param gamma: 难易样本调节参数，论文中设置为2
        :param reduction: 损失计算方式，默认"mean"返回均值，"sum"返回和
        """
        super(FocalLoss, self).__init__()
        self._reduction = reduction
        self._ignore_index = ignore_index
        self.gamma = gamma
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.tensor(alpha)
        else:
            assert  alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        损失计算
        :param y_pred: 预测输出，shape为 [B, N, C] 或者 [B, C]，N表示检测框数
        :param y_true: 实际类别，shape为 [B, N] 或者 [B,]
        :return:
        """
        y_pred = y_pred.view(-1, y_pred.size(-1))
        self.alpha = self.alpha.to(y_pred.device)
        # 计算softmax
        y_prob = F.softmax(y_pred, dim=1)

        ## 获取每个类别对应位置的概率
        y_prob = y_prob.gather(1, y_true.view(-1, 1))
        ## 获取每一个类别对应位置的对数概率
        y_logprob = torch.log(y_prob)
        ## 对忽略的索引进行mask
        num_labels = (y_true != self._ignore_index).sum()  # 计算有效的标签的数量
        y_logprob.masked_fill_((y_true == self._ignore_index).unsqueeze(1), 0)
        self.alpha = self.alpha.gather(0, y_true.view(-1))
        loss = - (torch.pow((1-y_prob), self.gamma) * y_logprob)
        loss = torch.mul(self.alpha, loss.t())

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
    focal = FocalLoss(ignore_index=0)
    y_pred = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
    y_true = torch.tensor([1, 0])
    print(focal(y_pred, y_true))
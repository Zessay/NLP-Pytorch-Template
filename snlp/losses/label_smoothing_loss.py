#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/2/14 18:04
# @Author : Zessay
# @File : label_smoothing_loss.py
# @Software: PyCharm

# -----------------------------
import torch
import torch.nn as nn

## ----------------- 定义LabelSmoothLoss ---------------------------
class LabelSmoothLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, reduction="mean", ignore_index=-100):
        super(LabelSmoothLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction=reduction)
        else:
            self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_classes):
        one_hot = torch.randn(1, num_classes)
        one_hot.fill_(self.label_smoothing / (num_classes - 1))
        return one_hot

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        LabelSmooth函数的主逻辑
        :param logits: [B, num_classes]
        :param labels: [B,]，真实标签值
        :return:
        """
        scores = self.logsoftmax(logits)
        num_classes = scores.size(-1)

        # 进行标签平滑
        targets = labels.view(-1)
        if self.confidence < 1:
            tdata = targets.detach()
            one_hot = self._smooth_label(num_classes)
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(targets.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            tmp_.masked_fill_((targets == self.ignore_index).unsqueeze(1), 0)
            targets = tmp_.detach()
        loss = self.criterion(scores, targets)
        return loss


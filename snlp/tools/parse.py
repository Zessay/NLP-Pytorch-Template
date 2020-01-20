#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: parse.py
@time: 2019/12/2 19:04
@description: 用于解析激活函数，损失，优化器以及metric
'''
import typing
import torch
import math
import torch.nn as nn
import torch.optim as optim

from snlp.base import BaseMetric
from snlp.optimizer import RAdam, AdamW, PlainRAdam, \
    AdaBound, AdaFactor, Lamb, Lookahead, Nadam, NovoGrad, Ralamb, RaLars, SGDW


activation = nn.ModuleDict([
    ['relu', nn.ReLU()],
    ['hardtanh', nn.Hardtanh()],
    ['relu6', nn.ReLU6()],
    ['sigmoid', nn.Sigmoid()],
    ['tanh', nn.Tanh()],
    ['softmax', nn.Softmax()],
    ['softmax2d', nn.Softmax2d()],
    ['logsoftmax', nn.LogSoftmax()],
    ['elu', nn.ELU()],
    ['selu', nn.SELU()],
    ['celu', nn.CELU()],
    ['hardshrink', nn.Hardshrink()],
    ['leakyrelu', nn.LeakyReLU()],
    ['logsigmoid', nn.LogSigmoid()],
    ['softplus', nn.Softplus()],
    ['softshrink', nn.Softshrink()],
    ['prelu', nn.PReLU()],
    ['softsign', nn.Softsign()],
    ['softmin', nn.Softmin()],
    ['tanhshrink', nn.Tanhshrink()],
    ['rrelu', nn.RReLU()],
    ['glu', nn.GLU()],
])

loss = nn.ModuleDict([
    ['l1', nn.L1Loss()],
    ['nll', nn.NLLLoss()],
    ['kldiv', nn.KLDivLoss()],
    ['mse', nn.MSELoss()],
    ['bce', nn.BCELoss()],
    ['bce_with_logits', nn.BCEWithLogitsLoss()],
    ['cosine_embedding', nn.CosineEmbeddingLoss()],
    ['ctc', nn.CTCLoss()],
    ['hinge_embedding', nn.HingeEmbeddingLoss()],
    ['margin_ranking', nn.MarginRankingLoss()],
    ['multi_label_margin', nn.MultiLabelMarginLoss()],
    ['multi_label_soft_margin', nn.MultiLabelSoftMarginLoss()],
    ['multi_margin', nn.MultiMarginLoss()],
    ['smooth_l1', nn.SmoothL1Loss()],
    ['soft_margin', nn.SoftMarginLoss()],
    ['cross_entropy', nn.CrossEntropyLoss()],
    ['triplet_margin', nn.TripletMarginLoss()],
    ['poisson_nll', nn.PoissonNLLLoss()]
])

optimizer = dict({
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'sparse_adam': optim.SparseAdam,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'sgd': optim.SGD,
    'radam': RAdam,
    'adamw': AdamW,
    'plainrdam': PlainRAdam,
    'adabound': AdaBound,
    'adafactor': AdaFactor,
    'lamb': Lamb,
    'lookahead': Lookahead,
    'nadam': Nadam,
    'novograd': NovoGrad,
    'ralamb': Ralamb,
    'ralars': RaLars,
    'sgdw': SGDW
})


def _parse(
        identifier: typing.Union[str, typing.Type[nn.Module], nn.Module],
        dictionary: nn.ModuleDict,
        target: str
) -> nn.Module:
    if isinstance(identifier, str):
        if identifier in dictionary:
            return dictionary[identifier]
        else:
            raise ValueError(
                f"Could not interpret {target} identifier: " + str(identifier)
            )
    elif isinstance(identifier, nn.Module):
        return identifier
    elif issubclass(identifier, nn.Module):
        return identifier()
    else:
        raise ValueError(
            f"Could not interpret {target} identifier: " + str(identifier)
        )

# ------------------------ 下面3个输入可以是str, 类或实例，返回是实例 -----------------------------------

# 解析激活函数
def parse_activation(
        identifier: typing.Union[str, typing.Type[nn.Module], nn.Module]
) -> nn.Module:
    return _parse(identifier, activation, "activation")

# 解析损失函数
def parse_loss(
        identifier: typing.Union[str, typing.Type[nn.Module], nn.Module],
) -> nn.Module:
    return _parse(identifier, loss, "loss")

# 解析metric
def parse_metric(
        metric: typing.Union[str, typing.Type[BaseMetric], BaseMetric]
) -> BaseMetric:
    if isinstance(metric, str):
        metric = metric.lower()
        for subclass in BaseMetric.__subclasses__():
            if metric in subclass.ALIAS:
                return subclass()
    elif isinstance(metric, BaseMetric):
        return metric
    elif issubclass(metric, BaseMetric):
        return metric()
    else:
        raise ValueError(f"`{metric}` can not be used. ")

# -------------------------- 下面的输入只能是str或者类，返回也是类 ---------------------------------------

def parse_optimizer(
        identifier: typing.Union[str, typing.Type[optim.Optimizer]]
) -> optim.Optimizer:
    if isinstance(identifier, str):
        identifier = identifier.lower()
        if identifier in optimizer:
            return optimizer[identifier]
        else:
            raise ValueError(
                f"Could not interpret optimizer identifier: " + str(identifier)
            )
    elif issubclass(identifier, optim.Optimizer):
        return identifier
    else:
        raise ValueError(
            f"Could not interpret optimizer identifier: " + str(identifier)
        )
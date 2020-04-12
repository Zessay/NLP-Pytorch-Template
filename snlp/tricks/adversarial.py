#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: adversarial.py
@time: 2020/4/8 21:00
@description: 定义对方训练方法FGM和PGD
'''
import torch

class FGM:
    """
    使用示例：
    fgm = FGM(model)
    for batch_x, batch_y in data:
        # 正常训练
        loss = model(batch_x, batch_y)
        loss.backward()  # 方向传播，得到正常的grad
        # 对抗训练
        fgm.attack()  # 在embedding上添加扰动
        loss_adv = model(batch_x, batch_y)
        # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        loss_adv.backward()
        fgm.restore()  # 恢复embedding参数原始值
        optimizer.step()
        model.zero_grad()
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embedding'):
        """
        :param epsilon:
        :param emb_name: 表示模型中embedding向量定义的名称
        :return:
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                # 计算梯度的范数
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="embedding"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup, f"{name}没有进行扰动"
                param.data = self.backup[name]
        self.backup = {}


# -------------------------------------------------------------

class PGD:
    """
    示例
    pgd = PGD(model)
    K = 3  # 表示前进的总步数
    for batch_x, batch_y in data:
        # 正常训练
        loss = model(batch_x, batch_y)
        loss.backward()  # 方向传播，得到正常的梯度
        pgd.backup_grad()  # 保存正常梯度
        # 对抗训练
        for t in range(K):
            pgd.attack(is_first_attack=(t==0))
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            ## 计算对抗损失
            loss_adv = model(batch_x, batch_y)
            ## 将对抗梯度累加到原始梯度上
            loss_adv.backward()
    pgd.restore()   # 恢复embedding的参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()
    """
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name="embedding", is_first_attck=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attck:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name="embedding"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup, f"{name}没有进行扰动"
                param.data = self.emb_backup[name]

        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        # 计算当前扰动和原始参数的差值
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return param_data + r

    def backup_grad(self):
        # 保存梯度值
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        # 还原梯度值
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: dense_net.py
@time: 2019/12/2 10:08
@description: 相当于残差网络的过程
'''
import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 growth_rate: int=20,
                 kernel_size: tuple=(2,2),
                 layers_per_dense_block: int=3):
        super().__init__()
        dense_block = []
        ## 按照残差的方式连接
        for _ in range(layers_per_dense_block):
            conv_block = self._make_conv_block(in_channels, growth_rate, kernel_size)
            dense_block.append(conv_block)
            in_channels += growth_rate
        self._dense_block = nn.ModuleList(dense_block)

    def forward(self, x):
        """
        对输入进行处理
        :param x: 输入的大小为[N, C, H, W]
        :return:
        """
        for layer in self._dense_block:
            conv_out =layer(x)
            x = torch.cat([x, conv_out], dim=1)
        return x


    @classmethod
    def _make_conv_block(cls,
                         in_channels: int,
                         out_channels: int,
                         kernel_size: tuple
                         ) -> nn.Module:
        return nn.Sequential(
            ## 这里进行pad，确保输入和输出的长宽保持不变
            ## 分别对两边填充时是(k-1)/2，只对一边填充则是(k-1)
            ## 这里padding参数元组分别表示左右上下侧
            nn.ConstantPad2d(
                (0, kernel_size[1]-1, 0, kernel_size[0]-1), 0
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ),
            nn.ReLU()
        )

class DenseNet(nn.Module):
    """DenseNet Module"""
    def __init__(self,
                 in_channels,
                 nb_dense_blocks: int=3,
                 layers_per_dense_block: int=3,
                 growth_rate: int=10,
                 transition_scale_down_ratio: float=0.5,
                 conv_kernel_size: tuple=(2,2),
                 pool_kernel_size: tuple=(2,2)):
       super().__init__()
       dense_blocks = []
       transition_blocks = []
       for _ in range(nb_dense_blocks):
           dense_block = DenseBlock(
               in_channels, growth_rate, conv_kernel_size, layers_per_dense_block)
           in_channels += layers_per_dense_block * growth_rate
           dense_blocks.append(dense_block)

           ## 表示dense的逆过程
           ## channel数量减少
           transition_block = self._make_transition_block(
               in_channels, transition_scale_down_ratio, pool_kernel_size)
           in_channels = int(in_channels * transition_scale_down_ratio)
           transition_blocks.append(transition_block)

       self._dense_blocks = nn.ModuleList(dense_blocks)
       self._transition_blocks = nn.ModuleList(transition_blocks)
       self._out_channels = in_channels


    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x):
        """
        前向传播
        :param x: 维度为[B, C, H, W]
        :return:
        """
        for dense_block, trans_block in zip(self._dense_blocks, self._transition_blocks):
            ## 先进行dense加深网络层，再进行trans减少channel
            ## denseblock中一次有多层，transblock中一次只有一层
            x = dense_block(x)
            x = trans_block(x)
        return x



    @classmethod
    def _make_transition_block(cls,
                               in_channels: int,
                               transition_scale_down_ratio: float,
                               pool_kernel_size: tuple
                               ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels*transition_scale_down_ratio),
                kernel_size=1
            ),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
        )
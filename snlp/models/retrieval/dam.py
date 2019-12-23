#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: dam.py
@time: 2019/12/20 13:54
@description: 来自论文 https://www.aclweb.org/anthology/P18-1103.pdf
'''
import torch
import torch.nn as nn

from snlp.params.param_table import ParamTable
from snlp.params.param import Param
from snlp.tools.parse import parse_activation
from snlp.base.base_model import BaseModel
from snlp.modules.transformer import StackedEncoder
from snlp.modules.transformer.layers import CrossAttentionLayer
from snlp.modules.transformer.models import get_pad_mask

class DAM(BaseModel):
    def __init__(self):
        super(DAM, self).__init__()

    @classmethod
    def get_default_params(cls) -> ParamTable:
        # 定义包含Emebedding层和MLP层
        params = super().get_default_params(with_embedding=True,
                                            with_multi_layer_perceptron=True)

        # ----------------- 定义关于输入长度的参数 ----------------------
        params.add(Param(
            name='uttrs_len', value=20,
            desc="The number of words of per utterance."
        ))
        params.add(Param(
            name='resp_len', value=20,
            desc="The number of words of response."
        ))
        params.add(Param(
            name='turns', value=4,
            desc="The prev turns of the dialogue to consider to select the response."
        ))

        # --------------- Encoder and Cross Attention -----------------
        # 定义自编码器和Cross Attention的参数
        params.add(Param(
            name='n_layers', value=4,
            desc="The layer number of encoder."
        ))
        params.add(Param(
            name='d_model', value=512,
            desc="The input and output tensor dim of encoder."
        ))
        params.add(Param(
            name='n_head', value=8,
            desc="The multi-head number of encoder."
        ))
        params.add(Param(
            name='d_k', value=64,
            desc="The key size of per head."
        ))
        params.add(Param(
            name='d_v', value=64,
            desc="The value size of per head."
        ))
        params.add(Param(
            name='d_inner', value=768,
            desc="The hidden size of FFN."
        ))
        params.add(Param(
            name='dropout', value=0.1,
            desc="The dropout of encoder."
        ))
        params.add(Param(
            name='n_position', value=512,
            desc="The positions of a sentence."
        ))

        # ----------------- The two 3D Conv and Pool ---------------------

        # 关于两个3D卷积层的参数
        ## 第一个3D卷积层
        params.add(Param(
            name='conv1_channels', value=256,
            desc="The output channels of first Conv3D."
        ))
        params.add(Param(
            name='conv1_kernel', value=(1, 3, 3),
            desc="The kernel size of first Conv3D, keep the output size immutable."
        ))
        params.add(Param(
            name='conv1_stride', value=(1, 1, 1),
            desc="The stride of first Conv3D, keep the output size immutable."
        ))
        params.add(Param(
            name='conv1_padding', value=(0, 1, 1),
            desc="The padding of first Conv3D, keep the output size immutable."
        ))
        params.add(Param(
            name='conv1_activation', value="relu",
            desc="The activation first Conv3D."
        ))
        ## 第一个3D池化层
        params.add(Param(
            name='pool1_kernel', value=(2, 4, 4),
            desc="The kernel size of first Pool3D, the output size is 1/4 of the origin."
        ))
        params.add(Param(
            name='pool1_stride', value=(2, 4, 4),
            desc="The stride of first Pool3D, the output size is 1/4 of the origin."
        ))
        params.add(Param(
            name='pool1_padding', value=(0, 0, 0),
            desc="The padding of first Pool3D, the output size is 1/4 of the origin."
        ))

        ## 第二个3D卷积层
        params.add(Param(
            name='conv2_channels', value=512,
            desc="The output channels of first Conv3D."
        ))
        params.add(Param(
            name='conv2_kernel', value=(1, 3, 3),
            desc="The stride of second Conv3D, keep the output size immutable."
        ))
        params.add(Param(
            name='conv2_stride', value=(1, 1, 1),
            desc="The stride of second Conv3D, keep the output size immutable."
        ))
        params.add(Param(
            name='conv2_padding', value=(0, 1, 1),
            desc="The padding of second Conv3D, keep the output size immutable."
        ))
        params.add(Param(
            name='conv2_activation', value="relu",
            desc="The activation second Conv3D."
        ))
        ## 第二个3D池化层
        params.add(Param(
            name='pool2_kernel', value=(2, 4, 4),
            desc="The kernel size of second Pool3D, the output size is 1/4 of the origin."
        ))
        params.add(Param(
            name='pool2_stride', value=(2, 4, 4),
            desc="The stride of second Pool3D, the output size is 1/4 of the origin."
        ))
        params.add(Param(
            name='pool2_padding', value=(0, 0, 0),
            desc="The padding of second Pool3D, the output size is 1/4 of the origin."
        ))
        return params

    def build(self):
        """DAM architecture"""
        # 由于已有的词向量是200维的，所以使用线性层先将词向量转化为512维
        ## 返回embedding层
        self.word_emb = self._make_default_embedding_layer()
        self.word_proj = nn.Linear(self._params['embedding_output_dim'],
                                   self._params['d_model'])


        # 得到多粒度表征，n_layers表示表征的数量，记为L
        self.encoder = StackedEncoder(
            n_layers=self._params['n_layers'],
            n_head=self._params['n_head'],
            d_k=self._params['d_k'],
            d_v=self._params['d_v'],
            d_model=self._params['d_model'],
            d_inner=self._params['d_inner'],
            dropout=self._params['dropout'],
            n_position=self._params['n_position'],
            add_position=True
        )

        # utterance和response的交叉attention
        self.U_R_att = CrossAttentionLayer(
            d_model=self._params['d_model'],
            d_inner=self._params['d_inner'],
            n_head=self._params['n_head'],
            d_k=self._params['d_k'],
            d_v=self._params['d_v'],
            dropout=self._params['dropout']
        )

        self.R_U_att = CrossAttentionLayer(
            d_model=self._params['d_model'],
            d_inner=self._params['d_inner'],
            n_head=self._params['n_head'],
            d_k=self._params['d_k'],
            d_v=self._params['d_v'],
            dropout=self._params['dropout']
        )

        self.conv1 = nn.Conv3d(2*(self._params['n_layers']+1), self._params['conv1_channels'],
                               kernel_size=self._params['conv1_kernel'],
                               stride=self._params['conv1_stride'],
                               padding=self._params['conv1_padding'])
        self.pool1 = nn.MaxPool3d(kernel_size=self._params['pool1_kernel'],
                                  stride=self._params['pool1_stride'])
        self.conv2 = nn.Conv3d(self._params['conv1_channels'], self._params['conv2_channels'],
                               kernel_size=self._params['conv2_kernel'],
                               stride=self._params['conv2_stride'],
                               padding=self._params['conv2_padding'])
        self.pool2 = nn.MaxPool3d(kernel_size=self._params['pool2_kernel'],
                                  stride=self._params['pool2_stride'])

        self.mlps = self._make_multi_perceptron_layer(
            self._params['conv2_channels']
        )

        self.output = self._make_output_layer(
            self._params['mlp_num_fan_out']
        )

    def forward(self, inputs):
        # [B, NU, Lu] 这里的length是所有utterance合并的
        utterances = inputs['utterances']
        # [B, Lr]
        response = inputs['response']
        b_size, L_r = response.size()  # 获取batch的大小
        L_u = utterances.size(-1)  # 每一个utterance的长度

        ## 使用torch.split对所有的utterances进行分隔
        ## 如果指定split_size_or_sections是一个int型，则表明每个uttr的大小
        ## 这里为了之后拼接，所有的utterances的长度必须相同
        ## 如果是list型，则表示具体指明每一个utterances的大小
        if self._params['uttrs_len'] is not None:
            uttrs = torch.unbind(utterances, dim=1)

        # ---------------- 得到response的所有层的结果 -------------------
        ## [B, Lr, d_model]

        rep_emb = self.word_proj(self.word_emb(response))
        ## [B, 1, Lr]
        rep_mask = get_pad_mask(response, self._params['padding_idx'])
        # [NL, B, Lr, d_model]
        rep_es, *_ = self.encoder(rep_emb, rep_mask)
        # [NL+1, B, Lr, d_model]
        rep_es_list = [rep_emb] + rep_es
        L_layer = len(rep_es_list)

        # --------------- 得到uttrs所有层的结果 -----------------------
        uttrs_encs_list = []
        UR_attns_list = [] # 保存U-R attention的结果
        RU_attns_list = [] # 保存R-U attention的结果
        for uttr in uttrs:
            # [B, Lu, d_model]
            uttr_emb = self.word_proj(self.word_emb(uttr))
            # [B, 1, Lu]
            uttr_mask = get_pad_mask(uttr, self._params['padding_idx'])
            # [NL, B, Lu, d_model]
            uttr_es_list, *_ = self.encoder(uttr_emb, uttr_mask)
            # [NL+1, B, Lu, d_model]
            uttr_es_list = [uttr_emb] + uttr_es_list

            # 计算和每一层response的cross attention结果
            ur_att_list, ru_att_list = [], []
            ## 对NL+1层分别进行cross-attention
            for i in range(len(rep_es_list)):
                ## [B, Lu, d_model]
                ur_, _ = self.U_R_att(uttr_es_list[i], rep_es_list[i], rep_es_list[i], rep_mask)
                ## [B, Lr, d_model]
                ru_, _ = self.R_U_att(rep_es_list[i], uttr_es_list[i], uttr_es_list[i], uttr_mask)
                ur_att_list += [ur_]
                ru_att_list += [ru_]
            # [B, NL+1, L, d_model]
            ur_concat = torch.cat(ur_att_list, dim=0).view(L_layer, b_size, L_u, self._params['d_model']).transpose(0, 1)
            ru_concat = torch.cat(ru_att_list, dim=0).view(L_layer, b_size, L_r, self._params['d_model']).transpose(0, 1)
            # 将同一个utterance的所有层进行拼接
            # [B, NL+1, L, d_model]
            uttr_concat = torch.cat(uttr_es_list, dim=0).view(L_layer, b_size, L_u, self._params['d_model']).transpose(0, 1)

            UR_attns_list += [ur_concat]
            RU_attns_list += [ru_concat]
            uttrs_encs_list += [uttr_concat]
        # ** 准备用于计算self-att match的张量 **
        ## 上面得到所有utterances，维度为[B, NU, NL+1, Lu, d_model]，NU表示utterance轮数
        Uttrs_encs = torch.cat(uttrs_encs_list, dim=0).view(-1, b_size, L_layer, L_u, self._params['d_model']).transpose(0, 1)
        ## 张量维度[B, 1, NL+1, L_r, d_model]
        Rep_encs = torch.cat(rep_es_list, dim=0).view(L_layer, b_size, L_r, self._params['d_model']).transpose(0, 1).unsqueeze(1)

        # ** 准备用于计算cross-attn match的张量 **
        ## 对attns进行拼接 [B, NU, NL+1, Lu, d_model]和[B, NU, NL+1, Lr, d_model]
        UR_attns = torch.cat(UR_attns_list, dim=0).view(-1, b_size,L_layer, L_u, self._params['d_model']).transpose(0, 1)
        RU_attns = torch.cat(RU_attns_list, dim=0).view(-1, b_size, L_layer, L_r, self._params['d_model']).transpose(0, 1)

        # ------- 计算self-att match --------
        ## [B, NU, NL+1, Lu, Lr]
        M_self = torch.matmul(Uttrs_encs, Rep_encs.transpose(3, 4))

        # ------- 计算cross-att match -------
        ## [B, NU, NL+1, Lu, Lr]
        M_cross = torch.matmul(UR_attns, RU_attns.transpose(3, 4))

        # [B, 2(NL+1), NU, Lu, Lr]
        Q = torch.cat([M_self, M_cross], dim=2).transpose(1, 2)

        conv1_activation = parse_activation(self._params['conv1_activation'])
        conv2_activation = parse_activation(self._params['conv2_activation'])
        # [B, conv1_channels, NU/2, Lu/4, Lr/4]
        Q = self.pool1(conv1_activation(self.conv1(Q)))
        # [B, conv2_channels, 1, 1, 1]
        Q = self.pool2(conv2_activation(self.conv2(Q)))
        # [B, conv2_channels]
        Q = Q.view(b_size, -1)
        # [B, fan_output]
        out = self.mlps(Q)
        # [B, num_classes] or [B, 1]
        out = self.output(out)
        return out
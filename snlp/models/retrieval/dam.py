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
import torch.nn.functional as F
import torch.nn.init as init

from snlp.params.param_table import ParamTable
from snlp.params.param import Param
from snlp.tools.parse import parse_activation
from snlp.base.base_model import BaseModel
from snlp.modules.transformer import StackedEncoder
from snlp.modules.transformer.layers import CrossAttentionLayer
from snlp.modules.transformer.models import get_pad_mask, PositionalEncoding

class DAM(BaseModel):
    def __init__(self, uttr_len=20, resp_len=20, turns=5):
        self.uttr_len = uttr_len
        self.resp_len = resp_len
        self.turns = turns
        super(DAM, self).__init__()

    def get_default_params(self) -> ParamTable:
        # 定义包含Emebedding层和MLP层
        params = super().get_default_params(with_embedding=True,
                                            with_multi_layer_perceptron=False)

        params['embedding_freeze'] = False
        # ----------------- 定义是否需要turn-embedding的参数 ----------------------
        params.add(Param(
            name='is_turn_emb', value=False,
            desc="Whether to add turn embedding."
        ))

        # ---------------- 定义是否需要position-embedding的参数 -------------------
        params.add(Param(
            name='is_position_emb', value=True,
            desc="Whether to add position embedding."
        ))
        params.add(Param(
            name='n_position', value=200,
            desc="The positions of position embedding."
        ))
        # -------------- 通用的参数和标志 -------------------
        params.add(Param(
            name='dropout', value=0.2,
            desc="The dropout of encoder."
        ))
        params.add(Param(
            name='is_layer_norm', value=True,
            desc="Whether to use layer norm before output."
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

        # ----------------- The two 3D Conv and Pool ---------------------

        # 关于两个3D卷积层的参数
        ## 第一个3D卷积层
        params.add(Param(
            name='conv1_channels', value=64,
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

        ## 第一个池化之前的pad层
        ## 如果utterance是奇数，池化之前需要一层padding
        ## 如果turns和uttr_len都是奇数
        if self.turns & 1 and self.uttr_len & 1:
            value = (0, 0, 0, 1, 1, 0)
        elif self.turns & 1:
            value = (0, 0, 0, 0, 1, 0)
        elif self.uttr_len & 1:
            value = (0, 0, 0, 1, 0, 0)
        else:
            value = (0, 0, 0, 0, 0, 0)
        params.add(Param(
            name='pad_size', value=value,
            desc= "Padding before the first Pool3D if the turn is odd."
        ))

        ## 第一个3D池化层
        params.add(Param(
            name='pool1_kernel', value=(2, 4, 4),
            desc="The kernel size of first Pool3D, the output size is 1/2 or 1/4 of the origin."
        ))
        params.add(Param(
            name='pool1_stride', value=(2, 4, 4),
            desc="The stride of first Pool3D, the output size is 1/2 or 1/4 of the origin."
        ))
        params.add(Param(
            name='pool1_padding', value=(0, 0, 0),
            desc="The padding of first Pool3D, the output size is 1/2 or 1/4 of the origin."
        ))

        ## 第二个3D卷积层
        params.add(Param(
            name='conv2_channels', value=128,
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
        value = (self.turns // 2, self.uttr_len // 4, self.resp_len // 4)
        params.add(Param(
            name='pool2_kernel', value=value,
            desc="The kernel size of second Pool3D, all dim size of the output is 1."
        ))
        params.add(Param(
            name='pool2_stride', value=value,
            desc="The stride of second Pool3D, all dim size of the output is 1."
        ))
        params.add(Param(
            name='pool2_padding', value=(0, 0, 0),
            desc="The padding of second Pool3D, all dim size of the output is 1."
        ))

        # --------- Output Layer --------------------
        # params['mlp_num_units'] = 512
        # params['mlp_num_layers'] = 1
        # params['mlp_num_fan_out'] = 128
        return params

    def build(self):
        """DAM architecture"""
        # 由于已有的词向量是200维的，所以使用线性层先将词向量转化为512维
        ## 返回embedding层
        self.word_emb = self._make_default_embedding_layer()
        self.word_proj = nn.Linear(self._params['embedding_output_dim'],
                                   self._params['d_model'], bias=False)

        if self._params['is_position_emb']:
            self.position_emb = PositionalEncoding(d_hid=self._params['d_model'],
                                                   n_position=self._params['n_position'])

        # 用于turn embedding
        if self._params['is_turn_emb']:
            self.turn_emb = nn.Embedding(self.turns+1, self._params['d_model'],
                                         padding_idx=self._params['padding_idx'])


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
            add_position=False
        )

        # utterance和response的交叉attention
        ## 共享权重
        self.cross_att = CrossAttentionLayer(
            d_model=self._params['d_model'],
            d_inner=self._params['d_inner'],
            n_head=self._params['n_head'],
            d_k=self._params['d_k'],
            d_v=self._params['d_v'],
            is_layer_norm=self._params['is_layer_norm'],
            dropout=self._params['dropout']
        )

        # self.R_U_att = CrossAttentionLayer(
        #     d_model=self._params['d_model'],
        #     d_inner=self._params['d_inner'],
        #     n_head=self._params['n_head'],
        #     d_k=self._params['d_k'],
        #     d_v=self._params['d_v'],
        #     dropout=self._params['dropout']
        # )

        self.conv1 = nn.Conv3d(2*(self._params['n_layers']+1), self._params['conv1_channels'],
                               kernel_size=self._params['conv1_kernel'],
                               stride=self._params['conv1_stride'],
                               padding=self._params['conv1_padding'])
        self.pool1 = nn.MaxPool3d(kernel_size=self._params['pool1_kernel'],
                                  stride=self._params['pool1_stride'])
        # 如果是5轮，需要进行一次pad防止丢失最后一轮信息
        self.cons_pad = nn.ConstantPad3d(padding=self._params['pad_size'], value=0)

        self.conv2 = nn.Conv3d(self._params['conv1_channels'], self._params['conv2_channels'],
                               kernel_size=self._params['conv2_kernel'],
                               stride=self._params['conv2_stride'],
                               padding=self._params['conv2_padding'])
        self.pool2 = nn.MaxPool3d(kernel_size=self._params['pool2_kernel'],
                                  stride=self._params['pool2_stride'])

        self.conv1_activation = parse_activation(self._params['conv1_activation'])
        self.conv2_activation = parse_activation(self._params['conv2_activation'])

        # self.mlps = self._make_multi_perceptron_layer(
        #     self._params['conv2_channels']
        # )

        # self.output = self._make_output_layer(
        #     self._params['mlp_num_fan_out']
        # )
        self.output = self._make_output_layer(
            self._params['conv2_channels']
        )
        self.dropout = nn.Dropout(self._params['dropout'])

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.word_proj.weight)
        init.xavier_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv2.weight)
        init.xavier_normal_(self.output.weight)
        if self._params['is_turn_emb']:
            init.orthogonal_(self.turn_emb.weight)

    def forward(self, inputs):
        """
        NU表示utterance的轮次，Lu表示一个utterance的长度，Lr表示一个response的长度
        """
        # ------------- 对utterance进行处理 -------------------
        ## [B, NU, Lu] 这里的Lu表示每一个utterance经过padding的长度
        utterances = inputs['utterances']
        bsz = utterances.size(0)  ## 获取当前的batch_size
        ## [B*Nu, 1, Lu]
        uttrs_mask = get_pad_mask(utterances, self._params['padding_idx']).view(bsz*self.turns, 1, self.uttr_len)
        ## [B*Nu, Lu, 1]
        uttrs_emb_mask = uttrs_mask.squeeze(dim=-2).unsqueeze(dim=-1)
        ## [B, Nu, Lu, emb_output]
        uttrs_emb = self.word_emb(utterances)
        ## [B*Nu, Lu, d_model]
        uttrs_emb = self.word_proj(uttrs_emb)
        ## 定义用于表示turn的张量
        ### [1, Nu]
        turns_num = inputs['turns']
        if self._params['is_turn_emb']:
            ## [1, Nu, 1, d_model]
            turns_embedding = self.turn_emb(turns_num).unsqueeze(dim=-2)
            ## [B, Nu, Lu, d_model]
            uttrs_emb = uttrs_emb + turns_embedding
        ## [B*Nu, Lu, d_model]
        uttrs_emb = uttrs_emb.view(bsz*self.turns, self.uttr_len, -1)
        if self._params['is_position_emb']:
            ### [B*Nu, Lu, d_model]
            uttrs_emb = self.position_emb(uttrs_emb)
        ## [B*Nu, Lu, d_model]，对embedding进行mask
        uttrs_emb = uttrs_emb * uttrs_emb_mask
        ## [NL, B*Nu, Lu, d_model]，自编码
        uttrs_es, *_ = self.encoder(uttrs_emb, uttrs_mask)
        uttrs_es_list = [uttrs_emb] + uttrs_es
        L_layer = len(uttrs_es_list)
        ## [B*Nu*(NL+1), Lu, d_model]
        uttrs_stack = torch.stack(uttrs_es_list, dim=0).transpose(0, 1).contiguous().view(bsz*self.turns*L_layer, self.uttr_len, -1)
        # ---------------- 得到response的所有层的结果 -------------------
        # [B, Lr]
        response = inputs['response']
        ## [B, 1, Lr]
        resp_mask = get_pad_mask(response, self._params['padding_idx'])
        ## [B, Lr, 1]
        resp_emb_mask = resp_mask.squeeze(dim=-2).unsqueeze(dim=-1)
        ## [B, Lr, emb_output]
        resp_emb = self.word_emb(response)
        ## [B, Lr, d_model]
        resp_emb = self.word_proj(resp_emb)
        if self._params['is_position_emb']:
            resp_emb = self.position_emb(resp_emb)
        resp_emb = resp_emb * resp_emb_mask
        ## [NL, B, Lr, d_model]
        resp_es, *_ = self.encoder(resp_emb, resp_mask)
        ## [NL+1, B, Lr, d_model]
        rep_es_list = [resp_emb] + resp_es
        ## [NL+1, B, Lr, d_model]的tensor
        resp_stack = torch.stack(rep_es_list, dim=0)
        ## [B*Nu*(NL+1), Lr, d_model]
        resp_stack = resp_stack.repeat(1, self.turns, 1, 1).transpose(0, 1).contiguous().view(bsz*self.turns*L_layer, self.resp_len, -1)

        # -------------------- 计算 Cross Attention --------------------------
        ## [B*Nu*(NL+1), Lu, d_model]
        UR_att, *_ = self.cross_att(uttrs_stack, resp_stack, resp_stack, resp_mask.repeat(self.turns*L_layer, 1, 1))
        ## [B*Nu*(NL+1), Lr, d_model]
        RU_att, *_ = self.cross_att(resp_stack, uttrs_stack, uttrs_stack, uttrs_mask.repeat(L_layer, 1, 1))
        # ------- 计算self-att match --------
        ## [B, NU, NL+1, Lu, Lr]
        ### **这里因为要计算相似度，注意要使用激活函数，避免发生梯度爆炸**
        M_self = F.tanh(torch.matmul(uttrs_stack, resp_stack.transpose(1, 2)).view(bsz, self.turns, L_layer, self.uttr_len, self.resp_len))
        # ------- 计算cross-att match -------
        ## [B, NU, NL+1, Lu, Lr]
        ### **这里因为要计算相似度，注意要使用激活函数，避免发生梯度爆炸**
        M_cross = F.tanh(torch.matmul(UR_att, RU_att.transpose(1, 2)).view(bsz, self.turns, L_layer, self.uttr_len, self.resp_len))
        # [B, 2(NL+1), NU, Lu, Lr]
        output = torch.cat([M_self, M_cross], dim=2).transpose(1, 2)

        # [B, conv1_channels, NU/2, Lu/4, Lr/4]
        output = self.pool1(self.cons_pad(self.conv1_activation(self.conv1(output))))
        # [B, conv2_channels, 1, 1, 1]
        output = self.pool2(self.conv2_activation(self.conv2(output)))
        # [B, conv2_channels]
        output = output.view(bsz, -1)
        output = self.dropout(output)
        # [B, fan_output]
        # output = self.mlps(output)
        # [B, num_classes] or [B, 1]
        output = self.output(output)
        return output
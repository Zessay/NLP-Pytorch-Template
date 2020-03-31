#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: albert_imn.py
@time: 2020/1/8 11:29
@description: Albert  + IMN
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from snlp.base.base_model import BaseModel
from snlp.params.param import Param
from snlp.params.param_table import ParamTable
from snlp.modules.stacked_brnn import StackedBRNN
from snlp.modules.transformer.layers import CrossAttentionLayer
from snlp.modules.transformer.models import get_pad_mask, PositionalEncoding
import snlp.tools.constants as constants

from albert_pytorch.model.modeling_albert_bright import AlbertModel


class AlbertIMN(BaseModel):
    def __init__(self, uttr_len=30, resp_len=30, turns=5,
                 config=None, model_path=None, data_type="uur", freeze_bert=True):
        self.config = config
        self.model_path = model_path
        self.uttr_len = uttr_len
        self.resp_len = resp_len
        self.turns = turns
        self.freeze_bert = freeze_bert
        self.data_type = data_type
        super(AlbertIMN, self).__init__()

    def get_default_params(self) -> ParamTable:
        params = super().get_default_params(with_embedding=False,
                                            with_multi_layer_perceptron=True)
        params.add(Param(
            name="padding_idx", value=0,
            desc="Whether to add turn embedding."
        ))

        # --------------------- 定义标志位 ---------------------
        params.add(Param(
            name='is_layer_norm', value=True,
            desc="Whether to use layer normalization when output from cross attention layer."
        ))

        params.add(Param(
            name='is_ur_embed', value=True,
            desc="Whether to use ur embedding add to the albert embedding."
        ))
        # --------------------- 设置模型参数 -------------------
        ## char embedding
        params.add(Param(
            name='d_model', value=312,
            desc="The input dim of the stack BRNN."
        ))
        ## dropout
        params.add(Param(
            name='dropout', value=0.1,
            desc="The global dropout of the model."
        ))
        ## stack BRNN
        hidden_size = 200

        params.add(Param(
            name='stack_brnn_hid', value=hidden_size,
            desc="The hidden size of Stacked BRNN."
        ))
        params.add(Param(
            name='stack_brnn_layers', value=1,
            desc="The layers of the Stacked BRNN."
        ))
        # params.add(Param(
        #     name='d_inner', value=hidden_size,
        #     desc="The FFN inner hidden size."
        # ))
        # params.add(Param(
        #     name='n_head', value=1,
        #     desc="The number of the head of the cross attention layer."
        # ))
        # params.add(Param(
        #     name='d_k', value=hidden_size,
        #     desc="The key size of the cross attention layer."
        # ))
        # params.add(Param(
        #     name='d_v', value=hidden_size,
        #     desc="The value size of the cross attention layer."
        # ))
        # mlp
        params['mlp_num_units'] = 512
        params['mlp_num_layers'] = 1
        params['mlp_num_fan_out'] = 256

        return params


    # 构建模型
    def build(self):
        # 用于对单词的强化编码
        self.albert = AlbertModel.from_pretrained(self.model_path, config=self.config)
        ## 如果需要，则冻结bert层的参数
        if self.freeze_bert:
            for child in self.albert.children():
                for param in child.parameters():
                    param.requires_grad=False
        if self.params['is_ur_embed'] and self.data_type == 'uru':
            self.ur_embed = nn.Embedding(2, self.params['d_model'])


        # 堆叠的RNN网络
        self.stack_brnn = StackedBRNN(input_size=self.params['d_model'],
                                      hidden_size=self.params['stack_brnn_hid'],
                                      num_layers=self.params['stack_brnn_layers'],
                                      dropout_rate=self.params['dropout'],
                                      stack_layers=True)

        # 定义对多层Attention时的参数，初始化为全0
        # self.w_m = nn.Parameter(data=torch.zeros(self._params['stack_brnn_layers']))

        # 交叉attention
        # self.cross_attn = CrossAttentionLayer(d_model=2*self._params['stack_brnn_hid'],
        #                                     d_inner=self._params['d_inner'],
        #                                     n_head=self._params['n_head'],
        #                                     d_k=self._params['d_k'],
        #                                     d_v=self._params['d_v'],
        #                                     is_layer_norm=self._params['is_layer_norm'],
        #                                     dropout=self._params['dropout'])

        # 定义一个用于获取句子表征的GRU层
        self.sent_gru = nn.GRU(input_size=8*self.params['stack_brnn_hid'],
                               hidden_size=self.params['stack_brnn_hid'],
                               batch_first=True,
                               dropout=self.params['dropout'],
                               bidirectional=True)
        # 定义用于uttr表征的GRU层
        self.uttr_gru = nn.GRU(input_size=6*self.params['stack_brnn_hid'],
                               hidden_size=self.params['stack_brnn_hid'],
                               batch_first=True,
                               dropout=self.params['dropout'],
                               bidirectional=True)

        self.mlps = self._make_multi_perceptron_layer(in_features=12*self.params['stack_brnn_hid'])
        self.output = self._make_output_layer(in_features=self.params['mlp_num_fan_out'])
        self.dropout = nn.Dropout(self.params['dropout'])
        self.init_weights()

    def init_weights(self):
        # init.xavier_normal_(self.word_proj.weight)
        init.xavier_normal_(self.output.weight)
        for weight in [self.sent_gru.weight_hh_l0, self.sent_gru.weight_ih_l0,
                       self.uttr_gru.weight_hh_l0, self.uttr_gru.weight_hh_l0]:
            init.orthogonal_(weight)

    def get_matching_tensor(self, slf_attn, cross_attn):
        minus_t = slf_attn - cross_attn
        dot_t = slf_attn * cross_attn
        t_ = torch.cat([slf_attn, cross_attn, minus_t, dot_t], dim=-1)
        return t_

    def attention(self, U_emb, R_emb, U_mask, R_mask):
        """
        :param U_emb: [B, Nu*Lu, 2*hidden_size]
        :param R_emb: [B, Lr, 2*hidden_size]
        :param U_mask: [B, 1, Nu*Lu]
        :param R_mask: [B, 1, Lr]
        :return:
        """
        # [B, Nu*Lu, Lr]
        cross_matrix = torch.einsum("bik,bjk->bij", U_emb, R_emb)
        cross_UR = cross_matrix.masked_fill(mask=(R_mask==0), value=1e-10)
        cross_RU = cross_matrix.transpose(1, 2).contiguous().masked_fill(mask=(U_mask==0), value=1e-10)
        U_cross_attn = torch.matmul(F.softmax(cross_UR, dim=-1), R_emb)
        R_cross_attn = torch.matmul(F.softmax(cross_RU, dim=-1), U_emb)
        return U_cross_attn, R_cross_attn

    def forward(self, inputs):
        # 首先获取输入的utterance和response
        ## [B, Nu, Lu]
        utterances = inputs[constants.UTTRS]
        ## [B, Lr]
        response = inputs[constants.RESP]

        bsz = utterances.size(0)
        ## 对utterances进行reshape [B*Nu, Lu]
        utterances = utterances.view(bsz*self.turns, self.uttr_len)
        ## 如果是uru的形式，获取UR的position标记
        if self.data_type == "uru":
            ## [B, Nu, Lu]
            ur_pos = inputs[constants.UR_POS]
            ur_pos = ur_pos.view(bsz*self.turns, self.uttr_len)


        # 获取utterance和response的mask
        ## [B*Nu, 1, Lu]
        uttrs_mask = get_pad_mask(utterances, self.params['padding_idx'])
        uttrs_albert_mask = uttrs_mask.squeeze(dim=-2) ## [B*Nu, Lu]
        ## [B, 1, Lr]
        resp_mask = get_pad_mask(response, self.params['padding_idx'])
        resp_albert_mask = resp_mask.squeeze(dim=-2) ## [B, Lr]

        ## 获取utterance和response的embeding
        U_emb = self.albert(input_ids=utterances, attention_mask=uttrs_albert_mask)[0] ## [B*Nu, Lu, d_model]
        R_emb = self.albert(input_ids=response, attention_mask=resp_albert_mask)[0] ## [B, Lr, d_model]

        if self.params['is_ur_embed'] and self.data_type == "uru":
            ur_emb = self.ur_embed(ur_pos)
            U_emb = U_emb + ur_emb

        ## 对输出的padding进行mask
        uttrs_embed_mask = uttrs_albert_mask.unsqueeze(dim=-1)
        resp_embed_mask = resp_albert_mask.unsqueeze(dim=-1)
        U_emb = U_emb * uttrs_embed_mask
        R_emb = R_emb * resp_embed_mask

        # ------------------ Attentive HR Encoder ----------------------
        ## [B*Nu, Lu, NL, 2*hid]
        U_stack = self.stack_brnn(U_emb, uttrs_embed_mask).squeeze(dim=-2)
        ## [B, Lr, NL, 2*hid]
        R_stack = self.stack_brnn(R_emb, resp_embed_mask).squeeze(dim=-2)
        # ## [NL, 1]
        # wm = F.softmax(self.w_m, dim=0).unsqueeze(-1)
        # ## 对utterance各个层进行Attention [B*Nu, Lu, 2*hid]
        # U_attn = torch.matmul(U_stack.transpose(2, 3), wm).squeeze(dim=-1)
        # ## 对response的各个层进行Attention [B, Lr, 2*hid]
        # R_attn = torch.matmul(R_stack.transpose(2, 3), wm).squeeze(dim=-1)

        U_attn = U_stack
        R_attn = R_stack

        # ------------------ Matching Layer -------------------------

        U_attn_reshape = U_attn.view(bsz, self.turns*self.uttr_len, -1)
        uttrs_mask_reshape = uttrs_mask.squeeze(dim=1).view(bsz, self.turns*self.uttr_len).unsqueeze(dim=-2)
        # ## [B, Nu*Lu, 2*hid]
        # U_R_attn, *_ = self.cross_attn(U_attn_reshape, R_attn, R_attn, resp_mask)
        # ## [B*Nu, Lu, 2*hid]
        # U_R_attn = U_R_attn.view(bsz*self.turns, self.uttr_len, -1)
        # ## [B, Lr, 2*hid]
        # R_U_attn, *_ = self.cross_attn(R_attn, U_attn_reshape, U_attn_reshape, uttrs_mask.squeeze(dim=1).view(bsz, self.turns*self.uttr_len).unsqueeze(dim=-2))
        ## [B, Nu*Lu, 2*hid]和[B, Lr, 2*hid]
        U_R_attn, R_U_attn = self.attention(U_attn_reshape, R_attn, U_mask=uttrs_mask_reshape, R_mask=resp_mask)
        U_R_attn = U_R_attn.view(bsz * self.turns, self.uttr_len, -1)

        ## [B*Nu, Lu, 8*hid]
        C_mat = self.get_matching_tensor(U_attn, U_R_attn)
        ## [B, Lr, 8*hid]
        R_mat = self.get_matching_tensor(R_attn, R_U_attn)

        # ----------------- Aggregation Layer -----------------------

        ## [B*Nu, Lu, 2*hid]以及[2, B*Nu, hid]
        C_out, C_state = self.sent_gru(C_mat)
        ## [B, Lr, 2*hid]以及[2, B, hid]
        R_out, R_state = self.sent_gru(R_mat)
        ## 最大池化和平均池化
        ### 下面3个为 [B*Nu, 2*hid]
        C_mean = torch.mean(C_out, dim=1)
        C_max = torch.max(C_out, dim=1)[0]
        C_state = C_state.transpose(0, 1).contiguous().view(bsz*self.turns, -1)
        ## [B, Nu, 6*hid]
        C_out = torch.cat([C_mean, C_max, C_state], dim=-1).view(bsz, self.turns, -1)
        ## [B, Nu, 2*hid]以及[2, B, hid]
        C_out, C_state = self.uttr_gru(C_out)
        ## utterance和response的聚合
        C_mean = torch.mean(C_out, dim=1)  # [B, 2*hid]
        C_max = torch.max(C_out, dim=1)[0]   # [B, 2*hid]
        C_state = C_state.transpose(0, 1).contiguous().view(bsz, -1)

        R_mean = torch.mean(R_out, dim=1) # [B, 2*hid]
        R_max = torch.max(R_out, dim=1)[0]   # [B, 2*hid]
        R_state = R_state.transpose(0, 1).contiguous().view(bsz, -1)

        ## [B, 12*hid]
        M_agr = torch.cat([C_mean, C_max, C_state, R_mean, R_max, R_state], dim=-1)

        # ------------------ Output Layer ---------------------
        output = self.dropout(M_agr)
        output = self.mlps(output)
        output = self.output(output)
        return output
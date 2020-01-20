#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: imn.py
@time: 2020/1/2 15:47
@description: Paper: https://arxiv.org/pdf/1901.01824.pdf | Reference Code: https://github.com/JasonForJoy/IMN/blob/master/Ecommerce/model/model_IMN.py
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

class IMN(BaseModel):
    def __init__(self, uttr_len=20, resp_len=20, turns=5):
        self.uttr_len = uttr_len
        self.resp_len = resp_len
        self.turns = turns
        super(IMN, self).__init__()

    def get_default_params(self) -> ParamTable:
        params = super().get_default_params(with_embedding=True,
                                            with_multi_layer_perceptron=True)
        params['embedding_freeze'] = False
        # ------------------- 定义一些标志位 ------------------
        params.add(Param(
            name='is_turn_emb', value=False,
            desc="Whether to add turn embedding."
        ))
        params.add(Param(
            name='is_position_emb', value=True,
            desc="Whether to add position embedding."
        ))
        params.add(Param(
            name='is_layer_norm', value=True,
            desc="Whether to use layer normalization when output from cross attention layer."
        ))

        # ----------------- 设置模型参数 --------------------
        ## position embedding
        params.add(Param(
            name='n_position', value=200,
            desc="The positions of position embedding."
        ))
        ## word proj
        params.add(Param(
            name='d_model', value=200,
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
            name='stack_brnn_layers', value=3,
            desc="The layers of the Stacked BRNN."
        ))
        params.add(Param(
            name='d_inner', value=hidden_size,
            desc="The FFN inner hidden size."
        ))
        params.add(Param(
            name='n_head', value=1,
            desc="The number of the head of the cross attention layer."
        ))
        params.add(Param(
            name='d_k', value=hidden_size,
            desc="The key size of the cross attention layer."
        ))
        params.add(Param(
            name='d_v', value=hidden_size,
            desc="The value size of the cross attention layer."
        ))
        # mlp
        params['mlp_num_units'] = 512
        params['mlp_num_layers'] = 1
        params['mlp_num_fan_out'] = 256

        return params

    def build(self):
        # 构建网络
        ## 构建Embedding层，默认输入向量的维度为200
        self.word_emb = self._make_default_embedding_layer()
        # self.train_word_emb = self._make_embedding_layer(
        #     num_embeddings=self._params['embedding_input_dim'],
        #     embedding_dim=self._params['embedding_output_dim'],
        #     freeze=False,
        #     padding_idx=self._params['padding_idx']
        # )

        if self._params['is_turn_emb']:
            self.turn_emb = nn.Embedding(self.turns+1,
                                         self._params['d_model'],
                                         padding_idx=self._params['padding_idx'])
        if self._params['is_position_emb']:
            self.position_emb = PositionalEncoding(self._params['d_model'],
                                                   n_position=self._params['n_position'])

        # self.word_proj = nn.Linear(self._params['embedding_output_dim'],
        #                            self._params['d_model'], bias=False)
        self.stack_brnn = StackedBRNN(input_size=self._params['d_model'],
                                      hidden_size=self._params['stack_brnn_hid'],
                                      num_layers=self._params['stack_brnn_layers'],
                                      dropout_rate=self._params['dropout'],
                                      stack_layers=True)

        # 定义对多层Attention时的参数，初始化为全0
        self.w_m = nn.Parameter(data=torch.zeros(self._params['stack_brnn_layers']))

        # 交叉attention
        self.cross_attn = CrossAttentionLayer(d_model=2*self._params['stack_brnn_hid'],
                                            d_inner=self._params['d_inner'],
                                            n_head=self._params['n_head'],
                                            d_k=self._params['d_k'],
                                            d_v=self._params['d_v'],
                                            is_layer_norm=self._params['is_layer_norm'],
                                            dropout=self._params['dropout'])
        # self.R_U_attn = CrossAttentionLayer(d_model=2*self._params['stack_brnn_hid'],
        #                                     d_inner=self._params['d_inner'],
        #                                     n_head=self._params['n_head'],
        #                                     d_k=self._params['d_k'],
        #                                     d_v=self._params['d_v'],
        #                                     is_layer_norm=self._params['is_layer_norm'],
        #                                     dropout=self._params['dropout'])

        # 定义一个用于获取句子表征的GRU层
        self.sent_gru = nn.GRU(input_size=8*self._params['stack_brnn_hid'],
                               hidden_size=self._params['stack_brnn_hid'],
                               batch_first=True,
                               dropout=self._params['dropout'],
                               bidirectional=True)
        # 定义用于uttr表征的GRU层
        self.uttr_gru = nn.GRU(input_size=6*self._params['stack_brnn_hid'],
                               hidden_size=self._params['stack_brnn_hid'],
                               batch_first=True,
                               dropout=self._params['dropout'],
                               bidirectional=True)

        self.mlps = self._make_multi_perceptron_layer(in_features=12*self._params['stack_brnn_hid'])
        self.output = self._make_output_layer(in_features=self._params['mlp_num_fan_out'])
        self.dropout = nn.Dropout(self._params['dropout'])
        self.init_weights()

    def init_weights(self):
        # init.xavier_normal_(self.word_proj.weight)
        init.xavier_normal_(self.output.weight)
        if self._params['is_turn_emb']:
            init.orthogonal_(self.turn_emb.weight)

        for weight in [self.sent_gru.weight_hh_l0, self.sent_gru.weight_ih_l0,
                       self.uttr_gru.weight_hh_l0, self.uttr_gru.weight_hh_l0]:
            init.orthogonal_(weight)

    def get_matching_tensor(self, slf_attn, cross_attn):
        minus_t = slf_attn - cross_attn
        dot_t = slf_attn * cross_attn
        t_ = torch.cat([slf_attn, cross_attn, minus_t, dot_t], dim=-1)
        return t_

    def forward(self, inputs):
        """
        NU表示utterance的轮次，Lu表示一个utterance的长度，Lr表示一个response的长度
        """
        # [B, Nu, Lu]
        utterances = inputs['utterances']
        # [B, Lr]
        response = inputs['response']
        bsz = utterances.size(0)
        # 获取表示turn的张量
        turns_num = inputs['turns']
        # 对utterance中的pad进行mask [B*Nu, 1, Lu]
        uttrs_mask = get_pad_mask(utterances.view(-1, self.uttr_len),
                                  self._params['padding_idx'])
        uttrs_embed_mask = uttrs_mask.squeeze(dim=-2).unsqueeze(dim=-1) # [B*Nu, Lu, 1]
        # 对response中的pad进行mask [B, 1, Lr]
        resp_mask = get_pad_mask(response, self._params['padding_idx'])
        resp_embed_mask = resp_mask.squeeze(dim=-2).unsqueeze(dim=-1) # [B, Lr, 1]


        # --------------- Embedding层 -----------------
        ## [B, Nu, Lu, embed_output_dim]
        uttrs_embedding = self.word_emb(utterances)
        ## [B, Lr, embed_output_dim]
        resp_embedding = self.word_emb(response)
        ## 是否加上turn embedding
        if self._params['is_turn_emb']:
            ### [1, Nu, 1, embed_output_dim]
            turns_embedding = self.turn_emb(turns_num).unsqueeze(dim=-2)
            uttrs_embedding = uttrs_embedding + turns_embedding

        if self._params['is_position_emb']:
            ### [B*Nu, Lu, embed_output_dim]
            uttrs_embedding = self.position_emb(uttrs_embedding.view(bsz*self.turns, self.uttr_len, -1))
            ### [B, Lr, embed_output_dim]
            resp_embedding = self.position_emb(resp_embedding)
        ## [B*Nu, Lu, d_model]
        # uttrs_embedding = self.word_proj(uttrs_embedding)
        # ## [B, Lr, d_model]
        # resp_embedding = self.word_proj(resp_embedding)
        U_emb = uttrs_embedding * uttrs_embed_mask
        R_emb = resp_embedding * resp_embed_mask

        # -------------------- Attentive HR Encoder -----------------
        ## [B*Nu, Lu, NL, 2*hid]
        U_stack = self.stack_brnn(U_emb, uttrs_embed_mask)
        ## [B, Lr, NL, 2*hid]
        R_stack = self.stack_brnn(R_emb, resp_embed_mask)
        ## [NL, 1]
        wm = F.softmax(self.w_m, dim=0).unsqueeze(-1)
        ## 对utterance各个层进行Attention [B*Nu, Lu, 2*hid]
        U_attn = torch.matmul(U_stack.transpose(2, 3), wm).squeeze(dim=-1)
        ## 对response的各个层进行Attention [B, Lr, 2*hid]
        R_attn = torch.matmul(R_stack.transpose(2, 3), wm).squeeze(dim=-1)

        # ------------------ Matching Layer -------------------------

        U_attn_reshape = U_attn.view(bsz, self.turns*self.uttr_len, -1)
        ## [B, Nu*Lu, 2*hid]
        U_R_attn, *_ = self.cross_attn(U_attn_reshape, R_attn, R_attn, resp_mask)
        ## [B*Nu, Lu, 2*hid]
        U_R_attn = U_R_attn.view(bsz*self.turns, self.uttr_len, -1)
        ## [B, Lr, 2*hid]
        R_U_attn, *_ = self.cross_attn(R_attn, U_attn_reshape, U_attn_reshape, uttrs_mask.squeeze(dim=1).view(bsz, self.turns*self.uttr_len).unsqueeze(dim=-2))
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
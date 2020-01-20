#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: msn.py
@time: 2019/12/25 15:03
@description: Paper - https://www.aclweb.org/anthology/D19-1011.pdf | Reference Code - https://github.com/chunyuanY/Dialogue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from snlp.base.base_model import BaseModel
from snlp.params.param import Param
from snlp.params.param_table import ParamTable
from snlp.tools.parse import parse_activation
from snlp.modules.transformer import Encoder
from snlp.modules.transformer.layers import CrossAttentionLayer
from snlp.modules.transformer.models import get_pad_mask, PositionalEncoding

class MSN(BaseModel):
    def __init__(self, uttr_len=20, resp_len=20, turns=5):
        self.uttr_len = uttr_len
        self.resp_len = resp_len
        self.turns = turns
        super(MSN, self).__init__()

    # 设置默认参数
    def get_default_params(self) -> ParamTable:
        params = super().get_default_params(with_embedding=True,
                                            with_multi_layer_perceptron=False)
        params['embedding_freeze'] = False
        # --------------------- 定义一些标志位 --------------------
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
            desc="Whether to use layer normalization when output from self attention."
        ))

        params.add(Param(
            name='bidirectional', value=True,
            desc="Whether to use bidirectional in the final GRU layer."
        ))
        params.add(Param(
            name='maxpool_output', value=True,
            desc="Whether to concat maxpool vector with the final hidden vector."
        ))

        # --------------------- 设置模型参数 ---------------------
        ## ***** 自注意力层和交叉注意力层 *****
        params.add(Param(
            name='n_position', value=200,
            desc="The max number of position embedding."
        ))
        params.add(Param(
            name='d_model', value=512,
            desc="The dim of the input vector of each word."
        ))
        params.add(Param(
            name='d_k', value=64,
            desc="The dim of key in each head."
        ))
        params.add(Param(
            name='d_v', value=64,
            desc="The dim of value in each head."
        ))
        params.add(Param(
            name='n_head', value=8,
            desc="The number of multi heads."
        ))
        params.add(Param(
            name='d_inner', value=768,
            desc="The hidden size of FFN."
        ))
        params.add(Param(
            name='dropout', value=0.1,
            desc="The dropout rate."
        ))

        ## ***** Word Selector参数 *****
        params.add(Param(
            name='ws_hid', value=20,
            desc="The hidden size use to catch word similarity."
        ))
        params.add(Param(
            name='hop_k', value=[1,2,3]
        ))

        ## ***** Context Selector参数 *****
        params.add(Param(
            name='alpha', value=0.5,
            desc="The ratio of the word selector, the rest is the utterance selector."
        ))
        params.add(Param(
            name='gamma', value=0.3,
            desc="The threshold of the context selector to preserve."
        ))


        ## ***** 卷积层和池化层 *****
        ## 第一个卷积层
        params.add(Param(
            name='conv1_out_channels', value=64,
            desc="The output channels of the first conv."
        ))
        params.add(Param(
            name='conv1_kernel_size', value=(3, 3),
            desc="The kernel size of the first conv, keep the size invariable."
        ))
        params.add(Param(
            name='conv1_padding', value=(1, 1),
            desc="The padding size of the first conv, keep the size invariable."
        ))

        ## 第一个constant padding和池化层
        cp1_padding_bottom = (3 - self.uttr_len % 3) if self.uttr_len % 3 !=0 else 0
        cp1_padding_right = (3 - self.resp_len % 3) if self.resp_len % 3 != 0 else 0
        cp1_padding = (0, cp1_padding_right, 0, cp1_padding_bottom)
        params.add(Param(
            name='cp1_padding', value=cp1_padding,
            desc="The padding size of the first constant padding."
        ))

        params.add(Param(
            name='maxpool1_kernel_size', value=(3, 3),
            desc="The kernel size of the first maxpool, make the size become 1/3."
        ))
        params.add(Param(
            name='maxpool1_stride', value=(3, 3),
            desc="The stride of the first maxpool, make the size become 1/3."
        ))

        ## 第二个卷积层
        params.add(Param(
            name='conv2_out_channels', value=256,
            desc="The output channels of the second conv."
        ))
        params.add(Param(
            name='conv2_kernel_size', value=(3, 3),
            desc="The kernel size of the second conv, keep the size invariable."
        ))
        params.add(Param(
            name='conv2_padding', value=(1, 1),
            desc="The padding size of the second conv, keep the size invariable."
        ))

        ## 第二个constant padding和池化层
        prev_output_h = (self.uttr_len + cp1_padding_bottom) // 3
        prev_output_w = (self.resp_len + cp1_padding_right) // 3
        cp2_padding_bottom = (2 - prev_output_h % 2) if prev_output_h % 2 != 0 else 0
        cp2_padding_right = (2 - prev_output_w % 2) if prev_output_w % 2 != 0 else 0
        cp2_padding = (0, cp2_padding_right, 0, cp2_padding_bottom)
        params.add(Param(
            name='cp2_padding', value=cp2_padding,
            desc="The padding size of the second constant padding."
        ))

        params.add(Param(
            name='maxpool2_kernel_size', value=(2, 2),
            desc="The kernel size of the second maxpool, make the size become 1/2."
        ))
        params.add(Param(
            name='maxpool2_stride', value=(2, 2),
            desc="The stride of the second maxpool, make the size become 1/2."
        ))

        ## 第三个卷积层
        params.add(Param(
            name='conv3_out_channels', value=512,
            desc="The output channels of the third conv."
        ))
        params.add(Param(
            name='conv3_kernel_size', value=(3, 3),
            desc="The kernel size of the third conv, keep the size invariable."
        ))
        params.add(Param(
            name='conv3_padding', value=(1, 1),
            desc="The padding size of the third conv, keep the size invariable."
        ))

        ## 第三个池化层
        last_output_h = (prev_output_h + cp2_padding_bottom) // 2
        last_output_w = (prev_output_w + cp2_padding_right) // 2
        params.add(Param(
            name='maxpool3_kernel_size', value=(last_output_h, last_output_w),
            desc="The kernel size of the third maxpool, make the (h, w) become (1, 1)."
        ))
        params.add(Param(
            name='maxpool3_stride', value=(last_output_h, last_output_w),
            desc="The stride of the third maxpool, make the (h, w) become (1, 1)."
        ))

        ## ***** 输出层 *****
        params.add(Param(
            name='linear_in', value=512,
            desc="The input size of the linear layer, equals to the covn3 output channels."
        ))
        params.add(Param(
            name='linear_out', value=300,
            desc="The output size of the linear layer."
        ))
        params.add(Param(
            name='gru_hid', value=128,
            desc="The hidden size of the last GRU."
        ))
        return params

    def build(self):
        # 构建Embedding层，默认的输入向量的维度为200，这里得到单词到向量的映射矩阵
        self.word_emb = self._make_default_embedding_layer()

        if self._params['is_turn_emb']:
            self.turn_emb = nn.Embedding(self.turns+1, self._params['d_model'],
                                         padding_idx=self._params['padding_idx'])
        if self._params['is_position_emb']:
            self.position_emb = PositionalEncoding(self._params['d_model'],
                                                   n_position=self._params['n_position'])

        self.word_proj = nn.Linear(self._params['embedding_output_dim'],
                                   self._params['d_model'], bias=False)
        # 用于对输入词向量的强化编码
        self.trans_emb = Encoder(n_src_vocab=0, d_word_vec=self._params['d_model'],
                                 n_layers=1,
                                 n_head=self._params['n_head'],
                                 d_k=self._params['d_k'],
                                 d_v=self._params['d_v'],
                                 d_model=self._params['d_model'],
                                 d_inner=self._params['d_inner'],
                                 pad_idx=self._params['padding_idx'],
                                 n_position=self._params['n_position'],
                                 is_word_embedding=False,
                                 is_pos_embedding=False,
                                 dropout=self._params['dropout'],
                                 is_layer_norm=self._params['is_layer_norm'])
        # ---------- Word Selector 参数---------------
        ## 计算Key和每一个utterance的对齐矩阵
        self.W_word = nn.Parameter(data=torch.Tensor(self._params['d_model'],
                                                     self._params['d_model'],
                                                     self._params['ws_hid']))
        self.v = nn.Parameter(data=torch.Tensor(self._params['ws_hid'], 1))
        self.ws_linear = nn.Linear(2*self.uttr_len, 1)
        self.ws_score = nn.Linear(len(self._params['hop_k']), 1)

        # --------- Matching and Aggregation ------------------
        self.slf_att = CrossAttentionLayer(d_model=self._params['d_model'],
                                         d_inner=self._params['d_inner'],
                                         n_head=self._params['n_head'],
                                         d_k=self._params['d_k'],
                                         d_v=self._params['d_v'],
                                         dropout=self._params['dropout'],
                                         is_layer_norm=self._params['is_layer_norm'])
        # self.R_slf = CrossAttentionLayer(d_model=self._params['d_model'],
        #                                  d_inner=self._params['d_inner'],
        #                                  n_head=self._params['n_head'],
        #                                  d_k=self._params['d_k'],
        #                                  d_v=self._params['d_v'],
        #                                  dropout=self._params['dropout'],
        #                                  is_layer_norm=self._params['is_layer_norm'])
        self.cross_att = CrossAttentionLayer(d_model=self._params['d_model'],
                                           d_inner=self._params['d_inner'],
                                           n_head=self._params['n_head'],
                                           d_k=self._params['d_k'],
                                           d_v=self._params['d_v'],
                                           dropout=self._params['dropout'],
                                           is_layer_norm=self._params['is_layer_norm'])
        # self.R_U_att = CrossAttentionLayer(d_model=self._params['d_model'],
        #                                    d_inner=self._params['d_inner'],
        #                                    n_head=self._params['n_head'],
        #                                    d_k=self._params['d_k'],
        #                                    d_v=self._params['d_v'],
        #                                    dropout=self._params['dropout'],
        #                                    is_layer_norm=self._params['is_layer_norm'])
        self.A1 = nn.Parameter(data=torch.Tensor(self._params['d_model'],
                                                 self._params['d_model']))
        self.A2 = nn.Parameter(data=torch.Tensor(self._params['d_model'],
                                                 self._params['d_model']))
        self.A3 = nn.Parameter(data=torch.Tensor(self._params['d_model'],
                                                 self._params['d_model']))

        # ------------------ Convolution Layer --------------------
        self.cnn2d_1 = nn.Conv2d(in_channels=6,
                                 out_channels=self._params['conv1_out_channels'],
                                 kernel_size=self._params['conv1_kernel_size'],
                                 padding=self._params['conv1_padding'])
        self.cp1 = nn.ConstantPad2d(padding=self._params['cp1_padding'], value=0)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=self._params['maxpool1_kernel_size'],
                                      stride=self._params['maxpool1_stride'])
        self.cnn2d_2 = nn.Conv2d(in_channels=self._params['conv1_out_channels'],
                                 out_channels=self._params['conv2_out_channels'],
                                 kernel_size=self._params['conv2_kernel_size'],
                                 padding=self._params['conv2_padding'])
        self.cp2 = nn.ConstantPad2d(padding=self._params['cp2_padding'], value=0)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=self._params['maxpool2_kernel_size'],
                                      stride=self._params['maxpool2_stride'])
        self.cnn2d_3 = nn.Conv2d(in_channels=self._params['conv2_out_channels'],
                                 out_channels=self._params['conv3_out_channels'],
                                 kernel_size=self._params['conv3_kernel_size'],
                                 padding=self._params['conv3_padding'])
        self.maxpool_3 = nn.MaxPool2d(kernel_size=self._params['maxpool3_kernel_size'],
                                      stride=self._params['maxpool3_stride'])

        # ----------------- Output Layer ------------------------------
        self.affine = nn.Linear(in_features=self._params['linear_in'],
                                out_features=self._params['linear_out'])
        self.gru = nn.GRU(input_size=self._params['linear_out'],
                          hidden_size=self._params['gru_hid'],
                          batch_first=True,
                          bidirectional=self._params['bidirectional'])
        ## 如果是单向的并且不拼接最终outputs的max-pool
        if not self._params['bidirectional'] and not self._params['maxpool_output']:
            in_features = self._params['gru_hid']
        ## 如果是双向的并且拼接最终outputs的max-pool
        elif self._params['bidirectional'] and self._params['maxpool_output']:
            in_features = 4 * self._params['gru_hid']
        else:
            in_features = 2 * self._params['gru_hid']

        self.output = self._make_output_layer(in_features=in_features)

        self.dropout = nn.Dropout(self._params['dropout'])
        self.init_weights()

    def init_weights(self):
        # 对一些参数矩阵进行初始化
        init.uniform_(self.W_word)
        init.uniform_(self.v)
        init.uniform_(self.ws_linear.weight)
        init.uniform_(self.ws_score.weight)

        init.xavier_normal_(self.A1)
        init.xavier_normal_(self.A2)
        init.xavier_normal_(self.A3)
        init.xavier_normal_(self.word_proj.weight)
        init.xavier_normal_(self.cnn2d_1.weight)
        init.xavier_normal_(self.cnn2d_2.weight)
        init.xavier_normal_(self.cnn2d_3.weight)
        init.xavier_normal_(self.affine.weight)
        init.xavier_normal_(self.output.weight)
        for weight in [self.gru.weight_hh_l0, self.gru.weight_ih_l0]:
            init.orthogonal_(weight)

    def word_selector(self, key, context):
        """
        进行单词级别的Selector
        :param key: [B, Lu, d_model]
        :param context: [B, Nu, Lu, d_model]
        :return:
        """
        dk = torch.sqrt(torch.Tensor([self._params['d_model']])).to(context.device)
        # [B, Nu, Lu, Lu, H]
        A = torch.tanh(torch.einsum("blrd,ddh,bud->blruh", context, self.W_word, key) / dk)
        # [B, Nu, Lu, Lu]
        A = torch.einsum("blruh,hp->blrup", A, self.v).squeeze()
        # [B, Nu, 2*Lu]
        a = torch.cat([A.max(dim=2)[0], A.max(dim=3)[0]], dim=-1)
        # [B, Nu]
        s1 = F.softmax(self.ws_linear(a).squeeze(), dim=-1)
        return s1

    def utterance_selector(self, key, context):
        """
        进行句子级别的selector
        :param key: [B, Lu, d_model]
        :param context: [B, Nu, Lu, d_model]
        :return:
        """
        # [B, d_model]
        key = key.mean(dim=1)
        # [B, Nu, d_model]
        context = context.mean(dim=2)
        # [B, Nu]
        s2 = torch.einsum("bld,bd->bl", context, key) / (1e-6 + torch.norm(context, dim=-1)*torch.norm(key,dim=-1,keepdim=True))
        return s2

    def context_selector(self, context, context_mask, hop=[1,2,3]):
        """
        上下文选择器，分别基于word和utterance两种方式
        :param context: [B, Nu, Lu, d_model]
        :param context_mask: [B*Nu, 1, Lu]
        :param hop:
        :return:
        """
        b, nu, lu, d_model = context.size()
        # [B*Nu, Lu, d_model]
        context_ = context.view(-1, lu, d_model)
        # 每个utterance进行自注意力编码
        context_ = self.trans_emb(context_, context_mask)[0]
        context_ = context_.view(b, nu, lu, d_model)
        context_mask = context_mask.view(b, nu, 1, lu)

        # 保存不同hop的加权值
        multi_match_score = []
        for hop_i in hop:
            ## [B, Lu, d_model]
            key = context[:, self.turns-hop_i:, :, :].mean(dim=1)
            ## [B, 1, Lu]
            key_mask = context_mask[:, self.turns-hop_i:, :, :].sum(dim=1)
            key_mask = (key_mask != self._params['padding_idx'])
            ## [B, Lu, d_model]
            key = self.trans_emb(key, key_mask)[0]
            ## [B, Nu]
            s1 = self.word_selector(key, context_)
            s2 = self.utterance_selector(key, context_)
            s = self._params['alpha'] * s1 + (1 - self._params['alpha']) * s2
            multi_match_score.append(s)
        # 多个hop选择器融合得到注意力之后的表征
        ## [B, Nu, hop_k]
        multi_match_score = torch.stack(multi_match_score, dim=-1)
        ## [B, Nu]
        match_score = self.ws_score(multi_match_score).squeeze()
        mask = (match_score.sigmoid() >= self._params['gamma']).float()
        match_score = match_score * mask
        ## [B, Nu, Lu, d_model]
        context = context * match_score.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return context

    def distance(self, A, B, C, epsilon=1e-6):
        """
        计算bilinear和cosine，表示向量的相似度矩阵，并进行拼接
        :param A: [B*Nu, Lu, d_model]，也就是utterance编码的结果
        :param B: bilinear的参数矩阵
        :param C: [B*Nu, Lr, d_model]，也就是response编码的结果
        :param epsilon: 计算cosine相似度时的平滑参数
        :return:
        """
        # 计算bilinear积
        ## [B*Nu, Lu, Lr]
        M1 = torch.einsum("bud,dd,brd->bur", A, B, C)
        # 计算cosine
        A_norm = A.norm(dim=-1)
        C_norm = C.norm(dim=-1)
        ## [B*Nu, Lu, Lr]
        M2 = torch.einsum("bud,brd->bur", A, C) / (torch.einsum("bu,br->bur", A_norm, C_norm) + epsilon)
        return M1, M2

    def get_matching_map(self, U_embed, R_embed, U_mask, R_mask):
        """
        获取匹配向量
        :param U_embed: [B*Nu, Lu, d_model]
        :param R_embed: [B*Nu, Lr, d_model]
        :param U_mask: [B*Nu, 1, Lu]，忽略自编码和交叉编码时key中的padding
        :param R_mask: [B*Nu, 1, Lr]，忽略自编码和交叉编码时response中的padding
        :return: [B*Nu, Lu, Lr]
        """
        # [B*Nu, Lu, Lr]
        M1, M2 = self.distance(U_embed, self.A1, R_embed)

        slf_U, _ = self.slf_att(U_embed, U_embed, U_embed, U_mask)
        slf_R, _ = self.slf_att(R_embed, R_embed, R_embed, R_mask)
        # [B*Nu, Lu, Lr]
        M3, M4 = self.distance(slf_U, self.A2, slf_R)

        cross_UR, _ = self.cross_att(U_embed, R_embed, R_embed, R_mask)
        cross_RU, _ = self.cross_att(R_embed, U_embed, U_embed, U_mask)
        # [B*Nu, Lu, Lr]
        M5, M6 = self.distance(cross_UR, self.A3, cross_RU)
        # [B*Nu, channel, Lu, Lr]
        M = torch.stack([M1, M2, M3, M4, M5, M6], dim=1)
        return M

    def UR_matching(self, U_embed, R_embed, U_mask, R_mask):
        """
        获取匹配张量
        :param U_embed: [B*Nu, Lu, d_model]
        :param R_embed: [B*Nu, Lr, d_model]
        :param U_mask: [B*Nu, 1, Lu]
        :param R_mask: [B*Nu, 1, Lr]
        :return: [B*Nu, final_channel]
        """
        # [B*Nu, channel, Lu, Lr]
        M = self.get_matching_map(U_embed, R_embed, U_mask, R_mask)

        # 第一层卷积层
        Z = F.relu(self.cnn2d_1(M))
        Z = self.maxpool_1(self.cp1(Z))

        # 第二层卷积层
        Z = F.relu(self.cnn2d_2(Z))
        Z = self.maxpool_2(self.cp2(Z))

        # 第三层卷积层
        Z = F.relu(self.cnn2d_3(Z))
        Z = self.maxpool_3(Z)

        # 转换成二维张量 [B*Nu, linear_in]
        Z = Z.view(Z.size(0), -1)

        # 转换成线性层的神经元 [B*Nu, linear_out]
        V = F.tanh(self.affine(Z))
        return V

    def forward(self, inputs):
        """
        NU表示utterance的轮次，Lu表示一个utterance的长度，Lr表示一个response的长度
        """
        # [B, Nu, Lu]
        utterances = inputs['utterances']
        # [B, Lr]
        response = inputs['response']
        bsz = utterances.size(0)
        # 获取用于表示turn的张量 [1, Nu]
        turns_num = inputs['turns']
        # 对utterance中的pad进行mask [B*Nu, 1, Lu]
        uttrs_mask = get_pad_mask(utterances.view(-1, self.uttr_len), self._params['padding_idx'])
        # 对response中的pad进行mask [B, 1, Lr]
        resp_mask = get_pad_mask(response, self._params['padding_idx'])
        # [B, Nu, Lu, embed_output_dim]
        word_embedding = self.word_emb(utterances)
        # [B*Nu, Lu, d_model]
        U_emb = self.word_proj(word_embedding)
        # [B, Lr, embed_output_dim]
        resp_embedding = self.word_emb(response)
        # [B, Lr, d_model]
        R_emb = self.word_proj(resp_embedding)
        # 加上turn embedding
        if self._params['is_turn_emb']:
            ## [1, Nu, 1, embed_output_dim]
            turns_embedding = self.turn_emb(turns_num).unsqueeze(dim=-2)
            ## [B, Nu, Lu, embed_output_dim]
            U_emb = U_emb + turns_embedding
        if self._params['is_position_emb']:
            # [B*Nu, Lu, embed_output_dim]
            U_emb = self.position_emb(U_emb.view(bsz*self.turns, self.uttr_len, -1))
            # [B, Lr, embed_output_dim]
            R_emb = self.position_emb(R_emb)

        U_emb = U_emb * (uttrs_mask.squeeze(dim=-2).unsqueeze(dim=-1))
        R_emb = R_emb * (resp_mask.squeeze(dim=-2).unsqueeze(dim=-1))

        # 1. 首先经过context-selector，选择有用的上下文
        ## [B, Nu, Lu, d_model]
        multi_context = self.context_selector(U_emb.view(-1, self.turns, self.uttr_len,
                                                         self._params['d_model']),
                                              uttrs_mask,
                                              self._params['hop_k'])
        ## [B*Nu, Lu, d_model]
        multi_context = multi_context.view(-1, self.uttr_len,
                                           self._params['d_model'])
        ## [B*Nu, Lr, d_model]
        R_emb = R_emb.unsqueeze(dim=1).repeat(1, self.turns, 1, 1).view(-1, self.resp_len, self._params['d_model'])
        resp_mask = resp_mask.unsqueeze(dim=1).repeat(1, self.turns, 1, 1).view(-1, 1, self.resp_len)

        # 2. 经过几层卷积层，提取特征
        ## [B*Nu, linear_out]
        V = self.UR_matching(multi_context, R_emb, U_mask=uttrs_mask, R_mask=resp_mask)
        ## [B, Nu, linear_out]
        V = V.view(bsz, self.turns, -1)

        # 3. 经过GRU，并得到最终的分类结果
        ## [B, Nu, direction*hidden_size] 和 [direction, B, hidden_size]
        outputs, h = self.gru(V)
        ## [B, direction*hidden_size]
        output = h.transpose(0, 1).contiguous().view(bsz, -1)
        if self._params['maxpool_output']:
            ## [B, direction*hidden_size]
            maxpool_output = outputs.max(dim=1)[0]
            ## [B, 2*direction*hidden_size]
            output = torch.cat([output, maxpool_output], dim=-1)
        output = self.dropout(output)
        # [B, num_classes]
        output = self.output(output)
        return output
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: translator.py
@time: 2019/12/18 19:22
@description: 使用Beam Search进行文本生成
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from snlp.modules.transformer.models import Transformer, get_pad_mask, get_subsequent_mask


class Translator(nn.Module):
    """
    加载模型并按照Beam-Search的方式进行解码
    :param model: 模型，也就是Transformer
    :param beam_size: int型，beam的大小
    :param max_seq_len: int型，解码序列的最大长度
    :param src_pad_idx: int型，src中pad的索引
    :param trg_pad_idx: int型，tgt中pad的索引
    :param trg_bos_idx: int型，tgt中bos的索引
    :param trg_eos_idx: int型，tgt中eos的索引
    """

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        # 注册初始化序列，维度为[1, 1]，其中的元素为bos_idx
        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        # 注册空白序列，维度为[Beam_size, max_len]，用pad进行填充
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        # 将第一列设置为bos
        self.blank_seqs[:, 0] = self.trg_bos_idx
        # 长度映射 [1, max_len+1]
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        """
        表示根据目标输入进行解码
        :param trg_seq: 维度为 [B, Ls]
        :param enc_output: 表示编码的输出 [B, Lq, d_model]
        :param src_mask: 表示编码的mask [1, 1, Lq]
        :return:
        """
        # 获取对当前之后单词的mask
        ## 由于这里只包含当前词和其前面的词，不包含后面的词，所以不需要对pad进行mask
        ## 维度为 [B, 1, Ls]
        trg_mask = get_subsequent_mask(trg_seq)
        # 解码的结果，输出为 [B, Ls, d_model]
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        # 解码结果，输出为 [B, Ls, vocab_size]，表示每一个单词的概率分布
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)

    def _get_init_state(self, src_seq, src_mask):
        """
        获取初始状态
        :param src_seq: [1, 1]
        :param src_mask: [1, 1, 1]
        :return:
        """
        beam_size = self.beam_size
        # 计算得到编码输出 [1, 1, d_model]
        enc_output, *_ = self.model.encoder(src_seq, src_mask)
        # 由于只输入了初始状态，所以只能得到下一个词的概率 [1, 1, vocab_size]
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        # 获取最后一个位置概率最大的前beam_size个词，以及对应的概率
        # 这里的维度为[1, beam_size]和[1, beam_size]
        best_k_probs, best_k_idx = dec_output[:, -1].topk(beam_size)

        # 计算log概率，便于相加计算，得到[1, beam_size]
        scores = torch.log(best_k_probs)
        # scores = torch.log(best_k_probs.unsqueeze(0))
        # 生成序列的维度为 [beam_size, max_len]，第0列都是bos
        gen_seq = self.blank_seqs.clone().detach()
        # 第1列设置为概率最大的beam个单词对应的索引
        gen_seq[:, 1] = best_k_idx[0]
        # [beam_size, Lq, d_model]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        """
        :param gen_seq: 维度为[beam_size, max_len]
        :param dec_output: [beam_size, Ls, vocab_size]
        :param scores: [1, beam_size]
        :param step: int型，当前解码的单词的索引，从0开始，0位置表示bos
        :return:
        """

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        # [beam_size, vocab_size]，每一个beam下一个单词的概率分布
        ## 两个输出[beam_size, beam_size]
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)
        # 得到和当前单词相加之后的概率 [beam_size, beam_size]
        best_k2_probs = torch.log(best_k2_probs).view(beam_size, -1) + scores.transpose(0, 1)

        # Get the best k candidates from k^2 candidates.
        ## 计算最有的topk个值
        best_k_probs, best_k_idx_in_k2 = best_k2_probs.view(-1).topk(beam_size)
        # 计算这k个值在原矩阵中的行和列
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        # 得到topk个索引
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # 获取最大概率所在beam的前面所有step对应的索引
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # 保存当前step的最大概率索引
        gen_seq[:, step] = best_k_idx

        # 得到累计最佳概率值[1, beam_size]
        scores = best_k_probs.unsqueeze(0)

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        """
        对一句话进行解码
        :param src_seq: 维度为[1, Lq]
        :return:
        """
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            # 获取对src进行self-attention时的mask矩阵 [1, 1, Lq]
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            # 得到初始单词的索引，[beam_size, Lq, d_model]和[beam_size, max_seq_len]和[1, 1, beam_size]
            # 其中gen_seq的第0列都是bos，第1列都是最大概率索引
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0  # default
            # 直到最大长度
            for step in range(2, max_seq_len):  # decode up to max length
                # 输入为[beam_size, step]和[beam_size, Lq, d_model]和[1, 1, Lq]
                ## 输出为 [beam_size, step, vocab_size]，表示beam中每一个词的概率分布
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                # 更新gen_seq之前的位置以及当前的位置，更新scores中的值
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                ## 寻找是否存在结束标志符eos
                ## [beam_size, max_len]
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                ## 通过mask获取每一个序列eos对应的长度值 [beam_size, ]
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                ## 检查是否所有的beams都包含eos，这是解码停止的条件
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    # 将概率值除以对长度的惩罚，由于是除以长度的指数，所以是希望长度越短越好
                    ## 先经过除法得到的结果维度为[1, beam_size]，输出分数最大值对应的路径索引
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(1)
                    ans_idx = ans_idx.item()
                    break
        # 返回路径最大值对应的解码结果
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
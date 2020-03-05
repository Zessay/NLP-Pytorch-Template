#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: crf.py
@time: 2020/3/5 10:18
@description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_scalar(var):
    """将tensor转化为标量"""
    return var.view(-1).detach().tolist()[0]

def argmax(vec):
    """计算最大值索引"""
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx

def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_

# -------------------------- 定义CRF模块 --------------------------------------

class CRF(nn.Module):
    def __init__(self, tagset_size, tag_dict, device, is_bert=None):
        """
        定义CRF模块
        :param tagset_size: int型，表示标签的数量
        :param tag_dict: dict型，表示标签到索引的字典
        :param device:
        :param is_bert:
        """
        super(CRF, self).__init__()

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        if is_bert:
            self.START_TAG = "[CLS]"
            self.STOP_TAG = "[SEP]"

        self.tag_dict = tag_dict
        self.tagset_size = tagset_size
        self.device = device
        ## 定义状态转移矩阵
        ### 0轴表示目标tag，1轴表示源tag，转移概率矩阵表示从源tag转移到目标tag的概率分布
        self.transitions= torch.randn(tagset_size, tagset_size)
        ## 任何状态不可能转移到START
        self.transitions.detach()[self.tag_dict[self.START_TAG], :] = -10000
        ## STOP不可能转移到其他状态
        self.transitions.detach()[:, self.tag_dict[self.STOP_TAG]] = -10000
        self.transitions = self.transitions.to(device)
        self.transitions = nn.Parameter(self.transitions)

    def _forward_alg(self, feats, lens_):
        """
        计算前向概率
        :param feats: [B, L, tag_size]，表示BERT输出的每一个标签的概率分布，表示发射概率
        :param lens_: [B, ]，表示每一个句子的实际长度
        :return: [B, tag_size]，每个句子概率分布对应的得分
        """
        # 初始化概率分布
        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dict[self.START_TAG]] = 0.0

        # [B, L+1, tag_size]
        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=self.device
        )
        # 每一个句子的起始字符都是START，对应位置的发射概率为1，赋给初始位置
        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)
        ## [B, tag_size, tag_size]
        transitions = self.transitions.view(1, self.transitions.shape[0],
                                            self.transitions.shape[1]).repeat(feats.shape[0], 1, 1)
        # 对长度进行遍历
        for i in range(feats.shape[1]):
            ## 获取发射概率，[B, tag_size]
            emit_score = feats[:, i, :]
            ## 当前位置的发射概率+转移概率+前一个位置到当前位置为止的累计概率
            ## [B, tag_size, tag_size]
            tag_var = (
                emit_score[:,:, None].repeat(1, 1, transitions.shape[2])   # [B, tag_size, tag_size]
                + transitions                                              # [B, tag_size, tag_size]
                + forward_var[:, i, :][:, :, None].repeat(1, 1, transitions.shape[2]).transpose(2, 1)   # [B, tag_size, tag_size]
            )
            # 获取最有可能tag以及对应的转移概率
            ## [B, tag_size]
            max_tag_var, _ = torch.max(tag_var, dim=2)
            ## [B, tag_size, tag_size]
            tag_var = tag_var - max_tag_var[:, :, None].repeat(1, 1, transitions.shape[2])
            ## [B, tag_size]
            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))
            cloned = forward_var.clone()
            cloned[:, i+1, :] = max_tag_var + agg_
            forward_var = cloned
        # 计算到每一个结束位置的概率分布 [B, tag_size]
        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
        ## 加上转移到结束符的概率 [B, tag_size]
        terminal_var = forward_var + self.transitions[self.tag_dict[self.START_TAG]][None, :].repeat(forward_var.shape[0], 1)
        ## 得到对数概率和 [B, tag_size]
        alpha = log_sum_exp_batch(terminal_var)
        return alpha

    def _viterbi_decode(self, feats):
        """
        维特比解码
        :param feats: [L, tag_size]
        :return:
        """
        # 回溯点，记录每个位置对应的前一个最大概率的位置
        backpointers = []
        backscores = []
        scores = []
        ## 初始化概率，对应START概率最大，[1, tagset_size]
        init_vars = torch.FloatTensor(1, self.tagset_size).to(self.device).fill_(-10000.0)
        init_vars[0][self.tag_dict[self.START_TAG]] = 0
        forward_var = init_vars

        for feat in feats:
            ## 得到下一个标签对应的概率分布，[tag_size, tag_size]
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            ## 获取转移到当前tag的最大值对应的索引，[tag_size]
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            ## 获取排序好的解码序列 [tag_size]
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            ## [tag_size]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        # [tag_size]
        terminal_var = (
            forward_var + self.transitions[self.tag_dict[self.START_TAG]]
        )
        ## 由于原始序列中删除了开始和结束符，所以最后一个标签不可能是这两个
        terminal_var.detach()[self.tag_dict[self.START_TAG]] = -10000.0
        terminal_var.detach()[self.tag_dict[self.STOP_TAG]] = -10000.0
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        # 回溯获取最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id.item())
        best_scores = []
        ## 获取每一个位置对应的概率分布
        for backscore  in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            ## 对应位置的最大概率值
            best_scores.append(softmax[prediction].item())
            scores.append([elem.item() for elem in softmax.flatten()])
        ## 交换最大概率位置以及对应的分数
        swap_best_path, swap_max_score = (best_path[0], scores[-1].index(max(scores[-1])))
        scores[-1][swap_best_path], scores[-1][swap_max_score] = (
            scores[-1][swap_max_score],
            scores[-1][swap_best_path],
        )
        start = best_path.pop()
        assert start == self.tag_dictionary[self.START_TAG]
        ## 最后将路径翻转
        best_path.reverse()
        return best_scores, best_path, scores

    def _score_sentence(self, feats, tags, lens_):
        """

        :param feats: [B, L, tag_size]，表示BERT输出的结果，即每个位置的发射概率
        :param tags: [B, L]，表示实际的标签值
        :param lens_: [B]，表示每个句子的实际长度
        :return:
        """
        start = torch.LongTensor([self.tag_dict[self.START_TAG]]).to(self.device)
        stop = torch.LongTensor([self.tag_dict[self.STOP_TAG]]).to(self.device)
        # 起始和结束位置
        start = start[None, :].repeat(tags.shape[0], 1)
        stop = stop[None, :].repeat(tags.shape[0], 1)
        ## [B, L+1]
        pad_start_tags = torch.cat([start, tags], dim=1)
        pad_stop_tags = torch.cat([tags, stop], dim=1)
        for i in range(len(lens_)):
            ## 将实际长度之后的tag全部设置为STOP
            pad_stop_tags[i, lens_[i]:] = self.tag_dict[self.STOP_TAG]
        score = torch.FloatTensor(feats.shape[0]).to(self.device)
        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(self.device)
            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, :lens_[i]+1], pad_start_tags[i, :lens_[i]+1]
                ]
            ) + torch.sum(feats[i, r, tags[i, :lens_[i]]])
        # [B, ]
        return score

    def _obtain_labels(self, feature, id2label, input_lens):
        """
        获取解码结果
        :param feature:
        :param id2label:
        :param input_lens:
        :return:
        """
        tags = []
        all_tags = []
        for feats, length in zip(feature, input_lens):
            confidences, tag_seq, scores = self._viterbi_decode(feats[:length])
            tags.append([id2label[tag] for tag in tag_seq])
            all_tags.append([[id2label[score_id] for score_id, score in enumerate(score_dist)] for score_dist in scores])
        return tags, all_tags

    def _calculate_loss_old(self, features, lengths, tags):
        forward_score = self._forward_alg(features, lengths)
        gold_score = self._score_sentence(features, tags, lengths)
        score = forward_score - gold_score
        return score.mean()

    def calculate_loss(self, scores, tag_list, lengths):
        return self._calculate_loss_old(scores, lengths, tag_list)
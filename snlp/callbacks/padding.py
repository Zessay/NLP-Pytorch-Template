#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: padding.py
@time: 2019/12/3 14:10
@description: 填充对齐
'''
import typing
import numpy as np
from snlp.base.base_callback import BaseCallback
import snlp.tools.constants as constants

# 通常用于词级别的padding
def _padding_2D(input, output, mode: str='post'):
    """
    Pad the input 2D-tensor to the output 2D-tensor.

    :param input: 表示原始的二维list
    :param output: 表示定义的固定长度的二维Tensor
    :param mode: padding mode, can be 'pre' or 'post'
    :return:
    """
    batch_size = min(output.shape[0], len(input))
    pad_length = output.shape[1]
    if mode == 'post':
        for i in range(batch_size):
            end_pos = min(len(input[i]), pad_length)
            if end_pos > 0:
                output[i][:end_pos] = input[i][:end_pos]
    elif mode == 'pre':
        for i in range(batch_size):
            start_pos = min(len(input[i]), pad_length)
            if start_pos > 0:
                output[i][-start_pos:] = input[i][-start_pos:]
    else:
        raise ValueError(f"{mode} is not a valid pad mode, "
                         f"only `pre` and `post` allow. ")

# 通常用于字符级别的padding
def _padding_3D(input, output, word_mode: str='post', char_mode: str='post'):
    """
    Pad the input 3D-tensor to the output 3D-tensor.
    :param word_mode: 表示针对第1维的填充方式，即单词维度或者turn维度
    :param char_mode: 表示针对第2维的填充方式，即字符维度或者word维度
    """
    if word_mode not in ['post', 'pre']:
        raise ValueError(f"{char_mode} is not a valid char pad mode.")
    if char_mode not in ['post', 'pre']:
        raise ValueError(f"{word_mode} is not a valid word pad mode.")


    batch_size = min(output.shape[0], len(input))
    pad_1d_length = output.shape[1]  # 通常表示一个sentence中单词的数量
    pad_2d_length = output.shape[2]  # 通常表示一个word中字符的数量
    if char_mode == 'post':
        for i in range(batch_size):
            # 这里表示words的数量，或者turns的数量
            origin_len = len(input[i])
            len_d1 = min(origin_len, pad_1d_length)
            origin_list=  range(origin_len)
            turns_list = range(pad_1d_length-len_d1, pad_1d_length) if word_mode=='pre' else range(len_d1)
            zip_list = zip(origin_list[::-1], turns_list[::-1])
            for inj, ouj in zip_list:
                ## 这里表示一个单词中chars的数量，或者一个turn中words的数量
                end_pos = min(len(input[i][inj]), pad_2d_length)
                if end_pos > 0:
                    output[i][ouj][:end_pos] = input[i][inj][:end_pos]
    else: # 'pre'
        for i in range(batch_size):
            origin_len = len(input[i])
            len_d1 = min(origin_len, pad_1d_length)
            origin_list=  range(origin_len)
            turns_list = range(pad_1d_length-len_d1, pad_1d_length) if word_mode=='pre' else range(len_d1)
            zip_list = zip(origin_list[::-1], turns_list[::-1])
            for inj, ouj in zip_list:
                start_pos = min(len(input[i][inj]), pad_2d_length)
                if start_pos > 0:
                    output[i][ouj][-start_pos:] = input[i][inj][-start_pos:]

# -----------------------------------------------------------------------------
# UR模式下的padding
def _padding_3D_UR(input, output, ur_output, word_mode: str='post', char_mode: str='post'):
    if word_mode not in ['post', 'pre']:
        raise ValueError(f"{char_mode} is not a valid char pad mode.")
    if char_mode not in ['post', 'pre']:
        raise ValueError(f"{word_mode} is not a valid word pad mode.")

    batch_size = min(output.shape[0], len(input))
    pad_1d_length = output.shape[1]  # 通常表示一个sentence中单词的数量，对应turn的长度
    pad_2d_length = output.shape[2]  # 通常表示一个word中字符的数量，对应sentence的长度

    if char_mode == "post":
        for i in range(batch_size):
            # 表示原始turn的长度
            origin_len = len(input[i])
            len_d1 = min(origin_len, pad_1d_length)
            origin_list = range(0, origin_len, 2)
            turns_list = range(pad_1d_length-len_d1, pad_1d_length) if word_mode=='pre' else range(len_d1)
            zip_list = zip(origin_list[::-1], turns_list[::-1])
            for inj, ouj in zip_list:
                ## 这里表示turn中单词的数量
                if inj == origin_len-1:
                    cur_sent = input[i][inj]
                    cur_len = len(cur_sent)
                    ur_ = [0] * len(input[i][inj])
                else:
                    cur_sent = list(input[i][inj]) + list(input[i][inj+1])
                    cur_len = len(cur_sent)
                    ur_ = [0] * len(input[i][inj]) + [1] * len(input[i][inj+1])
                end_pos = min(cur_len, pad_2d_length)
                if end_pos > 0:
                    output[i][ouj][:end_pos] = cur_sent[:end_pos]
                    ur_output[i][ouj][:end_pos] = ur_[:end_pos]
    else:
        for i in range(batch_size):
            origin_len = len(input[i])
            len_d1 = min(origin_len, pad_1d_length)
            origin_list= range(0, origin_len, 2)
            turns_list = range(pad_1d_length-len_d1, pad_1d_length) if word_mode=='pre' else range(len_d1)
            zip_list = zip(origin_list[::-1], turns_list[::-1])
            for inj, ouj in zip_list:
                ## 这里表示turn中单词的数量
                if inj == origin_len-1:
                    cur_sent = input[i][inj]
                    cur_len = len(cur_sent)
                    ur_ = [0] * len(input[i][inj])
                else:
                    cur_sent = list(input[i][inj]) + list(input[i][inj+1])
                    cur_len = len(cur_sent)
                    ur_ = [0] * len(input[i][inj]) + [1] * len(input[i][inj+1])
                start_pos = min(cur_len, pad_2d_length)
                if start_pos > 0:
                    output[i][ouj][-start_pos:] = cur_sent[-start_pos:]
                    ur_output[i][ouj][-start_pos] = ur_[-start_pos:]


# -------------------- 只存在一个文本字段情况时的padding方法 ----------------------

class SingleBasicPadding(BaseCallback):
    def __init__(self,
                 fixed_length: int=None,
                 pad_word_value: typing.Union[int, str] = 0,
                 pad_word_mode: str='post',
                 with_ngram: bool=False,
                 fixed_ngram_length: int=None,
                 pad_ngram_value: typing.Union[int, str]=0,
                 pad_ngram_mode: str='post',
                 dtype=np.int32):
        self._fixed_length = fixed_length
        self._pad_word_value = pad_word_value
        self._pad_word_mode = pad_word_mode
        self._with_ngram = with_ngram
        self._fixed_ngram_length = fixed_ngram_length
        self._pad_ngram_value = pad_ngram_value
        self._pad_ngram_mode = pad_ngram_mode
        self._dtype = dtype

    def on_batch(self, x: dict, y: np.ndarray):
        """
        对只有一个文本字段进行pad.
        :param x: dict型，必须包含字段`text_len`, `text`，可选`ngram`字段
        :param y:
        :return:
        """
        batch_size = len(x[constants.TEXT_LEN])
        pad_length = max(x[constants.TEXT_LEN])
        if self._with_ngram:
            ngram_length = max([len(w)
                                for k in x[constants.NGRAM] for w in k])
            if self._fixed_ngram_length:
                ngram_length = self._fixed_ngram_length

        if self._fixed_length is not None:
            pad_length = self._fixed_length

        for key, value in x.items():
            if key == constants.TEXT:
                padded_value = np.full([batch_size, pad_length],
                                       self._pad_word_value, dtype=self._dtype)
                _padding_2D(value, padded_value, self._pad_word_mode)
            elif key == constants.NGRAM:
                padded_value = np.full([batch_size, pad_length, ngram_length],
                                       self._pad_ngram_value, dtype=self._dtype)
                _padding_3D(value, padded_value, word_mode=self._pad_word_mode, char_mode=self._pad_ngram_mode)
            else:
                continue
            x[key] = padded_value

# -------------------- 存在两个文本字段情况时的padding方法 ------------------------

class DoubleBasicPadding(BaseCallback):
    """
    Pad data for basic preprocessors.
    The data have two columns.
    """
    def __init__(self,
                 fixed_length_left: int=None,
                 fixed_length_right: int=None,
                 pad_word_value: typing.Union[int, str]=0,
                 pad_word_mode: str='post',
                 with_ngram: bool=False,
                 fixed_ngram_length: int=None,
                 pad_ngram_value: typing.Union[int, str]=0,
                 pad_ngram_mode: str='post',
                 dtype=np.int32):
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._pad_word_value = pad_word_value
        self._pad_word_mode = pad_word_mode
        self._with_ngram = with_ngram
        self._fixed_ngram_length = fixed_ngram_length
        self._pad_ngram_value = pad_ngram_value
        self._pad_ngram_mode = pad_ngram_mode
        self._dtype = dtype

    def on_batch(self, x: dict, y: np.ndarray):
        """
        对每个batch的文本进行padding
        :param x: dict型，必须包含字段`id_left`, `text_left_len`, `text_right_len`, `text_left`, `text_right`，
                  还有`ngram_left`和`ngram_right`可选.
        :param y: ndarray型，表示数据的标签值
        :return:
        """
        batch_size = len(x[constants.LEFT_LEN])
        pad_length_left = max(x[constants.LEFT_LEN])
        pad_length_right = max(x[constants.RIGHT_LEN])

        if self._with_ngram:
            ngram_length_left = max([len(w)
                                     for k in x[constants.NGRAM_LEFT] for w in k])
            ngram_length_right = max([len(w)
                                      for k in x[constants.NGRAM_RIGHT] for w in k])
            ngram_length = max(ngram_length_left, ngram_length_right)
            if self._fixed_ngram_length:
                ngram_length = self._fixed_ngram_length

        if self._fixed_length_left is not None:
            pad_length_left = self._fixed_length_left
        if self._fixed_length_right is not None:
            pad_length_right = self._fixed_length_right


        for key, value in x.items():
            if key == constants.TEXT_LEFT:
                padded_value = np.full([batch_size, pad_length_left],
                                       self._pad_word_value, dtype=self._dtype)
                _padding_2D(value, padded_value, self._pad_word_mode)
            elif key == constants.TEXT_RIGHT:
                padded_value = np.full([batch_size, pad_length_right],
                                       self._pad_word_value, dtype=self._dtype)
                _padding_2D(value, padded_value, self._pad_word_mode)
            elif key == constants.NGRAM_LEFT:
                padded_value = np.full([batch_size, pad_length_left, ngram_length],
                                       self._pad_ngram_value, dtype=self._dtype)
                _padding_3D(value, padded_value, word_mode=self._pad_word_mode, char_mode=self._pad_ngram_mode)
            elif key == constants.NGRAM_RIGHT:
                padded_value = np.full([batch_size, pad_length_right, ngram_length],
                                       self._pad_ngram_value, dtype=self._dtype)
                _padding_3D(value, padded_value, word_mode=self._pad_word_mode, char_mode=self._pad_ngram_mode)
            else:
                continue
            x[key] = padded_value


# -------------------- 对于Multi-QA的padding ------------------------

class MultiQAPadding(BaseCallback):
    """
    Pad the data for Multi-QA data.
    """
    def __init__(self,
                 fixed_length_uttr: typing.Optional[int]=None,
                 fixed_length_resp: typing.Optional[int]=None,
                 fixed_length_turn: typing.Optional[int]=None,
                 char: bool=False,   # 是否需要对char进行填充
                 fixed_length_uttr_char: typing.Optional[int]=None,
                 fixed_length_resp_char: typing.Optional[int]=None,
                 pad_word_value: typing.Union[int, str]=0,
                 pad_word_mode: str='post',
                 data_type: str='uur',
                 dtype=np.int32):
        self._fixed_length_uttr = fixed_length_uttr
        self._fixed_length_resp = fixed_length_resp
        self._fixed_length_turn = fixed_length_turn
        self._fixed_length_uttr_char = fixed_length_uttr_char
        self._fixed_length_resp_char = fixed_length_resp_char
        self._pad_word_value = pad_word_value
        self._pad_word_mode = pad_word_mode
        self._char = char
        self._dtype = dtype
        self._data_type = data_type

    def on_batch(self, x: dict, y: np.ndarray):
        """
        对每个batch中的文本进行padding
        :param x: dict型，必须包含字段`utterances`, `response`, `utterances_len`, `response_len`
        :param y: ndarray型，表示数据的标签值
        :return:
        """
        batch_size = len(x[constants.RESP])
        pad_length_uttr = max(x[constants.UTTRS_LEN])
        pad_length_resp = max(x[constants.RESP_LEN])
        pad_length_turn = self._fixed_length_turn
        if self._char:
            pad_length_uttr_char = max(x[constants.UTTRS_CHAR_LEN])
            pad_length_resp_char = max(x[constants.RESP_CHAR_LEN])



        if self._fixed_length_uttr is not None:
            pad_length_uttr = self._fixed_length_uttr
        if self._fixed_length_resp is not None:
            pad_length_resp = self._fixed_length_resp
        if self._fixed_length_uttr_char is not None:
            pad_length_uttr_char = self._fixed_length_uttr_char
        if self._fixed_length_resp_char is not None:
            pad_length_resp_char = self._fixed_length_resp_char


        for key, value in x.items():
            if key == constants.UTTRS:
                padded_value = np.full([batch_size, pad_length_turn, pad_length_uttr],
                                       self._pad_word_value, dtype=self._dtype)
                if self._data_type == "uru":
                    ur_output = np.full([batch_size, pad_length_turn, pad_length_uttr],
                                       self._pad_word_value, dtype=self._dtype)

                    _padding_3D_UR(value, padded_value, ur_output=ur_output, word_mode="pre", char_mode=self._pad_word_mode)
                else:
                    _padding_3D(value, padded_value, word_mode='pre', char_mode=self._pad_word_mode)
            elif key == constants.RESP:
                padded_value = np.full([batch_size, pad_length_resp],
                                       self._pad_word_value, dtype=self._dtype)
                _padding_2D(value, padded_value, mode=self._pad_word_mode)
            elif key == constants.UTTRS_CHAR:
                padded_value = np.full([batch_size, pad_length_turn, pad_length_uttr_char],
                                       self._pad_word_value, dtype=self._dtype)
                _padding_3D(value, padded_value, word_mode='pre', char_mode=self._pad_word_mode)
            elif key == constants.RESP_CHAR:
                padded_value = np.full([batch_size, pad_length_resp_char],
                                       self._pad_word_value, dtype=self._dtype)
                _padding_2D(value, padded_value, mode=self._pad_word_mode)
            else:
                continue
            x[key] = padded_value


        if self._data_type == "uru":
            x[constants.UR_POS] = ur_output
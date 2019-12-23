#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: embedding.py
@time: 2019/11/27 20:17
@description: 用于构建embedding矩阵的类
'''

import typing
import numpy as np
from snlp.base.units.vocabulary import TermIndex


class Embedding(object):
    """Building embedding matrix accorading to term_index."""
    def __init__(self, embedding: dict, output_dim: int):
        self.word_vectors = embedding
        self.emb_size = output_dim


    def build_matrix(self, term_index: typing.Union[dict, TermIndex]) -> np.ndarray:
        vocab_size = len(term_index)
        matrix = np.empty((vocab_size, self.emb_size))
        valid_words = self.word_vectors.keys()

        for term, index in term_index.items():
            if term in valid_words:
                matrix[index] = self.word_vectors[term]
            else:
                matrix[index] = np.random.uniform(-0.2, 0.2, size=self.emb_size)
        return matrix


def load_from_file(file_path: str, mode: str='word2vec') -> Embedding:
    """加载各种形式的词向量"""
    embedding_data = {}
    if mode == 'word2vec' or mode == 'fasttext':
        with open(file_path, 'r', encoding='utf-8') as f:
            output_dim = int(f.readline().strip().split()[-1])
            for line in f:
                current_line = line.rstrip().split()
                embedding_data[current_line[0]] = np.array(current_line[1:], dtype=np.float32)
    elif mode == 'glove':
        with open(file_path, 'r', encoding='utf-8') as f:
            output_dim = len(f.readline().rstrip().split()) - 1
            f.seek(0)
            for line in f:
                current_line = line.rstrip().split()
                embedding_data[current_line[0]] = np.array(current_line[1:], dtype=np.float32)
    else:
        raise TypeError(f"{mode} is not a supported embedding type. "
                        f"'word2vec', 'fasttext' or 'glove' expected.")

    return Embedding(embedding_data, output_dim)
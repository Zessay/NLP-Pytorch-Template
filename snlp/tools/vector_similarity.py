#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: vector_similarity.py
@time: 2020/1/9 14:57
@description: 
'''
import numpy as np
from scipy.spatial.distance import cdist


def cosine_similarity(vec1, vec2, epsilon=1e-5):
    '''
    计算两个向量之间的余弦相似度
    :param vec1:
    :param vec2:
    :return:
    '''
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(np.sum(tx**2))
    cos22 = np.sqrt(np.sum(ty**2))
    cosine_value = cos1 / (float(cos21*cos22) + epsilon)
    cosine_value = (cosine_value + 1) / 2   # 标准化到0-1之间
    return cosine_value


# 计算wmd的函数
def relaxed_wmd(doc1_vectors, doc2_vectors, use_distance=False, distance_matrix=None):
    '''
    @params: list型
    doc1_vectors: doc1的词向量矩阵
    doc2_vectors: doc2的词向量矩阵
    use_distance: 如果是True，则使用欧式距离；否则使用cosine相似度
    '''
    if distance_matrix is None:
        if len(doc1_vectors) == 0 or len(doc2_vectors) == 0:
            if use_distance:
                return np.inf
            else:
                return 0

        if use_distance:
            distance_matrix = cdist(doc1_vectors, doc2_vectors, "euclidean")
        else:
            distance_matrix = cdist(doc1_vectors, doc2_vectors, "cosine")
            # 对相似度的值进行规范化
            distance_matrix = np.round(distance_matrix, 5)
            distance_matrix = np.nan_to_num(distance_matrix)
            distance_matrix[distance_matrix == np.inf] = 0.
            distance_matrix = distance_matrix.clip(min=0.)
            # 将距离转化为相似度
            distance_matrix = 1 - distance_matrix

    # 如果使用欧氏距离，需要将距离转换为相似度
    if use_distance:
        score = np.mean(np.min(distance_matrix, 1))
        return 1. / (1. + score)
    else:
        return max(0, np.mean(np.max(distance_matrix, 1)))


def relaxed_wmd_combined(doc1_vectors, doc2_vectors, distance=False,
                         combination="mean", return_parts=False):
    if len(doc1_vectors) == 0 or len(doc2_vectors) == 0:
        if distance:
            return np.inf if not return_parts else np.inf, np.inf, np.inf
        else:
            return 0. if not return_parts else 0., 0., 0.

    if distance:
        D = cdist(doc1_vectors, doc2_vectors, "euclidean")
    else:
        D = cdist(doc1_vectors, doc2_vectors, "cosine")
        D = np.round(D, 5)
        D = np.nan_to_num(D)
        D[D == np.inf] = 0.
        D = D.clip(min=0.)
        D = 1 - D

    # 计算前向和反向的RWMD
    l1 = relaxed_wmd(doc1_vectors, doc2_vectors, use_distance=distance, distance_matrix=D)
    l2 = relaxed_wmd(doc1_vectors, doc2_vectors, use_distance=distance, distance_matrix=D.T)

    # 将两种形式的RWMD进行合并
    if combination == "mean":
        combined = np.mean([l1, l2])
    elif combination == "min":
        combined = np.min([l1, l2])
    elif combination == "max":
        combined = np.max([l1, l2])

    if return_parts:
        return combined, l1, l2
    else:
        return combined



if __name__ == "__main__":
    a = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 5], [2, 3, 4, 2, 1], [3, 4, 5, 2, 1]]
    b = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 3, 4 ,1, 2], [2, 1, 6, 2, 1]]
    print(relaxed_wmd_combined(a, b))
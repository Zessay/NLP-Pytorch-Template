#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: negtive_sampling.py
@time: 2020/3/30 16:14
@description: 
'''
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

def neg_sampling(data:pd.DataFrame, n_neg: int=4,
                 text_left: str="text_left", text_right: str="text_right",
                 label: str="label") -> pd.DataFrame:

    data.rename(columns={text_left: 'text_left', text_right: 'text_right', label: 'label'}, inplace=True)
    result = pd.DataFrame(data={'text_left': ['placeholder'], 'text_right': ['placeholder'], 'label': [0]})
    candidates = data['text_right'].values
    length = len(candidates)

    for row in tqdm(data.itertuples()):
        text = row.text_left
        cur_label = row.label
        if cur_label:
            ## 将样本添加到结果中
            result.loc[result.shape[0]] = {'text_left': text, 'text_right': row.text_right,
                                           'label': row.label}
            pos_samples = data[data['text_left']==text]['text_right'].values
            for _ in range(n_neg):
                neg_sample = pos_samples[0]
                while neg_sample in pos_samples:
                    seg_index = np.random.randint(length)
                    low = max(0, seg_index-20)
                    high = min(length, seg_index+20)
                    index = np.random.randint(low, high)
                    neg_sample = candidates[index]
                result.loc[result.shape[0]] = {'text_left': text, 'text_right': neg_sample, 'label': 0}

    result = result[1:]
    return result



if __name__ == "__main__":
    basename = "/home/speech/data/sing_qa"
    file = "train_persona.csv"

    data = pd.read_csv(Path(basename) / file)

    n_neg = 4
    print("正在进行负采样...")
    result = neg_sampling(data, n_neg)
    to_file = file[0:-4]+'_rank.csv'
    print(f"保存数据到 {to_file}")
    result.to_csv(Path(basename) / to_file, index=False)


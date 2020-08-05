# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-05

import pandas as pd
from pathlib import Path

basename = "/home/xxx/data/train_data/"
train_file = "multi_chitchat_train.csv"
val_file = "multi_chitchat_val.csv"


def main():
    train = pd.read_csv(Path(basename) / train_file)
    val = pd.read_csv(Path(basename) / val_file)

    print(f"训练集的数量为：{train.shape[0]} | 测试集的数量为：{val.shape[0]}")

    train = train[train["turns"] > 1].reset_index(drop=True)
    val = val[val["turns"] > 1].reset_index(drop=True)

    train = train[:10000]
    val = val[:10000]
    print(f"过滤后训练集的数量为：{train.shape[0]} | 测试集的数量为：{val.shape[0]}")
    train.to_csv(Path(basename) / "multi_tmp_train_multi.csv", index=False)
    val.to_csv(Path(basename) / "multi_tmp_val_multi.csv", index=False)

if __name__ == '__main__':
    main()
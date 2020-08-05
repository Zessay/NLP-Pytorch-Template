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

    use_cols = ["last", "response", "label"]
    train = train[use_cols]
    val = val[use_cols]
    train = train.rename(columns={"last": "text_left", "response": "text_right"})
    val = val.rename(columns={"last": "text_left", "response": "text_right"})

    train.to_csv(Path(basename) / "multi_chitchat_train_single.csv", index=False)
    val.to_csv(Path(basename) / "multi_chitchat_val_single.csv", index=False)

if __name__ == '__main__':
    main()
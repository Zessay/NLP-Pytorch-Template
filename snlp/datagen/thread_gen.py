#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: thread_gen.py
@time: 2020/4/8 19:49
@description: 使用不同的线程并行处理数据，不同的线程并发读取数据，将batches放到一个共享的queue中
参考： https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/
'''
import threading
import numpy as np
import random
import torch
import torch.nn as nn
import time
from threading import Thread
from queue import Queue, Full, Empty

class threadsafe_iter:
    """it是一个可迭代的对象或者生成器"""
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()  # 线程锁

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def get_path_i(paths_count):
    """
    循环生成线程的ID
    :param paths_count: 表示使用的线程数
    :return:
    """
    current_path_id = 0
    while True:
        yield current_path_id
        current_path_id = (current_path_id + 1) % paths_count


class ThreadGen:
    def __init__(self, paths, batch_size):
        """
        :param paths: 每一条表示一个数据
        :param batch_size:
        """
        self.paths = paths  # 每一个元素是(data, label)的元组
        self.index = 0
        self.batch_size = batch_size
        self.init_count = 0
        self.lock = threading.Lock()   # 防止生成Path的不同的线程相互干扰
        self.yield_lock = threading.Lock()  # 防止生成batch的不同线程的干扰
        self.path_id_generator = threadsafe_iter(get_path_i(len(self.paths)))
        self.data = []
        self.labels = []

    def get_samples_count(self):
        return len(self.paths)

    def get_batches_count(self):
        return int(self.get_batches_count() / self.batch_size)

    def pre_process_input(self, data, label):
        # 用来对数据和标签进行处理，注意需要是线程安全的
        """"""
        return data, label

    def next(self):
        return self.__iter__()

    def __iter__(self):
        while True:
            ## 每个epoch开始，随机打乱数据
            with self.lock:
                if (self.init_count == 0):
                    random.shuffle(self.paths)
                    self.data, self.labels, self.batch_paths = [], [], []
                    self.init_count = 1

            # 以一种线程安全的方式迭代输入
            for path_id in self.path_id_generator:
                dat, label = self.paths[path_id]
                dat, label = self.pre_process_input(dat, label)

                # 并发访问
                with self.yield_lock:
                    if (len(self.data) < self.batch_size):
                        self.data.append(dat)
                        self.labels.append(label)
                    if (len(self.data) % self.batch_size == 0):
                        yield np.float(self.data), np.float(self.labels)
                        self.data, self.labels = [], []

            with self.lock:
                self.init_count = 0

    def __call__(self):
        return self.__iter__()

class thread_killer(object):
    """用来返回是否需要终止一个线程的bool型标志"""
    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, to_kill):
        self.to_kill = to_kill

def threaded_batches_feeder(to_kill, batches_queue, dataset_generator):
    """不同线程对数据进行预处理，向队列中填充，直到达到最大长度"""
    while not to_kill():
        for batch, (batch_data, batch_label) in enumerate(dataset_generator):
            batches_queue.put((batch, (batch_data, batch_label)), block=True)

            if to_kill():
                return

def threaded_cuda_batches(to_kill, cuda_batches_queue, batches_queue, device):
    """将numpy型的转化为tensor型的"""
    while not to_kill():
        batch, (batch_data, batch_label) = batches_queue.get(block=True)
        batch_data = torch.from_numpy(batch_data).to(device)
        batch_label = torch.from_numpy(batch_label).to(device)

        cuda_batches_queue.put((batch, (batch_data, batch_label)), block=True)
        if to_kill():
            return

# 示例

def main():

    def train_batch(batch_data, batch_label):
        return 0.0, 0.0


    num_epoches = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(100, 100).to(device)
    model.train()

    batches_per_epoch = 64
    preprocess_workers = 4
    # 定义训练集列表，每一个元素是 (数据，标签) 的元组
    training_set_list = None
    train_batches_queue = Queue(maxsize=12)
    cuda_batches_queue = Queue(maxsize=3)

    # 得到数据的迭代器
    training_set_generator = ThreadGen(training_set_list, batches_per_epoch)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)

    # 启动4个线程
    for _ in range(preprocess_workers):
        t = Thread(target=threaded_batches_feeder,
                   args=(train_thread_killer, train_batches_queue, training_set_generator,))
        t.start()

    cuda_transfers_thread_killer = thread_killer()
    cuda_transfers_thread_killer.set_tokill(False)
    cudathread = Thread(target=threaded_cuda_batches,
                        args=(cuda_transfers_thread_killer, cuda_batches_queue, train_batches_queue))
    cudathread.start()
    ## 保证训练开始之前，queue被填满
    time.sleep(8)
    for epoch in range(num_epoches):
        for batch in range(batches_per_epoch):
            # 获取数据
            _, (batch_data, batch_label) = cuda_batches_queue.get(block=True)

            # 计算损失和指标
            loss, accuracy = train_batch(batch_data, batch_label)

    ## 杀掉相关线程
    train_thread_killer.set_tokill(True)
    cuda_transfers_thread_killer.set_tokill(True)

    for _ in range(preprocess_workers):
        try:
            ## 强制线程停止
            train_batches_queue.get(block=True, timeout=1)
            cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass
    print("training done!")
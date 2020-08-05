# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-04

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import time
from predictors.multiQA_predictor import MultiQAPredictor

model_path = "/home/xxx/models/multiQA/albert_imn"

predictor = MultiQAPredictor(model_path, model_file="model.pt")

utterance = "这袜子可以的\t还可以吧"
responses = ["确实不错", "还可以吧"]

start = time.time()
result = predictor.predict(utterance, responses)
print("time cost: {} ms".format((time.time() - start) * 1000))

print(result)
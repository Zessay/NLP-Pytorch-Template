# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-05
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import time
from predictors.singleQA_predictor import SingleQAPredictor

model_path = "/home/xxx/models/singleQA/albert_tiny_single_qa"

predictor = SingleQAPredictor(model_path)

utterance = "这袜子可以的"
responses = ["确实不错", "还可以吧"]

start = time.time()
result = predictor.predict(utterance, responses)
print("time cost: {} ms".format((time.time() - start) * 1000))

print(result)
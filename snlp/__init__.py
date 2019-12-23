#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: __init__.py.py
@time: 2019/12/2 20:07
@description: 
'''

import sys
import os

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from snlp import base
from snlp import callbacks
from snlp import datagen
from snlp import embedding
from snlp import losses
from snlp import metrics
from snlp import modules
from snlp import preprocessors
from snlp import tools
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: __init__.py.py
@time: 2019/12/24 11:15
@description: 
'''

from snlp.schedule.lambda_schedules import get_constant_schedule, get_constant_schedule_with_warmup, \
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
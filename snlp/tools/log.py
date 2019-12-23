#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: log.py
@time: 2019/12/4 11:37
@description: 
'''

from pathlib import Path
import logging
from snlp.tools.constants import LOG_FILE, PACKAGE_NAME

def init_logger(log_file=None, log_file_level=logging.NOTSET, log_on_console=True):
    """用于初始化log对象，可以选择是否记录在文件中以及是否在屏幕上显示"""
    if isinstance(log_file, Path):
        log_file = str(log_file)

    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(logging.INFO)
    if log_on_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.handlers = [console_handler]
    if log_file and log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

logger = init_logger(LOG_FILE)
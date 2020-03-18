#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: common.py
@time: 2019/11/21 21:31
@description: 定义一些工具类
'''
import os
import functools
import pickle
import json
import inspect
import numpy as np
from pathlib import Path
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import cProfile
import pstats
import logging


logger = logging.getLogger(__name__)

## ------------- 定义一些通用的的工具类 ---------------------

def seed_everything(seed=2020):
    """设置整个开发环境的seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    # 如果n_gpu_use为数字，则使用range生成list
    # 如果输入的是一个list，则默认使用list[0]作为controller
     """
    if not n_gpu_use:
        device_type = 'cpu'
    else:
        n_gpu_use = n_gpu_use.split(",")
        device_type = f"cuda:{n_gpu_use[0]}"
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        device_type = 'cpu'
    if len(n_gpu_use) > n_gpu:
        msg = f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine."
        logger.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device(device_type)
    list_ids = n_gpu_use
    return device, list_ids

def model_device(n_gpu, model):
    '''
    判断环境 cpu还是gpu
    支持单机多卡
    :param n_gpu: str型，以,分隔每个数字
    :param model: 模型
    :return:
    '''
    device, device_ids = prepare_device(n_gpu)
    if len(device_ids) > 1:
        logger.info(f"current {len(device_ids)} GPUs")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device

def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data, file_path):
    '''
    保存成json文件
    :param data:
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # if isinstance(data,dict):
    #     data = json.dumps(data)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def load_json(file_path):
    '''
    加载json文件
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data

def is_chinese_char(ch):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    cp = ord(ch)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def is_alphabet(ch):
    """判断是否是英文字符"""
    PLACESYM = set([u'{', u'}', u'[', u']', u'-', u'_'])
    cp = ord(ch)
    if (cp >= 0x0041 and cp <= 0x005a) or (cp >= 0x0061 and cp <= 0x007a) or (ch in PLACESYM):
        return True
    return False



def is_chinese_punc(ch):
    # 关于中文常用标点unicode编码 http://blog.chinaunix.net/uid-12348673-id-3335307.html
    puncs = [0xb7, 0xd7, 0x2014, 0x2018, 0x2019, 0x201c,
            0x201d, 0x2026, 0x3001, 0x3002, 0x300a, 0x300b,
            0x300e, 0x300f, 0x3010, 0x3011, 0xff01, 0xff08,
            0xff09, 0xff0c, 0xff1a, 0xff1b, 0xff1f]
    if ord(ch) in puncs:
        return True
    return False


def save_valid(text):
    """
    去除文本中的无效字符，只保留中文字符，英文字母和中英文常见的符号
    :param text:
    :return:
    """
    valid_start_puncs = [0x2018, 0x201c, 0x300a, 0x300e, 0x3010, 0xff08]
    string = ""
    for ch in text:
        if is_chinese_char(ch) or is_chinese_punc(ch) or is_alphabet(ch):
            string += ch

    while (string != "" and not is_chinese_char(string[0])
           and not is_alphabet(string[0])  and (string[0] not in valid_start_puncs)):
        string = string[1:]
    return string


def summary(model, *inputs, batch_size=-1, show_input=True):
    '''
    打印模型结构信息
    :param model:
    :param inputs:
    :param batch_size:
    :param show_input:
    :return:
    Example:
        >>> print("model summary info: ")
        >>> for step,batch in enumerate(train_data):
        >>>     summary(self.model,*batch,show_input=True)
        >>>     break
    '''

    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if show_input is False and output is not None:
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out.size())[1:]
                            ][0]
                        else:
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out[0].size())[1:]
                            ][0]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    logger.info("-----------------------------------------------------------------------")
    if show_input is True:
        line_new = f"{'Layer (type)':>25}  {'Input Shape':>25} {'Param #':>15}"
    else:
        line_new = f"{'Layer (type)':>25}  {'Output Shape':>25} {'Param #':>15}"
    logger.info(line_new)
    logger.info("=======================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if show_input is True:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
        else:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )

        total_params += summary[layer]["nb_params"]
        if show_input is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        logger.info(line_new)

    logger.info("=======================================================================")
    logger.info(f"Total params: {total_params:0,}")
    logger.info(f"Trainable params: {trainable_params:0,}")
    logger.info(f"Non-trainable params: {(total_params - trainable_params):0,}")
    logger.info("-----------------------------------------------------------------------")


def do_cprofile(filename, do_prof=True, sort_key="tottime"):
    """
    用于性能分析的装饰器函数
    :param filename: 表示分析结果保存的文件路径和名称
    """
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            # 获取环境变量表示
            if do_prof:
                profile = cProfile.Profile()
                ## 开启性能分析的对象
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                ## 默认按照总计用时排序
                ps = pstats.Stats(profile).sort_stats(sort_key)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
    return wrapper


## ------------- 定义一些模型中使用的工具方法 ---------------------

def is_number(token: str) -> bool:
    """判断输入的字符串是否是数字"""
    try:
        float(token)
        return True
    except:
        return False

def flatten_list(input_: list) -> list:
    """将多层嵌套的列表结构展开"""
    output = []
    for l in input_:
        if (isinstance(l, list)) or (isinstance(l, np.ndarray) or isinstance(l, tuple)):
            output.extend(flatten_list(l))
        else:
            output.append(l)
    return output

def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """将输入的一系列类别转换为onehot形式"""
    one_vec = np.eye(num_classes)
    one_hot = one_vec[indices]
    return one_hot

def label_smoothing(indices: np.ndarray, num_classes: int, pos_sub: float) -> np.ndarray:
    """对样本标签进行平滑"""
    # 先转换成二维的One-hot形式的矩阵
    onehot_matrix = one_hot(indices, num_classes)
    ## 相当于对上面的矩阵所有元素取反
    invert_matrix = np.negative(onehot_matrix) + 1
    neg_add = round(pos_sub / (num_classes - 1), 3)
    result = (1 - pos_sub) * onehot_matrix + neg_add * invert_matrix
    return result




def sort_and_couple(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """根据预测的概率值从大到小进行排序"""
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))


## ------------- 用于Preprocessor的工具方法 ---------------------

def validate_context(func):
    """Validate context in the preprocessors"""
    @functools.wraps(func)
    def transform_wrapper(self, *args, **kwargs):
        if not self.context:
            raise ValueError("Please call 'fit' before calling 'transform'.")
        return func(self, *args, **kwargs)
    return transform_wrapper



## ------------------ 列出一个类的所有子类 -------------------------

def list_recursive_concrete_subclasses(base):
    """ List all concrete subclasses of `base` recursively. """
    return _filter_concrete(_bfs(base))

def _filter_concrete(classes):
    return list(filter(lambda c: not inspect.isabstract(c), classes))

def _bfs(base):
    return base.__subclasses__() + sum([
        _bfs(subclass) for subclass in base.__subclasses__()], [])
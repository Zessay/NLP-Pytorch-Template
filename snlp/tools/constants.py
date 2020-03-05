#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: constants.py
@time: 2019/12/4 11:40
@description: 定义一些常量
'''
import os

# ----------------- log文件相关配置 -------------------
PACKAGE_NAME = os.path.basename(os.path.dirname(os.getcwd()))
LOG_FILE = "./train.log" # log保存的路径

# ----------------- DataFrame字段配置 -----------------
## 标签字段的名称
LABEL = 'label'

######## ------- 对于单个文本的字段名 -------
TEXT = 'text'
TEXT_LEN = 'text_len'
# ****** 以上是必须包含的字段，下面是可选字段 ********
NGRAM = 'ngram'

######## -------- 两个文本的字段名 ----------
TEXT_LEFT = 'text_left'
TEXT_RIGHT = 'text_right'
LEFT_LEN = 'text_left_len' # 必须是原列名加上`_len`
RIGHT_LEN = 'text_right_len'
# ****** 以上是必须包含的字段，下面是可选字段 *******
ID_LEFT = 'id_left'
ID_RIGHT = 'id_right'
NGRAM_LEFT = 'ngram_left'
NGRAM_RIGHT = 'ngram_right'

######## -------- 用于多轮QA的字段名 ----------
UTTRS = 'utterances'
RESP = 'response'
LAST = 'last'  # 表示最近的utterance
UTTRS_LEN = 'utterances_len'
RESP_LEN = 'response_len'
TURNS = 'turns'
UTTRS_CHAR = 'utterances_char'
RESP_CHAR = 'response_char'
UTTRS_CHAR_LEN = 'utterances_char_len'
RESP_CHAR_LEN = 'response_char_len'


## 用于排序比较文本长度的字段
SORT_LEN = 'response_len'

# ----------------- 填充词汇相关配置 -------------------

PAD = 0  # 必须为0
UNK = 1  # 必须为1
BOS = 2
EOS = 3

PAD_WORD = "<pad>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"


# --------------- 表情符 --------------------------

wechat_face = {"/::)": "微笑", "/::~": "撇嘴", "/::B": "好色", "/::|": "发呆", "/:8-)": "得意", "/::<": "流泪", "/::$": "害羞",
              "/::X": "闭嘴", "/::Z": "睡觉", "/::'(": "大哭", "/::-|": "尴尬", "/::@": "发怒", "/::P": "调皮", "/::D": "呲牙",
              "/::O": "惊讶", "/::(": "难过", "[囧]": "囧", "/::Q": "抓狂", "/::T": "恶心", "/:,@P": "偷笑", "/:,@-D": "愉快",
              "/::d": "白眼", "/:,@o": "傲慢", "/:|-)": "困", "/::!": "惊恐", "/::L": "好吧", "/::>": "憨笑", "/::,@": "神气",
              "/:,@f": "加油", "/::-S": "咒骂", "/:?": "疑惑", "/:,@x": "安静", "/:,@@": "晕了", "/:,@!": "衰", "/:!!!": "骷髅",
              "/:xx": "笨蛋", "/:bye": "再见", "/:wipe": "擦汗", "/:dig": "抠鼻", "/:handclap": "鼓掌", "/:B-)": "坏笑", "/:<@": "左哼哼",
              "/:@>": "右哼哼", "/::-O": "打哈欠", "/:>-|": "鄙视", "/:P-(": "委屈", "/::'|": "快哭了", "/:X-)": "阴险", "/::*": "亲一个",
              "/:8*": "可怜", "/:pd": "菜刀", "/:<W>": "西瓜", "/:beer": "啤酒", "/:coffee": "咖啡", "/:pig": "猪头", "/:rose": "玫瑰",
              "/:fade": "枯萎", "/:showlove": "嘴唇", "/:heart": "爱心", "/:break": "心碎了", "/:cake": "蛋糕", "/:bome": "炸弹",
              "/:shit": "便便", "/:moon": "晚安", "/:sun": "太阳", "/:hug": "拥抱", "/:strong": "强", "/:weak": "弱", "/:share": "握手",
              "/:v": "胜利", "/:@)": "抱拳", "/:jj": "勾引", "/:@@": "握拳", "/:ok": "好的", "/:jump": "高兴", "/:shake": "害怕",
              "/:<O>": "怄火", "/:circle": "开心地转圈", "[Hey]": "得意", "[Facepalm]": "捂脸", "[Smirk]": "坏坏地笑", "[Smart]": "看好你",
              "[Concerned]": "皱眉", "[Yeah!]": "高兴", "[吃瓜]": "看热闹", "[加油]": "加油", "[汗]": "无语", "[天啊]": "惊讶",
              "[Emm]": "好吧", "[社会社会]": "厉害了", "[旺柴]": "嘿嘿", "[好的]": "好的", "[打脸]": "打脸", "[加油加油]": "加油",
              "[哇]": "惊喜", "[Packet]": "红包", "[發]": "发财", "[Blessing]": "福"}

emoji_face = {"\ue415": ["笑脸", "\U0001f604"], "\ue40c": ["生病", "\U0001f637"], "\ue412": ["笑哭", "\U0001f602"],
             "\ue409": ["吐舌头", "\U0001f61d"], "\ue40d": ["耿直", "\U0001f633"], "\ue107": ["恐惧", "\U0001f631"],
             "\ue403": ["失望", "\U0001f614"], "\ue40e": ["无语", "\U0001f612"], "\ue11b": ["吐舌头", "\U0001f47b"],
             "\ue41d": ["祈祷", "\U0001f64f"], "\ue14c": ["加油", "\U0001f4aa"], "\ue312": ["庆祝", "\U0001f389"],
             "\ue112": ["礼物", "\U0001f381"]}
#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: translation.py
@time: 2019/11/22 10:46
@description: 多种API的中英互译，用于文本增强
'''
# post请求
import json
import requests
import execjs
import re
## 用于google翻译
from googletrans import Translator


############################################
#--------- 金山词霸翻译：中英互译 -----------#
############################################
class King:
    def __init__(self):
        self.url = 'http://fy.iciba.com/ajax.php?a=fy'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
        }
        # 构造post请求的参数
        self.post_data = {
            'f': 'auto',
            't': 'auto'
        }

    # 发送请求
    def request_post(self, word):
        # 构造post请求的参数
        self.post_data['w'] = word
        res = requests.post(url=self.url, headers=self.headers, data=self.post_data)
        # print(res.content.decode())
        return res.content.decode()

    # 解析数据
    @staticmethod
    def parse_data(data):
        dict_data = json.loads(data)
        if 'out' in dict_data['content']:
            return dict_data['content']['out']
        elif 'word_mean' in dict_data['content']:
            return dict_data['content']['word_mean']

    def run(self, word):
        data = self.request_post(word)
        result = self.parse_data(data)
        return result


############################################
#----------- Bing翻译：中英互译 ------------#
############################################
class Biying:
    def __init__(self):
        self.url = 'https://cn.bing.com/ttranslatev3?'
        # self.url = 'https://cn.bing.com/ttranslatev3?isVertical=1&&IG=E3F2E74779804936A4B134F621FE89FB&IID=translator.5028.12'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
        }
        # 构造post请求的参数
        self.post_data = {
            'fromLang': 'auto-detect',
            'to': 'zh-Hans',
        }

    # 判断post参数
    def judge_post(self, word):
        if self.is_chinese(word):
            self.post_data['to'] = 'en'
            # print(self.word.encode().isalpha())

    # 判断是否为汉字
    @staticmethod
    def is_chinese(uchar):
        if u'\u4e00' <= uchar <= u'\u9fa5':
            return True
        else:
            return False

    # 发送请求
    def request_post(self, word):
        self.post_data['text'] = word
        res = requests.post(url=self.url, headers=self.headers, data=self.post_data)
        # print(res.content.decode())
        return res.content.decode()

    # 解析数据
    @staticmethod
    def parse_data(data):
        dict_data = json.loads(data)
        return dict_data[0]['translations'][0]['text']

    def run(self, word):
        self.judge_post(word)
        data = self.request_post(word)
        result = self.parse_data(data)
        return result

############################################
#------------ 百度翻译：中英互译 ------------#
############################################
JS_CODE = """
function a(r, o) {
    for (var t = 0; t < o.length - 2; t += 3) {
        var a = o.charAt(t + 2);
        a = a >= "a" ? a.charCodeAt(0) - 87 : Number(a),
        a = "+" === o.charAt(t + 1) ? r >>> a: r << a,
        r = "+" === o.charAt(t) ? r + a & 4294967295 : r ^ a
    }
    return r
}
var C = null;
var token = function(r, _gtk) {
    var o = r.length;
    o > 30 && (r = "" + r.substr(0, 10) + r.substr(Math.floor(o / 2) - 5, 10) + r.substring(r.length, r.length - 10));
    var t = void 0,
    t = null !== C ? C: (C = _gtk || "") || "";
    for (var e = t.split("."), h = Number(e[0]) || 0, i = Number(e[1]) || 0, d = [], f = 0, g = 0; g < r.length; g++) {
        var m = r.charCodeAt(g);
        128 > m ? d[f++] = m: (2048 > m ? d[f++] = m >> 6 | 192 : (55296 === (64512 & m) && g + 1 < r.length && 56320 === (64512 & r.charCodeAt(g + 1)) ? (m = 65536 + ((1023 & m) << 10) + (1023 & r.charCodeAt(++g)), d[f++] = m >> 18 | 240, d[f++] = m >> 12 & 63 | 128) : d[f++] = m >> 12 | 224, d[f++] = m >> 6 & 63 | 128), d[f++] = 63 & m | 128)
    }
    for (var S = h,
    u = "+-a^+6",
    l = "+-3^+b+-f",
    s = 0; s < d.length; s++) S += d[s],
    S = a(S, u);
    return S = a(S, l),
    S ^= i,
    0 > S && (S = (2147483647 & S) + 2147483648),
    S %= 1e6,
    S.toString() + "." + (S ^ h)
}
"""


class Baidu:
    def __init__(self):
        self.sess = requests.Session()
        self.headers = {
            'User-Agent':
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
        }
        self.token = None
        self.gtk = None
        self.javascript = execjs.compile(JS_CODE)

        # 获得token和gtk
        # 必须要加载两次保证token是最新的，否则会出现998的错误
        self.loadMainPage()
        self.loadMainPage()

    def loadMainPage(self):
        """
            load main page : https://fanyi.baidu.com/
            and get token, gtk
        """
        url = 'https://fanyi.baidu.com'

        try:
            r = self.sess.get(url, headers=self.headers)
            self.token = re.findall(r"token: '(.*?)',", r.text)[0]
            self.gtk = re.findall(r"window.gtk = '(.*?)';", r.text)[0]
        except Exception as e:
            raise e

    def langdetect(self, query):
        """
            post query to https://fanyi.baidu.com/langdetect
            return json like
            {"error":0,"msg":"success","lan":"en"}
        """
        url = 'https://fanyi.baidu.com/langdetect'
        data = {'query': query}
        try:
            r = self.sess.post(url=url, data=data)
        except Exception as e:
            raise e

        json = r.json()
        if 'msg' in json and json['msg'] == 'success':
            return json['lan']
        return None

    def dictionary(self, query, dst='zh', src=None):
        """
            get translate result from https://fanyi.baidu.com/v2transapi
        """
        url = 'https://fanyi.baidu.com/v2transapi'

        sign = self.javascript.call('token', query, self.gtk)

        if not src:
            src = self.langdetect(query)

        data = {
            'from': src,
            'to': dst,
            'query': query,
            'simple_means_flag': 3,
            'sign': sign,
            'token': self.token,
        }
        try:
            r = self.sess.post(url=url, data=data)
        except Exception as e:
            raise e

        if r.status_code == 200:
            json = r.json()
            if 'error' in json:
                raise Exception('baidu sdk error: {}'.format(json['error']))
                # 998错误则意味需要重新加载主页获取新的token
            return json
        return None

    def run(self, word, dst='zh', src=None):
        dic = self.dictionary(word, dst=dst, src=src)
        try:
            result = dic['trans_result']['data'][0]['dst']
            return result
        except:
            print("没有得到正确的返回结果")
            return None

############################################
#----------- Google翻译：中英互译 -----------#
############################################

class Google:
    def __init__(self):
        self.trans = Translator()

    def detect(self, word):
        self.lang = self.trans.detect(word).lang
        if self.lang == 'en':
            self.dest = 'zh-CN'
        else:
            self.dest = 'en'

    def get_result(self, word):
        self.detect(word)
        tran_res = self.trans.translate(word, dest=self.dest)
        return tran_res

    def run(self, word):
        tran_res = self.get_result(word)
        if type(tran_res) == str:
            result = tran_res.text
            return result
        elif type(tran_res) == list:
            results = []
            for tran in tran_res:
                results.append(tran.text)
            return results

############################################
#------------- 有道翻译：中英互译 -----------#
############################################

class Youdao:
    def __init__(self):
        self.url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
        self.key = {
        'type': "AUTO",
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "false"}

    def get_response(self, word):
        self.key['i'] = word
        response = requests.post(self.url, data=self.key)
        if response.status_code == 200:
            return response.text
        else:
            print("有道词典没有返回正确的响应")
            return None

    def parse_response(self, word):
        response = self.get_response(word)
        if response is not None:
            result = json.loads(response)
            return result['translateResult'][0][0]['tgt']
        else:
            return None

    def run(self, word):
        result = self.parse_response(word)
        return result



if __name__ == '__main__':
    ## 可以实现中英互译
    word = input("翻译：")
    king = King()
    result = king.run(word)
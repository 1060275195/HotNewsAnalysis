# -*- coding: utf-8 -*-

import re
# import opencc
import jieba
import jieba.posseg as pseg
import json
from datetime import datetime
from datetime import timedelta


def data_filter(df):
    """数据过滤"""
    # 过滤掉没有内容的新闻
    df = df[df['content'] != ''].copy()
    df = df.dropna(subset=['content']).copy()
    # 去重
    df = df.drop_duplicates(subset=['url'])
    df = df.reset_index(drop=True)
    return df


def get_data(df, last_time, delta):
    """
    获取某段时间的新闻数据
    :param df: 原始数据
    :param last_time: 指定要获取数据的最后时间
    :param delta: 时间间隔
    :return: last_time前timedelta的数据
    """
    last_time = datetime.strptime(last_time, '%Y-%m-%d %H:%M')
    delta = timedelta(delta)
    df['time'] = df['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))
    df = df[df['time'].map(lambda x: (x <= last_time) and (x > last_time - delta))].copy()
    df = df.sort_values(by=['time'], ascending=[0])
    df['time'] = df['time'].map(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M'))
    df = df.reset_index(drop=True)
    return df


def clean_title(title):
    """清理新闻标题"""
    # 清理未知字符和空白字符
    title = re.sub(r'\?+', ' ', title)
    title = re.sub(r'( *\n+)+', '\n', title)
    title = re.sub(r'\u3000', '', title)
    # 清理…和|
    title = re.sub(r'…|\|', ' ', title)
    # 中文繁体转简体
    # x = opencc.OpenCC('t2s').convert(x)
    # 英文大写转小写
    title = title.lower()
    return title


def clean_content(content):
    """清理新闻内容"""
    # 清理未知字符和空白字符
    content = re.sub(r'\?+', ' ', content)
    content = re.sub(r'( *\n+)+', '\n', content)
    content = re.sub(r'\u3000', '', content)
    # 清理责任编辑
    content = content.split('\n责任编辑')[0]
    content = content.split('返回搜狐，查看更多')[0]
    # 清理原标题等
    list1 = ['原标题', '新浪财经讯[ ，]', '新浪美股讯[ ，]', '新浪外汇讯[ ，]', '新浪科技讯[ ，]',
             r'[（\(].{,20}来源[:：].{,30}[）\)]',
             r'(文章|图片|数据|资料)来源[:：].{,30}\n',
             r'(文章|图片|数据|资料)来源[:：].{,30}$',
             r'来源[:：].{,30}\n', r'来源[:：].{,30}$',
             r'作者：.{,11}\n', r'编辑：.{,11}\n',
             r'作者：.{,11}$', r'编辑：.{,11}$',
             r'[（\(].{,20}记者 .{,20}[）\)]']
    for i in list1:
        content = re.sub(i, '', content)
    # 中文繁体转简体
    # x = opencc.OpenCC('t2s').convert(x)
    # 英文大写转小写
    content = content.lower()
    return content


def get_num_eng_ch(text):
    # 提取数字英文中文
    text = re.sub(r'[^0-9A-Za-z\u4E00-\u9FFF]+', ' ', text)
    text = text.strip()
    return text


def pseg_cut(content, userdict_path=None):
    """词性标注"""
    if userdict_path is not None:
        jieba.load_userdict(userdict_path)
    words = pseg.lcut(content)
    return words


def get_words_by_flags(words, flags=None):
    """获取指定词性的词"""
    flags = ['n.*', 'v.*'] if flags is None else flags
    words = [w for w, f in words if w != ' ' and re.match('|'.join(['(%s$)' % flag for flag in flags]), f)]
    return words


def userdict_cut(x, userdict_path):
    # 用户词词典
    jieba.load_userdict(userdict_path)
    words = jieba.cut(x)
    return words


def stop_words_cut(words, stop_words_path):
    # 停用词处理
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
        stopwords.append(' ')
        words = [word for word in words if word not in stopwords]
    return words


def disambiguation_cut(words, disambiguation_dict_path):
    # 消歧词典
    with open(disambiguation_dict_path, 'r', encoding='utf-8') as f:
        disambiguation_dict = json.load(f)
        words = [(disambiguation_dict[word]
                  if disambiguation_dict.get(word) else word) for word in words]
    return words


def individual_character_cut(words, individual_character_dict_path):
    # 删除无用单字
    with open(individual_character_dict_path, 'r', encoding='utf-8') as f:
        individual_character = [line.strip() for line in f.readlines()]
        words = [word for word in words
                 if ((len(word) > 1) or ((len(word) == 1) and (word in individual_character)))]
    return words

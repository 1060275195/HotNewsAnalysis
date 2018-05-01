# -*- coding: utf-8 -*-

import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from wordcloud import WordCloud
from textrank4zh import TextRank4Sentence


def feature_extraction(series, vectorizer='CountVectorizer', vec_args=None):
    """
    对原文本进行特征提取
    :param series: pd.Series，原文本
    :param vectorizer: string，矢量化器，如'CountVectorizer'或者'TfidfVectorizer'
    :param vec_args: dict，矢量化器参数
    :return: 稀疏矩阵
    """
    vec_args = {'max_df': 1.0, 'min_df': 1} if vec_args is None else vec_args
    vec_args_list = ['%s=%s' % (i[0],
                                "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                ) for i in vec_args.items()]
    vec_args_str = ','.join(vec_args_list)
    vectorizer1 = eval("%s(%s)" % (vectorizer, vec_args_str))
    matrix = vectorizer1.fit_transform(series)
    return matrix


def get_cluster(matrix, cluster='DBSCAN', cluster_args=None):
    """
    对数据进行聚类，获取训练好的聚类器
    :param matrix: 稀疏矩阵
    :param cluster: string，聚类器
    :param cluster_args: dict，聚类器参数
    :return: 训练好的聚类器
    """
    cluster_args = {'eps': 0.5, 'min_samples': 5, 'metric': 'cosine'} if cluster_args is None else cluster_args
    cluster_args_list = ['%s=%s' % (i[0],
                                    "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                    ) for i in cluster_args.items()]
    cluster_args_str = ','.join(cluster_args_list)
    cluster1 = eval("%s(%s)" % (cluster, cluster_args_str))
    cluster1 = cluster1.fit(matrix)
    return cluster1


def get_labels(cluster):
    """
    获取聚类标签
    :param cluster: 训练好的聚类器
    :return: list，聚类标签
    """
    labels = cluster.labels_
    return labels


def label2rank(labels_list):
    """
    按标签的数量将标签转换为排行
    :param labels_list: list，聚类标签
    :return: list，聚类排行
    """
    series = pd.Series(labels_list)
    list1 = series[series != -1].tolist()
    n = len(set(list1))
    cnt = Counter(list1)
    key = [cnt.most_common()[i][0] for i in range(n)]
    value = [i for i in range(1, n + 1)]
    my_dict = dict(zip(key, value))
    my_dict[-1] = -1
    rank_list = [my_dict[i] for i in labels_list]
    return rank_list


def get_non_outliers_data(df, label_column='label'):
    """获取属于某个聚类簇的数据"""
    df = df[df[label_column] != -1].copy()
    return df


def get_data_sort_labelnum(df, label_column='label', top=1):
    """
    获取按标签数量排行的第top组数据
    :param df: pd.DataFrame，带有标签列的数据
    :param label_column: string，标签列名
    :param top: int
    :return: pd.DataFrame，前top组的数据
    """
    assert top > 0, 'top不能小于等于0！'
    labels = df[label_column].tolist()
    cnt = Counter(labels)
    label = cnt.most_common()[top - 1][0] if top <= len(set(labels)) else -2
    df = df[df[label_column] == label].copy() if label != -2 else pd.DataFrame(columns=df.columns)
    return df


def list2wordcloud(list1, save_path, font_path):
    """
    将文本做成词云
    :param list1: list，文本列表
    :param save_path: string，词云图片保存的路径
    :param font_path: string，用于制作词云所需的字体路径
    """
    text = ' '.join(list1)
    wc = WordCloud(font_path=font_path, width=800, height=600, margin=2,
                   ranks_only=True, max_words=200, collocations=False).generate(text)
    wc.to_file(save_path)


def get_key_sentences(text, num=1):
    """
    利用textrank算法，获取文本摘要
    :param text: string，原文本
    :param num: int，指定摘要条数
    :return: string，文本摘要
    """
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    abstract = '\n'.join([item.sentence for item in tr4s.get_key_sentences(num=num)])
    return abstract

# -*- coding: utf-8 -*-

import pandas as pd
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
from wordcloud import WordCloud
from textrank4zh import TextRank4Sentence


def feature_extraction(series, vectorizer='CountVectorizer', vec_args=None):
    vec_args = {'max_df': 1.0, 'min_df': 1} if vec_args is None else vec_args
    vec_args_list = ['%s=%s' % (i[0],
                                "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                ) for i in vec_args.items()]
    vec_args_str = ','.join(vec_args_list)
    vectorizer1 = eval("%s(%s)" % (vectorizer, vec_args_str))
    matrix = vectorizer1.fit_transform(series.map(lambda x: ' '.join(x)))
    return matrix


def get_cluster(matrix, cluster='DBSCAN', cluster_args=None):
    cluster_args = {'eps': 0.5, 'min_samples': 5, 'metric': 'cosine'} if cluster_args is None else cluster_args
    cluster_args_list = ['%s=%s' % (i[0],
                                    "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                    ) for i in cluster_args.items()]
    cluster_args_str = ','.join(cluster_args_list)
    cluster1 = eval("%s(%s)" % (cluster, cluster_args_str))
    cluster1 = cluster1.fit(matrix)
    return cluster1


def get_labels(cluster):
    labels = cluster.labels_
    return labels


def label2rank(labels):
    series = pd.Series(labels)
    list1 = series[series != -1].tolist()
    n = get_labelnum(list1)
    cnt = Counter(list1)
    key = [cnt.most_common()[i][0] for i in range(n)]
    value = [i+1 for i in range(n)]
    my_dict = dict(zip(key, value))
    my_dict[-1] = -1
    rank = [my_dict[i] for i in labels]
    return rank


def get_labelnum(labels):
    labelnum = len(set(labels))
    return labelnum


def get_non_outliers_data(df, label_column='label'):
    df = df[df[label_column] != -1].copy()
    return df


def get_data_sort_labelnum(df, label_column='label', top=1):
    assert top > 0, 'top不能小于等于0！'
    labels = df[label_column].tolist()
    cnt = Counter(labels)
    label = cnt.most_common()[top-1][0] if top <= get_labelnum(labels) else -2
    df = df[df[label_column] == label].copy() if label != -2 else pd.DataFrame(columns=df.columns)
    return df


def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


def get_most_common(series, n=10):
    list1 = series.tolist()
    list2 = flat(list1)
    cnt = Counter(list2)
    list3 = [i[0] for i in cnt.most_common(n) if cnt[i[0]] > 1]
    return list3


def list2wordcloud(list1, save_path, font_path):
    text = ' '.join(list1)
    wc = WordCloud(font_path=font_path, width=800, height=600, margin=2,
                   ranks_only=True, max_words=200, collocations=False).generate(text)
    wc.to_file(save_path)


def key_sentences(text, num=1):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    abstract = '\n'.join([item.sentence for item in tr4s.get_key_sentences(num=num)])
    return abstract

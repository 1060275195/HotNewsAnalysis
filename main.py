# -*- coding: utf-8 -*-

import os
import pandas as pd
from datetime import datetime
from utils import news_crawler
from utils import preprocessing
from utils import modeling

# 获取项目路径
project_path = os.path.dirname(os.path.realpath(__file__))
# 获取数据存放目录路径
data_path = os.path.join(project_path, 'data')
news_path = os.path.join(data_path, 'news')
extra_dict_path = os.path.join(data_path, 'extra_dict')
fonts_path = os.path.join(data_path, 'fonts')
results_path = os.path.join(data_path, 'results')


def my_crawler():
    """爬取新闻数据"""
    # sina_news_df = news_crawler.get_latest_news('sina', top=1000, show_content=True)
    # sohu_news_df = news_crawler.get_latest_news('sohu', top=1000, show_content=True)
    # xinhuanet_news_df = news_crawler.get_latest_news('xinhuanet', top=100, show_content=True)
    # news_crawler.save_news(sina_news_df, os.path.join(news_path, 'sina_latest_news.csv'))
    # news_crawler.save_news(sohu_news_df, os.path.join(news_path, 'sohu_latest_news.csv'))
    # news_crawler.save_news(xinhuanet_news_df, os.path.join(news_path, 'xinhuanet_latest_news.csv'))
    news_crawler.threaded_crawler()


def load_data():
    """加载数据"""
    # sina_news_df = news_crawler.load_news(os.path.join(news_path, 'sample_sina_latest_news.csv'))
    # sohu_news_df = news_crawler.load_news(os.path.join(news_path, 'sample_sohu_latest_news.csv'))
    # xinhuanet_news_df = news_crawler.load_news(os.path.join(news_path, 'sample_xinhuanet_latest_news.csv'))
    sina_news_df = news_crawler.load_news(os.path.join(news_path, 'sina_latest_news.csv'))
    sohu_news_df = news_crawler.load_news(os.path.join(news_path, 'sohu_latest_news.csv'))
    xinhuanet_news_df = news_crawler.load_news(os.path.join(news_path, 'xinhuanet_latest_news.csv'))
    news_df = pd.concat([sina_news_df, sohu_news_df, xinhuanet_news_df], ignore_index=True)
    return news_df


def filter_data(news_df):
    """过滤数据"""
    df = preprocessing.data_filter(news_df)
    now_time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
    # now_time = '2018-04-06 23:59'
    df = preprocessing.get_data(df, last_time=now_time, delta=5)
    return df


def title_preprocess(df):
    """标题分词预处理"""
    df['title1'] = df['title'].map(lambda x: preprocessing.clean_title(x))
    df['title1'] = df['title1'].map(lambda x: preprocessing.get_num_eng_ch(x))
    df['title1'] = df['title1'].map(lambda x: preprocessing.pseg_cut(
        x, userdict_path=os.path.join(extra_dict_path, 'self_userdict.txt')))
    df['title_cut'] = df['title1'].map(lambda x: preprocessing.get_words_by_flags(
        x, flags=['n.*', '.*n', 'v.*', 't', 's', 'j', 'l', 'i', 'eng']))
    df['title_cut'] = df['title_cut'].map(lambda x: preprocessing.stop_words_cut(
        x, os.path.join(extra_dict_path, 'HIT_stop_words.txt')))
    df['title_cut'] = df['title_cut'].map(lambda x: preprocessing.stop_words_cut(
        x, os.path.join(extra_dict_path, 'self_stop_words.txt')))
    df['title_cut'] = df['title_cut'].map(lambda x: preprocessing.disambiguation_cut(
        x, os.path.join(extra_dict_path, 'self_disambiguation_dict.json')))
    return df


def title_cluster(df, save=False):
    df = title_preprocess(df)
    matrix = modeling.feature_extraction(df['title_cut'], vectorizer='CountVectorizer',
                                         vec_args={'max_df': 1.0, 'min_df': 1})
    dbscan = modeling.get_cluster(matrix, cluster='DBSCAN',
                                  cluster_args={'eps': 0.5, 'min_samples': 5, 'metric': 'cosine'})

    labels = modeling.get_labels(dbscan)
    df['title_label'] = labels
    df_non_outliers = modeling.get_non_outliers_data(df, label_column='title_label')
    labelnum = modeling.get_labelnum(df_non_outliers['title_label'].tolist())
    print('一共有%d个簇(不包括离群点)' % labelnum)
    title_rank = modeling.label2rank(labels)
    df['title_rank'] = title_rank
    for i in range(labelnum):
        df1 = df[df['title_rank'] == i+1]
        list1 = modeling.flat(df1['title_cut'].tolist())
        modeling.list2wordcloud(list1, save_path=os.path.join(results_path, 'title_ranks', '%d.png' % (i+1)),
                                font_path=os.path.join(fonts_path, '禹卫书法行书简体.ttf'))
        top_list = modeling.get_most_common(df1['title_cut'], n=15)
        print(top_list)
    if save:
        df.drop(['content', 'title1', 'title_cut', 'title_label'], axis=1, inplace=True)
        news_crawler.save_news(df, os.path.join(results_path, 'df_title_rank.csv'))
    return df


def content_preprocess(df):
    """新闻内容分词预处理"""
    df['content1'] = df['content'].map(lambda x: preprocessing.clean_content(x))
    df['content1'] = df['content1'].map(lambda x: preprocessing.get_num_eng_ch(x))
    df['content1'] = df['content1'].map(lambda x: preprocessing.pseg_cut(
        x, userdict_path=os.path.join(extra_dict_path, 'self_userdict.txt')))
    df['content_cut'] = df['content1'].map(lambda x: preprocessing.get_words_by_flags(
        x, flags=['n.*', '.*n', 'v.*', 't', 's', 'j', 'l', 'i', 'eng']))
    df['content_cut'] = df['content_cut'].map(lambda x: preprocessing.stop_words_cut(
        x, os.path.join(extra_dict_path, 'HIT_stop_words.txt')))
    df['content_cut'] = df['content_cut'].map(lambda x: preprocessing.stop_words_cut(
        x, os.path.join(extra_dict_path, 'self_stop_words.txt')))
    df['content_cut'] = df['content_cut'].map(lambda x: preprocessing.disambiguation_cut(
        x, os.path.join(extra_dict_path, 'self_disambiguation_dict.json')))
    return df


def content_cluster(df, save=False):
    df = content_preprocess(df)
    matrix = modeling.feature_extraction(df['content_cut'], vectorizer='CountVectorizer',
                                         vec_args={'max_df': 0.95, 'min_df': 1, 'max_features': None})
    dbscan = modeling.get_cluster(matrix, cluster='DBSCAN',
                                  cluster_args={'eps': 0.5, 'min_samples': 5, 'metric': 'cosine'})

    labels = modeling.get_labels(dbscan)
    df['content_label'] = labels
    df_non_outliers = modeling.get_non_outliers_data(df, label_column='content_label')
    labelnum = modeling.get_labelnum(df_non_outliers['content_label'].tolist())
    print('一共有%d个簇(不包括离群点)' % labelnum)
    content_rank = modeling.label2rank(labels)
    df['content_rank'] = content_rank
    for i in range(labelnum):
        df1 = df[df['content_rank'] == i+1]
        list1 = modeling.flat(df1['content_cut'].tolist())
        modeling.list2wordcloud(list1, save_path=os.path.join(results_path, 'content_ranks', '%d.png' % (i+1)),
                                font_path=os.path.join(fonts_path, '禹卫书法行书简体.ttf'))
        top_list = modeling.get_most_common(df1['content_cut'], n=15)
        print(top_list)
    if save:
        df.drop(['content1', 'content_cut', 'content_label'], axis=1, inplace=True)
        news_crawler.save_news(df, os.path.join(results_path, 'df_content_rank.csv'))
    return df


def key_content(df, save=False):
    def f(x):
        x = preprocessing.clean_content(x)
        x = modeling.key_sentences(x, num=1)
        return x
    df['abstract'] = df['content'].map(f)
    if save:
        df.drop(['content'], axis=1, inplace=True)
        news_crawler.save_news(df, os.path.join(results_path, 'df_abstract.csv'))
    return df


def main():
    import multiprocessing
    my_crawler()
    news_df = load_data()
    df = filter_data(news_df)
    p1 = multiprocessing.Process(target=title_cluster, args=(df, True))
    p2 = multiprocessing.Process(target=content_cluster, args=(df, True))
    p3 = multiprocessing.Process(target=key_content, args=(df, True))
    p1.start()
    p2.start()
    p3.start()
    processes = [p1, p2, p3]
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()

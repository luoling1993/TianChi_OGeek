#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import json
import logging
import os
import time
import warnings
from operator import itemgetter

import jieba
import numpy as np
import pandas as pd
from gensim import matutils
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from logconfig import config_logging
from utils import char_cleaner, char_list_cheaner
from w2v import build_model

config_logging()
logger = logging.getLogger('w2v_features')

warnings.filterwarnings('ignore')

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_PATH = os.path.join(BASE_PATH, "RawData")
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")
TEMP_DATA_PATH = os.path.join(BASE_PATH, "TempData")


class PreProcessing(object):
    def __init__(self, size, w2v_model):
        self.size = size
        self.w2v_model = w2v_model

    def to_csv(self, df, col):
        file_name = '{col}_w2v.csv'.format(col=col)
        file_path = os.path.join(TEMP_DATA_PATH, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        columns = ['w2v_{}'.format(i) for i in range(self.size)]

        with open(file_path, 'a', encoding='utf-8') as f:
            # write columns
            f.write(','.join(columns) + '\n')

            for idx, item in tqdm(df[col].items()):
                item = char_cleaner(item)
                if item == 'null':
                    item_list = [''] * self.size
                elif not item:
                    item_list = [''] * self.size
                else:
                    seg_cut = jieba.lcut(str(item))
                    seg_cut = char_list_cheaner(seg_cut)

                    w2v_array = list()
                    for word in seg_cut:
                        try:
                            similar_list = self.w2v_model[word]
                            w2v_array.append(similar_list)
                        except KeyError:
                            pass

                    if not w2v_array:
                        item_list = [''] * self.size
                    else:
                        item_list = matutils.unitvec(np.array(w2v_array).mean(axis=0))

                f.write(','.join(map(str, item_list)) + '\n')


class Procossing(object):
    def __init__(self, size, force):
        self.size = size
        self.force = force
        self.w2v_model = self._get_w2v_model()

    def _get_w2v_model(self):
        w2v_model_name = "w2v_{}.bin".format(self.size)
        w2v_model_path = os.path.join("resources", w2v_model_name)
        if not os.path.exists(w2v_model_path):
            build_model(fname=['train', 'vali', 'test'], size=self.size)
        w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True, unicode_errors="ignore")
        return w2v_model

    @staticmethod
    def _get_data(name):
        if name == "test":
            columns = ['prefix', 'query_prediction', 'title', 'tag']
        else:
            columns = ['prefix', 'query_prediction', 'title', 'tag', 'label']

        data_name = os.path.join(RAW_DATA_PATH, "oppo_round1_{}.txt".format(name))
        df = pd.read_csv(data_name, names=columns, sep="\t", header=None, encoding="utf-8")

        return df

    def _get_jieba_array(self, words):
        words = char_cleaner(words)
        seg_cut = jieba.lcut(words)
        seg_cut = char_list_cheaner(seg_cut)

        w2v_array = list()
        for word in seg_cut:
            try:
                similar_list = self.w2v_model[word]
                w2v_array.append(similar_list)
            except KeyError:
                continue

        if not w2v_array:
            w2v_array = [None] * self.size
        else:
            w2v_array = matutils.unitvec(np.array(w2v_array).mean(axis=0))

        return w2v_array

    def _get_w2v_similar(self, item):
        item_dict = dict()

        query_predict = item["query_prediction"]
        title = item['title']

        if not query_predict:
            item_dict["max_similar"] = None
            item_dict["mean_similar"] = None
            item_dict["weight_similar"] = None
            return item_dict

        query_predict = sorted(query_predict.items(), key=itemgetter(1), reverse=True)
        query_predict = query_predict[:3]

        similar_list = list()
        weight_similar_list = list()

        title_array = self._get_jieba_array(title)
        for key, value in query_predict:
            query_cut_array = self._get_jieba_array(key)

            try:
                w2v_similar = np.dot(query_cut_array, title_array)
            except (KeyError, ZeroDivisionError, TypeError):
                w2v_similar = np.nan

            similar_list.append(w2v_similar)
            weight_w2v_similar = w2v_similar * float(value)
            weight_similar_list.append(weight_w2v_similar)

        max_similar = np.nanmax(similar_list)
        mean_similar = np.nanmean(similar_list)
        weight_similar = np.nansum(weight_similar_list)

        item_dict["max_similar"] = max_similar
        item_dict["mean_similar"] = mean_similar
        item_dict["weight_similar"] = weight_similar

        return item_dict

    @staticmethod
    def _get_help_flag(item):
        if np.isnan(item):
            return 0
        return 1

    def _get_w2v_df(self, df, col):
        file_name = '{col}_w2v.csv'.format(col=col)
        file_path = os.path.join(TEMP_DATA_PATH, file_name)

        if os.path.exists(file_path) and not self.force:
            pass
        else:
            pre_processing = PreProcessing(self.size, self.w2v_model)
            pre_processing.to_csv(df, col)

        w2v_df = pd.read_csv(file_path, header=0)
        w2v_df['help_index'] = w2v_df.index
        w2v_df['help_flag'] = w2v_df['w2v_0'].apply(self._get_help_flag)

        return w2v_df

    def _get_query_df(self, df):
        query_df = pd.DataFrame()

        query_df["item_dict"] = df[['query_prediction', 'title']].apply(self._get_w2v_similar, axis=1)
        query_df["max_similar"] = query_df["item_dict"].apply(lambda item: item.get("max_similar"))
        query_df["mean_similar"] = query_df["item_dict"].apply(lambda item: item.get("mean_similar"))
        query_df["weight_similar"] = query_df["item_dict"].apply(lambda item: item.get("weight_similar"))
        query_df = query_df.drop(columns=["item_dict"])

        return query_df

    @staticmethod
    def _get_prefix_df(prefix_w2v_df, title_w2v_df, col_name):
        prefix_df = pd.DataFrame()

        remove_columns = ['help_index', 'help_flag']

        prefix_w2v_df = prefix_w2v_df.copy()
        prefix_w2v_df = prefix_w2v_df.drop(columns=remove_columns)

        title_w2v_df = title_w2v_df.copy()
        title_w2v_df = title_w2v_df.drop(columns=remove_columns)

        prefix_w2v_list = list()
        for idx, prefix in prefix_w2v_df.iterrows():
            if np.isnan(prefix[0]):
                prefix_w2v_list.append(None)
                continue

            title = title_w2v_df.loc[idx]
            if np.isnan(title[0]):
                prefix_w2v_list.append(None)
                continue

            similar = np.dot(prefix, title)
            prefix_w2v_list.append(similar)

        prefix_df[col_name] = prefix_w2v_list
        return prefix_df

    @staticmethod
    def _get_kmeans_dict(df, size=20):
        df = df.copy()
        df = df[df['help_flag'] == 1]
        help_index = df['help_index'].tolist()

        df = df.drop(columns=['help_index', 'help_flag'])

        kmeans = MiniBatchKMeans(n_clusters=size, reassignment_ratio=0.001)
        preds = kmeans.fit_predict(df)

        kmeans_dict = dict(zip(help_index, preds))
        return kmeans_dict

    @staticmethod
    def _loads(item):
        try:
            return json.loads(item)
        except (json.JSONDecodeError, TypeError):
            return json.loads("{}")

    @staticmethod
    def _mapping_kmeans(item, mapping_dict):
        return mapping_dict.get(item, -1)

    @staticmethod
    def _get_ctr_df(df, train_df_length, columns=None):
        df = df.copy()

        train_df = df[:train_df_length]

        if columns is None:
            columns = ['prefix_kmeans', 'title_kmeans', 'complete_prefix_kmeans']

        ctr_df = df[columns]

        # click count and ctr
        for idx, column in enumerate(columns):
            click_column = "{column}_click".format(column=column)
            count_column = "{column}_count".format(column=column)
            ctr_column = "{column}_ctr".format(column=column)

            agg_dict = {click_column: "sum", count_column: "count"}
            column_apriori_df = train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
            column_apriori_df[ctr_column] = column_apriori_df[click_column] / (column_apriori_df[count_column] + 5)
            ctr_df = pd.merge(ctr_df, column_apriori_df, how='left', on=column)

        ctr_df = ctr_df.drop(columns=columns)

        return ctr_df

    @staticmethod
    def _get_pca_df(df, name, n_components=5):
        df = df.copy()

        remove_columns = ['help_flag', 'help_index']

        df_effective = df[df['help_flag'] == 1]
        df_invalid = df[df['help_flag'] == 0]

        df_effective = df_effective.drop(columns=remove_columns)
        df_invalid = df_invalid.drop(columns=remove_columns)

        pca_columns = ['{}_pca_{}'.format(name, i) for i in range(n_components)]

        pca = PCA(n_components=n_components)

        pca_data = pca.fit_transform(df_effective)
        pca_df = pd.DataFrame(pca_data, index=df_effective.index, columns=pca_columns)
        none_df = pd.DataFrame(index=df_invalid.index, columns=pca_columns)

        pca_df = pd.concat([pca_df, none_df], axis=0, ignore_index=False, sort=False)
        pca_df = pca_df.sort_index()

        return pca_df

    @staticmethod
    def _get_complete_prefix(item):
        prefix = item['prefix']
        query_prediction = item['query_prediction']

        if not query_prediction:
            return prefix

        predict_word_dict = dict()
        prefix = str(prefix)

        for query_item, query_ratio in query_prediction.items():
            query_item_cut = jieba.lcut(query_item)
            item_word = ""
            for item in query_item_cut:
                if prefix not in item_word:
                    item_word += item
                else:
                    if item_word not in predict_word_dict.keys():
                        predict_word_dict[item_word] = 0.0
                    predict_word_dict[item_word] += float(query_ratio)

        if not predict_word_dict:
            return prefix

        predict_word_dict = sorted(predict_word_dict.items(), key=itemgetter(1), reverse=True)
        complete_prefix = predict_word_dict[0][0]
        return complete_prefix

    def get_processing(self):
        train_df = self._get_data(name="train")
        validate_df = self._get_data(name="vali")
        test_df = self._get_data(name="test")
        logger.info('finish load data!')

        train_df_length = train_df.shape[0]
        validate_df_length = validate_df.shape[0]
        df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True, sort=False)

        del train_df, validate_df, test_df
        gc.collect()

        # make query prediction to json
        df["query_prediction"] = df["query_prediction"].apply(self._loads)

        # complete prefix
        df['complete_prefix'] = df[['prefix', 'query_prediction']].apply(self._get_complete_prefix, axis=1)

        # clearn prefix and title
        df["prefix"] = df["prefix"].apply(char_cleaner)
        df["title"] = df["title"].apply(char_cleaner)
        df["complete_prefix"] = df["complete_prefix"].apply(char_cleaner)

        w2v_df = df[['label']]

        prefix_w2v_df = self._get_w2v_df(df, col='prefix')
        title_w2v_df = self._get_w2v_df(df, col='title')
        complete_prefix_w2v_df = self._get_w2v_df(df, col='complete_prefix')
        logger.info('finish get prefix and title w2v df!')

        prefix_pca_df = self._get_pca_df(prefix_w2v_df, 'prefix')
        title_pca_df = self._get_pca_df(title_w2v_df, 'title')
        complete_prefix_pca_df = self._get_pca_df(complete_prefix_w2v_df, 'complete_prefix')
        w2v_df = pd.concat([w2v_df, prefix_pca_df, title_pca_df, complete_prefix_pca_df], axis=1)

        del prefix_pca_df, title_pca_df, complete_prefix_pca_df
        gc.collect()

        prefix_kmeans_dict = self._get_kmeans_dict(prefix_w2v_df)
        title_kmeans_dict = self._get_kmeans_dict(title_w2v_df)
        complete_prefix_kmeans_dict = self._get_kmeans_dict(complete_prefix_w2v_df)
        logger.info('finish make kmeans!')

        w2v_df['prefix_kmeans'] = prefix_w2v_df['help_index'].apply(self._mapping_kmeans, args=(prefix_kmeans_dict,))
        w2v_df['title_kmeans'] = title_w2v_df['help_index'].apply(self._mapping_kmeans, args=(title_kmeans_dict,))
        w2v_df['complete_prefix_kmeans'] = complete_prefix_w2v_df['help_index'].apply(
            self._mapping_kmeans, args=(complete_prefix_kmeans_dict,))

        ctr_df = self._get_ctr_df(w2v_df, train_df_length)
        w2v_df = pd.concat([w2v_df, ctr_df], axis=1)

        del ctr_df, prefix_kmeans_dict, title_kmeans_dict, complete_prefix_kmeans_dict
        gc.collect()

        prefix_df = self._get_prefix_df(prefix_w2v_df, title_w2v_df, 'prefix_w2v')
        omplete_prefix_df = self._get_prefix_df(complete_prefix_w2v_df, title_w2v_df, 'complete_prefix_w2v')
        logger.info('finish get prefix  df!')
        w2v_df = pd.concat([w2v_df, prefix_df, omplete_prefix_df], axis=1)

        del prefix_df, omplete_prefix_df, prefix_w2v_df, title_w2v_df
        gc.collect()

        query_df = self._get_query_df(df)
        logger.info('finish get query_df!')
        w2v_df = pd.concat([w2v_df, query_df], axis=1)

        w2v_df = w2v_df.drop(columns=['label'])

        train_data = w2v_df[:train_df_length]
        validate_data = w2v_df[train_df_length:train_df_length + validate_df_length]
        test_data = w2v_df[train_df_length + validate_df_length:]

        train_data_name = os.path.join(ETL_DATA_PATH, "train_w2v.csv")
        validate_data_name = os.path.join(ETL_DATA_PATH, "validate_w2v.csv")
        test_data_name = os.path.join(ETL_DATA_PATH, "test_w2v.csv")

        train_data.to_csv(train_data_name, index=False)
        validate_data.to_csv(validate_data_name, index=False)
        test_data.to_csv(test_data_name, index=False)


if __name__ == "__main__":
    t0 = time.time()
    processing = Procossing(size=100, force=False)
    processing.get_processing()
    print(time.time() - t0)

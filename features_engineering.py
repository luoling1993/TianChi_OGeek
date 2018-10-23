#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import time
import warnings

import jieba
import numpy as np
import pandas as pd
from gensim import matutils
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from utils import char_cleaner, char_list_cheaner
from w2v import build_model

warnings.filterwarnings('ignore')

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_PATH = os.path.join(BASE_PATH, "RawData")
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")
TEMP_DATA_PATH = os.path.join(BASE_PATH, "TempData")

w2v_model_name = "./w2v.bin"
if not os.path.exists(w2v_model_name):
    build_model(fname="train")
w2v_model = KeyedVectors.load_word2vec_format("w2v.bin", binary=True, unicode_errors="ignore")


class PrefixProcessing(object):
    def __init__(self, prefix_w2v_df, title_w2v_df):
        self.prefix_w2v_df = prefix_w2v_df.copy()
        self.title_w2v_df = title_w2v_df.copy()

    @staticmethod
    def _is_in_title(item):
        prefix = item["prefix"]
        title = item["title"]

        if not isinstance(prefix, str):
            prefix = "null"

        if prefix in title:
            return 1
        return 0

    @staticmethod
    def _levenshtein_distance(item):
        str1 = item["prefix"]
        str2 = item["title"]

        if not isinstance(str1, str):
            str1 = "null"

        x_size = len(str1) + 1
        y_size = len(str2) + 1

        matrix = np.zeros((x_size, y_size), dtype=np.int_)

        for x in range(x_size):
            matrix[x, 0] = x

        for y in range(y_size):
            matrix[0, y] = y

        for x in range(1, x_size):
            for y in range(1, y_size):
                if str1[x - 1] == str2[y - 1]:
                    matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1)
                else:
                    matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1)

        return matrix[x_size - 1, y_size - 1]

    @staticmethod
    def _distince_rate(item):
        str1 = item["prefix"]
        str2 = item["title"]
        leven_distance = item["leven_distance"]

        if not isinstance(str1, str):
            str1 = "null"

        length = max(len(str1), len(str2))

        return leven_distance / (length + 5)  # 平滑

    def _get_prefix_w2v(self):
        remove_columns = ['help_index', 'help_flag']

        prefix_w2v_df = self.prefix_w2v_df.copy()
        prefix_w2v_df = prefix_w2v_df.drop(columns=remove_columns)

        title_w2v_df = self.title_w2v_df.copy()
        title_w2v_df = title_w2v_df.drop(columns=remove_columns)

        prefix_w2v_list = list()
        for idx, prefix in prefix_w2v_df.items():
            if not prefix[0]:
                prefix_w2v_list.append(None)
                continue

            title = title_w2v_df.loc[idx]
            if not title[0]:
                prefix_w2v_list.append(None)
                continue

            similar = np.dot(prefix, title)
            prefix_w2v_list.append(similar)

        return prefix_w2v_list

    def get_prefix_df(self, df):
        prefix_df = pd.DataFrame()

        prefix_df[["prefix", "title"]] = df[["prefix", "title"]]
        prefix_df["is_in_title"] = prefix_df.apply(self._is_in_title, axis=1)
        prefix_df["leven_distance"] = prefix_df.apply(self._levenshtein_distance, axis=1)
        prefix_df["distance_rate"] = prefix_df.apply(self._distince_rate, axis=1)
        prefix_df["prefix_w2v"] = self._get_prefix_w2v()
        return prefix_df


class QueryProcessing(object):
    def __init__(self, title_w2v_df):
        self.title_w2v_df = title_w2v_df.copy()
        self.title_w2v_dict = self._get_title_w2v_dict()

    def _get_title_w2v_dict(self):
        title_w2v_df = self.title_w2v_df.copy()
        title_w2v_df = title_w2v_df.drop(columns=['help_flag'])

        title_w2v_dict = title_w2v_df.set_index('help_index').T.to_dict(orient='list')
        return title_w2v_dict

    @staticmethod
    def _get_jieba_array(words, size=500):
        seg_cut = jieba.lcut(words)
        seg_cut = char_list_cheaner(seg_cut)

        w2v_array = list()
        for word in seg_cut:
            try:
                similar_list = w2v_model[word]
                w2v_array.append(similar_list)
            except KeyError:
                continue

        if not w2v_array:
            w2v_array = [None] * size
        else:
            w2v_array = matutils.unitvec(np.array(w2v_array).mean(axis=0))

        return w2v_array

    def _get_w2v_similar(self, item):
        item_dict = dict()

        query_predict = item["query_prediction"]

        if not query_predict:
            item_dict["max_similar"] = None
            item_dict["mean_similar"] = None
            item_dict["weight_similar"] = None
            return item_dict

        similar_list = list()
        weight_similar_list = list()

        index = item.name
        title_array = self.title_w2v_dict[index]

        for _, value in query_predict.iterrows():
            query_cut_array = self._get_jieba_array(value)
            w2v_similar = np.dot(query_cut_array, title_array)

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

    def get_query_df(self, df):
        query_df = pd.DataFrame()

        query_df["item_dict"] = df.apply(self._get_w2v_similar, axis=1)
        query_df["max_similar"] = query_df["item_dict"].apply(lambda item: item.get("max_similar"))
        query_df["mean_similar"] = query_df["item_dict"].apply(lambda item: item.get("mean_similar"))
        query_df["weight_similar"] = query_df["item_dict"].apply(lambda item: item.get("weight_similar"))
        query_df = query_df.drop(columns=["item_dict"])

        return query_df


class Processing(object):
    @staticmethod
    def _get_data(name):
        if name == "test":
            name = "test_A"
            columns = ['prefix', 'query_prediction', 'title', 'tag']
        else:
            columns = ['prefix', 'query_prediction', 'title', 'tag', 'label']

        data_name = os.path.join(RAW_DATA_PATH, "oppo_round1_{}.txt".format(name))
        df = pd.read_csv(data_name, names=columns, sep="\t", header=None, encoding="utf-8")
        return df

    @staticmethod
    def _get_ctr_df(df, train_df_length):
        df = df.copy()

        train_df = df[:train_df_length]

        labels_columns = ["prefix", "title", "tag", 'prefix_kmeans', 'title_kmeans']
        for column in labels_columns:
            click_column = "{column}_click".format(column=column)
            count_column = "{column}_count".format(column=column)
            ctr_column = "{column}_ctr".format(column=column)

            agg_dict = {click_column: "sum", count_column: "count"}
            ctr_df = train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
            ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

            df = pd.merge(df, ctr_df, how="left", on=column)

        # combine features
        for idx, column1 in enumerate(labels_columns):
            for column2 in labels_columns[idx + 1:]:
                if column1 in column2:
                    continue

                group_column = [column1, column2]
                click_column = "{column}_click".format(column="_".join(group_column))
                count_column = "{column}_count".format(column="_".join(group_column))
                ctr_column = "{column}_ctr".format(column="_".join(group_column))

                agg_dict = {click_column: "sum", count_column: "count"}
                ctr_df = train_df.groupby(group_column, as_index=False)["label"].agg(agg_dict)
                ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

                df = pd.merge(df, ctr_df, how="left", on=group_column)

        return df

    @staticmethod
    def _to_csv(df, col, size):
        file_name = '{col}_w2v.csv'.format(col=col)
        file_path = os.path.join(TEMP_DATA_PATH, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        columns = ['{}_w2v_{}'.format(col, i) for i in range(size)]
        none_index_list = list()

        with open(file_path, 'a', encoding='utf-8') as f:
            # write columns
            f.write(','.join(columns) + '\n')

            for idx, item in tqdm(df[col].items()):
                if item == 'null':
                    item_list = [''] * size
                    none_index_list.append(idx)
                elif not item:
                    item_list = [''] * size
                    none_index_list.append(idx)
                else:
                    seg_cut = jieba.lcut(item)
                    seg_cut = char_list_cheaner(seg_cut)

                    w2v_array = list()
                    for word in seg_cut:
                        try:
                            similar_list = w2v_model[word]
                            w2v_array.append(similar_list)
                        except KeyError:
                            pass

                    if not w2v_array:
                        item_list = [''] * size
                        none_index_list.append(idx)
                    else:
                        item_list = matutils.unitvec(np.array(w2v_array).mean(axis=0))

                f.write(','.join(map(str, item_list)) + '\n')

        return none_index_list

    def _get_w2v_df(self, df, col, size=500):
        none_index_list = self._to_csv(df, col, size)

        file_name = '{col}_w2v.csv'.format(col=col)
        file_path = os.path.join(TEMP_DATA_PATH, file_name)

        w2v_df = pd.read_csv(file_path, header=0)
        w2v_df['help_index'] = w2v_df.index
        w2v_df['help_flag'] = w2v_df['help_index'].apply(lambda _item: 0 if _item in none_index_list else 1)

        return w2v_df

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
    def _mapping_kmeans(item, mapping_dict):
        return mapping_dict.get(item, -1)

    def get_processing(self):
        train_df = self._get_data(name="train")
        validate_df = self._get_data(name="vali")
        test_df = self._get_data(name="test")

        train_df_length = train_df.shape[0]
        validate_df_length = validate_df.shape[0]
        df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True, sort=False)

        # make query prediction to json
        df["query_prediction"] = df["query_prediction"].apply(json.loads)

        # clearn prefix and title
        df["prefix"] = df["prefix"].apply(char_cleaner)
        df["title"] = df["title"].apply(char_cleaner)

        prefix_w2v_df = self._get_w2v_df(df, col='prefix')
        title_w2v_df = self._get_w2v_df(df, col='title')

        prefix_kmeans_dict = self._get_kmeans_dict(prefix_w2v_df)
        title_kmeans_dict = self._get_kmeans_dict(title_w2v_df)
        df['prefix_kmeans'] = prefix_w2v_df['help_index'].apply(self._mapping_kmeans, args=(prefix_kmeans_dict,))
        df['title_kmeans'] = title_w2v_df['help_index'].apply(self._mapping_kmeans, args=(title_kmeans_dict,))

        df = self._get_ctr_df(df, train_df_length)

        prefix_processing = PrefixProcessing(prefix_w2v_df, title_w2v_df)
        prefix_df = prefix_processing.get_prefix_df(df)

        query_processing = QueryProcessing(title_w2v_df)
        query_df = query_processing.get_query_df(df)

        df = pd.concat([df, prefix_df, query_df], axis=1)

        drop_columns = ['prefix', 'query_prediction', 'title']
        df = df.drop(columns=drop_columns)

        # one hot
        df = pd.get_dummies(df, columns=['tag', 'prefix_kmeans', 'title_kmeans'])

        train_data = df[:train_df_length]
        train_data["label"] = train_data["label"].apply(int)

        validate_data = df[train_df_length:train_df_length + validate_df_length]
        validate_data["label"] = validate_data["label"].apply(int)

        test_data = df[train_df_length + validate_df_length:]
        test_data = test_data.drop(columns=["label"])

        train_data_name = os.path.join(ETL_DATA_PATH, "train.csv")
        validate_data_name = os.path.join(ETL_DATA_PATH, "validate.csv")
        test_data_name = os.path.join(ETL_DATA_PATH, "test.csv")

        train_data.to_csv(train_data_name, index=False)
        validate_data.to_csv(validate_data_name, index=False)
        test_data.to_csv(test_data_name, index=False)


if __name__ == "__main__":
    t0 = time.time()
    Processing().get_processing()
    print(time.time() - t0)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import logging
import os
import time
import warnings
from operator import itemgetter

import jieba
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from logconfig import config_logging
from utils import char_cleaner

config_logging()
logger = logging.getLogger('stat_features')

warnings.filterwarnings('ignore')
np.random.seed(2018)

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_PATH = os.path.join(BASE_PATH, "RawData")
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")


class PrefixProcessing(object):
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

    def get_prefix_df(self, df):
        prefix_df = pd.DataFrame()

        prefix_df[["prefix", "title"]] = df[["prefix", "title"]]
        prefix_df["is_in_title"] = prefix_df.apply(self._is_in_title, axis=1)
        prefix_df["leven_distance"] = prefix_df.apply(self._levenshtein_distance, axis=1)
        prefix_df["distance_rate"] = prefix_df.apply(self._distince_rate, axis=1)

        return prefix_df


class QueryProcessing(object):
    @staticmethod
    def _get_query_dict(item):
        item_dict = dict()

        query_predict = item["query_prediction"]

        if not query_predict:
            item_dict["query_length"] = 0
            item_dict["prob_sum"] = None
            item_dict["prob_max"] = None
            item_dict["prob_mean"] = None
            return item_dict

        prob_list = list()
        for _, prob in query_predict.items():
            prob = float(prob)
            prob_list.append(prob)

        item_dict["query_length"] = len(prob_list)
        item_dict["prob_sum"] = np.sum(prob_list)
        item_dict["prob_max"] = np.max(prob_list)
        item_dict["prob_mean"] = np.mean(prob_list)

        return item_dict

    def get_query_df(self, df):
        query_df = pd.DataFrame()

        query_df["item_dict"] = df.apply(self._get_query_dict, axis=1)
        query_df["query_length"] = query_df["item_dict"].apply(lambda item: item.get("query_length"))
        query_df["prob_sum"] = query_df["item_dict"].apply(lambda item: item.get("prob_sum"))
        query_df["prob_max"] = query_df["item_dict"].apply(lambda item: item.get("prob_max"))
        query_df["prob_mean"] = query_df["item_dict"].apply(lambda item: item.get("prob_mean"))
        query_df = query_df.drop(columns=["item_dict"])

        return query_df


class Processing(object):

    @staticmethod
    def _get_data(name):
        if name == "test":
            columns = ['prefix', 'query_prediction', 'title', 'tag']
        else:
            columns = ['prefix', 'query_prediction', 'title', 'tag', 'label']

        data_name = os.path.join(RAW_DATA_PATH, "oppo_round1_{}.txt".format(name))
        df = pd.read_csv(data_name, names=columns, sep="\t", header=None, encoding="utf-8")

        return df

    @staticmethod
    def _loads(item):
        try:
            return json.loads(item)
        except (json.JSONDecodeError, TypeError):
            return json.loads("{}")

    @staticmethod
    def _get_apriori_df(df, train_df_length, columns=None):
        df = df.copy()

        train_df = df[:train_df_length]

        if columns is None:
            columns = ['prefix', 'complete_prefix', 'title', 'tag']

        ctr_columns = columns.copy()
        ctr_columns.extend(['prefix_title', 'prefix_tag', 'complete_prefix_title', 'complete_prefix_tag', 'title_tag'])
        apriori_df = df[ctr_columns]

        # click count and ctr
        for idx, column in enumerate(ctr_columns):
            click_column = "{column}_click".format(column=column)
            count_column = "{column}_count".format(column=column)
            ctr_column = "{column}_ctr".format(column=column)

            agg_dict = {click_column: "sum", count_column: "count"}
            column_apriori_df = train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
            column_apriori_df[ctr_column] = column_apriori_df[click_column] / (column_apriori_df[count_column] + 5)
            apriori_df = pd.merge(apriori_df, column_apriori_df, how='left', on=column)

        length = apriori_df.shape[0]
        all_columns = apriori_df.columns

        # apriori
        for column1 in columns:
            for column2 in columns:
                if column1 == column2:
                    continue

                if column1 in column2:
                    continue

                if column2 in column1:
                    continue

                temp_click_column = "{}_{}_click".format(column1, column2)
                if temp_click_column in all_columns:
                    click_column = temp_click_column
                else:
                    click_column = "{}_{}_click".format(column2, column1)

                temp_count_column = "{}_{}_count".format(column1, column2)
                if temp_count_column in all_columns:
                    count_column = temp_count_column
                else:
                    count_column = "{}_{}_count".format(column2, column1)

                click_column1 = "{column}_click".format(column=column1)
                count_column1 = "{column}_count".format(column=column1)
                click_column2 = "{column}_click".format(column=column2)
                count_column2 = "{column}_count".format(column=column2)

                click_confidence_column = "click_{}_{}_confidence".format(column1, column2)
                count_confidence_column = "count_{}_{}_confidence".format(column1, column2)
                click_lift_column = "click_{}_{}_lift".format(column1, column2)
                count_lift_column = "count_{}_{}_lift".format(column1, column2)

                # confidence = P(A&B)/P(A)
                apriori_df[click_confidence_column] = apriori_df[click_column] * 100 / (apriori_df[click_column1] + 5)
                apriori_df[count_confidence_column] = apriori_df[count_column] * 100 / (apriori_df[count_column1] + 5)

                # lift = P(A&B)/(P(A)*P(B))
                apriori_df[click_lift_column] = (apriori_df[click_column] / length) / (
                        (apriori_df[click_column1] * apriori_df[click_column2]) / (length * length))
                apriori_df[count_lift_column] = (apriori_df[count_column] / length) / (
                        (apriori_df[count_column1] * apriori_df[count_column2]) / (length * length))

        apriori_df = apriori_df.drop(columns=ctr_columns)
        return apriori_df

    @staticmethod
    def _get_expose_df(df, columns=None):
        df = df.copy()

        if columns is None:
            columns = ['prefix', 'complete_prefix', 'title', 'tag']

        expose_df = df[columns]

        for column1 in columns:
            for column2 in columns:

                if column1 == column2:
                    continue

                nunique_column_name = "{}_{}_nunique".format(column1, column2)
                temp_df = expose_df.groupby(column1)[column2].nunique().reset_index().rename(
                    columns={column2: nunique_column_name})
                expose_df = pd.merge(expose_df, temp_df, how='left', on=column1)

        expose_df = expose_df.drop(columns=columns)
        return expose_df

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
                    break

        if not predict_word_dict:
            return prefix

        predict_word_dict = sorted(predict_word_dict.items(), key=itemgetter(1), reverse=True)
        complete_prefix = predict_word_dict[0][0]
        return complete_prefix

    @staticmethod
    def _get_max_query_ratio(item):
        query_prediction = item['query_prediction']
        title = item['title']

        if not query_prediction:
            return 0

        for query_wrod, ratio in query_prediction.items():
            if title == query_wrod:
                if float(ratio) > 0.1:
                    return 1

        return 0

    @staticmethod
    def _get_word_length(item):
        item = str(item)

        word_cut = jieba.lcut(item)
        length = len(word_cut)
        return length

    @staticmethod
    def _get_small_query_num(item):
        small_query_num = 0

        for _, ratio in item.items():
            if float(ratio) <= 0.08:
                small_query_num += 1

        return small_query_num

    def _get_length_df(self, df):
        df = df.copy()

        columns = ['query_prediction', 'prefix', 'title']
        length_df = df[columns]

        length_df['max_query_ratio'] = length_df.apply(self._get_max_query_ratio, axis=1)
        length_df['prefix_word_num'] = length_df['prefix'].apply(self._get_word_length)
        length_df['title_word_num'] = length_df['title'].apply(self._get_word_length)
        length_df['title_len'] = length_df['title'].apply(len)
        length_df['small_query_num'] = length_df['query_prediction'].apply(self._get_small_query_num)

        length_df = length_df.drop(columns=columns)
        return length_df

    def get_processing(self):
        train_df = self._get_data(name="train")
        validate_df = self._get_data(name="vali")
        test_df = self._get_data(name="test")
        logger.info('finish load data!')

        train_df_length = train_df.shape[0]
        validate_df_length = validate_df.shape[0]
        df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True, sort=False)

        # make query prediction to json
        df["query_prediction"] = df["query_prediction"].apply(self._loads)

        # complete prefix
        df['complete_prefix'] = df[['prefix', 'query_prediction']].apply(self._get_complete_prefix, axis=1)
        logger.info('finish get complete prefix!')

        length_df = self._get_length_df(df)
        logger.info('finish get length df!')

        # clearn prefix and title
        df["prefix"] = df["prefix"].apply(char_cleaner)
        df["title"] = df["title"].apply(char_cleaner)
        df["complete_prefix"] = df["complete_prefix"].apply(char_cleaner)
        logger.info('finish clearn columns')

        # combine columns
        df['prefix_title'] = df[['prefix', 'title']].apply(lambda item: '_'.join(map(str, item)), axis=1)
        df['prefix_tag'] = df[['prefix', 'tag']].apply(lambda item: '_'.join(map(str, item)), axis=1)
        df['complete_prefix_title'] = df[['complete_prefix', 'title']].apply(lambda item: '_'.join(map(str, item)),
                                                                             axis=1)
        df['complete_prefix_tag'] = df[['complete_prefix', 'tag']].apply(lambda item: '_'.join(map(str, item)), axis=1)
        df['title_tag'] = df[['title', 'tag']].apply(lambda item: '_'.join(map(str, item)), axis=1)
        logger.info('finish combine columns')

        apriori_df = self._get_apriori_df(df, train_df_length)
        logger.info('finish get apriori df!')

        drop_columns = ['prefix_title', 'prefix_tag', 'title_tag', 'complete_prefix_title', 'complete_prefix_tag']
        df = df.drop(columns=drop_columns)

        expose_df = self._get_expose_df(df)
        logger.info('finish get expose df!')

        prefix_processing = PrefixProcessing()
        prefix_df = prefix_processing.get_prefix_df(df)
        logger.info('finish get prefix df!')

        query_processing = QueryProcessing()
        query_df = query_processing.get_query_df(df)
        logger.info('finish get query df!')

        df = pd.concat([df, length_df, apriori_df, expose_df, prefix_df, query_df], axis=1)
        logger.info('finish combine all df!')

        drop_columns = ['prefix', 'complete_prefix', 'query_prediction', 'title']
        df = df.drop(columns=drop_columns)

        # label encoder
        label_encoder = LabelEncoder()
        df['tag'] = label_encoder.fit_transform(df['tag'])
        logger.info('finish label encoder tag!')

        train_data = df[:train_df_length]
        train_data["label"] = train_data["label"].apply(int)

        validate_data = df[train_df_length:train_df_length + validate_df_length]
        validate_data["label"] = validate_data["label"].apply(int)

        test_data = df[train_df_length + validate_df_length:]
        test_data = test_data.drop(columns=["label"])

        train_data_name = os.path.join(ETL_DATA_PATH, "train_stat.csv")
        validate_data_name = os.path.join(ETL_DATA_PATH, "validate_stat.csv")
        test_data_name = os.path.join(ETL_DATA_PATH, "test_stat.csv")

        train_data.to_csv(train_data_name, index=False)
        validate_data.to_csv(validate_data_name, index=False)
        test_data.to_csv(test_data_name, index=False)


if __name__ == "__main__":
    t0 = time.time()
    processing = Processing()
    processing.get_processing()
    print(time.time() - t0)

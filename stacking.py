#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class Stacking(object):
    def __init__(self, kflod, df, train_df_length):
        self.kflod = kflod
        self.df = df.copy()
        self.train_df_length = train_df_length

    def _get_kflod(self, list_):
        list_array = np.array(list_)
        np.random.shuffle(list_array)

        list_part = np.array_split(list_array, self.kflod)
        for idx, list_item in enumerate(list_part):
            list_part_copy = list_part.copy()
            list_part_copy.pop(idx)

            other_list_part = np.concatenate(list_part_copy).ravel()

            yield other_list_part, list_item

    def get_stacking_df(self, columns=None):
        if columns is None:
            columns = ['prefix', 'title', 'tag', 'prefix_title', 'prefix_tag', 'title_tag']

        train_df = self.df[:self.train_df_length]
        train_df_index = train_df.index

        validate_test_df = self.df[self.train_df_length:]

        stacking_df = pd.DataFrame()
        stacking_columns = ['stacking_{}'.format(column) for column in columns]

        kfloder = self._get_kflod(train_df_index)
        kflod_list = list()
        for kflod_item in kfloder:
            kflod_list.append(kflod_item)

        for column in columns:
            stacking_train_df = pd.DataFrame()
            stacking_test_list = list()

            for train_index, test_index in kflod_list:
                k_train_df = train_df.loc[train_index]
                k_test_df = train_df.loc[test_index]

                click_column = "{column}_click".format(column=column)
                count_column = "{column}_count".format(column=column)
                stacking_column = "{column}_stacking".format(column=column)

                agg_dict = {click_column: "sum", count_column: "count"}
                _stacking_df = k_train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
                _stacking_df[stacking_column] = _stacking_df[click_column] / (_stacking_df[count_column] + 5)

                k_test_df = pd.merge(k_test_df, _stacking_df, how='left', on=column)
                stacking_train_df = pd.concat([stacking_train_df, k_test_df[stacking_column]],
                                              axis=0, ignore_index=False, sort=False)

                temp_df = pd.merge(validate_test_df, _stacking_df, how='left', on=column)
                temp_column_list = temp_df[stacking_column].tolist()
                stacking_test_list.append(temp_column_list)

            # train data
            stacking_train_df.sort_index(inplace=True)

            # validate + test data
            length = len(stacking_test_list)
            stacking_test_columns = ["stacking_{id}".format(id=i) for i in range(length)]
            stacking_test_df = pd.DataFrame(data=stacking_test_list)
            stacking_test_df = stacking_test_df.T
            stacking_test_df.columns = stacking_test_columns
            stacking_test_df['mean'] = stacking_test_df.mean(axis=1)

            # contact train validate test
            column_stacking_df = pd.concat([stacking_train_df, stacking_test_df['mean']],
                                           axis=0, ignore_index=True, sort=False)

            # contact column to stacking df
            stacking_df = pd.concat([stacking_df, column_stacking_df], axis=1)

        stacking_df.columns = stacking_columns
        return stacking_df

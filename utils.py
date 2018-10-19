#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re

import pandas as pd


def get_data(name):
    etl_path = os.path.join("data", "EtlData")

    if name == "train":
        file_name = "train.csv"
    elif name == "validate":
        file_name = "validate.csv"
    elif name == "test":
        file_name = "test.csv"
    else:
        raise FileNotFoundError()

    data_name = os.path.join(etl_path, file_name)

    df = pd.read_csv(data_name, header=0)

    return df


def char_cleaner(char):
    if not isinstance(char, str):
        char = "null"

    pattern = re.compile("[^a-zA-Z\u4E00-\u9FA5 ]")
    char = re.sub(pattern, "", char)
    char = char.lower()
    return char


def char_list_cheaner(char_list, stop_words=None):
    new_char_list = list()
    for char in char_list:
        if len(char) <= 1:
            continue
        if stop_words and char in stop_words:
            continue
        new_char_list.append(char)

    return new_char_list

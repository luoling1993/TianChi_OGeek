#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")
RESOURCE_PATH = os.path.join('resources')


def get_stop_words():
    stop_wrods_name = os.path.join(RESOURCE_PATH, 'stop_words.txt')
    _stop_words_list = list()
    with open(stop_wrods_name, encoding='utf-8') as f:
        for line in f:
            _stop_words_list.append(line.strip())

    _stop_words_set = set(_stop_words_list)
    return _stop_words_set


stop_words_set = get_stop_words()


def char_cleaner(char):
    if not isinstance(char, str):
        char = "null"
    pattern = re.compile("[^0-9a-zA-Z\u4E00-\u9FA5 ]")
    char = re.sub(pattern, "", char)
    char = char.lower()
    return char


def char_list_cheaner(char_list):
    new_char_list = list()
    for char in char_list:
        if len(char) == 0:
            continue
        if char in stop_words_set:
            continue
        new_char_list.append(char)

    return new_char_list

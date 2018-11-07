#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import os
import time

import jieba
from gensim.models import Word2Vec

from logconfig import config_logging
from utils import char_cleaner, char_list_cheaner

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
RawData = os.path.join(BASE_PATH, "RawData")

config_logging()
logger = logging.getLogger('w2v')


def get_sentence(name):
    if isinstance(name, list):
        name_list = name
    else:
        name_list = [name]

    for name in name_list:
        name = "oppo_round1_{fname}.txt".format(fname=name)
        file_path = os.path.join(RawData, name)
        if not os.path.exists(file_path):
            raise FileNotFoundError("{} Not Found!".format(file_path))

        with open(file_path, "r", encoding="utf-8") as f:
            line = f.readline()

            while line:
                line_arr = line.split("\t")

                query_prediction = line_arr[1]
                try:
                    sentences = json.loads(query_prediction)
                except json.JSONDecodeError:
                    sentences = json.loads("{}")

                for sentence in sentences:
                    yield sentence

                title = line_arr[2]
                yield title

                line = f.readline()


class MySentence(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for sentence in get_sentence(self.fname):
            sentence = char_cleaner(sentence)
            seg_list = jieba.lcut(sentence)
            seg_list = char_list_cheaner(seg_list)

            if seg_list:
                yield seg_list


def build_model(fname, size):
    sentences = MySentence(fname)
    model_name = "w2v_{}.bin".format(size)
    model_path = os.path.join("resources", model_name)
    my_model = Word2Vec(sentences, size=size, window=5, sg=1, hs=1, min_count=5, workers=10)
    my_model.wv.save_word2vec_format(model_path, binary=True)


if __name__ == "__main__":
    t0 = time.time()
    build_model(fname=['train', 'vali', 'test'], size=100)
    print(time.time() - t0)

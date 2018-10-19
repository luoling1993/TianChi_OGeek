#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import time

import jieba
from gensim.models import Word2Vec

from utils import char_cleaner, char_list_cheaner

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
RawData = os.path.join(BASE_PATH, "RawData")


def get_sentence(fname="train"):
    fname = "oppo_round1_{fname}.txt".format(fname=fname)
    file_path = os.path.join(RawData, fname)
    if not os.path.exists(file_path):
        raise FileNotFoundError("{} Not Found!".format(file_path))

    with open(file_path, "r", encoding="utf-8") as f:
        line = f.readline()

        while line:
            line_arr = line.split("\t")

            query_prediction = line_arr[1]
            sentences = json.loads(query_prediction)
            for sentence in sentences:
                yield char_cleaner(sentence)

            title = line_arr[2]
            yield char_cleaner(title)

            line = f.readline()


class MySentence(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for sentence in get_sentence(self.fname):
            seg_list = jieba.cut(sentence)
            seg_list = list(seg_list)
            seg_list = char_list_cheaner(seg_list)
            if seg_list:
                yield seg_list


def build_model(fname):
    sentences = MySentence(fname)
    model_name = "w2v.bin"
    my_model = Word2Vec(sentences, size=500, window=5, sg=1, hs=1, min_count=2, workers=10)

    my_model.wv.save_word2vec_format(model_name, binary=True)


if __name__ == "__main__":
    t0 = time.time()
    build_model(fname="train")
    print(time.time() - t0)

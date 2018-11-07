#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from stat_engineering import Processing as Sp
from w2v_engineering import Procossing as Wp

if __name__ == "__main__":
    # 统计特征
    stat_processing = Sp()
    stat_processing.get_processing()

    # w2v特征
    w2v_processing = Wp(force=False, size=100)
    w2v_processing.get_processing()

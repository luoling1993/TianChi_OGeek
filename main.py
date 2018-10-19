#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from features_engineering import Processing
from lgb_models import model_main

if __name__ == "__main__":
    processing = Processing()
    processing.get_processing()

    model_main()

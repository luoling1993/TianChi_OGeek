#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from logconfig import config_logging
from utils import char_cleaner, char_list_cheaner
import logging
import warnings

config_logging()
logger = logging.getLogger('embed_features')

warnings.filterwarnings('ignore')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from logging.config import dictConfig


def config_logging():
    logging_config = {
        'version': 1,
        'formatters': {
            'default': {'format': '%(asctime)s  %(name)s : %(levelname)s  %(message)s',
                        'datefmt': '%Y-%m-%d %H:%M:%S'}
        },
        'handlers': {
            'console': {
                'level': logging.DEBUG,
                'class': 'logging.StreamHandler',
                'formatter': 'default',
            },
            'file': {
                'level': logging.DEBUG,
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': 'ogeek.log',
                'maxBytes': 1024 * 1024 * 10,
                'backupCount': 1
            }
        },
        'loggers': {
            'test': {
                'level': logging.DEBUG,
                'handlers': ['console', 'file']
            }
        },
        'root': {
            'level': logging.DEBUG,
            'handlers': ['console', 'file']
        },
        'disable_existing_loggers': False
    }
    dictConfig(logging_config)

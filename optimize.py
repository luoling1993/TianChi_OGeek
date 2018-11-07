#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import warnings
from collections import namedtuple

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from logconfig import config_logging

warnings.filterwarnings('ignore')

config_logging()
logger = logging.getLogger('optimize')

Property = namedtuple('Property', ['min', 'max', 'type'])


class Optimize(object):
    _num_leaves = None
    _learning_rate = None
    _n_estimators = None
    _min_child_weight = None
    _min_child_samples = None
    _reg_alpha = None
    _reg_lambda = None
    _colsample_bytree = None
    _subsample = None

    def __init__(self, x_train, y_train, params, grid_params, iter_num=1):
        self.x_train = x_train
        self.y_train = y_train
        self.params = params
        self.grid_params = grid_params
        self.iter_num = iter_num

        # init property
        self.num_leaves = None
        self.learning_rate = None
        self.n_estimators = None
        self.min_child_weight = None
        self.min_child_samples = None
        self.reg_alpha = None
        self.reg_lambda = None
        self.colsample_bytree = None
        self.subsample = None

        # zip property as a dict
        self.property_dict = dict(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_weight=self.min_child_weight,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample
        )

    @property
    def num_leaves(self):
        return self._num_leaves

    @num_leaves.setter
    def num_leaves(self, value=None):
        default = [10, 1000, 'int']

        if value is None:
            self._num_leaves = Property._make(default)
        elif isinstance(value, list):
            self._num_leaves = Property._make(value)
        elif isinstance(value, dict):
            self._num_leaves = Property(**value)
        else:
            raise ValueError()

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value=None):
        default = [0.01, 0.5, 'float']

        if value is None:
            self._learning_rate = Property._make(default)
        elif isinstance(value, list):
            self._learning_rate = Property._make(value)
        elif isinstance(value, dict):
            self._learning_rate = Property(**value)
        else:
            raise ValueError()

    @property
    def n_estimators(self):
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, value=None):
        default = [500, 20000, 'int']

        if value is None:
            self._n_estimators = Property._make(default)
        elif isinstance(value, list):
            self._n_estimators = Property._make(value)
        elif isinstance(value, dict):
            self._n_estimators = Property(**value)
        else:
            raise ValueError()

    @property
    def min_child_weight(self):
        return self._min_child_weight

    @min_child_weight.setter
    def min_child_weight(self, value=None):
        default = [0.1, 10, 'float']

        if value is None:
            self._min_child_weight = Property._make(default)
        elif isinstance(value, list):
            self._min_child_weight = Property._make(value)
        elif isinstance(value, dict):
            self._min_child_weight = Property(**value)
        else:
            raise ValueError()

    @property
    def min_child_samples(self):
        return self._min_child_samples

    @min_child_samples.setter
    def min_child_samples(self, value=None):
        default = [50, 1000, 'int']

        if value is None:
            self._min_child_samples = Property._make(default)
        elif isinstance(value, list):
            self._min_child_samples = Property._make(value)
        elif isinstance(value, dict):
            self._min_child_samples = Property(**value)
        else:
            raise ValueError()

    @property
    def reg_alpha(self):
        return self._reg_alpha

    @reg_alpha.setter
    def reg_alpha(self, value=None):
        default = [0, 10, 'float']

        if value is None:
            self._reg_alpha = Property._make(default)
        elif isinstance(value, list):
            self._reg_alpha = Property._make(value)
        elif isinstance(value, dict):
            self._reg_alpha = Property(**value)
        else:
            raise ValueError()

    @property
    def reg_lambda(self):
        return self._reg_lambda

    @reg_lambda.setter
    def reg_lambda(self, value=None):
        default = [0, 10, 'float']

        if value is None:
            self._reg_lambda = Property._make(default)
        elif isinstance(value, list):
            self._reg_lambda = Property._make(value)
        elif isinstance(value, dict):
            self._reg_lambda = Property(**value)
        else:
            raise ValueError()

    @property
    def colsample_bytree(self):
        return self._colsample_bytree

    @colsample_bytree.setter
    def colsample_bytree(self, value=None):
        default = [0.5, 1, 'float']

        if value is None:
            self._colsample_bytree = Property._make(default)
        elif isinstance(value, list):
            self._colsample_bytree = Property._make(value)
        elif isinstance(value, dict):
            self._colsample_bytree = Property(**value)
        else:
            raise ValueError()

    @property
    def subsample(self):
        return self._subsample

    @subsample.setter
    def subsample(self, value=None):
        default = [0.5, 1, 'float']

        if value is None:
            self._subsample = Property._make(default)
        elif isinstance(value, list):
            self._subsample = Property._make(value)
        elif isinstance(value, dict):
            self._subsample = Property(**value)
        else:
            raise ValueError()

    @staticmethod
    def _get_values_list(low, high, dtype, size):
        linspace = np.linspace(low, high, size, dtype=dtype)

        if dtype == 'float':
            linspace = list(map(lambda item: round(item, 4), linspace))

        return linspace

    def _get_grid_params(self, values, key, best_value, size):
        max_value = max(values)
        min_value = min(values)

        property_item = self.property_dict[key]

        if best_value == max_value:
            if best_value == property_item.max:
                return [best_value]
            low = best_value
            high = property_item.max
            linspace = self._get_values_list(low, high, property_item.type, size)
        elif best_value == min_value:
            if best_value == property_item.min:
                return [best_value]
            low = min_value
            high = best_value
            linspace = self._get_values_list(low, high, property_item.type, size)
        else:
            best_index = values.index(best_value)
            low = values[best_index - 1]
            high = values[best_index + 1]
            linspace = self._get_values_list(low, high, property_item.type, size)

        linspace = list(set(linspace))
        return linspace

    def _update_params(self, best_params):
        for key, value in best_params.items():
            self.params[key] = value

    def _update_grid_params(self, best_params, size=4):
        for key, value in best_params.items():
            values = self.grid_params[key]

            values_list = self._get_grid_params(values, key, value, size)
            self.grid_params[key] = values_list

    def _optimize(self, params, grid_params):
        clf = lgb.LGBMClassifier(**params)
        grid_clf = GridSearchCV(clf, grid_params, cv=5, scoring='neg_log_loss', n_jobs=1, verbose=100)
        grid_clf.fit(self.x_train, self.y_train)
        return grid_clf

    def optimize(self):
        best_params = None

        while self.iter_num > 0:
            grid_clf = self._optimize(self.params, self.grid_params)

            best_params = grid_clf.best_params_
            best_score = grid_clf.best_score_

            logger.info('iter_num: {} best_params: {}'.format(self.iter_num, best_params))
            logger.info('iter_num: {} best_score: {}'.format(self.iter_num, best_score))

            self._update_params(best_params)
            self._update_grid_params(best_params)

            self.iter_num -= 1

        return best_params


class SimpleOptimize(object):
    def __init__(self, x_train, y_train, params, opt_params):
        self.x_train = x_train
        self.y_train = y_train
        self.params = params
        self.opt_params = opt_params

    def _update_params(self, best_params):
        for key, value in best_params.items():
            self.params[key] = value

    def optimize(self, grid=True, random=False):
        gbm = lgb.LGBMClassifier(**self.params)
        if grid:
            opt_gbm = GridSearchCV(gbm, self.opt_params, cv=5, scoring='neg_log_loss', refit="binary_logloss",
                                   n_jobs=1, verbose=100)
        elif random:
            opt_gbm = RandomizedSearchCV(gbm, self.opt_params, cv=5, scoring='neg_log_loss', refit="binary_logloss",
                                         n_jobs=1, verbose=100)
        else:
            raise ValueError()

        opt_gbm.fit(self.x_train, self.y_train)
        best_params = opt_gbm.best_params_
        best_score = opt_gbm.best_score_

        logger.info('best_params: {}'.format(best_params))
        logger.info('best_score: {}'.format(best_score))

        self._update_params(best_params)

        logger.info('update best params: {}'.format(self.params))
        return self.params

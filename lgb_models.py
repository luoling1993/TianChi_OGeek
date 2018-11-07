#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from logconfig import config_logging

warnings.filterwarnings('ignore')

config_logging()
logger = logging.getLogger('models')

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")


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

    one_hot_columns = ['tag', 'prefix_kmeans', 'title_kmeans', 'complete_prefix_kmeans']
    df = pd.get_dummies(df, columns=one_hot_columns)

    return df


def combine():
    names = ['train', 'test', 'validate']
    for name in names:
        stat_name = os.path.join(ETL_DATA_PATH, '{}_stat.csv'.format(name))
        stat_df = pd.read_csv(stat_name)

        w2v_name = os.path.join(ETL_DATA_PATH, '{}_w2v.csv'.format(name))
        w2v_df = pd.read_csv(w2v_name)

        df = pd.concat([stat_df, w2v_df], axis=1)

        df_name = os.path.join(ETL_DATA_PATH, '{}.csv'.format(name))
        df.to_csv(df_name, index=False)


def lgb_model(train_data, validate_data, test_data, parms, threshold, n_folds=5):
    columns = train_data.columns
    remove_columns = ["label"]
    features_columns = [column for column in columns if column not in remove_columns]

    train_data = pd.concat([train_data, validate_data], axis=0, ignore_index=True, sort=False)
    train_features = train_data[features_columns]
    train_labels = train_data["label"]

    validate_labels = validate_data["label"]

    test_data = pd.concat([validate_data, test_data], axis=0, ignore_index=True, sort=False)
    validate_data_length = validate_data.shape[0]
    test_features = test_data[features_columns]

    kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_features, train_labels)

    preds_list = list()
    for train_index, test_index in kfold:
        k_x_train = train_features.loc[train_index]
        k_y_train = train_labels.loc[train_index]
        k_x_test = train_features.loc[test_index]
        k_y_test = train_labels.loc[test_index]

        gbm = lgb.LGBMClassifier(**parms)
        gbm = gbm.fit(k_x_train, k_y_train,
                      eval_metric="logloss",
                      eval_set=[(k_x_train, k_y_train),
                                (k_x_test, k_y_test)],
                      eval_names=["train", "valid"],
                      early_stopping_rounds=100,
                      verbose=True)

        preds = gbm.predict_proba(test_features, num_iteration=gbm.best_iteration_)[:, 1]

        preds_list.append(preds)

    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df["mean"] = preds_df.mean(axis=1)

    preds_df["mean"] = preds_df["mean"].apply(lambda item: 1 if item >= threshold else 0)

    validate_preds = preds_df[:validate_data_length]
    test_preds = preds_df[validate_data_length:]

    logger.info('the avg of test is {}'.format(np.mean(test_preds["mean"])))

    f_score = f1_score(validate_labels, validate_preds["mean"])
    logger.info('validate f_score is {}'.format(f_score))
    logger.info('validate the avg of validate is {}'.format(np.mean(validate_preds["mean"])))

    predictions = pd.DataFrame({"predicted_score": test_preds["mean"]})

    predictions.to_csv("predict.csv", index=False, header=False)


def lgb_lr_model(train_data, validate_data, test_data, threshold, n_folds=5):
    columns = train_data.columns
    remove_columns = ["label"]
    features_columns = [column for column in columns if column not in remove_columns]

    validate_data_length = validate_data.shape[0]

    train_data = pd.concat([train_data, validate_data], axis=0, ignore_index=True, sort=False)
    train_features = train_data[features_columns]
    train_labels = train_data["label"]

    validate_labels = validate_data["label"]

    test_data = pd.concat([validate_data, test_data], axis=0, ignore_index=True, sort=False)
    test_features = test_data[features_columns]

    gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                             num_leaves=127,
                             reg_alpha=3,
                             reg_lambda=5,
                             max_depth=-1,
                             n_estimators=80,
                             objective='binary',
                             subsample=0.8,
                             colsample_bytree=0.8,
                             subsample_freq=1,
                             min_child_weight=0.1,
                             learning_rate=0.1,
                             random_state=2018,
                             n_jobs=-1,
                             min_child_samples=200)

    gbm.fit(train_features, train_labels, eval_metric='binary_logloss', early_stopping_rounds=100)

    lgb_train_leaf = gbm.predict(train_features, pred_leaf=True)
    lgb_test_leaf = gbm.predict(test_features, pred_leaf=True)

    leaf_columns = ['leaf_{}'.format(i) for i in range(lgb_train_leaf.shape[1])]

    train_leaf_df = pd.DataFrame(lgb_train_leaf, columns=leaf_columns)
    test_leaf_df = pd.DataFrame(lgb_test_leaf, columns=leaf_columns)

    train_features = pd.concat([train_features, train_leaf_df], axis=1)
    test_features = pd.concat([test_features, test_leaf_df], axis=1)

    df_features = pd.concat([train_features, test_features], ignore_index=True, sort=False, axis=0)
    cate_columns = ['tag', 'prefix_kmeans', 'title_kmeans', 'complete_kmeans']
    cate_columns.extend(leaf_columns)

    df_columns = df_features.columns
    num_columns = [column for column in df_columns if column not in cate_columns]

    train_csr = sparse.csr_matrix(train_features.shape[0], 0)
    test_csr = sparse.csr_matrix(test_features.shape[0], 0)

    # cate columns one-hot
    one_hot_encoder = OneHotEncoder()
    for col in cate_columns:
        one_hot_encoder.fit(df_features[col].values.reshape(-1, 1))

        train_encoder = one_hot_encoder.transform(train_features[col].values.reshape(-1, 1))
        train_csr = sparse.hstack((train_csr, train_encoder), 'csr', 'bool')

        test_encoder = one_hot_encoder.transform(test_features[col].values.reshape(-1, 1))
        test_csr = sparse.hstack((test_csr, test_encoder), 'csr', 'bool')

    # num columns min-max scaler
    min_max_scaler = MinMaxScaler()
    for col in num_columns:
        df_features[col].fillna(0, inplace=True)
        train_features[col].fillna(0, inplace=True)
        test_features[col].fillna(0, inplace=True)

        min_max_scaler.fit(np.array(df_features[col].values.tolist()).reshape(-1, 1))

        train_features[col] = min_max_scaler.transform(np.array(train_features[col].values.tolist()).reshape(-1, 1))
        test_features[col] = min_max_scaler.transform(np.array(test_features[col].values.tolist()).reshape(-1, 1))

    # combine num features
    train_csr = sparse.hstack(sparse.csr_matrix(train_features[num_columns], train_csr), 'csr').astype('float32')
    test_csr = sparse.hstack(sparse.csr_matrix(test_features[num_columns], test_csr), 'csr').astype('float32')

    lr_clf = LogisticRegression(penalty='l2', solver='sag', C=0.1, n_jobs=-1)

    kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_csr, train_labels)

    preds_list = list()
    for train_index, test_index in kfold:
        k_x_train = train_csr.loc[train_index]
        k_y_train = train_labels.loc[train_index]
        k_x_test = train_csr.loc[test_index]
        k_y_test = train_labels.loc[test_index]

        lr_clf.fit(k_x_train, k_y_train)

        eval_pred = lr_clf.predict_proba(k_x_test)[:, 1]
        eval_loss = log_loss(k_y_test, eval_pred)
        logger.info('eval log loss: {}'.format(eval_loss))

        test_preds = lr_clf.predict_proba(test_csr)[:, 1]
        preds_list.append(test_preds)

    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df["mean"] = preds_df.mean(axis=1)

    preds_df["mean"] = preds_df["mean"].apply(lambda item: 1 if item >= threshold else 0)

    validate_preds = preds_df[:validate_data_length]
    test_preds = preds_df[validate_data_length:]

    logger.info('the avg of test is {}'.format(np.mean(test_preds["mean"])))

    f_score = f1_score(validate_labels, validate_preds["mean"])
    logger.info('validate f_score is {}'.format(f_score))
    logger.info('validate the avg of validate is {}'.format(np.mean(validate_preds["mean"])))

    predictions = pd.DataFrame({"predicted_score": test_preds["mean"]})

    predictions.to_csv("predict.csv", index=False, header=False)


def model_main(model='lgb', threshold=0.5):
    lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "max_bin": 425,
        "subsample_for_bin": 20000,
        "objective": 'binary',
        "metric": 'logloss',
        "min_split_gain": 0,
        "min_child_weight": 0.001,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 3,
        "reg_lambda": 5,
        "seed": 2018,
        "n_jobs": -1,
        "verbose": 1,
        "silent": False
    }

    train_df = get_data(name="train")
    validate_df = get_data(name="validate")
    test_df = get_data(name="test")

    if model == 'lgb':
        lgb_model(train_df, validate_df, test_df, lgb_parms, threshold=threshold)
    elif model == 'lgb_lr':
        lgb_lr_model(train_df, validate_df, test_df, threshold=threshold)
    else:
        raise ValueError()


if __name__ == "__main__":
    combine()  # features combine, ignore it if features not change
    model_main(model='lgb', threshold=0.4)

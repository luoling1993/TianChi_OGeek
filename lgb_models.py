#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from utils import get_data

warnings.filterwarnings('ignore')


def lgb_model(train_data, validate_data, test_data, parms, n_folds=5):
    columns = train_data.columns
    remove_columns = ["label"]
    features_columns = [column for column in columns if column not in remove_columns]

    train_data = pd.concat([train_data, validate_data], axis=0, ignore_index=True, sort=False)
    train_features = train_data[features_columns]
    train_labels = train_data["label"]

    validate_data_length = validate_data.shape[0]
    validate_features = validate_data[features_columns]
    validate_labels = validate_data["label"]
    test_features = test_data[features_columns]
    test_features = pd.concat([validate_features, test_features], axis=0, ignore_index=True, sort=False)

    clf = lgb.LGBMClassifier(**parms)
    kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_features, train_labels)

    preds_list = list()
    for train_index, test_index in kfold:
        k_x_train = train_features.loc[train_index]
        k_y_train = train_labels.loc[train_index]
        k_x_test = train_features.loc[test_index]
        k_y_test = train_labels.loc[test_index]

        lgb_clf = clf.fit(k_x_train, k_y_train,
                          eval_names=["train", "valid"],
                          eval_metric="logloss",
                          eval_set=[(k_x_train, k_y_train),
                                    (k_x_test, k_y_test)],
                          early_stopping_rounds=100,
                          verbose=True)

        preds = lgb_clf.predict_proba(test_features, num_iteration=lgb_clf.best_iteration_)[:, 1]

        preds_list.append(preds)

    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df = preds_df.copy()
    preds_df["mean"] = preds_df.mean(axis=1)

    preds_df["mean"] = preds_df["mean"].apply(lambda item: 1 if item >= 0.5 else 0)

    validate_preds = preds_df[:validate_data_length]
    test_preds = preds_df[validate_data_length:]

    f_score = f1_score(validate_labels, validate_preds["mean"])
    print("The validate data's f1_score is {}".format(f_score))

    predictions = pd.DataFrame({"predicted_score": test_preds["mean"]})

    predictions.to_csv("predict.csv", index=False, header=False)


def model_main():
    lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "max_bin": 425,
        "subsample_for_bin": 20000,
        "objective": 'binary',
        "min_split_gain": 0,
        "min_child_weight": 0.001,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 1,
        "reg_alpha": 3,
        "reg_lambda": 5,
        "seed": 2018,
        "n_jobs": 5,
        "verbose": 1,
        "silent": False
    }

    train_df = get_data(name="train")
    validate_df = get_data(name="validate")
    test_df = get_data(name="test")

    lgb_model(train_df, validate_df, test_df, lgb_parms)


if __name__ == "__main__":
    model_main()

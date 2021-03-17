from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from data.fea_eng import rescale, normalization_scaler


features = [
    "SeedA",
    "SeedB",
    "WinRatioA",
    "GapAvgA",
    "WinRatioB",
    "GapAvgB",
    "OrdinalRankA",
    "OrdinalRankB",
    "SeedDiff",
    "OrdinalRankDiff",
    "WinRatioDiff",
    "GapAvgDiff",
    "adjoA",
    "adjoB",
    "adjdA",
    "adjdB",
    "luckA",
    "luckB",
    "adjoDiff",
    "adjdDiff",
    "luckDiff",
]
target = "WinA"


def lgb_kfold_model(
    params: Dict[str, float],
    df: pd.DataFrame,
    df_test_: pd.DataFrame = None,
    verbose: int = 1,
) -> np.ndarray:
    seasons = np.array([2015, 2016, 2017, 2018, 2019])
    weights = np.array([0.1, 0.1, 0.3, 0.3, 0.2])
    cvs = np.array([])
    pred_tests = np.zeros(df_test_.shape[0])

    lgb_params = {
        "num_leaves": 24,
        "colsample_bytree": 0.897413960991818,
        "subsample": 0.5028604067583243,
        "subsample_freq": 1,
        "min_child_samples": 76,
    }
    lgb_params["objective"] = "binary"
    lgb_params["boosting_type"] = "gbdt"
    lgb_params["n_estimators"] = 20000
    lgb_params["learning_rate"] = 0.05
    lgb_params["random_state"] = 42

    for season, weight in zip(seasons, weights):
        if verbose:
            print(f"\n Validating on season {season}")

        df_train = df[df["Season"] < season].reset_index(drop=True).copy()
        df_val = df[df["Season"] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()

        df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)

        model = LGBMClassifier(**lgb_params)
        model.fit(
            df_train[features],
            df_train[target],
            eval_set=[
                (df_train[features], df_train[target]),
                (df_val[features], df_val[target]),
            ],
            early_stopping_rounds=100,
            eval_metric="logloss",
            verbose=20,
        )

        pred = model.predict_proba(
            df_val[features], num_iteration=model.best_iteration_
        )[:, 1]

        if df_test is not None:
            pred_test = model.predict_proba(
                df_test[features], num_iteration=model.best_iteration_
            )[:, 1]

        pred_tests += weight * pred_test
        loss = log_loss(df_val[target].values, pred)
        cvs = np.append(cvs, loss)

        if verbose:
            print(f"\t -> Scored {loss:.5f}")

    print(f"\n Local CV is {np.sum(weights * cvs):.5f}")

    return pred_tests


def xgb_kfold_model(
    fold: int, df: pd.DataFrame, df_test_: pd.DataFrame = None, verbose: int = 1
) -> np.ndarray:
    seasons = df["Season"].unique()
    cvs = []
    pred_tests = []

    for season in seasons[fold:]:
        if verbose:
            print(f"\n Validating on season {season}")

        df_train = df[df["Season"] < season].reset_index(drop=True).copy()
        df_val = df[df["Season"] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()

        df_train, df_val, df_test = normalization_scaler(
            features, df_train, df_val, df_test
        )

        xgb_params = {
            "max_depth": 25,
            "learning_rate": 0.34329158619800576,
            "gamma": 3.0889803941034155,
            "subsample": 0.5042152969925682,
            "min_child_weight": 20,
            "colsample_bytree": 0.3,
        }
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "logloss"
        xgb_params["use_label_encoder"] = False
        xgb_params["n_estimators"] = 30000
        xgb_params["random_state"] = 42

        model = XGBClassifier(**xgb_params)
        model.fit(
            df_train[features],
            df_train[target],
            eval_set=[
                (df_train[features], df_train[target]),
                (df_val[features], df_val[target]),
            ],
            early_stopping_rounds=100,
            verbose=20,
        )

        pred = model.predict_proba(df_val[features])[:, 1]

        if df_test is not None:
            pred_test = model.predict_proba(df_test[features])[:, 1]

        pred_tests.append(pred_test)
        loss = log_loss(df_val[target].values, pred)
        cvs.append(loss)

        if verbose:
            print(f"\t -> Scored {loss:.5f}")

    print(f"\n Local CV is {np.mean(cvs):.5f}")

    pred_test = np.mean(pred_tests, 0)
    return pred_test


def logistic_kfold_model(
    df: pd.DataFrame, df_test_: pd.DataFrame = None, verbose: int = 1
) -> np.ndarray:
    seasons = np.array([2015, 2016, 2017, 2018, 2019])
    weights = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
    cvs = []
    pred_tests = np.zeros(df_test_.shape[0])

    for season, weight in zip(seasons, weights):
        if verbose:
            print(f"\n Validating on season {season}")

        df_train = df[df["Season"] < season].reset_index(drop=True).copy()
        df_val = df[df["Season"] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()

        df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)

        model = LogisticRegression(C=6)
        model.fit(df_train[features], df_train[target])

        pred = model.predict_proba(df_val[features])[:, 1]

        if df_test is not None:
            pred_test = model.predict_proba(df_test[features])[:, 1]

        pred_tests += weight * pred_test
        loss = log_loss(df_val[target].values, pred)
        cvs = np.append(cvs, loss)

        if verbose:
            print(f"\t -> Scored {loss:.5f}")

    print(f"\n Local CV is {np.sum(weights * cvs):.5f}")
    return pred_tests


def knn_kfold_model(
    fold: int, df: pd.DataFrame, df_test_: pd.DataFrame = None, verbose: int = 1
) -> np.ndarray:
    seasons = df["Season"].unique()
    cvs = []
    pred_tests = []

    for season in seasons[fold:]:
        if verbose:
            print(f"\n Validating on season {season}")

        df_train = df[df["Season"] < season].reset_index(drop=True).copy()
        df_val = df[df["Season"] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()

        df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)

        model = KNeighborsClassifier(n_neighbors=250)
        model.fit(df_train[features], df_train[target])

        pred = model.predict_proba(df_val[features])[:, 1]

        if df_test is not None:
            pred_test = model.predict_proba(df_test[features])[:, 1]

        pred_tests.append(pred_test)
        loss = log_loss(df_val[target].values, pred)
        cvs.append(loss)

        if verbose:
            print(f"\t -> Scored {loss:.5f}")

    print(f"\n Local CV is {np.mean(cvs):.5f}")

    pred_test = np.mean(pred_tests, 0)
    return pred_test


def forest_kfold_model(
    fold: int, df: pd.DataFrame, df_test_: pd.DataFrame = None, verbose: int = 1
) -> np.ndarray:
    seasons = df["Season"].unique()
    cvs = []
    pred_tests = []

    for season in seasons[fold:]:
        if verbose:
            print(f"\n Validating on season {season}")

        df_train = df[df["Season"] < season].reset_index(drop=True).copy()
        df_val = df[df["Season"] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()

        df_train, df_val, df_test = normalization_scaler(
            features, df_train, df_val, df_test
        )

        model = RandomForestClassifier()
        model.fit(
            df_train[features],
            df_train[target],
            eval_set=[
                (df_train[features], df_train[target]),
                (df_val[features], df_val[target]),
            ],
            early_stopping_rounds=100,
            verbose=20,
        )

        pred = model.predict_proba(df_val[features])[:, 1]

        if df_test is not None:
            pred_test = model.predict_proba(df_test[features])[:, 1]

        pred_tests.append(pred_test)
        loss = log_loss(df_val[target].values, pred)
        cvs.append(loss)

        if verbose:
            print(f"\t -> Scored {loss:.5f}")

    print(f"\n Local CV is {np.mean(cvs):.5f}")

    pred_test = np.mean(pred_tests, 0)
    return pred_test
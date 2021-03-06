import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

from data.fea_eng import rescale, normalization_scaler


features = [
    "SeedA",
    "SeedB",
    # "WinRatioA",
    "GapAvgA",
    # "WinRatioB",
    "GapAvgB",
    # "OrdinalRankA",
    # "OrdinalRankB",
    "SeedDiff",
    "OrdinalRankDiff",
    "WinRatioDiff",
    "GapAvgDiff",
]
target = "WinA"


def lgb_kfold_model(
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

        lgb_params = {
            "num_leaves": 62,
            "reg_alpha": 0.010128063599316449,
            "reg_lambda": 0.5824200175810507,
            "colsample_bytree": 0.7521104418815311,
            "subsample": 0.4004123428146916,
            "subsample_freq": 3,
            "min_child_samples": 85,
        }

        lgb_params["objective"] = "binary"
        lgb_params["boosting_type"] = "gbdt"
        lgb_params["n_estimators"] = 20000
        lgb_params["learning_rate"] = 0.05
        lgb_params["random_state"] = 42

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

        pred_tests.append(pred_test)
        loss = log_loss(df_val[target].values, pred)
        cvs.append(loss)

        if verbose:
            print(f"\t -> Scored {loss:.5f}")

    print(f"\n Local CV is {np.mean(cvs):.5f}")

    pred_test = np.mean(pred_tests, 0)
    return pred_test


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
            "max_depth": 18,
            "learning_rate": 0.5857698987440044,
            "reg_lambda": 0.001180513673871078,
            "reg_alpha": 0.42412647846599194,
            "gamma": 3.3593638447345113,
            "subsample": 0.4069684849403186,
            "min_child_weight": 18,
            "colsample_bytree": 0.2,
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

        model = LogisticRegression(C=6)
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


def voting_kfold_model(
    fold: int,
    df: pd.DataFrame,
    df_test_: pd.DataFrame = None,
    verbose: int = 1,
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

        lgb_params = {
            "num_leaves": 47,
            "reg_alpha": 2.0322145576381225,
            "reg_lambda": 1.4276863976112468,
            "colsample_bytree": 0.8903033132580764,
            "subsample": 0.46383593231999987,
            "subsample_freq": 3,
            "min_child_samples": 89,
        }

        lgb_params["objective"] = "binary"
        lgb_params["boosting_type"] = "gbdt"
        lgb_params["n_estimators"] = 20000
        lgb_params["learning_rate"] = 0.05
        lgb_params["random_state"] = 42

        lgb_model = LGBMClassifier(**lgb_params)

        xgb_params = {
            "max_depth": 18,
            "learning_rate": 0.5857698987440044,
            "reg_lambda": 0.001180513673871078,
            "reg_alpha": 0.42412647846599194,
            "gamma": 3.3593638447345113,
            "subsample": 0.4069684849403186,
            "min_child_weight": 18,
            "colsample_bytree": 0.2,
        }
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "logloss"
        xgb_params["use_label_encoder"] = False
        xgb_params["n_estimators"] = 30000
        xgb_params["random_state"] = 42

        xgb_model = XGBClassifier(**xgb_params)

        knn_model = KNeighborsClassifier(n_neighbors=250)
        logistic_model = LogisticRegression(C=6)

        model = VotingClassifier(
            estimators=[
                ("LGBM", lgb_model),
                ("XGB", xgb_model),
                ("LR", logistic_model),
                ("knn", knn_model),
            ],
            voting="soft",
        )
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


def stacking_kfold_model(
    fold: int,
    df: pd.DataFrame,
    df_test_: pd.DataFrame = None,
    verbose: int = 1,
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

        lgb_params = {
            "num_leaves": 47,
            "reg_alpha": 2.0322145576381225,
            "reg_lambda": 1.4276863976112468,
            "colsample_bytree": 0.8903033132580764,
            "subsample": 0.46383593231999987,
            "subsample_freq": 3,
            "min_child_samples": 89,
        }

        lgb_params["objective"] = "binary"
        lgb_params["boosting_type"] = "gbdt"
        lgb_params["n_estimators"] = 20000
        lgb_params["learning_rate"] = 0.05
        lgb_params["random_state"] = 42

        lgb_model = LGBMClassifier(**lgb_params)

        xgb_params = {
            "max_depth": 18,
            "learning_rate": 0.5857698987440044,
            "reg_lambda": 0.001180513673871078,
            "reg_alpha": 0.42412647846599194,
            "gamma": 3.3593638447345113,
            "subsample": 0.4069684849403186,
            "min_child_weight": 18,
            "colsample_bytree": 0.2,
        }
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "logloss"
        xgb_params["use_label_encoder"] = False
        xgb_params["n_estimators"] = 30000
        xgb_params["random_state"] = 42

        xgb_model = XGBClassifier(**xgb_params)

        knn_model = KNeighborsClassifier(n_neighbors=250)
        logistic_model = LogisticRegression(C=6)

        model = StackingClassifier(
            estimators=[
                ("XGB", xgb_model),
                ("LR", logistic_model),
                ("knn", knn_model),
            ],
            final_estimator=lgb_model,
        )
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

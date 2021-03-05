import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from data.fea_eng import maxabs_scaler, robust_transformer_scaler, rescale

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

        df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)

        lgb_params = {
            "num_leaves": 46,
            "reg_alpha": 0.004835967944598257,
            "reg_lambda": 0.09941680956568132,
            "colsample_bytree": 0.4813842319363184,
            "subsample": 0.6502207469424413,
            "subsample_freq": 6,
            "min_child_samples": 30,
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

        df_train, df_val, df_test = maxabs_scaler(features, df_train, df_val, df_test)

        xgb_params = {
            "max_depth": 17,
            "learning_rate": 0.2998036559586344,
            "reg_lambda": 3.4863807958372117,
            "reg_alpha": 0.5380496049743049,
            "gamma": 1.2676788981943639,
            "subsample": 0.8,
            "min_child_weight": 31,
            "colsample_bytree": 0.4,
        }

        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "logloss"
        xgb_params["use_label_encoder"] = False
        xgb_params["n_estimators"] = 3000
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


def cat_kfold_model(
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

        df_train, df_val, df_test = maxabs_scaler(features, df_train, df_val, df_test)

        cat_params = {
            "verbose": 20,
            "eval_metric": "Logloss",
            "random_state": 42,
            "num_boost_round": 20000,
            "max_depth": 4,
            "learning_rate": 0.010155932673991064,
            "l2_leaf_reg": 9.083564967810792,
            "bagging_temperature": 1.5604104823748561,
            "penalties_coefficient": 2.0037171127502633,
        }
        model = CatBoostClassifier(**cat_params)
        model.fit(
            df_train[features],
            df_train[target],
            eval_set=[
                (df_train[features], df_train[target]),
                (df_val[features], df_val[target]),
            ],
            early_stopping_rounds=200,
            verbose=500,
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

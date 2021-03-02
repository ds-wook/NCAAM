from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

from data.fea_eng import rescale

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


def kfold_model(
    fold: int, df: pd.DataFrame, df_test_: pd.DataFrame = None, verbose: int = 1
) -> List[float]:
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
            "num_leaves": 65,
            "colsample_bytree": 0.6432045758305666,
            "subsample": 0.5733933267821268,
            "subsample_freq": 1,
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
            pred_test = model.predict_proba(df_test[features])[:, 1]

        pred_tests.append(pred_test)
        loss = log_loss(df_val[target].values, pred)
        cvs.append(loss)

        if verbose:
            print(f"\t -> Scored {loss:.3f}")

    print(f"\n Local CV is {np.mean(cvs):.3f}")

    pred_test = np.mean(pred_tests, 0)
    return pred_test

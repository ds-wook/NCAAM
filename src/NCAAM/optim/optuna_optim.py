import numpy as np
from optuna.trial import Trial
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

from data.dataset import load_dataset
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

df, df_test = load_dataset()


def objective(trial: Trial) -> float:
    global df, df_test
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 20000,
        "learning_rate": 0.05,
        "random_state": 42,
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        # "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 1.0),
        # "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        # "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }
    seasons = df["Season"].unique()
    cvs = []
    pred_tests = []

    for season in seasons[12:]:
        df_train = df[df["Season"] < season].reset_index(drop=True).copy()
        df_val = df[df["Season"] == season].reset_index(drop=True).copy()
        df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)

        model = LGBMClassifier(**params)
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
        loss = np.mean(cvs)
    return loss

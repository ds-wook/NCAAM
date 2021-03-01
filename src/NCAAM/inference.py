import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from data.dataset import tourney, path
from data.fea_eng import train_test_dataset
from model.group_fold import group_kfold_model

tourney, test = train_test_dataset(tourney, stage=True)

X = tourney.drop(["Season", "TeamID1", "TeamID2", "result"], axis=1)
y = tourney["result"]
s = tourney["Season"]

X_test = test.drop(["ID", "Season", "TeamID1", "TeamID2"], axis=1)
MSampleSubmission = pd.read_csv(path + "/MSampleSubmissionStage1.csv")

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../res/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    parse.add_argument("--fold", type=int, help="Input num_fold", default=5)
    args = parse.parse_args()
    lgb_params = {
        "num_leaves": 81,
        "max_depth": 4,
        "reg_alpha": 0.0010290634955717258,
        "reg_lambda": 0.01829600991070637,
        "colsample_bytree": 0.8323239215532339,
        "subsample": 0.560337580160405,
        "subsample_freq": 3,
        "min_child_samples": 22,
    }

    lgb_params["objective"] = "binary"
    lgb_params["boosting"] = "gbdt"
    lgb_params["n_estimators"] = 20000
    lgb_params["learning_rate"] = 0.05
    lgb_params["random_state"] = 42

    lgb_model = LGBMClassifier(**lgb_params)
    lgb_preds = group_kfold_model(lgb_model, args.fold, s, X, y, X_test)
    idx = lgb_preds.shape[0] // 2
    lgb_preds[idx:] = 1 - lgb_preds[idx:]

    pred = (
        pd.concat([test.ID, pd.Series(lgb_preds)], axis=1)
        .groupby("ID")[0]
        .mean()
        .reset_index()
        .rename(columns={0: "Pred"})
    )
    sub = MSampleSubmission.drop(["Pred"], axis=1).merge(pred, on="ID")
    sub.to_csv(args.path + args.file, index=False)
    sub.head()

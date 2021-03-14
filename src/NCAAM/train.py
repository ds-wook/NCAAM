import argparse

import numpy as np

from data.dataset import load_dataset
from model.cv_fold import (
    lgb_kfold_model,
    xgb_kfold_model,
    logistic_kfold_model,
    knn_kfold_model,
    # voting_kfold_model,
    # stacking_kfold_model,
)

df, df_test = load_dataset()

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../res/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    parse.add_argument("--fold", type=int, help="Input num_fold", default=5)
    args = parse.parse_args()
    lgb_pred = lgb_kfold_model(args.fold, df, df_test)
    # xgb_pred = xgb_kfold_model(args.fold, df, df_test)
    # logistic_pred = logistic_kfold_model(args.fold, df, df_test)
    # knn_pred = knn_kfold_model(args.fold, df, df_test)
    # voting_pred = voting_kfold_model(args.fold, df, df_test)
    # stacking_pred = stacking_kfold_model(args.fold, df, df_test)
    sub = df_test[["ID", "Pred"]].copy()
    sub["Pred"] = lgb_pred
    # sub.loc[sub["Pred"] < 0.025, "Pred"] = 0.025
    # sub.loc[sub["Pred"] > 0.975, "Pred"] = 0.975
    sub.to_csv(args.path + args.file, index=False)

import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset import load_dataset
from model.cv_fold import (
    lgb_kfold_model,
    xgb_kfold_model,
    logistic_kfold_model,
    knn_kfold_model,
    voting_kfold_model,
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
    xgb_pred = xgb_kfold_model(args.fold, df, df_test)
    logistic_pred = logistic_kfold_model(args.fold, df, df_test)
    knn_pred = knn_kfold_model(args.fold, df, df_test)
    # voting_pred = voting_kfold_model(args.fold, df, df_test)
    # stacking_pred = stacking_kfold_model(args.fold, df, df_test)
    sub = df_test[["ID", "Pred"]].copy()
    fig, ax = plt.subplots()
    sub["Pred"] = np.average(
        [lgb_pred, xgb_pred, logistic_pred, knn_pred],
        axis=0,
        weights=[0.45, 0.45, 0.05, 0.05],
    )
    sns.histplot(sub["Pred"], ax=ax)
    plt.savefig("../../image/ensemble_pred.png")
    sub.to_csv(args.path + args.file, index=False)

import argparse

import numpy as np
import pandas as pd

from data.dataset import load_dataset
from model.cv_fold import lgb_kfold_model


df, df_test = load_dataset()

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../submission/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    args = parse.parse_args()
    lgb_pred = lgb_kfold_model(df, df_test)
    sub = df_test[["ID", "Pred"]].copy()
    sub["Pred"] = lgb_pred
    sub.to_csv(args.path + args.file, index=False)

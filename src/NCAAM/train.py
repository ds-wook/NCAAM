import argparse

from data.dataset import load_dataset
from model.cv_fold import lgb_kfold_model

df, df_test = load_dataset()

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../submission/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    parse.add_argument("--fold", type=int, help="Input num_fold", default=5)
    parse.add_argument(
        "--params", type=str, help="Input parameters", default="params.pkl"
    )
    args = parse.parse_args()
    lgb_pred = lgb_kfold_model(args.params, df, df_test)
    sub = df_test[["ID", "Pred"]].copy()
    sub["Pred"] = lgb_pred
    sub.loc[sub["Pred"] < 0.025, "Pred"] = 0.025
    sub.loc[sub["Pred"] > 0.975, "Pred"] = 0.975
    sub.to_csv(args.path + args.file, index=False)

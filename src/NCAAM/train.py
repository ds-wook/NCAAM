import argparse

from data.dataset import load_dataset
from model.cv_fold import lgb_kfold_model, xgb_kfold_model, cat_kfold_model

df, df_test = load_dataset()

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../res/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    parse.add_argument("--fold", type=int, help="Input num_fold", default=5)
    args = parse.parse_args()
    # lgb_pred = lgb_kfold_model(args.fold, df, df_test)
    xgb_pred = xgb_kfold_model(args.fold, df, df_test)
    # cat_pred = cat_kfold_model(args.fold, df, df_test)
    sub = df_test[["ID", "Pred"]].copy()
    sub["Pred"] = xgb_pred
    sub.to_csv(args.path + args.file, index=False)

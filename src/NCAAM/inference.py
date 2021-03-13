import argparse
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../res/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    args = parse.parse_args()

    winner_solution = pd.read_csv("../../res/MPred_1.csv")
    my_solution = pd.read_csv("../../res/3fold_lgbm_final.csv")
    sub = pd.read_csv("../../input/ncaam-march-mania-2021/MSampleSubmissionStage1.csv")
    sub["Pred"] = np.average(
        [winner_solution["Pred"], my_solution["Pred"]], axis=0, weights=[0.1, 0.9]
    )
    sub.to_csv(args.file + args.file, index=False)

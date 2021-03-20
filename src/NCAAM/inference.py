import argparse
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../submission/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    args = parse.parse_args()
    path = "../../input/ncaam-march-mania-2021/MDataFiles_Stage2/"
    winner_solution = pd.read_csv("../../submission/MPred_1.csv")
    winner_solution2 = pd.read_csv("../../submission/logistic.csv")
    my_solution = pd.read_csv("../../submission/lgbm_weights_prob2.csv")
    sub = pd.read_csv(path + "MSampleSubmissionStage2.csv")

    sub["Pred"] = np.average(
        [winner_solution["Pred"], winner_solution2["Pred"], my_solution["Pred"]],
        axis=0,
        weights=[0.3, 0.4, 0.3],
    )
    sub.to_csv(args.path + args.file, index=False)

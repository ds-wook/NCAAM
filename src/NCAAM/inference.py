import argparse

import numpy as np
import pandas as pd


parse = argparse.ArgumentParser("Training!")
parse.add_argument(
    "--path", type=str, help="Input data save path", default="../../res/"
)
parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
args = parse.parse_args()

sub = pd.read_csv("../../input/ncaam-march-mania-2021/MSampleSubmissionStage1.csv")
winner_solution = pd.read_csv("../../res/MPred_1.csv")
my_solution = pd.read_csv("../../res/lgb_kfold2.csv")

sub["Pred"] = np.average(
    [winner_solution["Pred"], my_solution["Pred"]], axis=0, weights=[0.7, 0.3]
)
sub.to_csv(args.path + args.file, index=False)

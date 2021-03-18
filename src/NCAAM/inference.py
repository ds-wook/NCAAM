import argparse

import numpy as np
import pandas as pd


parse = argparse.ArgumentParser("Training!")
parse.add_argument(
    "--path", type=str, help="Input data save path", default="../../submission/"
)
parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
args = parse.parse_args()

sub = pd.read_csv(
    "../../input/ncaam-march-mania-2021/MDataFiles_Stage1/MSampleSubmissionStage1.csv"
)
winner_solution = pd.read_csv("../../res/MPred_1.csv")
my_solution = pd.read_csv("../../res/test_model5.csv")

sub["Pred"] = 0.7 * winner_solution["Pred"] + 0.3 * my_solution["Pred"]
sub.to_csv(args.path + args.file, index=False)

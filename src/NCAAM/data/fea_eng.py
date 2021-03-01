from typing import Tuple

import numpy as np
import pandas as pd

from data.dataset import wl_1, wl_2, scr_1, scr_2, dt_1, dt_2, MOR_1, MOR_2, path


def merge_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        wl_1, how="left", left_on=["Season", "TeamID1"], right_on=["Season", "TeamID"]
    )
    df = df.merge(
        wl_2, how="left", left_on=["Season", "TeamID2"], right_on=["Season", "TeamID"]
    )
    df = df.drop(["TeamID_x", "TeamID_y"], axis=1)

    df = df.merge(
        scr_1, how="left", left_on=["Season", "TeamID1"], right_on=["Season", "TeamID"]
    )
    df = df.merge(
        scr_2, how="left", left_on=["Season", "TeamID2"], right_on=["Season", "TeamID"]
    )
    df = df.drop(["TeamID_x", "TeamID_y"], axis=1)
    # df['win_pct_A_diff'] = df['win_pct_A_1'] - df['win_pct_A_2']
    # df['win_pct_N_diff'] = df['win_pct_N_1'] - df['win_pct_N_2']
    # df['win_pct_H_diff'] = df['win_pct_H_1'] - df['win_pct_H_2']
    # df['win_pct_All_diff'] = df['win_pct_All_1'] - df['win_pct_All_2']

    # df['Score_A_diff'] = df['Score_A_1'] - df['Score_A_2']
    # df['Score_N_diff'] = df['Score_N_1'] - df['Score_N_2']
    # df['Score_H_diff'] = df['Score_H_1'] - df['Score_H_2']
    # df['Score_All_diff'] = df['Score_All_1'] - df['Score_All_2']

    # df['relScore_A_diff'] = df['relScore_A_1'] - df['relScore_A_2']
    # df['relScore_N_diff'] = df['relScore_N_1'] - df['relScore_N_2']
    # df['relScore_H_diff'] = df['relScore_H_1'] - df['relScore_H_2']
    df['relScore_All_diff'] = df['relScore_All_1'] - df['relScore_All_2']
    df = df.merge(
        dt_1, how="left", left_on=["Season", "TeamID1"], right_on=["Season", "TeamID"]
    )
    df = df.merge(
        dt_2, how="left", left_on=["Season", "TeamID2"], right_on=["Season", "TeamID"]
    )
    df = df.drop(["TeamID_x", "TeamID_y"], axis=1)

    df = df.merge(
        MOR_1, how="left", left_on=["Season", "TeamID1"], right_on=["Season", "TeamID"]
    )
    df = df.merge(
        MOR_2, how="left", left_on=["Season", "TeamID2"], right_on=["Season", "TeamID"]
    )
    df = df.drop(["TeamID_x", "TeamID_y"], axis=1)

    df["OrdinalRank_127_128_diff"] = (
        df["OrdinalRank_127_128_1"] - df["OrdinalRank_127_128_2"]
    )

    df = df.fillna(-1)

    for col in df.columns:
        if (df[col] == np.inf).any() or (df[col] == -np.inf).any():
            df[col][(df[col] == np.inf) | (df[col] == -np.inf)] = -1

    return df


def train_test_dataset(
    tourney: pd.DataFrame, stage: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    tourney = merge_data(tourney)
    tourney = tourney.loc[tourney.Season >= 2003, :].reset_index(drop=True)

    if stage:
        tourney = tourney.loc[tourney.Season < 2015, :]
        MSampleSubmission = pd.read_csv(path + "/MSampleSubmissionStage1.csv")
    else:
        MSampleSubmission = pd.read_csv(
            path + None
        )  # put stage 2 submission file link here

    test1 = MSampleSubmission.copy()
    test1["Season"] = test1.ID.apply(lambda x: int(x[0:4]))
    test1["TeamID1"] = test1.ID.apply(lambda x: int(x[5:9]))
    test1["TeamID2"] = test1.ID.apply(lambda x: int(x[10:14]))

    test2 = MSampleSubmission.copy()
    test2["Season"] = test2.ID.apply(lambda x: int(x[0:4]))
    test2["TeamID1"] = test2.ID.apply(lambda x: int(x[10:14]))
    test2["TeamID2"] = test2.ID.apply(lambda x: int(x[5:9]))

    test = pd.concat([test1, test2]).drop(["Pred"], axis=1)
    test = merge_data(test)

    return tourney, test

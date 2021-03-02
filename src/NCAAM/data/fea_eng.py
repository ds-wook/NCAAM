import re

import pandas as pd


def get_round(day: int) -> int:
    round_dic = {
        134: 0,
        135: 0,
        136: 1,
        137: 1,
        138: 2,
        139: 2,
        143: 3,
        144: 3,
        145: 4,
        146: 4,
        152: 5,
        154: 6,
    }
    try:
        return round_dic[day]
    except KeyError:
        print(f"Unknow day : {day}")
        return 0


def treat_seed(seed: int) -> int:
    return int(re.sub("[^0-9]", "", seed))


def add_loosing_matches(win_df: pd.DataFrame) -> pd.DataFrame:
    win_rename = {
        "WTeamID": "TeamIdA",
        "WScore": "ScoreA",
        "LTeamID": "TeamIdB",
        "LScore": "ScoreB",
        "SeedW": "SeedA",
        "SeedL": "SeedB",
        "WinRatioW": "WinRatioA",
        "WinRatioL": "WinRatioB",
        "GapAvgW": "GapAvgA",
        "GapAvgL": "GapAvgB",
        "OrdinalRankW": "OrdinalRankA",
        "OrdinalRankL": "OrdinalRankB",
    }

    lose_rename = {
        "WTeamID": "TeamIdB",
        "WScore": "ScoreB",
        "LTeamID": "TeamIdA",
        "LScore": "ScoreA",
        "SeedW": "SeedB",
        "SeedL": "SeedA",
        "GapAvgW": "GapAvgB",
        "GapAvgL": "GapAvgA",
        "WinRatioW": "WinRatioB",
        "WinRatioL": "WinRatioA",
        "OrdinalRankW": "OrdinalRankB",
        "OrdinalRankL": "OrdinalRankA",
    }

    win_df = win_df.copy()
    lose_df = win_df.copy()

    win_df = win_df.rename(columns=win_rename)
    lose_df = lose_df.rename(columns=lose_rename)

    return pd.concat([win_df, lose_df], 0, sort=False)


def rescale(features, df_train, df_val, df_test=None):
    min_ = df_train[features].min()
    max_ = df_train[features].max()

    df_train[features] = (df_train[features] - min_) / (max_ - min_)
    df_val[features] = (df_val[features] - min_) / (max_ - min_)

    if df_test is not None:
        df_test[features] = (df_test[features] - min_) / (max_ - min_)

    return df_train, df_val, df_test

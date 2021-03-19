from typing import Tuple

import numpy as np
import pandas as pd
from data.fea_eng import (
    get_round,
    treat_seed,
    add_loosing_matches,
)


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = "../../input/ncaam-march-mania-2021/MDataFiles_Stage2/"

    df_seeds = pd.read_csv(path + "MNCAATourneySeeds.csv")

    df_season_results = pd.read_csv(path + "MRegularSeasonCompactResults.csv")
    df_season_results.drop(["NumOT", "WLoc"], axis=1, inplace=True)
    df_season_results["ScoreGap"] = (
        df_season_results["WScore"] - df_season_results["LScore"]
    )

    num_win = df_season_results.groupby(["Season", "WTeamID"]).count()
    num_win = num_win.reset_index()[["Season", "WTeamID", "DayNum"]].rename(
        columns={"DayNum": "NumWins", "WTeamID": "TeamID"}
    )

    num_loss = df_season_results.groupby(["Season", "LTeamID"]).count()
    num_loss = num_loss.reset_index()[["Season", "LTeamID", "DayNum"]].rename(
        columns={"DayNum": "NumLosses", "LTeamID": "TeamID"}
    )

    gap_win = df_season_results.groupby(["Season", "WTeamID"]).mean().reset_index()
    gap_win = gap_win[["Season", "WTeamID", "ScoreGap"]].rename(
        columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"}
    )

    gap_loss = df_season_results.groupby(["Season", "LTeamID"]).mean().reset_index()
    gap_loss = gap_loss[["Season", "LTeamID", "ScoreGap"]].rename(
        columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"}
    )

    df_features_season_w = (
        df_season_results.groupby(["Season", "WTeamID"])
        .count()
        .reset_index()[["Season", "WTeamID"]]
        .rename(columns={"WTeamID": "TeamID"})
    )
    df_features_season_l = (
        df_season_results.groupby(["Season", "LTeamID"])
        .count()
        .reset_index()[["Season", "LTeamID"]]
        .rename(columns={"LTeamID": "TeamID"})
    )

    df_features_season = (
        pd.concat([df_features_season_w, df_features_season_l], 0)
        .drop_duplicates()
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )

    df_features_season = df_features_season.merge(
        num_win, on=["Season", "TeamID"], how="left"
    )
    df_features_season = df_features_season.merge(
        num_loss, on=["Season", "TeamID"], how="left"
    )
    df_features_season = df_features_season.merge(
        gap_win, on=["Season", "TeamID"], how="left"
    )
    df_features_season = df_features_season.merge(
        gap_loss, on=["Season", "TeamID"], how="left"
    )
    df_features_season.fillna(0, inplace=True)

    df_features_season["WinRatio"] = df_features_season["NumWins"] / (
        df_features_season["NumWins"] + df_features_season["NumLosses"]
    )

    df_features_season["GapAvg"] = (
        df_features_season["NumWins"] * df_features_season["GapWins"]
        - df_features_season["NumLosses"] * df_features_season["GapLosses"]
    ) / (df_features_season["NumWins"] + df_features_season["NumLosses"])

    df_features_season.drop(
        ["NumWins", "NumLosses", "GapWins", "GapLosses"], axis=1, inplace=True
    )

    df_tourney_results = pd.read_csv(path + "MNCAATourneyCompactResults.csv")
    df_tourney_results.drop(["NumOT", "WLoc"], axis=1, inplace=True)
    df_tourney_results["Round"] = df_tourney_results["DayNum"].apply(get_round)

    df_massey = pd.read_csv(path + "MMasseyOrdinals.csv")
    df_massey = (
        df_massey[df_massey["RankingDayNum"] == 133]
        .drop("RankingDayNum", axis=1)
        .reset_index(drop=True)
    )  # use first day of the tournament

    systems = []
    for year in range(2003, 2019):
        r = df_massey[df_massey["Season"] == year]
        systems.append(r["SystemName"].unique())

    all_systems = list(set(list(np.concatenate(systems))))

    common_systems = []
    # ['RTH', 'RPI', 'COL', 'AP', 'POM', 'USA', 'DOL', 'MOR', 'SAG', 'WOL', 'WLK']
    for system in all_systems:
        common = True
        for system_years in systems:
            if system not in system_years:
                common = False
        if common:
            common_systems.append(system)

    df_massey = df_massey[df_massey["SystemName"].isin(common_systems)].reset_index(
        drop=True
    )

    df = df_tourney_results.copy()
    df = df[df["Season"] >= 2003].reset_index(drop=True)

    df = (
        pd.merge(
            df,
            df_seeds,
            how="left",
            left_on=["Season", "WTeamID"],
            right_on=["Season", "TeamID"],
        )
        .drop("TeamID", axis=1)
        .rename(columns={"Seed": "SeedW"})
    )

    df = (
        pd.merge(
            df,
            df_seeds,
            how="left",
            left_on=["Season", "LTeamID"],
            right_on=["Season", "TeamID"],
        )
        .drop("TeamID", axis=1)
        .rename(columns={"Seed": "SeedL"})
    )

    df["SeedW"] = df["SeedW"].apply(treat_seed)
    df["SeedL"] = df["SeedL"].apply(treat_seed)

    df = (
        pd.merge(
            df,
            df_features_season,
            how="left",
            left_on=["Season", "WTeamID"],
            right_on=["Season", "TeamID"],
        )
        .rename(
            columns={
                "NumWins": "NumWinsW",
                "NumLosses": "NumLossesW",
                "GapWins": "GapWinsW",
                "GapLosses": "GapLossesW",
                "WinRatio": "WinRatioW",
                "GapAvg": "GapAvgW",
            }
        )
        .drop(columns="TeamID", axis=1)
    )
    df = (
        pd.merge(
            df,
            df_features_season,
            how="left",
            left_on=["Season", "LTeamID"],
            right_on=["Season", "TeamID"],
        )
        .rename(
            columns={
                "NumWins": "NumWinsL",
                "NumLosses": "NumLossesL",
                "GapWins": "GapWinsL",
                "GapLosses": "GapLossesL",
                "WinRatio": "WinRatioL",
                "GapAvg": "GapAvgL",
            }
        )
        .drop(columns="TeamID", axis=1)
    )
    avg_ranking = df_massey.groupby(["Season", "TeamID"]).mean().reset_index()

    df = (
        pd.merge(
            df,
            avg_ranking,
            how="left",
            left_on=["Season", "WTeamID"],
            right_on=["Season", "TeamID"],
        )
        .drop("TeamID", axis=1)
        .rename(columns={"OrdinalRank": "OrdinalRankW"})
    )

    df = (
        pd.merge(
            df,
            avg_ranking,
            how="left",
            left_on=["Season", "LTeamID"],
            right_on=["Season", "TeamID"],
        )
        .drop("TeamID", axis=1)
        .rename(columns={"OrdinalRank": "OrdinalRankL"})
    )

    df = add_loosing_matches(df)
    df["SeedDiff"] = df["SeedA"] - df["SeedB"]
    df["OrdinalRankDiff"] = df["OrdinalRankA"] - df["OrdinalRankB"]
    df["WinRatioDiff"] = df["WinRatioA"] - df["WinRatioB"]
    df["GapAvgDiff"] = df["GapAvgA"] - df["GapAvgB"]
    df["ScoreDiff"] = df["ScoreA"] - df["ScoreB"]
    df["ratingA"] = 100 - 4 * np.log1p(df["OrdinalRankA"]) - df["OrdinalRankA"] / 22
    df["ratingB"] = 100 - 4 * np.log1p(df["OrdinalRankB"]) - df["OrdinalRankB"] / 22
    df["Prob"] = 1 / (1 + 10 ** ((df["ratingB"] - df["ratingA"]) / 15))
    df["WinA"] = (df["ScoreDiff"] > 0).astype(int)

    df_test = pd.read_csv(path + "MSampleSubmissionStage2.csv")
    df_test["Season"] = df_test["ID"].apply(lambda x: int(x.split("_")[0]))
    df_test["TeamIdA"] = df_test["ID"].apply(lambda x: int(x.split("_")[1]))
    df_test["TeamIdB"] = df_test["ID"].apply(lambda x: int(x.split("_")[2]))

    df_test = (
        pd.merge(
            df_test,
            df_seeds,
            how="left",
            left_on=["Season", "TeamIdA"],
            right_on=["Season", "TeamID"],
        )
        .drop("TeamID", axis=1)
        .rename(columns={"Seed": "SeedA"})
    )
    df_test = (
        pd.merge(
            df_test,
            df_seeds,
            how="left",
            left_on=["Season", "TeamIdB"],
            right_on=["Season", "TeamID"],
        )
        .drop("TeamID", axis=1)
        .rename(columns={"Seed": "SeedB"})
    )
    df_test["SeedA"] = df_test["SeedA"].apply(treat_seed)
    df_test["SeedB"] = df_test["SeedB"].apply(treat_seed)

    df_test = (
        pd.merge(
            df_test,
            df_features_season,
            how="left",
            left_on=["Season", "TeamIdA"],
            right_on=["Season", "TeamID"],
        )
        .rename(
            columns={
                "NumWins": "NumWinsA",
                "NumLosses": "NumLossesA",
                "GapWins": "GapWinsA",
                "GapLosses": "GapLossesA",
                "WinRatio": "WinRatioA",
                "GapAvg": "GapAvgA",
            }
        )
        .drop(columns="TeamID", axis=1)
    )
    df_test = (
        pd.merge(
            df_test,
            df_features_season,
            how="left",
            left_on=["Season", "TeamIdB"],
            right_on=["Season", "TeamID"],
        )
        .rename(
            columns={
                "NumWins": "NumWinsB",
                "NumLosses": "NumLossesB",
                "GapWins": "GapWinsB",
                "GapLosses": "GapLossesB",
                "WinRatio": "WinRatioB",
                "GapAvg": "GapAvgB",
            }
        )
        .drop(columns="TeamID", axis=1)
    )
    df_test = (
        pd.merge(
            df_test,
            avg_ranking,
            how="left",
            left_on=["Season", "TeamIdA"],
            right_on=["Season", "TeamID"],
        )
        .drop("TeamID", axis=1)
        .rename(columns={"OrdinalRank": "OrdinalRankA"})
    )
    df_test = (
        pd.merge(
            df_test,
            avg_ranking,
            how="left",
            left_on=["Season", "TeamIdB"],
            right_on=["Season", "TeamID"],
        )
        .drop("TeamID", axis=1)
        .rename(columns={"OrdinalRank": "OrdinalRankB"})
    )
    df_test["SeedDiff"] = df_test["SeedA"] - df_test["SeedB"]
    df_test["OrdinalRankDiff"] = df_test["OrdinalRankA"] - df_test["OrdinalRankB"]
    df_test["WinRatioDiff"] = df_test["WinRatioA"] - df_test["WinRatioB"]
    df_test["GapAvgDiff"] = df_test["GapAvgA"] - df_test["GapAvgB"]
    df_test["ratingA"] = (
        100 - 4 * np.log1p(df_test["OrdinalRankA"]) - df_test["OrdinalRankA"] / 22
    )
    df_test["ratingB"] = (
        100 - 4 * np.log1p(df_test["OrdinalRankB"]) - df_test["OrdinalRankB"] / 22
    )
    df_test["Prob"] = 1 / (1 + 10 ** ((df_test["ratingB"] - df_test["ratingA"]) / 15))

    return df, df_test

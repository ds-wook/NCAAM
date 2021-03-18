# %%
import pandas as pd
import numpy as np
from trueskill import TrueSkill, Rating, rate_1vs1
import trueskill

from typing import List, Tuple
import re

# %%


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


def rescale(
    features: List[str],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    min_ = df_train[features].min()
    max_ = df_train[features].max()

    df_train[features] = (df_train[features] - min_) / (max_ - min_)
    df_val[features] = (df_val[features] - min_) / (max_ - min_)

    if df_test is not None:
        df_test[features] = (df_test[features] - min_) / (max_ - min_)

    return df_train, df_val, df_test


# %%
path = "../input/ncaam-march-mania-2021/MDataFiles_Stage2/"

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
print(df.shape)
df.head()

df["SeedW"] = df["SeedW"].astype(str)
df["SeedL"] = df["SeedL"].astype(str)

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
# %%
ts = TrueSkill(draw_probability=0.01)  # 0.01 is arbitary small number
beta = 25 / 6  # default value


def win_probability(p1, p2):
    delta_mu = p1.mu - p2.mu
    sum_sigma = p1.sigma * p1.sigma + p2.sigma * p2.sigma
    denom = np.sqrt(2 * (beta * beta) + sum_sigma)
    return ts.cdf(delta_mu / denom)


# %%

path = "../input/ncaam-march-mania-2021/MDataFiles_Stage1/"
submit = pd.read_csv(path + "MSampleSubmissionStage1.csv")
submit[["Season", "Team1", "Team2"]] = submit.apply(
    lambda r: pd.Series([int(t) for t in r.ID.split("_")]), axis=1
)
submit.head()
# %%
df_tour = df
df_tour.head()
# %%
team_ids = np.unique(
    np.concatenate([df_tour["TeamIdA"].values, df_tour["TeamIdB"].values])
)
ratings = {tid: ts.Rating() for tid in team_ids}

# %%


def feed_season_results(season: int):
    print("season = {}".format(season))
    df1 = df_tour[df_tour.Season == season]
    for r in df1.itertuples():
        ratings[r.TeamIdA], ratings[r.TeamIdB] = rate_1vs1(
            ratings[r.TeamIdA], ratings[r.TeamIdB]
        )


def update_pred(season: int):
    beta = np.std([r.mu for r in ratings.values()])
    print("beta = {}".format(beta))
    submit.loc[submit.Season == season, "Pred"] = submit[submit.Season == season].apply(
        lambda r: win_probability(ratings[r.Team1], ratings[r.Team2]), axis=1
    )


# %%
for season in sorted(df_tour["Season"].unique())[:-5]:
    feed_season_results(season)

# %%
update_pred(2015)
feed_season_results(2015)
update_pred(2016)
feed_season_results(2016)
update_pred(2017)
feed_season_results(2017)
update_pred(2018)
feed_season_results(2018)
update_pred(2019)
# %%
submit.drop(["Season", "Team1", "Team2"], axis=1, inplace=True)
y_preds_ts = submit["Pred"]
submit
# %%
submit.to_csv("../submission/true_skills_test.csv", index=False)
submit.head()
# %%
df_tour
# %%
def feed_season_results(season: int):
    print("season = {}".format(season))
    df1 = df_tour[df_tour.Season == season]
    for r in df1.itertuples():
        ratings[r.TeamIdA], ratings[r.TeamIdB] = rate_1vs1(
            ratings[r.TeamIdA], ratings[r.TeamIdB]
        )


def update_pred(season: int):
    beta = np.std([r.mu for r in ratings.values()])
    print("beta = {}".format(beta))
    df_tour.loc[df_tour.Season == season, "Pred"] = df_tour[
        df_tour.Season == season
    ].apply(lambda r: win_probability(ratings[r.TeamIdA], ratings[r.TeamIdB]), axis=1)

# %%
for season in range(2003, 2020):
    update_pred(season)
df_tour
# %%

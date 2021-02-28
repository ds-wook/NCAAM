import pandas as pd

path = "../../input/ncaam-march-mania-2021/"
MRSCResults = pd.read_csv(path + "/MRegularSeasonCompactResults.csv")

A_w = (
    MRSCResults[MRSCResults.WLoc == "A"]
    .groupby(["Season", "WTeamID"])["WTeamID"]
    .count()
    .to_frame()
    .rename(columns={"WTeamID": "win_A"})
)
N_w = (
    MRSCResults[MRSCResults.WLoc == "N"]
    .groupby(["Season", "WTeamID"])["WTeamID"]
    .count()
    .to_frame()
    .rename(columns={"WTeamID": "win_N"})
)
H_w = (
    MRSCResults[MRSCResults.WLoc == "H"]
    .groupby(["Season", "WTeamID"])["WTeamID"]
    .count()
    .to_frame()
    .rename(columns={"WTeamID": "win_H"})
)
win = A_w.join(N_w, how="outer").join(H_w, how="outer").fillna(0)

H_l = (
    MRSCResults[MRSCResults.WLoc == "A"]
    .groupby(["Season", "LTeamID"])["LTeamID"]
    .count()
    .to_frame()
    .rename(columns={"LTeamID": "lost_H"})
)
N_l = (
    MRSCResults[MRSCResults.WLoc == "N"]
    .groupby(["Season", "LTeamID"])["LTeamID"]
    .count()
    .to_frame()
    .rename(columns={"LTeamID": "lost_N"})
)
A_l = (
    MRSCResults[MRSCResults.WLoc == "H"]
    .groupby(["Season", "LTeamID"])["LTeamID"]
    .count()
    .to_frame()
    .rename(columns={"LTeamID": "lost_A"})
)
lost = A_l.join(N_l, how="outer").join(H_l, how="outer").fillna(0)

win.index = win.index.rename(["Season", "TeamID"])
lost.index = lost.index.rename(["Season", "TeamID"])
wl = win.join(lost, how="outer").reset_index()
wl["win_pct_A"] = wl["win_A"] / (wl["win_A"] + wl["lost_A"])
wl["win_pct_N"] = wl["win_N"] / (wl["win_N"] + wl["lost_N"])
wl["win_pct_H"] = wl["win_H"] / (wl["win_H"] + wl["lost_H"])
wl["win_pct_All"] = (wl["win_A"] + wl["win_N"] + wl["win_H"]) / (
    wl["win_A"] + wl["win_N"] + wl["win_H"] + wl["lost_A"] + wl["lost_N"] + wl["lost_H"]
)

del A_w, N_w, H_w, H_l, N_l, A_l, win, lost


MRSCResults["relScore"] = MRSCResults.WScore - MRSCResults.LScore

w_scr = MRSCResults.loc[:, ["Season", "WTeamID", "WScore", "WLoc", "relScore"]]
w_scr.columns = ["Season", "TeamID", "Score", "Loc", "relScore"]
l_scr = MRSCResults.loc[:, ["Season", "LTeamID", "LScore", "WLoc", "relScore"]]
l_scr["WLoc"] = l_scr.WLoc.apply(
    lambda x: "H" if x == "A" else "A" if x == "H" else "N"
)
l_scr["relScore"] = -1 * l_scr.relScore
l_scr.columns = ["Season", "TeamID", "Score", "Loc", "relScore"]
wl_scr = pd.concat([w_scr, l_scr])

A_scr = (
    wl_scr[wl_scr.Loc == "A"]
    .groupby(["Season", "TeamID"])[["Score", "relScore"]]
    .mean()
    .rename(columns={"Score": "Score_A", "relScore": "relScore_A"})
)
N_scr = (
    wl_scr[wl_scr.Loc == "N"]
    .groupby(["Season", "TeamID"])[["Score", "relScore"]]
    .mean()
    .rename(columns={"Score": "Score_N", "relScore": "relScore_N"})
)
H_scr = (
    wl_scr[wl_scr.Loc == "H"]
    .groupby(["Season", "TeamID"])[["Score", "relScore"]]
    .mean()
    .rename(columns={"Score": "Score_H", "relScore": "relScore_H"})
)
All_scr = (
    wl_scr.groupby(["Season", "TeamID"])[["Score", "relScore"]]
    .mean()
    .rename(columns={"Score": "Score_All", "relScore": "relScore_All"})
)
scr = (
    A_scr.join(N_scr, how="outer")
    .join(H_scr, how="outer")
    .join(All_scr, how="outer")
    .fillna(0)
    .reset_index()
)

del w_scr, l_scr, wl_scr, A_scr, H_scr, N_scr, All_scr

MRSDetailedResults = pd.read_csv(path + "/MRegularSeasonDetailedResults.csv")

w = MRSDetailedResults.loc[
    :,
    [
        "Season",
        "WTeamID",
        "WFGM",
        "WFGA",
        "WFGM3",
        "WFGA3",
        "WFTM",
        "WFTA",
        "WOR",
        "WDR",
        "WAst",
        "WTO",
        "WStl",
        "WBlk",
        "WPF",
    ],
]
w.columns = [
    "Season",
    "TeamID",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
]
lo = MRSDetailedResults.loc[
    :,
    [
        "Season",
        "LTeamID",
        "LFGM",
        "LFGA",
        "LFGM3",
        "LFGA3",
        "LFTM",
        "LFTA",
        "LOR",
        "LDR",
        "LAst",
        "LTO",
        "LStl",
        "LBlk",
        "LPF",
    ],
]
lo.columns = [
    "Season",
    "TeamID",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
]

detail = pd.concat([w, lo])
detail["goal_rate"] = detail.FGM / detail.FGA
detail["3p_goal_rate"] = detail.FGM3 / detail.FGA3
detail["ft_goal_rate"] = detail.FTM / detail.FTA

dt = (
    detail.groupby(["Season", "TeamID"])[
        [
            "FGM",
            "FGA",
            "FGM3",
            "FGA3",
            "FTM",
            "FTA",
            "OR",
            "DR",
            "Ast",
            "TO",
            "Stl",
            "Blk",
            "PF",
            "goal_rate",
            "3p_goal_rate",
            "ft_goal_rate",
        ]
    ]
    .mean()
    .fillna(0)
    .reset_index()
)

del w, lo, detail

MMOrdinals = pd.read_csv(path + "/MMasseyOrdinals.csv")

MOR_127_128 = MMOrdinals[
    (MMOrdinals.SystemName == "MOR")
    & ((MMOrdinals.RankingDayNum == 127) | (MMOrdinals.RankingDayNum == 128))
][["Season", "TeamID", "OrdinalRank"]]
MOR_50_51 = MMOrdinals[
    (MMOrdinals.SystemName == "MOR")
    & ((MMOrdinals.RankingDayNum == 50) | (MMOrdinals.RankingDayNum == 51))
][["Season", "TeamID", "OrdinalRank"]]
MOR_15_16 = MMOrdinals[
    (MMOrdinals.SystemName == "MOR")
    & ((MMOrdinals.RankingDayNum == 15) | (MMOrdinals.RankingDayNum == 16))
][["Season", "TeamID", "OrdinalRank"]]

MOR_127_128 = MOR_127_128.rename(columns={"OrdinalRank": "OrdinalRank_127_128"})
MOR_50_51 = MOR_50_51.rename(columns={"OrdinalRank": "OrdinalRank_50_51"})
MOR_15_16 = MOR_15_16.rename(columns={"OrdinalRank": "OrdinalRank_15_16"})

MOR = MOR_127_128.merge(MOR_50_51, how="left", on=["Season", "TeamID"]).merge(
    MOR_15_16, how="left", on=["Season", "TeamID"]
)

# # normalizing Rank values by its season maxium as it varies by seasons
MOR_max = (
    MOR.groupby("Season")[
        ["OrdinalRank_127_128", "OrdinalRank_50_51", "OrdinalRank_15_16"]
    ]
    .max()
    .reset_index()
)
MOR_max.columns = ["Season", "maxRank_127_128", "maxRank_50_51", "maxRank_15_16"]

MOR_tmp = MMOrdinals[
    (MMOrdinals.SystemName == "MOR") & (MMOrdinals.RankingDayNum < 133)
]
MOR_stats = (
    MOR_tmp.groupby(["Season", "TeamID"])["OrdinalRank"]
    .agg(["max", "min", "std", "mean"])
    .reset_index()
)
MOR_stats.columns = ["Season", "TeamID", "RankMax", "RankMin", "RankStd", "RankMean"]

MOR = MOR.merge(MOR_max, how="left", on="Season").merge(
    MOR_stats, how="left", on=["Season", "TeamID"]
)
MOR["OrdinalRank_127_128"] = MOR["OrdinalRank_127_128"] / MOR["maxRank_127_128"]
MOR["OrdinalRank_50_51"] = MOR["OrdinalRank_50_51"] / MOR["maxRank_50_51"]
MOR["OrdinalRank_15_16"] = MOR["OrdinalRank_15_16"] / MOR["maxRank_15_16"]
MOR["RankTrans_50_51_to_127_128"] = (
    MOR["OrdinalRank_127_128"] - MOR["OrdinalRank_50_51"]
)
MOR["RankTrans_15_16_to_127_128"] = (
    MOR["OrdinalRank_127_128"] - MOR["OrdinalRank_15_16"]
)

# MOR['RankMax'] = MOR['RankMax'] / MOR['maxRank_127_128']
# MOR['RankMin'] = MOR['RankMin'] / MOR['maxRank_127_128']
# MOR['RankStd'] = MOR['RankStd'] / MOR['maxRank_127_128']
# MOR['RankMean'] = MOR['RankMean'] / MOR['maxRank_127_128']

MOR.drop(
    ["OrdinalRank_50_51", "OrdinalRank_15_16", "maxRank_50_51", "maxRank_15_16"],
    axis=1,
    inplace=True,
)

del MOR_127_128, MOR_50_51, MOR_15_16, MOR_max, MOR_tmp, MOR_stats

wl_1 = wl.loc[
    :, ["Season", "TeamID", "win_pct_A", "win_pct_N", "win_pct_H", "win_pct_All"]
]
wl_1.columns = [
    str(col) + "_1" if col not in ["Season", "TeamID"] else str(col)
    for col in wl_1.columns
]

wl_2 = wl.loc[
    :, ["Season", "TeamID", "win_pct_A", "win_pct_N", "win_pct_H", "win_pct_All"]
]
wl_2.columns = [
    str(col) + "_2" if col not in ["Season", "TeamID"] else str(col)
    for col in wl_2.columns
]

scr_1 = scr.copy()
scr_1.columns = [
    str(col) + "_1" if col not in ["Season", "TeamID"] else str(col)
    for col in scr_1.columns
]

scr_2 = scr.copy()
scr_2.columns = [
    str(col) + "_2" if col not in ["Season", "TeamID"] else str(col)
    for col in scr_2.columns
]

dt_1 = dt.copy()
dt_1.columns = [
    str(col) + "_1" if col not in ["Season", "TeamID"] else str(col)
    for col in dt_1.columns
]

dt_2 = dt.copy()
dt_2.columns = [
    str(col) + "_2" if col not in ["Season", "TeamID"] else str(col)
    for col in dt_2.columns
]

MOR_1 = MOR.copy()
MOR_1.columns = [
    str(col) + "_1" if col not in ["Season", "TeamID"] else str(col)
    for col in MOR_1.columns
]

MOR_2 = MOR.copy()
MOR_2.columns = [
    str(col) + "_2" if col not in ["Season", "TeamID"] else str(col)
    for col in MOR_2.columns
]

TCResults = pd.read_csv(path + "/MNCAATourneyCompactResults.csv")

tourney1 = TCResults.loc[:, ["Season", "WTeamID", "LTeamID"]]
tourney1.columns = ["Season", "TeamID1", "TeamID2"]
tourney1["result"] = 1

tourney2 = TCResults.loc[:, ["Season", "LTeamID", "WTeamID"]]
tourney2.columns = ["Season", "TeamID1", "TeamID2"]
tourney2["result"] = 0

tourney = pd.concat([tourney1, tourney2])
del tourney1, tourney2

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

path = "../input/ncaam-march-mania-2021/"

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [markdown]
"""
## Data Section 1 - The Basics
### This section provides everything
you need to build a simple prediction model and submit predictions.

+ Team ID's and Team Names
+ Tournament seeds since 1984-85 season
+ Final scores of all regular season, conference tournament,
and NCAA® tournament games since 1984-85 season
+ Season-level details including dates and region names
+ Example submission file for stage 1
"""

# %% [markdown]
"""
## Data Section 1 file: Teams.csv
TeamID - a 4 digit id number, from 1000-1999,
uniquely identifying each NCAA® men's team.
TeamName - a compact spelling of the team's college name, 16 characters or fewer.
FirstD1Season - the first season in our dataset that the school was a Division-I school.
LastD1Season - the last season in our dataset that the school was a Division-I school.
"""
teams = pd.read_csv(path + "MTeams.csv")
teams.head()
# %%
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
sns.countplot(ax=axes[0], data=teams, x="FirstD1Season")
sns.countplot(ax=axes[1], data=teams, x="LastD1Season")
plt.show()

# %% [markdown]
"""
## Data Section 1 file: MSeasons.csv

+ Season - indicates the year in which the tournament was played.
+ DayZero - tells you the date corresponding to DayNum=0 during that season.
All game dates have been aligned upon a common scale so that (each year)
the Monday championship game of the men's tournament is on DayNum=154.
+ RegionW, RegionX, Region Y, Region Z - by our contests' convention,
each of the four regions in the final tournament is assigned a letter of W, X, Y, or Z.
Whichever region's name comes first alphabetically, that region will be Region W.
"""
seasons = pd.read_csv(path + "MSeasons.csv")
seasons.head()
# %% [markdown]
"""
## Data Section 1 file: MRegularSeasonCompactResults.csv
+ Season - this is the year of the associated entry in MSeasons.csv
(the year in which the final tournament occurs).
+ DayNum - this integer always ranges from 0 to 132,
and tells you what day the game was played on.
It represents an offset from the "DayZero" date in the "MSeasons.csv" file.
+ WTeamID - this identifies the id number of the team that won the game,
as listed in the "MTeams.csv" file.
+ WScore - this identifies the number of points scored by the winning team.
+ LTeamID - this identifies the id number of the team that lost the game.
+ LScore - this identifies the number of points scored by the losing team.
+ WLoc - this identifies the "location" of the winning team.
If the winning team was the home team, this value will be "H".
If the winning team was the visiting team, this value will be "A".
If it was played on a neutral court, then this value will be "N".
+ NumOT - this indicates the number of overtime periods in the game,
an integer 0 or higher.
"""
MRegularSeasonCompactResults = pd.read_csv(path + "MRegularSeasonCompactResults.csv")
MRegularSeasonCompactResults.head()

# %%
fig = plt.figure(figsize=(5, 5))
sns.countplot(data=MRegularSeasonCompactResults, x="WLoc")
plt.show()
# %%
fig = plt.figure(figsize=(5, 5))
sns.countplot(data=MRegularSeasonCompactResults, x="NumOT")
plt.show()
# %%
fig = plt.figure(figsize=(15, 5))
sns.countplot(data=MRegularSeasonCompactResults, x="Season")
plt.show()
# %%
fig = plt.figure(figsize=(5, 50))
sns.countplot(data=MRegularSeasonCompactResults, y="WTeamID")
plt.show()
# %%
fig = plt.figure(figsize=(10, 5))
sns.histplot(
    MRegularSeasonCompactResults.groupby("WTeamID").apply(lambda x: len(x)).values,
    bins=25,
)
plt.title("Distribution of total number of wins by each teams")
plt.show()

# %%
fig = plt.figure(figsize=(10, 5))
sns.histplot(
    MRegularSeasonCompactResults.groupby("LTeamID").apply(lambda x: len(x)).values,
    bins=25,
)
plt.title("Distribution of total number of Losses by each teams")
plt.show()
# %%
fig = plt.figure(figsize=(10, 5))
sns.histplot(data=MRegularSeasonCompactResults, x="WScore")
plt.show()
# %%
fig = plt.figure(figsize=(10, 5))
sns.histplot(data=MRegularSeasonCompactResults, x="LScore")
plt.show()
# %%
MRegularSeasonCompactResults["score_difference"] = (
    MRegularSeasonCompactResults["WScore"] - MRegularSeasonCompactResults["LScore"]
)

fig = plt.figure(figsize=(10, 5))
sns.histplot(data=MRegularSeasonCompactResults, x="score_difference")
plt.show()
# %%
fig = plt.figure(figsize=(10, 5))
sns.histplot(data=MRegularSeasonCompactResults, x="DayNum")
plt.show()
# %%
MRegularSeasonDetailedResults = pd.read_csv(path + "/MRegularSeasonDetailedResults.csv")
MRegularSeasonDetailedResults.head()
# %%
mrsc_results = pd.read_csv(path + "MRegularSeasonCompactResults.csv")
A_w = (
    mrsc_results[mrsc_results.WLoc == "A"]
    .groupby(["Season", "WTeamID"])["WTeamID"]
    .count()
    .to_frame()
    .rename(columns={"WTeamID": "win_A"})
)
A_w
# %%

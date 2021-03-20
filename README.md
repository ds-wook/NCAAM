# NCAAM
This is a collection of my code from the [March Machine Learning Mania 2021 - NCAAM](https://www.kaggle.com/c/ncaam-march-mania-2021) Kaggle competition.

## Code Style
I follow [black](https://pypi.org/project/black/) for code style. Black is a PEP 8 compliant opinionated formatter.

## Feature Engineering
+ SeedA
+ SeedB
+ SeedDiff
+ WinRatioA
+ WinRatioB
+ WinRatioDiff
+ GapAvgA
+ GapAvgB
+ GapAvgDiff
+ OrdinalRankA
+ OrdinalRankB
+ OrdinalRankDiff
+ ratingA
+ ratingB
+ Prob
+ WinA: target

## Cross Validation Strategy
+ time series split cross-validation
<img src="image/time series split cross-validation.JPG"  width="700" height="370">


## Tree-structured Parzen Estimator (TPE) Approach Hyper Parameter Tunning
[Optuna](https://optuna.org/) is an open source hyperparameter optimization framework to automate hyperparameter search. I used TPE algorithm.

# NCAAM
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
This is a collection of my code from the [March Machine Learning Mania 2021 - NCAAM](https://www.kaggle.com/c/ncaam-march-mania-2021) Kaggle competition.

## Requirements
```
lightgbm==3.1.1
numpy==1.20.1
optuna==2.5.0
pandas==1.2.2
plotly==4.14.3
scikit-learn==0.24.1
scipy==1.6.1
```

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


## Model
Light GBM is very nice Ensemble model.

## Cross Validation Strategy
+ time series split cross-validation
It is a model that is weighted for each year.
<img src="image/time series split cross-validation.JPG"  width="700" height="370">


## Tree-structured Parzen Estimator (TPE) Approach Hyper Parameter Tunning
[Optuna](https://optuna.org/) is an open source hyperparameter optimization framework to automate hyperparameter search. I used TPE algorithm.

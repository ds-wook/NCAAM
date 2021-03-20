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

## Model
Light GBM is very nice Ensemble model.

## Cross Validation Strategy
+ time series split cross-validation
<img src="image/time series split cross-validation.JPG"  width="700" height="370">


## Tree-structured Parzen Estimator (TPE) Approach Hyper Parameter Tunning
[Optuna](https://optuna.org/) is an open source hyperparameter optimization framework to automate hyperparameter search. I used TPE algorithm.

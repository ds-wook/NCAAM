# NCAAM
This is a collection of my code from the [March Machine Learning Mania 2021 - NCAAM](https://www.kaggle.com/c/ncaam-march-mania-2021) Kaggle competition.

## Code Style
I follow [black](https://pypi.org/project/black/) for code style. Black is a PEP 8 compliant opinionated formatter.

## Benchmark

#### TPE Hyper Parameter Tunning
|method|baseline|OOF|Public LB|Private LB|
|------|:------:|:---------:|:--------:|:--------:|
|Non-LGBM optuna|0.55835|0.55575(5-fold)|0.52438|-|
|FE-LGB optuna|-|-|-|-|

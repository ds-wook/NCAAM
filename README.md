# NCAAM
This is a collection of my code from the [March Machine Learning Mania 2021 - NCAAM](https://www.kaggle.com/c/ncaam-march-mania-2021) Kaggle competition.

## Code Style
I follow [black](https://pypi.org/project/black/) for code style. Black is a PEP 8 compliant opinionated formatter.

## Benchmark

#### FE Hyper Parameter Tunning
|method|OOF(5-fold)|Public LB|Private LB|
|------|:---------:|:--------:|:--------:|
|LGBM optuna(4 params)|0.5134|0.41899|-|
|LGBM optuna(7 params)|0.5261|0.49364|-|
|LGBM - MaxAbs(6 params)|0.51270|0.50479|-|

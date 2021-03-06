# NCAAM
This is a collection of my code from the [March Machine Learning Mania 2021 - NCAAM](https://www.kaggle.com/c/ncaam-march-mania-2021) Kaggle competition.

## Code Style
I follow [black](https://pypi.org/project/black/) for code style.
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
Black is a PEP 8 compliant opinionated formatter.

## Benchmark

#### FE Hyper Parameter Tunning
|method|OOF|Public LB|Private LB|
|------|:---------:|:--------:|:--------:|
|LGBM optuna(4 params)|0.5134(5-fold)|0.41899|-|
|LGBM optuna(7 params)|0.5261(5-fold)|0.49364|-|
|LGBM optuna(7 params)|0.49390(3-fold)|0.38244|-|
|XGB optuna(6 params)|0.50106(3-fold)|0.48643|-|
|XGB-MaxAbs optuna(6 params)|0.50106(3-fold)|0.48643|-|
|XGB-robust optuna(8 params)|0.50313(3-fold)|0.50467|-|
|XGB-nomalization optuna(8 params)|0.50652(5-fold)|0.49091|-|
|XGB-nomalization optuna(8 params features)|0.49608(5-fold)|0.48286|-|
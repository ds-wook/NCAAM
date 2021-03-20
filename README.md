# NCAAM
This is a collection of my code from the [March Machine Learning Mania 2021 - NCAAM](https://www.kaggle.com/c/ncaam-march-mania-2021) Kaggle competition.

## Code Style
I follow [black](https://pypi.org/project/black/) for code style. Black is a PEP 8 compliant opinionated formatter.


## Tree-structured Parzen Estimator Approach (TPE) Hyper Parameter Tunning

Anticipating that our hyper-parameter optimization tasks will mean high dimensions and small fitness evaluation budgets, we now turn to another modeling strategy and EI optimization scheme for the SMBO algorithm. Whereas the Gaussian-process based approach modeled p(y|x) directly, this
strategy models p(x|y) and p(y).
Recall from the introduction that the configuration space X is described by a graph-structured generative process (e.g. first choose a number of DBN layers, then choose the parameters for each).
The tree-structured Parzen estimator (TPE) models p(x|y) by transforming that generative process,
replacing the distributions of the configuration prior with non-parametric densities. In the experimental section, we will see that the configuation space is described using uniform, log-uniform,
quantized log-uniform, and categorical variables. In these cases, the TPE algorithm makes the
following replacements: uniform → truncated Gaussian mixture, log-uniform → exponentiated
truncated Gaussian mixture, categorical → re-weighted categorical. Using different observations
{x(1), ..., x(k)} in the non-parametric densities, these substitutions represent a learning algorithm
that can produce a variety of densities over the configuration space X . The TPE defines p(x|y)
using two such densities:
p(x|y) = 
`e(x) if y < y∗ g(x) if y ≥ y∗`,
(2) where (x) is the density formed by using the observations {x(i)} such that corresponding loss
f(x(i)) was less than y∗ and g(x) is the density formed by using the remaining observations.
Whereas the GP-based approach favoured quite an aggressive y∗ (typically less than the best observed loss), the TPE algorithm depends on a y
∗that is larger than the best observed f(x) so that some points can be used to form `(x). The TPE algorithm chooses y∗
to be some quantile γ of the observed y values, so that p(y < y∗) = γ, but no specific model for p(y) is necessary. By maintaining sorted lists of observed variables in H, the runtime of each iteration of the TPE algorithm can scale linearly in |H| and linearly in the number of variables (dimensions) being optimized.

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


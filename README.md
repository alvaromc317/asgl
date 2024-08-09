
<!-- README.md is generated from README.Rmd. Please edit that file -->

# asgl <img src="figures/logo.png" align="right" height="150" alt="funq website" /></a>

[![Downloads](https://pepy.tech/badge/asgl)](https://pepy.tech/project/asgl)
[![Downloads](https://pepy.tech/badge/asgl/month)](https://pepy.tech/project/asgl)
[![License: GPL
v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Package
Version](https://img.shields.io/badge/version-1.0.5-blue.svg)](https://cran.r-project.org/package=asgl)

## Introduction

`asgl` is a Python package that solves several regression related models
for simultaneous variable selection and prediction, in low and high
dimensional frameworks. This package is directly related to research
work shown on [this
paper](https://link.springer.com/article/10.1007/s11634-020-00413-8) and
a full description of the capabilities of the package is shown on [this
paper](https://arxiv.org/abs/2111.00472). We also suggest accessing the
user_guide notebook provided in the [GitHub
repository](https://github.com/alvaromc317/asgl). Or you can find more
accesible explanations
[here](https://towardsdatascience.com/sparse-group-lasso-in-python-255e379ab892)
and
[here](https://towardsdatascience.com/an-adaptive-lasso-63afca54b80d).

The current version of the package supports:

- Linear regression models
- Quantile regression models

And considers the following penalizations for variable selection:

- No penalized models
- lasso
- group lasso
- sparse group lasso
- adaptive lasso
- adaptive group lassso
- adaptive sparse group lasso

## Requirements

The package makes use of some basic functions from `scikit-learn` and
`numpy`, and is built on top of the wonderful `cvxpy` convex
optimization module. It is higly encouraged to install `cvxpy` prior of
the installation of `asgl` following the instructions from the original
authors, that can be found [here](https://www.cvxpy.org/)).
Additionally, `asgl` makes use of python `multiprocessing` module,
allowing, if requested, for parallel execution of the code highly
reducing computation time.

## Usage example:

In the following example we will analyze the `BostonHousing` dataset
(available
[here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston)).
Even though the `asgl` package can easily deal with much more complex
datasets, we will work using this one so we are not affected by
computation time. We will show how to implement cross validation on a
grid of possible parameter values for an sparse group lasso linear
model, how to find the optimal parameter values and finally, how to
compute the test error.

#### Example:

The following code performs cross validation in a grid of different
parameter values for an sparse group lasso model on the well known
`BostonHousing` dataset:

``` python
# Import required packages
import numpy as np
import asgl
from sklearn.datasets import make_regression

x, y = make_regression(n_samples=100, n_features=30, random_state=42)
group_index = np.random.randint(1, 6, 30)

# Define parameters grid
lambda1 = (10.0 ** np.arange(-3, 2.01, 1)) # 8 values
alpha = np.arange(0, 1.01, 0.25) # 10 values

# Define a cross validation object
cv_class = asgl.CV(model='lm', penalization='sgl', lambda1=lambda1, alpha=alpha,
                   nfolds=5, error_type='MSE', parallel=True, random_state=99)

# Compute error using k-fold cross validation
error = cv_class.cross_validation(x=x, y=y, group_index=group_index)

num_models, k_folds = error.shape
# error is a matrix of shape (number_of_models, k_folds)
print(f'We are considering a grid of {num_models} models, optimized based on {k_folds}-folds cross validation')
#> We are considering a grid of 30 models, optimized based on 5-folds cross validation

# Obtain the mean error across different folds
error = np.mean(error, axis=1)
```

### Citing

------------------------------------------------------------------------

If you use ASGL for academic work, we encourage you to [cite our
paper](https://link.springer.com/article/10.1007/s11634-020-00413-8).
Thank you for your support and we hope you find this package useful!

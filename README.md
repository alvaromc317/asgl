
<!-- README.md is generated from README.Rmd. Please edit that file -->

# asgl <img src="figures/logo.png" align="right" height="150" alt="funq website" /></a>

[![Downloads](https://pepy.tech/badge/asgl)](https://pepy.tech/project/asgl)
[![Downloads](https://pepy.tech/badge/asgl/month)](https://pepy.tech/project/asgl)
[![License: GPL
v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Package
Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://cran.r-project.org/package=asgl)

## Introduction

The `asgl` package is a versatile and robust tool designed for fitting a
variety of regression models, including linear regression, quantile
regression, and various penalized regression models such as Lasso, Group
Lasso, Sparse Group Lasso, and their adaptive variants. The package is
especially useful for simultaneous variable selection and prediction in
both low and high-dimensional frameworks.

The primary class available to users is the `Regressor` class, which is
detailed later in this document.

`asgl` is based on cutting-edge research and methodologies, as outlined
in the following papers:

- [Adaptive Sparse Group Lasso in Quantile
  Regression](https://link.springer.com/article/10.1007/s11634-020-00413-8)
- [`asgl`: A Python Package for Penalized Linear and Quantile
  Regression](https://arxiv.org/abs/2111.00472)

For a practical introduction to the package, users can refer to the user
guide notebook available in the GitHub repository. Additional accessible
explanations can be found on [Towards Data Science: Sparse Group
Lasso](https://towardsdatascience.com/sparse-group-lasso-in-python-255e379ab892)
and [Towards Data Science: Adaptive
Lasso](https://towardsdatascience.com/an-adaptive-lasso-63afca54b80d).

## Dependencies

asgl requires:

- Python \>= 3.9
- cvxpy \>= 1.2.0
- numpy \>= 1.20.0
- scikit-learn \>= 1.0
- pytest \>= 7.1.2

## User installation

The easiest way to install asgl is using `pip`:

    pip install asgl

## Testing

After installation, you can launch the test suite from the source
directory (you will need to have `pytest >= 7.1.2` installed) by runnig:

    pytest

## What’s new?

With the release of version 2.0, the `asgl` package has undergone
significant enhancements and improvements. The most notable change is
the introduction of the `Regressor` object, which brings full
compatibility with scikit-learn. This means that the `Regressor` object
can now be used just like any other scikit-learn estimator, enabling
seamless integration with scikit-learn’s extensive suite of tools for
model evaluation, hyperparameter optimization, and performance metrics.

Key updates include:

- Scikit-learn Compatibility: The `Regressor` class is now fully
  compatible with scikit-learn. Users can leverage functionalities such
  as `sklearn.model_selection.GridSearchCV` for hyperparameter tuning
  and utilize various scikit-learn metrics and utilities to assess model
  performance.

- Deprecation of `ASGL` class: The old `ASGL` class is still included in
  the package for backward compatibility but is now deprecated. It will
  raise a `DeprecationWarning` when used, as it is no longer supported
  and will be removed in future versions. Users are strongly encouraged
  to transition to the new `Regressor` class to take advantage of the
  latest features and improvements.

For users currently utilizing the `ASGL` class, we recommend switching
to the `Regressor` class to ensure continued support and access to the
latest functionalities.

## Main parameters:

The `Regressor` class includes the following list of parameters:

- model: str, default=‘lm’
  - Type of model to fit. Options are ‘lm’ (linear regression) and ‘qr’
    (quantile regression).
- penalization: str or None, default=‘lasso’
  - Type of penalization to use. Options are ‘lasso’, ‘gl’ (group
    lasso), ‘sgl’ (sparse group lasso), ‘alasso’ (adaptive lasso), ‘agl’
    (adaptive group lasso), ‘asgl’ (adaptive sparse group lasso), or
    None.
- quantile: float, default=0.5
  - Quantile level for quantile regression models. Valid values are
    between 0 and 1.
- fit_intercept: bool, default=True
  - Whether to fit an intercept in the model.
- lambda1: float, default=0.1
  - Constant that multiplies the penalization, controlling the strength.
    Must be a non-negative float i.e. in `[0, inf)`. Larger values will
    result in larger penalizations.
- alpha: float, default=0.5
  - Constant that performs tradeoff between individual and group
    penalizations in sgl and asgl penalizations. `alpha=1` enforces a
    lasso penalization while `alpha=0` enforces a group lasso
    penalization.
- solver: str, default=‘default’
  - Solver to be used by `cvxpy`. Default uses optimal alternative
    depending on the problem. Users can check available solvers via the
    command `cvxpy.installed_solvers()`.
- weight_technique: str, default=‘pca_pct’
  - Technique used to fit adaptive weights. Options include ‘pca_1’,
    ‘pca_pct’, ‘pls_1’, ‘pls_pct’, ‘lasso’, ‘unpenalized’, and
    ‘sparse_pca’. For low dimensional problems (where the number of
    variables is smaller than the number of observations) the usage of
    the ‘unpenalized’ weight_technique alternative is encouraged. For
    high dimensional problems (where the number of variables is larger
    than the number of observations) the default alternative is
    encouraged.
- individual_power_weight: float, default=1
  - Power to which individual weights are raised. This parameter only
    has effect in adaptive penalizations. (‘alasso’ and ‘asgl’).
- group_power_weight: float, default=1
  - Power to which group weights are raised. This parameter only has
    effect in adaptive penalizations with a grouped structure (‘agl’ and
    ‘asgl’).
- variability_pct: float, default=0.9
  - Percentage of variability explained by PCA, PLS, and sparse PCA
    components. This parameter only has effect in adaptiv penalizations
    where `weight_technique` is equal to ‘pca_pct’, ‘pls_pct’ or
    ‘sparse_pca’.
- lambda1_weights: float, default=0.1
  - The value of the parameter `lambda1` used to solve the lasso model
    if `weight_technique='lasso'`
- spca_alpha: float, default=1e-5
  - Sparse PCA parameter. This parameter only has effect if
    `weight_technique='sparse_pca'`See scikit-learn implementation for
    more details.
- spca_ridge_alpha: float, default=1e-2
  - Sparse PCA parameter. This parameter only has effect if
    `weight_technique='sparse_pca'`See scikit-learn implementation for
    more details.
- individual_weights: array or None, default=None
  - Custom individual weights for adaptive penalizations. If this
    parameter is informed, it overrides the weight estimation process
    defined by parameter `weight_technique` and allows the user to
    provide custom weights. It must be either `None` or be an array with
    non-negative float values and length equal to the number of
    variables.
- group_weights: array or None, default=None
  - Custom group weights for adaptive penalizations. If this parameter
    is informed, it overrides the weight estimation process defined by
    parameter `weight_technique` and allows the user to provide custom
    weights. It must be either `None` or be an array with non-negative
    float values and length equal to the number of groups (as defined by
    `group_index`)
- tol: float, default=1e-4
  - Tolerance for coefficients to be considered zero.
- weight_tol: float, default=1e-4
  - Tolerance value used to avoid ZeroDivision errors when computing the
    weights.

## Examples

### Example 1: Linear Regression with Lasso Penalization.

``` python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from asgl import Regressor

X, y = make_regression(n_samples=1000, n_features=50, n_informative=25, bias=10, noise=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=250)

model = Regressor(model='lm', penalization='lasso', lambda1=0.1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(predictions, y_test)
```

This example illustrates how to:

- Generate synthetic regression data.
- Split the data into training and testing sets.
- Create a `Regressor` object configured for linear regression with
  Lasso penalization.
- Fit the model to the training data.
- Make predictions on the test data.
- Evaluate the model’s performance using mean squared error.

### Example 2: Quantile regression with Adaptive Sparse Group Lasso penalization.

Group-based penalizations like Group Lasso, Sparse Group Lasso, and
their adaptive variants, assume that there is a group structure within
the regressors. This structure can be useful in various applications,
such as when using dummy variables where all the dummies of the same
variable belong to the same group, or in genetic data analysis where
genes are grouped into genetic pathways.

For scenarios where the regressors have a known grouped structure, this
information can be passed to the `Regressor` class during model fitting
using the group_index parameter. The following example demonstrates this
with a synthetic group_index. The model will be optimized using
scikit-learn’s `RandomizedSearchCV` function.

``` python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from asgl import Regressor

X, y = make_regression(n_samples=1000, n_features=50, n_informative=25, bias=10, noise=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=250)

group_index = np.random.randint(1, 5, size=50)

model = Regressor(model='qr', penalization='asgl', quantile=0.5)

param_grid = {'lambda1': [1e-4, 1e-3, 1e-2, 1e-1, 1], 'alpha': [0, 0.2, 0.4, 0.6, 0.8, 1]}
rscv = RandomizedSearchCV(model, param_grid, scoring='neg_median_absolute_error')
rscv.fit(X_train, y_train, **{'group_index': group_index})
```

This example demonstrates how to fit a quantile regression model with
Adaptive Sparse Group Lasso penalization, utilizing scikit-learn’s
`RandomizedSearchCV` to optimize the model’s hyperparameters.

## Contributions

Contributions are welcome! Please submit a pull request or open an issue
to discuss your ideas.

### Citation

------------------------------------------------------------------------

If you use `asgl` in a scientific publication, we would appreciate you
[cite our
paper](https://link.springer.com/article/10.1007/s11634-020-00413-8).
Thank you for your support and we hope you find this package useful!

## License

This project is licensed under the GPL-3.0 license. This means that the
package is open source and that any copy or modification of the original
code must also be released under the GPL-3.0 license. In other words,
you can take the code, add to it or make major changes, and then openly
distribute your version, but not profit from it.

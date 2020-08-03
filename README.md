# `asgl` package

## Introduction

`asgl` is a Python package that solves several regression related models for simultaneous variable selection and prediction, in low and high dimensional frameworks. This package is directly related to research work shown on [this paper](https://link.springer.com/article/10.1007/s11634-020-00413-8).

The current version of the package supports:
* Linear regression models
* Quantile regression models

And considers the following penalizations for variable selection:

* No penalized models 
* lasso
* group lasso
* sparse group lasso
* adaptive sparse group lasso

## Requirements 
The package makes use of some basic functions from `scikit-learn` and `numpy`, and is built on top of the wonderful `cvxpy` convex optimization module. It is higly encouraged to install `cvxpy` prior of the installation of `asgl` following the instructions from the original authors, that can be found [here](https://www.cvxpy.org/)). Additionally,  `asgl` makes use of python `multiprocessing` module, allowing, if requested, for parallel execution of the code highly reducing computation time.

## Usage example:
In the following example we will analyze the `BostonHousing` dataset (available [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston)). Even though the `asgl` package can easily deal with much more complex datasets, we will work using this one so we are not affected by computation time. We will show how to implement cross validation on a grid of possible parameter values for an sparse group lasso linear model, how to find the optimal parameter values and finally, how to compute the test error.

#### Example:
The following code performs cross validation in a grid of
different parameter values for an sparse group lasso model on the well known 
`BostonHousing` dataset:

```python3
# Import required packages
import numpy as np
from sklearn.datasets import load_boston
import asgl

# Import test data #
boston = load_boston()
x = boston.data
y = boston.target
group_index = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5])

# Define parameters grid
lambda1 = (10.0 ** np.arange(-3, 1.51, 0.2)) # 23 possible values for lambda
alpha = np.arange(0, 1, 0.05) # 20 possible values for alpha

# Define model parameters
model = 'lm'  # linear model
penalization = 'sgl'  # sparse group lasso penalization
parallel = True  # Code executed in parallel
error_type = 'MSE'  # Error measuremente considered. MSE stands for Mean Squared Error.

# Define a cross validation object
cv_class = asgl.CV(model=model, penalization=penalization, lambda1=lambda1, alpha=alpha,
                   nfolds=5, error_type=error_type, parallel=parallel, random_state=99)

# Compute error using k-fold cross validation
error = cv_class.cross_validation(x=x, y=y, group_index=group_index)

num_models, k_folds = error.shape
# error is a matrix of shape (number_of_models, k_folds)
print(f'We are considering a grid of {num_models} models, optimized based on {k_folds}-folds cross validation')

# Obtain the mean error across different folds
error = np.mean(error, axis=1)
```       

For a full review on the capabilities of these package we suggest accessing the  user_guide notebook provided in the [GitHub repository](https://github.com/alvaromc317/asgl)

### Citing
___
If you use ASGL for academic work, we encourage you to [cite our paper](https://link.springer.com/article/10.1007/s11634-020-00413-8). Thank you for your support and we hope you find this package useful!


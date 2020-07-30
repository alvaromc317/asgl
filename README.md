# ASGL

ASGL is a Python package that solves several regression 
related models for simultaneous variable selection and prediction 
in low and high dimensional frameworks. This package is directly 
related with research work from the authors that can be found on [this paper](https://link.springer.com/article/10.1007/s11634-020-00413-8).

The current package supports:
1. Linear regression models
2. Quantile regression models

And considers the following penalizations for variable selection:
1. lasso
2. group lasso
3. sparse group lasso
4. adaptive sparse group lasso                                   

The package is built on top of the `cvxpy` convex optimization module (see more information on this module [here](https://www.cvxpy.org/)) and makes use of python `multiprocessing` module. This allows to easily go from sequential to parallel execution of the code.

#### Example:
The following code performs cross validation in a grid of
different parameter values for an sparse group lasso model on the well known 
`BostonHousing` dataset:

```python3
1 + 1     
```       

### Citing
___
If you use ASGL for academic work, we encourage you to [cite our paper](https://link.springer.com/article/10.1007/s11634-020-00413-8). 
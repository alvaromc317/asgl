from typing import Sequence, Optional, Union
import cvxpy as cp
import numpy as np

from .constants import (
  ArrayOrSparse,
  INDIV_ADAPTIVE,
  GROUP_ADAPTIVE,
)
from .utils import _get_group_info
from .base_model import BaseModel
from .adaptive_weights import AdaptiveWeights


class Regressor(BaseModel, AdaptiveWeights):
  """
  Parameters
  ----------
  model: str, default = 'lm'
      Model to be fit. Currently, accepts:
          - 'lm': linear regression models.
          - 'qr': quantile regression models.
          - 'logit': logistic regression for binary classification, output binary classification.
      Both 'lm' and 'qr' models support multivariate regression (multiple outputs),
      allowing for simultaneous fitting and coupled feature selection with grouped penalizations.
  penalization: str or None, default = 'lasso'
      Penalization to use. Currently, accepts:
          - None: unpenalized model.
          - 'lasso': lasso penalization.
          - 'ridge': ridge penalization.
          - 'gl': group lasso penalization.
          - 'sgl': sparse group lasso penalization.
          - 'alasso': adaptive lasso penalization.
          - 'aridge': adaptive ridge penalization.
          - 'agl': adaptive group lasso penalization.
          - 'asgl': adaptive sparse group lasso penalization.
  quantile: float, default=0.5
      quantile level in quantile regression models. Valid values are between 0 and 1. It only has effect if
      ``model='qr'``
  fit_intercept: bool, default=True,
      Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
  lambda1: float, defaul=0.1
      Constant that multiplies the penalization, controlling the strength. Must be a non-negative float
      i.e. in `[0, inf)`. Larger values will result in larger penalizations.
  alpha: float, default=0.5
      Constant that performs tradeoff between lasso and group lasso in sgl and asgl penalizations.
      ``alpha=1`` enforces a lasso while ``alpha=0`` enforces a group lasso.
  solver: str, default='CLARABEL'
      Solver to be used by cvxpy. Default uses open source convex programming solver CLARABEL.
      If a list is provided, the model will try each solver in order.
      If the specified solver(s) fail, the model falls back to other installed solvers.
      See `CVXPY Solvers <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_
      for more information.
      Users can check available solvers via the command `cp.installed_solvers()`.
  weight_technique: str, default='pca_pct'
      Weight technique used to fit the adaptive weights. Currently, accepts:
          - pca_1: Builds the weights using the first component from PCA.
          - pca_pct: Builds the weights using as many components from PCA as required to achieve the
          ``variability_pct``.
          - pls_1: Builds the weights using the first component from PLS.
          - pls_pct:  Builds the weights using as many components from PLS as indicated to achieve the
          ``variability_pct``.
          - lasso: Builds the weights using the lasso model.
          - ridge: Builds the weights using the ridge model.
          - unpenalized: Builds the weights using the unpenalized model.
          - sparse_pca: Similar to 'pca_pct' but it builds the weights using sparse PCA components.
  individual_power_weight: float, default=1
      Power at which the individual weights are raised.
  group_power_weight: float, default=1
      Power at which the group weights are raised.
  variability_pct: float, default=0.9
      Percentage of variability explained by pca, pls and sparse_pca components. It only has effect if
      `` weight_technique`` is one of the following: 'pca_pct', 'pls_pct', 'sparse_pca'.
      **Note:** For sparse input matrices, this value must be set to 1.
  lambda1_weights: float, default=0.1
      The value of the parameter ``lambda1`` used to solve the lasso model if ``weight_technique='lasso'`` or
      the ridge if ``weight_technique='ridge'``
  spca_alpha: float, default=1e-5
      sparse PCA parameter. See sklearn implementation of sparse PCA for more details.
  spca_ridge_alpha: float, default=1e-2
      sparse PCA parameter. See sklearn implementation of sparse PCA for more details.
  individual_weights: array or None, default=None
      An array containing the values of individual weights in adaptive penalizations. If this parameter is informed,
      it overrides the weight estimation process defined by parameter ``weight_technique`` and allows the user to
      provide custom weights.
  group_weights: array or None, default=None
      An array containing the values of group weights in adaptive penalizations. If this parameter is informed,
      it overrides the weight estimation process defined by parameter ``weight_technique``. and allows the user to
      provide custom weights.
  tol: float, default=1e-3
      The tolerance for a coefficient in the model to be considered as 0. Values smaller than ``tol`` are assumed to
      be 0.
  weight_tol: float, default=1e-4
      Tolerance value used to avoid ZeroDivision errors when computing the weights.
  canon_backend: str, default='CPP'
      Canonicalization backend to be used by ``cvxpy``.
      Options include 'CPP' (default), 'SCIPY', and 'COO'.
      See `CVXPY Canonicalization Backends <https://www.cvxpy.org/tutorial/advanced/index.html#canonicalization-backends>`_
      for more information.

  Attributes
  ----------
  coef_: ndarray of shape (n_features,)
      Estimated coefficients for the regression problem.
  intercept_: float
      Independent term in the regression model
  n_features_in_: int
      Number of features seen during fit.
  """

  def __init__(
    self,
    model: str = "lm",
    penalization: Optional[str] = "lasso",
    quantile: float = 0.5,
    fit_intercept: bool = True,
    lambda1: float = 0.1,
    alpha: float = 0.5,
    solver: Union[str, Sequence[str]] = "default",
    weight_technique: str = "pca_pct",
    individual_power_weight: float = 1,
    group_power_weight: float = 1,
    variability_pct: float = 0.9,
    lambda1_weights: float = 0.1,
    spca_alpha: float = 1e-5,
    spca_ridge_alpha: float = 1e-2,
    individual_weights: Optional[np.ndarray] = None,
    group_weights: Optional[np.ndarray] = None,
    weight_tol: float = 1e-4,
    tol: float = 1e-3,
    verbose: bool = False,
    canon_backend: str = "CPP",
  ):
    super().__init__(
      model=model,
      penalization=penalization,
      quantile=quantile,
      fit_intercept=fit_intercept,
      lambda1=lambda1,
      alpha=alpha,
      solver=solver,
      tol=tol,
      verbose=verbose,
      canon_backend=canon_backend,
    )
    self.weight_technique = weight_technique
    self.individual_power_weight = individual_power_weight
    self.group_power_weight = group_power_weight
    self.variability_pct = variability_pct
    self.lambda1_weights = lambda1_weights
    self.spca_alpha = spca_alpha
    self.spca_ridge_alpha = spca_ridge_alpha
    self.individual_weights = individual_weights
    self.group_weights = group_weights
    self.weight_tol = weight_tol

  # Penalized problems
  def _aridge(
    self, beta_var: cp.Variable, group_index: Optional[Sequence[int]]
  ) -> cp.Expression:
    mx, my = beta_var.shape
    # Reshape weights to (mx, 1) for proper broadcasting across my outputs
    weights = np.asarray(self.individual_weights_).reshape(-1, 1)
    pen = self.lambda1 * cp.sum_squares(cp.multiply(weights, beta_var))
    return pen

  def _alasso(
    self, beta_var: cp.Variable, group_index: Optional[Sequence[int]]
  ) -> cp.Expression:
    mx, my = beta_var.shape
    # Reshape weights to (mx, 1) for proper broadcasting across my outputs
    weights = np.asarray(self.individual_weights_).reshape(-1, 1)
    pen = self.lambda1 * cp.norm1(cp.multiply(weights, beta_var))
    return pen

  def _agl(self, beta_var: cp.Variable, group_index: Sequence[int]) -> cp.Expression:
    unique_groups, group_sizes, indices_per_group = _get_group_info(group_index)
    sqrt_sizes = np.sqrt(group_sizes)
    group_weights = sqrt_sizes * self.group_weights_
    mx, my = beta_var.shape
    # For each group, compute the norm of all features in that group across all outputs
    # This gives the 2-norm of the group's coefficients (treating each output separately)
    group_norms = cp.hstack(
      [cp.norm2(beta_var[indices_per_group[g], :]) for g in unique_groups]
    )
    pen = self.lambda1 * cp.sum(cp.multiply(group_weights, group_norms))
    return pen

  def _asgl(self, beta_var: cp.Variable, group_index: Sequence[int]) -> cp.Expression:
    individual_param = self.lambda1 * self.alpha
    mx, my = beta_var.shape
    # Reshape individual weights to (mx, 1) for proper broadcasting across my outputs
    weights = np.asarray(self.individual_weights_).reshape(-1, 1)
    group_param = self.lambda1 * (1 - self.alpha)
    unique_groups, group_sizes, indices_per_group = _get_group_info(group_index)
    sqrt_sizes = np.sqrt(group_sizes)
    group_weights = sqrt_sizes * self.group_weights_
    # For each group, compute the norm of all features in that group across all outputs
    group_norms = cp.hstack(
      [cp.norm2(beta_var[indices_per_group[g], :]) for g in unique_groups]
    )
    individual_penalization = individual_param * cp.norm1(
      cp.multiply(weights, beta_var)
    )
    group_penalization = group_param * cp.sum(cp.multiply(group_weights, group_norms))
    pen = individual_penalization + group_penalization
    return pen

  def fit(
    self,
    X: ArrayOrSparse,
    y: ArrayOrSparse,
    group_index: Optional[Sequence[int]] = None,
  ):
    self._check_attributes()
    if self.penalization in (INDIV_ADAPTIVE + GROUP_ADAPTIVE):
      self.fit_weights(X, y, group_index)
    # Call the fit method of the parent class (BaseModel) to perform the main fitting logic.
    super().fit(X, y, group_index)
    return self

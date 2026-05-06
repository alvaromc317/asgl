import warnings
from typing import Sequence, Optional, Tuple, Union
from sklearn.utils.validation import check_X_y
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from scipy import sparse

from .constants import (
  ArrayOrSparse,
  INDIV_ADAPTIVE,
  GROUP_ADAPTIVE,
  ALLOWED_WEIGHT_TECHNIQUES,
)
from .base_model import BaseModel


class AdaptiveWeights:
  def __init__(
    self,
    model: str = "lm",
    penalization: str = "alasso",
    quantile: float = 0.5,
    weight_technique: str = "pca_pct",
    individual_power_weight: float = 1,
    group_power_weight: float = 1,
    variability_pct: float = 0.9,
    lambda1_weights: float = 0.1,
    spca_alpha: float = 1e-5,
    spca_ridge_alpha: float = 1e-2,
    individual_weights=None,
    group_weights=None,
    solver: Union[str, Sequence[str]] = "CLARABEL",
    weight_tol: float = 1e-4,
    verbose: bool = False,
    canon_backend: str = "CPP",
  ):
    self.model = model
    self.penalization = penalization
    self.quantile = quantile
    self.weight_technique = weight_technique
    self.individual_power_weight = individual_power_weight
    self.group_power_weight = group_power_weight
    self.variability_pct = variability_pct
    self.lambda1_weights = lambda1_weights
    self.spca_alpha = spca_alpha
    self.spca_ridge_alpha = spca_ridge_alpha
    self.individual_weights = individual_weights
    self.group_weights = group_weights
    self.solver = solver
    self.weight_tol = weight_tol
    self.verbose = verbose
    self.canon_backend = canon_backend

  def _wpca_1(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    """
    Weights based on the first principal component
    """
    if sparse.issparse(X):
      pca = PCA(n_components=1, svd_solver="arpack")
    else:
      pca = PCA(n_components=1, svd_solver="auto")
    pca.fit(X)
    tmp_weight = np.abs(pca.components_).ravel()
    return tmp_weight

  def _wpca_pct(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    """
    Weights based on principal component analysis
    """
    if sparse.issparse(X) and np.min(X.shape) > 1:
      max_comp = np.min(X.shape) - 1
      # Run PCA once with max_comp
      pca = PCA(n_components=max_comp, svd_solver="arpack")
      t = pca.fit_transform(X)
      explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
      n_comp = (
        np.searchsorted(explained_variance_ratio_cumsum, self.variability_pct) + 1
      )
      n_comp = min(n_comp, max_comp)  # Safety upper bound
      t = t[:, :n_comp]
      p = pca.components_[:n_comp].T
    else:
      if sparse.issparse(X):
        # If min(X.shape) == 1, arpack PCA fails. Densify to use optimized full SVD logic.
        X = X.toarray()
      # For dense matrices, using svd_solver="full" with n_components=None is vastly
      # faster than arpack for computing all components and avoids ValueErrors when
      # min(X.shape) == 1.
      pca = PCA(n_components=None, svd_solver="full")
      t = pca.fit_transform(X)
      explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
      n_comp = (
        np.searchsorted(explained_variance_ratio_cumsum, self.variability_pct) + 1
      )
      t = t[:, :n_comp]
      p = pca.components_[:n_comp].T
    unpenalized_model = BaseModel(
      model=self.model,
      penalization=None,
      fit_intercept=True,
      quantile=self.quantile,
      solver=self.solver,
      verbose=self.verbose,
      canon_backend=self.canon_backend,
    )
    unpenalized_model.fit(X=t, y=y)
    beta_sol = unpenalized_model.coef_
    # Recover an estimation of the beta parameters and use it as weight
    tmp_weight = np.abs(np.dot(p, beta_sol))
    # If multi-output (2D coefficients), collapse to 1D by taking L2 norm across outputs
    if tmp_weight.ndim > 1:
      tmp_weight = np.linalg.norm(tmp_weight, axis=1)
    else:
      tmp_weight = tmp_weight.ravel()
    return tmp_weight

  def _wpls_1(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    """
    Weights based on the first partial least squares component
    """
    if sparse.issparse(X):
      raise ValueError(
        "weight_technique='pls_1' does not support sparse matrices. "
        "PLSRegression requires mean centering which would densify the matrix. "
        "Convert to dense with X = X.toarray() before fitting."
      )
    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(X, y)
    tmp_weight = np.abs(pls.x_rotations_).ravel()
    return tmp_weight

  def _wpls_pct(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    """
    Weights based on partial least squares
    """
    if sparse.issparse(X):
      raise ValueError(
        "weight_technique='pls_pct' does not support sparse matrices. "
        "PLSRegression requires mean centering which would densify the matrix. "
        "Convert to dense with X = X.toarray() before fitting."
      )
    total_variance_in_x = np.sum(np.var(X, axis=0))
    pls = PLSRegression(n_components=np.min(X.shape) - 1, scale=False)
    pls.fit(X, y)
    variance_in_pls = np.var(pls.x_scores_, axis=0)
    fractions_of_explained_variance = np.cumsum(variance_in_pls / total_variance_in_x)
    if self.variability_pct > np.max(fractions_of_explained_variance):
      warnings.warn(
        f"variability_pct={self.variability_pct} was requested, but PLS "
        f"attains a maximum of {np.max(fractions_of_explained_variance):.4f}",
        RuntimeWarning,
        stacklevel=2,
      )
    n_comp = np.searchsorted(fractions_of_explained_variance, self.variability_pct) + 1
    # Ensure n_comp is at least 1
    n_comp = np.clip(n_comp, 1, pls.x_rotations_.shape[1])

    # Calculate coefficients directly from the existing PLS model without refitting
    coef = np.dot(pls.x_rotations_[:, :n_comp], pls.y_loadings_[:, :n_comp].T)
    tmp_weight = np.abs(coef)
    # If multi-output (2D coefficients), collapse to 1D by taking L2 norm across outputs
    if tmp_weight.ndim > 1:
      tmp_weight = np.linalg.norm(tmp_weight, axis=1)
    else:
      tmp_weight = tmp_weight.ravel()
    return tmp_weight

  def _wsparse_pca(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    """
    Weights based on sparse principal component analysis.
    """
    if sparse.issparse(X):
      raise ValueError(
        "weight_technique='sparse_pca' does not support sparse matrices. "
        "SparsePCA requires dense input. "
        "Convert to dense with X = X.toarray() before fitting."
      )
    total_variance_in_x = np.sum(np.var(X, axis=0))
    spca = SparsePCA(
      n_components=np.min(X.shape) - 1,
      alpha=self.spca_alpha,
      ridge_alpha=self.spca_ridge_alpha,
    )
    t = spca.fit_transform(X)
    p = spca.components_.T
    # Obtain explained variance using spca as explained in the original paper (based on QR decomposition)
    _, r_spca = np.linalg.qr(t, mode="reduced")
    t_spca_variance = np.square(np.diag(r_spca)) / X.shape[0]
    fractions_of_explained_variance = np.cumsum(t_spca_variance / total_variance_in_x)
    if self.variability_pct > np.max(fractions_of_explained_variance):
      warnings.warn(
        f"variability_pct={self.variability_pct} was requested, but Sparse PCA "
        f"attains a maximum of {np.max(fractions_of_explained_variance):.4f}",
        RuntimeWarning,
        stacklevel=2,
      )
    n_comp = np.searchsorted(fractions_of_explained_variance, self.variability_pct) + 1
    n_comp = max(1, n_comp)
    unpenalized_model = BaseModel(
      model=self.model,
      penalization=None,
      fit_intercept=True,
      quantile=self.quantile,
      solver=self.solver,
      verbose=self.verbose,
      canon_backend=self.canon_backend,
    )
    unpenalized_model.fit(X=t[:, 0:n_comp], y=y)
    beta_sol = unpenalized_model.coef_
    # Recover an estimation of the beta parameters and use it as weight
    tmp_weight = np.abs(np.dot(p[:, 0:n_comp], beta_sol))
    # If multi-output (2D coefficients), collapse to 1D by taking L2 norm across outputs
    if tmp_weight.ndim > 1:
      tmp_weight = np.linalg.norm(tmp_weight, axis=1)
    else:
      tmp_weight = tmp_weight.ravel()
    return tmp_weight

  def _wunpenalized(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    """
    Only for low dimensional frameworks. Weights based on an unpenalized regression model
    """
    unpenalized_model = BaseModel(
      model=self.model,
      penalization=None,
      fit_intercept=True,
      quantile=self.quantile,
      solver=self.solver,
      verbose=self.verbose,
      canon_backend=self.canon_backend,
    )
    unpenalized_model.fit(X=X, y=y)
    tmp_weight = np.abs(unpenalized_model.coef_)
    # If multi-output (2D coefficients), collapse to 1D by taking L2 norm across outputs
    if tmp_weight.ndim > 1:
      tmp_weight = np.linalg.norm(tmp_weight, axis=1)
    return tmp_weight

  def _wlasso(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    lasso_model = BaseModel(
      model=self.model,
      penalization="lasso",
      lambda1=self.lambda1_weights,
      fit_intercept=True,
      quantile=self.quantile,
      solver=self.solver,
      verbose=self.verbose,
      canon_backend=self.canon_backend,
    )
    lasso_model.fit(X=X, y=y)
    tmp_weight = np.abs(lasso_model.coef_)
    # If multi-output (2D coefficients), collapse to 1D by taking L2 norm across outputs
    if tmp_weight.ndim > 1:
      tmp_weight = np.linalg.norm(tmp_weight, axis=1)
    return tmp_weight

  def _wridge(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    ridge_model = BaseModel(
      model=self.model,
      penalization="ridge",
      lambda1=self.lambda1_weights,
      fit_intercept=True,
      quantile=self.quantile,
      solver=self.solver,
      verbose=self.verbose,
      canon_backend=self.canon_backend,
    )
    ridge_model.fit(X=X, y=y)
    tmp_weight = np.abs(ridge_model.coef_)
    # If multi-output (2D coefficients), collapse to 1D by taking L2 norm across outputs
    if tmp_weight.ndim > 1:
      tmp_weight = np.linalg.norm(tmp_weight, axis=1)
    return tmp_weight

  def _check_type_penalization(self) -> Tuple[bool, bool]:
    return (
      self.penalization in INDIV_ADAPTIVE,
      self.penalization in GROUP_ADAPTIVE,
    )

  def fit_weights(
    self,
    X: ArrayOrSparse,
    y: ArrayOrSparse,
    group_index: Optional[Sequence[int]] = None,
  ):
    if (
      not isinstance(self.weight_technique, str)
      or self.weight_technique not in ALLOWED_WEIGHT_TECHNIQUES
    ):
      raise ValueError(
        f"weight_technique must be one of {sorted(ALLOWED_WEIGHT_TECHNIQUES)}; "
        f"got {self.weight_technique}."
      )
    X, y = check_X_y(
      X,
      y,
      accept_sparse=True,
      y_numeric=True,
      ensure_min_samples=2,
      multi_output=True,
    )
    bool_individual, bool_group = self._check_type_penalization()
    if bool_group and group_index is None:
      raise ValueError(
        "A group penalisation was requested but `group_index` is missing."
      )
    tmp_weight: Optional[np.ndarray] = None
    if bool_individual:
      if self.individual_weights is None:
        tmp_weight = getattr(self, "_w" + self.weight_technique)(X=X, y=y)
        self.individual_weights_ = 1 / (
          tmp_weight**self.individual_power_weight + self.weight_tol
        )
      else:
        self.individual_weights_ = self.individual_weights

    if bool_group:
      if self.group_weights is None:
        if tmp_weight is None:
          tmp_weight = getattr(self, "_w" + self.weight_technique)(X=X, y=y)
        group_index = np.asarray(group_index, dtype=int)

        # O(N log N) vectorized group norm calculations replacing the O(N*G) loop
        unique_groups, inverse_indices = np.unique(group_index, return_inverse=True)
        # Efficiently aggregate squared values by group using bincount
        group_sums = np.bincount(inverse_indices, weights=tmp_weight**2)
        norms = np.sqrt(group_sums)

        self.group_weights_ = 1.0 / (
          np.power(norms, self.group_power_weight) + self.weight_tol
        )
      else:
        self.group_weights_ = self.group_weights

    if bool_individual and (len(self.individual_weights_) != X.shape[1]):
      raise ValueError(
        "Number of individual weights does not match the number of columns in X"
      )
    if bool_group and (len(self.group_weights_) != len(np.unique(group_index))):
      raise ValueError(
        "Number of group weights does not match the number of groups in group_index"
      )
    return self

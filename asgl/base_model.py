import warnings
from typing import Sequence, Optional, Tuple, Union
from sklearn.utils.validation import check_is_fitted, check_X_y, check_scalar
import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import expit
from sklearn.metrics import accuracy_score
from scipy import sparse

from .constants import (
  ArrayOrSparse,
  ALLOWED_MODELS,
  ALL_PENALTIES,
  ALLOWED_CANON_BACKENDS,
  GROUP_NONADAPTIVE,
  GROUP_ADAPTIVE,
)
from .utils import _get_group_info


class BaseModel(BaseEstimator, RegressorMixin):
  """
  Base class for penalized regression models using cp.
  """

  def __init__(
    self,
    model: str = "lm",
    penalization: Optional[str] = "lasso",
    quantile: float = 0.5,
    fit_intercept: bool = True,
    lambda1: float = 0.1,
    alpha: float = 0.5,
    solver: Union[str, Sequence[str]] = "CLARABEL",
    tol: float = 1e-3,
    verbose: bool = False,
    canon_backend: str = "CPP",
  ):
    self.model = model
    self.penalization = penalization
    self.quantile = quantile
    self.fit_intercept = fit_intercept
    self.lambda1 = lambda1
    self.alpha = alpha
    self.solver = solver
    self.tol = tol
    self.verbose = verbose
    self.canon_backend = canon_backend

  @property
  def _estimator_type(self):
    if self.model == "logit":
      return "classifier"
    else:
      return "regressor"

  def _check_attributes(self) -> None:
    """
    Validate constructor arguments.
    Raises ValueError If any argument is outside the allowed domain.
    """
    # Numerical arguments
    check_scalar(
      self.lambda1,
      "lambda1",
      target_type=(int, float),
      min_val=0.0,
      include_boundaries="left",
    )
    check_scalar(
      self.alpha,
      "alpha",
      target_type=(int, float),
      min_val=0.0,
      max_val=1.0,
      include_boundaries="both",
    )
    check_scalar(
      self.quantile,
      "quantile",
      target_type=(int, float),
      min_val=0.0,
      max_val=1.0,
      include_boundaries="neither",
    )
    # string arguments
    check_scalar(self.model, "model", target_type=str)
    if self.model not in ALLOWED_MODELS:
      raise ValueError(
        f"model must be one of {sorted(ALLOWED_MODELS)}; got {self.model}."
      )
    # penalization may also be None
    check_scalar(self.penalization, "penalization", target_type=(str, type(None)))
    if (self.penalization is not None) and (self.penalization not in ALL_PENALTIES):
      raise ValueError(
        f"penalization must be one of {sorted(ALL_PENALTIES)}; got {self.penalization}."
      )
    # canon_backend validation
    check_scalar(self.canon_backend, "canon_backend", target_type=str)
    if self.canon_backend not in ALLOWED_CANON_BACKENDS:
      raise ValueError(
        f"canon_backend must be one of {sorted(ALLOWED_CANON_BACKENDS)}; got {self.canon_backend}."
      )

  def _quantile_function(self, X) -> cp.Expression:
    """cp quantile loss function."""
    # new implementation, should be more efficient avoiding abs
    # Uses a residual splitting approach
    q = float(self.quantile)
    return cp.sum(0.5 * cp.abs(X) + (q - 0.5) * X)

  def _define_quantile_objective(self, n, q, u, v):
    # objective: (1/n) * (q * sum(u) + (1-q) * sum(v))
    return (1.0 / n) * (q * cp.sum(u) + (1.0 - q) * cp.sum(v))

  def _define_objective_function(
    self, y: ArrayOrSparse, model_prediction: cp.Expression
  ) -> cp.Expression:
    # Define the objective function based on the problem to solve
    if self.model == "lm":
      return (1.0 / y.shape[0]) * cp.sum_squares(y - model_prediction)
    elif self.model == "qr":
      return (1.0 / y.shape[0]) * cp.sum(
        self._quantile_function(X=(y - model_prediction))
      )
    elif self.model == "logit":
      # Flatten both to 1D for universal element-wise multiply across single and multi-output
      y_flat = cp.reshape(y, (y.shape[0] * y.shape[1],), order="F")
      pred_flat = cp.reshape(
        model_prediction,
        (model_prediction.shape[0] * model_prediction.shape[1],),
        order="F",
      )
      return (1.0 / y.shape[0]) * cp.sum(
        cp.logistic(pred_flat) - cp.multiply(y_flat, pred_flat)
      )
    else:
      raise ValueError("Invalid value for model parameter.")

  def _solve_cp_problem(self, problem: cp.Problem) -> None:
    # Normalise solver to a list of strings; "default" means let cp choose
    if isinstance(self.solver, str):
      requested_solvers = [self.solver]
    else:
      requested_solvers = list(self.solver)

    installed_solvers = sorted(cp.installed_solvers())
    failed_solvers: set = set()
    solved = False

    # --- Phase 1: try each solver in the user-specified sequence ---
    for solver_name in requested_solvers:
      # "default" means pass solver=None so cvxpy picks its own default
      cp_solver = None if solver_name == "default" else solver_name
      try:
        with warnings.catch_warnings():
          warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="You are solving a parameterized problem that is not DPP",
          )
          problem.solve(
            solver=cp_solver,
            verbose=self.verbose,
            canon_backend=self.canon_backend,
          )
        if problem.status is not None and "optimal" in problem.status.lower():
          solved = True
          break
        else:
          # Solved without exception but status is not optimal
          warnings.warn(
            f"Solver {solver_name} returned status '{problem.status}'. Trying next solver.",
            RuntimeWarning,
            stacklevel=2,
          )
          failed_solvers.add(solver_name)
      except (ValueError, cp.error.SolverError, cp.error.DCPError):
        warnings.warn(
          f"Solver {solver_name} failed. Trying next solver.",
          RuntimeWarning,
          stacklevel=2,
        )
        failed_solvers.add(solver_name)

    # --- Phase 2: fall back to remaining installed solvers not yet tried ---
    if not solved:
      remaining = [
        s
        for s in installed_solvers
        if s not in failed_solvers and s not in requested_solvers
      ]
      if remaining:
        warnings.warn(
          f"Requested solver(s) {requested_solvers} failed. "
          f"Trying remaining installed solvers: {remaining}",
          RuntimeWarning,
          stacklevel=2,
        )
      for alt_solver in remaining:
        try:
          with warnings.catch_warnings():
            warnings.filterwarnings(
              action="ignore",
              category=UserWarning,
              message="You are solving a parameterized problem that is not DPP",
            )
            problem.solve(
              solver=alt_solver,
              verbose=self.verbose,
              canon_backend=self.canon_backend,
            )
          if problem.status is not None and "optimal" in problem.status.lower():
            warnings.warn(
              f"Successfully solved with fallback solver: {alt_solver}",
              RuntimeWarning,
              stacklevel=2,
            )
            solved = True
            break
          else:
            failed_solvers.add(alt_solver)
        except (ValueError, cp.error.SolverError, cp.error.DCPError):
          failed_solvers.add(alt_solver)

    if (
      problem.status is None
      or "infeasible" in problem.status.lower()
      or "unbounded" in problem.status.lower()
    ):
      warnings.warn(
        f"Optimization problem finished with status: {problem.status}. Solution may be unreliable.",
        RuntimeWarning,
        stacklevel=2,
      )
    # Store serializable stats
    stats = problem.solver_stats
    safelist = ["solver_name", "num_iters", "setup_time", "solve_time", "status"]
    solver_stats_ = {k: getattr(stats, k, None) for k in safelist}
    solver_stats_["status"] = problem.status if hasattr(problem, "status") else None
    self.solver_stats_ = solver_stats_

  # Penalized problems
  def _ridge(
    self, beta_var: cp.Variable, group_index: Optional[Sequence[int]]
  ) -> cp.Expression:
    pen = self.lambda1 * cp.sum_squares(beta_var)
    return pen

  def _lasso(
    self, beta_var: cp.Variable, group_index: Optional[Sequence[int]]
  ) -> cp.Expression:
    pen = self.lambda1 * cp.norm1(beta_var)
    return pen

  def _gl(self, beta_var: cp.Variable, group_index: Sequence[int]) -> cp.Expression:
    unique_groups, group_sizes, indices_per_group = _get_group_info(group_index)
    sqrt_sizes = np.sqrt(group_sizes)
    group_norms = cp.hstack(
      [cp.norm2(beta_var[indices_per_group[g]]) for g in unique_groups]
    )
    pen = self.lambda1 * cp.sum(cp.multiply(sqrt_sizes, group_norms))
    return pen

  def _sgl(self, beta_var: cp.Variable, group_index: Sequence[int]) -> cp.Expression:
    group_param = self.lambda1 * (1 - self.alpha)
    individual_param = self.lambda1 * self.alpha
    unique_groups, group_sizes, indices_per_group = _get_group_info(group_index)
    sqrt_sizes = np.sqrt(group_sizes)
    group_norms = cp.hstack(
      [cp.norm2(beta_var[indices_per_group[g]]) for g in unique_groups]
    )
    group_penalization = group_param * cp.sum(cp.multiply(sqrt_sizes, group_norms))
    individual_penalization = individual_param * cp.norm1(beta_var)
    pen = individual_penalization + group_penalization
    return pen

  def _obtain_beta(
    self, X: ArrayOrSparse, y: ArrayOrSparse, group_index: Optional[Sequence[int]]
  ) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    mx = X.shape[1]
    # Ensure y is 2D for CVXPY operations
    y_was_1d = y.ndim == 1
    if y_was_1d:
      y = y.reshape(-1, 1)
    my = y.shape[1]
    beta_var = cp.Variable((mx, my))
    intercept_var = cp.Variable() if self.fit_intercept else 0
    # wrap the X in a constant to avoid issues with sparse matrices in cvxpy expressions
    X_constant = cp.Constant(X)
    pred = X_constant @ beta_var + intercept_var
    if self.model == "qr":
      # residual splitting variables (nonnegative)
      # For multi-output: u,v shape (n, my), for single output: (n,)
      if my == 1:
        u = cp.Variable(n, nonneg=True)
        v = cp.Variable(n, nonneg=True)
        # Reshape pred to (n,) for single output constraint
        pred_reshaped = cp.reshape(pred, (n,), order="F")
        constraints = [y.ravel() - pred_reshaped == u - v]
      else:
        u = cp.Variable((n, my), nonneg=True)
        v = cp.Variable((n, my), nonneg=True)
        constraints = [y - pred == u - v]

      objective = self._define_quantile_objective(n, self.quantile, u, v)

      # add penalties if any (pen should operate on beta_var)
      if self.penalization is not None:
        pen = getattr(self, "_" + self.penalization)(beta_var, group_index)
        objective = objective + pen

      problem = cp.Problem(cp.Minimize(objective), constraints)

    else:
      # existing lm / logit handling (keeping X_const @ beta_var)
      objective = self._define_objective_function(y, pred)
      if self.penalization is not None:
        pen = getattr(self, "_" + self.penalization)(beta_var, group_index)
        objective = objective + pen
      problem = cp.Problem(cp.Minimize(objective))

    self._solve_cp_problem(problem)
    beta_sol = beta_var.value
    intercept_sol = intercept_var.value if self.fit_intercept else 0
    if (beta_sol is None) or (intercept_sol is None):
      raise ValueError("CVXPY optimization failed to find a solution")
    beta_sol[np.abs(beta_sol) < self.tol] = 0
    # Flatten beta if y was originally 1D
    if y_was_1d:
      beta_sol = beta_sol.ravel()
    return intercept_sol, beta_sol

  def fit(
    self,
    X: ArrayOrSparse,
    y: ArrayOrSparse,
    group_index: Optional[Sequence[int]] = None,
  ):
    self.feature_names_in_ = None
    if hasattr(X, "columns") and callable(getattr(X, "columns", None)):
      self.feature_names_in_ = np.asarray(X.columns, dtype=object)
    X, y = check_X_y(
      X,
      y,
      accept_sparse=True,
      y_numeric=True,
      ensure_min_samples=2,
      multi_output=True,
    )
    self.n_features_in_ = X.shape[1]
    self._check_attributes()
    # Check binary y
    if self._estimator_type == "classifier":
      if y.ndim > 1:
        raise ValueError("Logistic regression does not support multi-output y.")
      unique_y_values = set(np.unique(y))
      if not (unique_y_values.issubset({0, 1}) or unique_y_values.issubset({0.0, 1.0})):
        raise ValueError(
          "For logistic model, y must contain only 0 and 1 (or 0.0, 1.0)."
        )
      y = y.astype(int)
      self.classes_ = np.array([0, 1])  # Assuming 0 and 1 are the classes
    if (
      self.penalization in (GROUP_NONADAPTIVE + GROUP_ADAPTIVE) and group_index is None
    ):
      raise ValueError(
        "The penalization provided requires fitting the model with a group_index parameter but no group_index was detected."
      )
    if group_index is not None:
      group_index = np.asarray(group_index, dtype=int)
      if len(group_index) != X.shape[1]:
        raise ValueError(
          f"group_index length {len(group_index)} does not match number of features {X.shape[1]}"
        )
      if any(group_index < 0):
        raise ValueError(
          "group_index must be a positive integer array. Negative values detected"
        )
    # Solve the problem
    self.intercept_, self.coef_ = self._obtain_beta(X, y, group_index)
    self.is_fitted_ = True
    return self

  def decision_function(self, X: ArrayOrSparse) -> np.ndarray:
    check_is_fitted(self, ["coef_", "intercept_", "is_fitted_"])
    intercept = self.intercept_ if self.fit_intercept else 0
    predictions = (
      X @ self.coef_ + intercept
      if sparse.issparse(X)
      else np.dot(X, self.coef_) + intercept
    )
    return predictions

  def predict_proba(self, X: ArrayOrSparse) -> np.ndarray:
    if self._estimator_type != "classifier":
      raise AttributeError(
        f"predict_proba is not available when model is '{self.model}'. It is only available for classifier models."
      )
    check_is_fitted(self, "classes_")  # Ensure classes_ is available
    decision = self.decision_function(X)
    proba_pos_class = expit(decision)
    return np.vstack([1 - proba_pos_class, proba_pos_class]).T

  def predict(self, X: ArrayOrSparse) -> np.ndarray:
    check_is_fitted(self, ["coef_", "intercept_", "is_fitted_"])
    raw_predictions = self.decision_function(X)
    if self._estimator_type == "classifier":
      # self.classes_ should be [0, 1]
      # Threshold decision function output at 0 for class labels
      indices = (raw_predictions >= 0).astype(
        int
      )  # 0 if raw_pred <= 0, 1 if raw_pred > 0
      return self.classes_[indices]
    else:  # Regressor
      return raw_predictions

  def __sklearn_tags__(self):
    tags = super().__sklearn_tags__()
    tags.target_tags.required = True
    tags.target_tags.multi_output = True
    if self.model == "logit":
      tags.estimator_type = "classifier"
      from sklearn.utils._tags import ClassifierTags

      tags.classifier_tags = ClassifierTags(multi_class=False)
    else:
      tags.estimator_type = "regressor"
      from sklearn.utils._tags import RegressorTags

      tags.regressor_tags = RegressorTags()
    return tags

  def _more_tags(self):
    return {
      "allow_nan": False,
      "requires_y": True,
    }

  def score(self, X, y, sample_weight=None):
    if self._estimator_type == "regressor":
      return RegressorMixin.score(self, X, y, sample_weight)
    else:  # Classifier
      return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

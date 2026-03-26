import warnings
from typing import Sequence, Optional, Tuple, Union, Dict
from sklearn.utils.validation import check_is_fitted, check_X_y, check_scalar
import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from scipy.special import expit
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score

# sparse matrices
from scipy import sparse

ArrayOrSparse = Union[np.ndarray, sparse.spmatrix]

# Define constants for penalization types
INDIV_NONADAPTIVE = ["lasso", "ridge", "sgl"]
INDIV_ADAPTIVE = ["alasso", "aridge", "asgl"]
GROUP_NONADAPTIVE = ["gl", "sgl"]
GROUP_ADAPTIVE = ["agl", "asgl"]
ALL_PENALTIES = INDIV_NONADAPTIVE + INDIV_ADAPTIVE + GROUP_ADAPTIVE + GROUP_NONADAPTIVE
ALLOWED_MODELS = ["lm", "qr", "logit"]
ALLOWED_WEIGHT_TECHNIQUES = {
    "pca_1",
    "pca_pct",
    "pls_1",
    "pls_pct",
    "sparse_pca",
    "unpenalized",
    "lasso",
    "ridge",
}
ALLOWED_CANON_BACKENDS = {
    "CPP",
    "SCIPY",
    "COO",
}


def _get_group_info(group_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Efficiently computes group sizes and indices for each group.
    """
    argsort_indices = np.argsort(group_index, kind='mergesort')
    sorted_group_index = group_index[argsort_indices]
    unique_groups, group_starts, group_counts = np.unique(sorted_group_index, return_index=True, return_counts=True)
    indices_per_group = {
        g: argsort_indices[start : start + count]
        for g, start, count in zip(unique_groups, group_starts, group_counts)
    }
    return unique_groups, group_counts, indices_per_group


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
            return (-1.0 / y.shape[0]) * cp.sum(
                cp.multiply(y - 1, model_prediction) - cp.logistic(-model_prediction)
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
            remaining = [s for s in installed_solvers if s not in failed_solvers
                         and s not in requested_solvers]
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
            unique_y_values = set(np.unique(y))
            if not (unique_y_values.issubset({0, 1}) or unique_y_values.issubset({0.0, 1.0})):
                raise ValueError(
                    "For logistic model, y must contain only 0 and 1 (or 0.0, 1.0)."
                )
            y = y.astype(int)
            self.classes_ = np.array([0, 1])  # Assuming 0 and 1 are the classes
        if (
            self.penalization in (GROUP_NONADAPTIVE + GROUP_ADAPTIVE)
            and group_index is None
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
        if self.model == "logit":
            tags.estimator_type = "classifier"
        else:
            tags.estimator_type = "regressor"
        return tags

    def _more_tags(self):
        tags = {
            "allow_nan": False,  # declare the estimator does *not* accept NaNs
            "requires_y": True,  # fitting requires a target y
        }
        if self._estimator_type == "classifier":
            tags["binary_only"] = True
        return tags

    def score(self, X, y, sample_weight=None):
        if self._estimator_type == "regressor":
            return RegressorMixin.score(self, X, y, sample_weight)
        else:  # Classifier
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


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
        pca = PCA(n_components=1, svd_solver="auto")
        pca.fit(X)
        tmp_weight = np.abs(pca.components_).ravel()
        return tmp_weight

    def _wpca_pct(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
        """
        Weights based on principal component analysis
        """
        if sparse.issparse(X) and self.variability_pct < 1:
            max_comp = np.min(X.shape) - 1
            # Run PCA once with max_comp
            pca = PCA(n_components=max_comp, svd_solver="arpack")
            t = pca.fit_transform(X)
            explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_comp = np.searchsorted(explained_variance_ratio_cumsum, self.variability_pct) + 1
            t = t[:, :n_comp]
            p = pca.components_[:n_comp].T
        else:
            var_pct2 = (
                (np.min(X.shape) - 1) if self.variability_pct == 1 else self.variability_pct
            )
            pca = PCA(n_components=var_pct2, svd_solver="auto")
            t = pca.fit_transform(X)  # scores
            p = pca.components_.T  # loadings
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
        pls = PLSRegression(n_components=1, scale=False)
        pls.fit(X, y)
        tmp_weight = np.abs(pls.x_rotations_).ravel()
        return tmp_weight

    def _wpls_pct(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
        """
        Weights based on partial least squares
        """
        total_variance_in_x = np.sum(np.var(X, axis=0))
        pls = PLSRegression(n_components=np.min(X.shape) - 1, scale=False)
        pls.fit(X, y)
        variance_in_pls = np.var(pls.x_scores_, axis=0)
        fractions_of_explained_variance = np.cumsum(
            variance_in_pls / total_variance_in_x
        )
        if self.variability_pct > np.max(fractions_of_explained_variance):
            warnings.warn(
                f"The total explained variability using PLS reaches {np.max(fractions_of_explained_variance)}.",
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
        x_center = X - X.mean(axis=0)
        total_variance_in_x = np.sum(np.var(X, axis=0))
        spca = SparsePCA(
            n_components=np.min(X.shape) - 1,
            alpha=self.spca_alpha,
            ridge_alpha=self.spca_ridge_alpha,
        )
        t = spca.fit_transform(x_center)
        p = spca.components_.T
        # Obtain explained variance using spca as explained in the original paper (based on QR decomposition)
        _, r_spca = np.linalg.qr(t, mode="reduced")
        t_spca_variance = np.square(np.diag(r_spca)) / X.shape[0]
        fractions_of_explained_variance = np.cumsum(
            t_spca_variance / total_variance_in_x
        )
        if self.variability_pct > np.max(fractions_of_explained_variance):
            warnings.warn(
                f"The total explained variability using Sparse PCA reaches {np.max(fractions_of_explained_variance)}.",
                RuntimeWarning,
                stacklevel=2,
            )
        n_comp = np.searchsorted(fractions_of_explained_variance, self.variability_pct)
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
        if not isinstance(self.weight_technique, str) or self.weight_technique not in ALLOWED_WEIGHT_TECHNIQUES:
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

                self.group_weights_ = 1.0 / (np.power(norms, self.group_power_weight) + self.weight_tol)
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
        penalization: str | None = "lasso",
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
        individual_weights: np.ndarray | None = None,
        group_weights: np.ndarray | None = None,
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
        pen = self.lambda1 * cp.sum_squares(
            cp.multiply(weights, beta_var)
        )
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
        group_penalization = group_param * cp.sum(
            cp.multiply(group_weights, group_norms)
        )
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

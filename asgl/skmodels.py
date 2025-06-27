import warnings
from typing import Sequence, Optional, Tuple
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

# Define constants for penalization types
INDIV_NONADAPTIVE = ['lasso', 'ridge', 'sgl']
INDIV_ADAPTIVE = ['alasso', 'aridge', 'asgl']
GROUP_NONADAPTIVE = ['gl', 'sgl']
GROUP_ADAPTIVE = ['agl', 'asgl']
ALL_PENALTIES = INDIV_NONADAPTIVE + INDIV_ADAPTIVE + GROUP_ADAPTIVE + GROUP_NONADAPTIVE
ALLOWED_MODELS = ["lm", "qr", "logit"]

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="You are solving a parameterized problem that is not DPP")

class BaseModel(BaseEstimator, RegressorMixin):
    """
    Base class for penalized regression models using cp.
    """
    def __init__(self,
                 model: str = 'lm',
                 penalization: Optional[str] = 'lasso',
                 quantile: float = 0.5,
                 fit_intercept: bool = True,
                 lambda1: float = 0.1,
                 alpha: float = 0.5,
                 solver: str = 'CLARABEL',
                 tol: float = 1e-3):
        self.model = model
        self.penalization = penalization
        self.quantile = quantile
        self.fit_intercept = fit_intercept
        self.lambda1 = lambda1
        self.alpha = alpha
        self.solver = solver
        self.tol = tol

    @property
    def _estimator_type(self):
        if self.model == 'logit':
            return "classifier"
        else:
            return "regressor"

    def _check_attributes(self) -> None:
        """
        Validate constructor arguments.
        Raises ValueError If any argument is outside the allowed domain.
        """
        # Numerical arguments
        check_scalar(self.lambda1, "lambda1", target_type=(int, float), min_val=0.0, include_boundaries='left')
        check_scalar(self.alpha, "alpha", target_type=(int, float), min_val=0.0, max_val=1.0, include_boundaries='both')
        check_scalar(self.quantile, "quantile", target_type=(int, float), min_val=0.0, max_val=1.0, include_boundaries='neither')
        # string arguments
        check_scalar(self.model, "model", target_type=str)
        if self.model not in ALLOWED_MODELS:
            raise ValueError(f"model must be one of {sorted(ALLOWED_MODELS)}; got {self.model}.")
        # penalization may also be None
        check_scalar(self.penalization, "penalization", target_type=(str, type(None)))
        if (self.penalization is not None) and (self.penalization not in ALL_PENALTIES):
            raise ValueError(f"penalization must be one of {sorted(ALL_PENALTIES)}; got {self.penalization}.")

    def _quantile_function(self, X) -> cp.Expression:
        """cp quantile loss function."""
        return 0.5 * cp.abs(X) + (self.quantile - 0.5) * X

    def _define_objective_function(self,y: np.ndarray, model_prediction: cp.Expression) -> cp.Expression:
        # Define the objective function based on the problem to solve
        if self.model == 'lm':
            return (1.0 / y.shape[0]) * cp.sum_squares(y - model_prediction)
        elif self.model == 'qr':
            return (1.0 / y.shape[0]) * cp.sum(self._quantile_function(X=(y - model_prediction)))
        elif self.model == 'logit':
            return (-1.0 / y.shape[0]) * cp.sum(
                cp.multiply(y - 1, model_prediction) - cp.logistic(-model_prediction))
        else:
            raise ValueError('Invalid value for model parameter.')

    def _solve_cp_problem(self, problem: cp.Problem) -> None:
        # Solve a cvxpy problem
        solver_options = list(cp.installed_solvers())
        chosen_solver = self.solver if self.solver != 'default' else None # Let cp choose default
        try:
            problem.solve(solver=chosen_solver)
        except (ValueError, cp.error.SolverError, cp.error.DCPError):
            warnings.warn(f'Default solver {self.solver} failed. Using alternative options from {solver_options}',
                          RuntimeWarning, stacklevel=2)
            if chosen_solver in solver_options:
                solver_options.remove(chosen_solver) # Skip the one that already failed
            for alt_solver in solver_options:
                try:
                    problem.solve(solver=alt_solver)
                    if 'optimal' in problem.status.lower():
                        warnings.warn(f"Successfully solved with alternative solver: {alt_solver}",
                                      RuntimeWarning, stacklevel=2)
                        break
                except (ValueError, cp.error.SolverError, cp.error.DCPError):
                    pass
        if problem.status is None or "infeasible" in problem.status.lower() or "unbounded" in problem.status.lower():
            warnings.warn(f"Optimization problem finished with status: {problem.status}. Solution may be unreliable.",
                          RuntimeWarning, stacklevel=2)
        # Store serializable stats
        stats = problem.solver_stats
        safelist = ["solver_name", "num_iters", "setup_time", "solve_time", "status"]
        solver_stats_ = {k: getattr(stats, k, None) for k in safelist}
        solver_stats_['status'] = problem.status if hasattr(problem, 'status') else None
        self.solver_stats_ = solver_stats_

    # Penalized problems
    def _ridge(self, beta_var: cp.Variable, group_index: Optional[Sequence[int]]) -> cp.Expression:
        lambda_param = cp.Parameter(nonneg=True, value=self.lambda1)
        pen = lambda_param * cp.sum_squares(beta_var)
        return pen

    def _lasso(self, beta_var: cp.Variable, group_index: Optional[Sequence[int]]) -> cp.Expression:
        lambda_param = cp.Parameter(nonneg=True, value=self.lambda1)
        pen = lambda_param * cp.norm1(beta_var)
        return pen

    def _gl(self, beta_var: cp.Variable, group_index: Sequence[int]) -> cp.Expression:
        lambda_param = cp.Parameter(nonneg=True, value=self.lambda1)
        unique_groups, group_sizes = np.unique(group_index, return_counts=True)
        indices_per_group = {g: np.where(group_index == g)[0] for g in unique_groups}
        sqrt_sizes = np.sqrt(group_sizes)
        group_norms = cp.hstack([cp.norm2(beta_var[indices_per_group[g]]) for g in unique_groups])
        pen = lambda_param * cp.sum(cp.multiply(sqrt_sizes, group_norms))
        return pen

    def _sgl(self, beta_var: cp.Variable, group_index: Sequence[int]) -> cp.Expression:
        group_param = cp.Parameter(nonneg=True, value=self.lambda1 * (1 - self.alpha))
        individual_param = cp.Parameter(nonneg=True, value=self.lambda1 * self.alpha)
        unique_groups, group_sizes = np.unique(group_index, return_counts=True)
        indices_per_group = {g: np.where(group_index == g)[0] for g in unique_groups}
        sqrt_sizes = np.sqrt(group_sizes)
        group_norms = cp.hstack([cp.norm2(beta_var[indices_per_group[g]]) for g in unique_groups])
        group_penalization = group_param * cp.sum(cp.multiply(sqrt_sizes, group_norms))
        individual_penalization = individual_param  * cp.norm1(beta_var)
        pen = individual_penalization + group_penalization
        return pen

    def _obtain_beta(self, X: np.ndarray, y: np.ndarray, group_index: Optional[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        m = X.shape[1]
        beta_var = cp.Variable(m)
        intercept_var = cp.Variable() if self.fit_intercept else 0
        pred = X @ beta_var + intercept_var
        objective_function = self._define_objective_function(y, pred)
        # Handle unpenalized models
        if self.penalization is None:
            problem = cp.Problem(cp.Minimize(objective_function))
        else:
            pen = getattr(self, '_' + self.penalization)(beta_var, group_index)
            problem = cp.Problem(cp.Minimize(objective_function + pen))
        self._solve_cp_problem(problem)
        beta_sol = beta_var.value
        intercept_sol = intercept_var.value if self.fit_intercept else 0
        if (beta_sol is None) or (intercept_sol is None):
            raise ValueError("CVXPY optimization failed to find a solution")
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        beta_sol = np.ravel(beta_sol) # Ensure it is 1D
        return intercept_sol, beta_sol

    def fit(self, X: np.ndarray, y: np.ndarray, group_index: Optional[Sequence[int]] = None):
        self.feature_names_in_ = None
        if hasattr(X, "columns") and callable(getattr(X, "columns", None)):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, ensure_min_samples=2)
        self.n_features_in_ = X.shape[1]
        self._check_attributes()
        # Check binary y
        if self._estimator_type == "classifier":
            if type_of_target(y) != "binary":
                unique_y_values = np.unique(y)
                # check_estimator might pass float y like [0.0, 1.0]
                is_binary_int = len(unique_y_values) <= 2 and np.all(np.isin(unique_y_values, [0, 1]))
                is_binary_float = len(unique_y_values) <= 2 and np.all(np.isin(unique_y_values, [0.0, 1.0]))
                if (not is_binary_int) | (not is_binary_float):
                    raise ValueError(f"For logistic model, y must contain only 0 and 1 (or 0.0, 1.0).")
                y = y.astype(int)
            self.classes_ = np.array([0, 1])  # Assuming 0 and 1 are the classes
        if self.penalization in (GROUP_NONADAPTIVE+GROUP_ADAPTIVE) and group_index is None:
            raise ValueError(f'The penalization provided requires fitting the model with a group_index parameter but no group_index was detected.')
        if group_index is not None:
            group_index = np.asarray(group_index, dtype=int)
            if len(group_index) != X.shape[1]:
                raise ValueError(f"group_index length {len(group_index)} does not match number of features {X.shape[1]}")
            if any(group_index < 0):
                raise ValueError(f"group_index must be a positive integer array. Negative values detected")
        # Solve the problem
        self.intercept_, self.coef_ = self._obtain_beta(X, y, group_index)
        self.is_fitted_ = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["coef_", "intercept_", "is_fitted_"])
        intercept = self.intercept_ if self.fit_intercept else 0
        predictions = np.dot(X, self.coef_) + intercept
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._estimator_type != "classifier":
            raise AttributeError(f"predict_proba is not available when model is '{self.model}'. It is only available for classifier models.")
        check_is_fitted(self, "classes_")  # Ensure classes_ is available
        decision = self.decision_function(X)
        proba_pos_class = expit(decision)
        return np.vstack([1 - proba_pos_class, proba_pos_class]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["coef_", "intercept_", "is_fitted_"])
        raw_predictions = self.decision_function(X)
        if self._estimator_type == "classifier":
            # self.classes_ should be [0, 1]
            # Threshold decision function output at 0 for class labels
            indices = (raw_predictions >= 0).astype(int) # 0 if raw_pred <= 0, 1 if raw_pred > 0
            return self.classes_[indices]
        else: # Regressor
            return raw_predictions

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        if self.model == 'logit':
            tags.estimator_type = 'classifier'
        else:
            tags.estimator_type = 'regressor'
        return tags

    def _more_tags(self):
        tags = {
            "allow_nan": False,        # declare the estimator does *not* accept NaNs
            "requires_y": True,        # fitting requires a target y
        }
        if self._estimator_type == "classifier":
            tags["binary_only"] = True
        return tags

    def score(self, X, y, sample_weight=None):
        if self._estimator_type == "regressor":
            return  RegressorMixin.score(self, X, y, sample_weight)
        else: # Classifier
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

class AdaptiveWeights:
    def __init__(self,
                 model: str = 'lm',
                 penalization: str = 'alasso',
                 quantile: float = 0.5,
                 weight_technique: str = 'pca_pct',
                 individual_power_weight: float = 1,
                 group_power_weight: float = 1,
                 variability_pct: float = 0.9,
                 lambda1_weights: float = 0.1,
                 spca_alpha: float = 1e-5,
                 spca_ridge_alpha: float = 1e-2,
                 individual_weights=None,
                 group_weights=None,
                 weight_tol: float = 1e-4):
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
        self.weight_tol = weight_tol

    def _wpca_1(self, X: np.ndarray, y: np.ndarray)  -> np.ndarray:
        """
        Weights based on the first principal component
        """
        pca = PCA(n_components=1, svd_solver="full")
        pca.fit(X)
        tmp_weight = np.abs(pca.components_).ravel()
        return tmp_weight

    def _wpca_pct(self, X: np.ndarray, y: np.ndarray)  -> np.ndarray:
        """
        Weights based on principal component analysis
        """
        var_pct2 = np.min(X.shape) if self.variability_pct == 1 else self.variability_pct
        pca = PCA(n_components=var_pct2, svd_solver="full")
        t = pca.fit_transform(X) # scores
        p = pca.components_.T # loadings
        unpenalized_model = BaseModel(model=self.model, penalization=None, fit_intercept=True, quantile=self.quantile)
        unpenalized_model.fit(X=t, y=y)
        beta_sol = unpenalized_model.coef_
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p, beta_sol)).ravel()
        return tmp_weight

    def _wpls_1(self, X: np.ndarray, y: np.ndarray)  -> np.ndarray:
        """
        Weights based on the first partial least squares component
        """
        pls = PLSRegression(n_components=1, scale=False)
        pls.fit(X, y)
        tmp_weight = np.abs(pls.x_rotations_).ravel()
        return tmp_weight

    def _wpls_pct(self, X: np.ndarray, y: np.ndarray)  -> np.ndarray:
        """
        Weights based on partial least squares
        """
        total_variance_in_x = np.sum(np.var(X, axis=0))
        pls = PLSRegression(n_components=np.min(X.shape), scale=False)
        pls.fit(X, y)
        variance_in_pls = np.var(pls.x_scores_, axis=0)
        fractions_of_explained_variance = np.cumsum(variance_in_pls / total_variance_in_x)
        if self.variability_pct > np.max(fractions_of_explained_variance):
            warnings.warn(f'The total explained variability using PLS reaches {np.max(fractions_of_explained_variance)}.',
                          RuntimeWarning, stacklevel=2)
        n_comp = np.searchsorted(fractions_of_explained_variance, self.variability_pct)
        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X, y)
        tmp_weight = np.abs(np.asarray(pls.coef_).ravel())
        return tmp_weight

    def _wsparse_pca(self, X: np.ndarray, y: np.ndarray)  -> np.ndarray:
        """
        Weights based on sparse principal component analysis.
        """
        x_center = X - X.mean(axis=0)
        total_variance_in_x = np.sum(np.var(X, axis=0))
        spca = SparsePCA(n_components=np.min(X.shape), alpha=self.spca_alpha, ridge_alpha=self.spca_ridge_alpha)
        t = spca.fit_transform(x_center)
        p = spca.components_.T
        # Obtain explained variance using spca as explained in the original paper (based on QR decomposition)
        _, r_spca = np.linalg.qr(t, mode="reduced")
        t_spca_variance = np.square(np.diag(r_spca)) / X.shape[0]
        fractions_of_explained_variance = np.cumsum(t_spca_variance / total_variance_in_x)
        if self.variability_pct > np.max(fractions_of_explained_variance):
            warnings.warn(f'The total explained variability using Sparse PCA reaches {np.max(fractions_of_explained_variance)}.',
                          RuntimeWarning, stacklevel=2)
        n_comp = np.searchsorted(fractions_of_explained_variance, self.variability_pct)
        unpenalized_model = BaseModel(model=self.model, penalization=None, fit_intercept=True, quantile=self.quantile)
        unpenalized_model.fit(X=t[:, 0:n_comp], y=y)
        beta_sol = unpenalized_model.coef_
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p[:, 0:n_comp], beta_sol)).ravel()
        return tmp_weight

    def _wunpenalized(self, X: np.ndarray, y: np.ndarray)  -> np.ndarray:
        """
        Only for low dimensional frameworks. Weights based on an unpenalized regression model
        """
        unpenalized_model = BaseModel(model=self.model, penalization=None, fit_intercept=True, quantile=self.quantile)
        unpenalized_model.fit(X=X, y=y)
        tmp_weight = np.abs(unpenalized_model.coef_)
        return tmp_weight

    def _wlasso(self, X: np.ndarray, y: np.ndarray)  -> np.ndarray:
        lasso_model = BaseModel(model=self.model, penalization='lasso', lambda1=self.lambda1_weights,
                                fit_intercept=True, quantile=self.quantile)
        lasso_model.fit(X=X, y=y)
        tmp_weight = np.abs(lasso_model.coef_)
        return tmp_weight

    def _wridge(self, X: np.ndarray, y: np.ndarray)  -> np.ndarray:
        ridge_model = BaseModel(model=self.model, penalization='ridge', lambda1=self.lambda1_weights,
                                fit_intercept=True, quantile=self.quantile)
        ridge_model.fit(X=X, y=y)
        tmp_weight = np.abs(ridge_model.coef_)
        return tmp_weight

    def _check_type_penalization(self) -> Tuple[bool, bool]:
        return (self.penalization in INDIV_ADAPTIVE,
                self.penalization in GROUP_ADAPTIVE)

    def fit_weights(self, X: np.ndarray, y: np.ndarray, group_index: Optional[Sequence[int]] = None):
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, ensure_min_samples=2)
        bool_individual, bool_group = self._check_type_penalization()
        if bool_group and group_index is None:
            raise ValueError("A group penalisation was requested but `group_index` is missing.")
        tmp_weight: Optional[np.ndarray] = None
        if bool_individual and self.individual_weights is None:
            tmp_weight = getattr(self, '_w' + self.weight_technique)(X=X, y=y)
            self.individual_weights = 1 / (tmp_weight ** self.individual_power_weight + self.weight_tol)
        if bool_group and self.group_weights is None:
            if tmp_weight is None:
                tmp_weight = getattr(self, '_w' + self.weight_technique)(X=X, y=y)
            group_index = np.asarray(group_index, dtype=int)
            unique_index = np.unique(group_index)
            group_weights = []
            for g in unique_index:
                mask = group_index == g
                norm = np.linalg.norm(tmp_weight[mask], ord=2)
                group_weights.append(1.0 / (np.power(norm, self.group_power_weight) + self.weight_tol))
            self.group_weights = np.asarray(group_weights)
        if bool_individual and (len(self.individual_weights) != X.shape[1]):
            raise ValueError('Number of individual weights does not match the number of columns in X')
        if bool_group and (len(self.group_weights) != len(np.unique(group_index))):
            raise ValueError('Number of group weights does not match the number of groups in group_index')
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

    Attributes
    ----------
    coef_: ndarray of shape (n_features,)
        Estimated coefficients for the regression problem.
    intercept_: float
        Independent term in the regression model
    n_features_in_: int
        Number of features seen during fit.
    """
    def __init__(self,
                 model: str='lm',
                 penalization: str | None='lasso',
                 quantile: float=0.5,
                 fit_intercept: bool=True,
                 lambda1: float=0.1,
                 alpha: float=0.5,
                 solver: str='default',
                 weight_technique: str='pca_pct',
                 individual_power_weight: float=1,
                 group_power_weight: float=1,
                 variability_pct: float=0.9,
                 lambda1_weights: float=0.1,
                 spca_alpha: float=1e-5,
                 spca_ridge_alpha: float=1e-2,
                 individual_weights: np.ndarray | None=None,
                 group_weights: np.ndarray | None=None,
                 weight_tol: float=1e-4,
                 tol: float=1e-3):
        super().__init__(
            model=model,
            penalization=penalization,
            quantile=quantile,
            fit_intercept=fit_intercept,
            lambda1=lambda1,
            alpha=alpha,
            solver=solver,
            tol=tol)
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
    def _aridge(self, beta_var: cp.Variable, group_index: Optional[Sequence[int]]) -> cp.Expression:
        lambda_param = cp.Parameter(nonneg=True, value=self.lambda1)
        individual_weights_param = cp.Parameter(len(self.individual_weights), nonneg=True,
                                                value=self.individual_weights)
        pen = lambda_param * cp.sum_squares(cp.multiply(individual_weights_param, beta_var))
        return pen

    def _alasso(self, beta_var: cp.Variable, group_index: Optional[Sequence[int]]) -> cp.Expression:
        lambda_param = cp.Parameter(nonneg=True, value=self.lambda1)
        individual_weights_param = cp.Parameter(len(self.individual_weights), nonneg=True,
                                                value=self.individual_weights)
        pen = lambda_param * cp.norm1(cp.multiply(individual_weights_param, beta_var))
        return pen

    def _agl(self, beta_var: cp.Variable, group_index: Sequence[int]) -> cp.Expression:
        lambda_param = cp.Parameter(nonneg=True, value=self.lambda1)
        unique_groups, group_sizes = np.unique(group_index, return_counts=True)
        indices_per_group = {g: np.where(group_index == g)[0] for g in unique_groups}
        sqrt_sizes = np.sqrt(group_sizes)
        group_weights = cp.Parameter(len(sqrt_sizes), nonneg=True,
                                                value=sqrt_sizes * self.group_weights)
        group_norms = cp.hstack([cp.norm2(beta_var[indices_per_group[g]]) for g in unique_groups])
        pen = lambda_param * cp.sum(cp.multiply(group_weights, group_norms))
        return pen

    def _asgl(self, beta_var: cp.Variable, group_index: Sequence[int]) -> cp.Expression:
        individual_param = cp.Parameter(nonneg=True, value=self.lambda1 * self.alpha)
        individual_weights_param = cp.Parameter(len(self.individual_weights), nonneg=True,
                                                value=self.individual_weights)
        group_param = cp.Parameter(nonneg=True, value=self.lambda1 * (1 - self.alpha))
        unique_groups, group_sizes = np.unique(group_index, return_counts=True)
        indices_per_group = {g: np.where(group_index == g)[0] for g in unique_groups}
        sqrt_sizes = np.sqrt(group_sizes)
        group_weights = cp.Parameter(len(sqrt_sizes), nonneg=True,
                                                value=sqrt_sizes * self.group_weights)
        group_norms = cp.hstack([cp.norm2(beta_var[indices_per_group[g]]) for g in unique_groups])
        individual_penalization = individual_param * cp.norm1(cp.multiply(individual_weights_param, beta_var))
        group_penalization = group_param * cp.sum(cp.multiply(group_weights, group_norms))
        pen = individual_penalization + group_penalization
        return pen

    def fit(self, X: np.ndarray, y: np.ndarray, group_index: Optional[Sequence[int]] = None):
        self._check_attributes()
        if self.penalization in (INDIV_ADAPTIVE+GROUP_ADAPTIVE):
            self.fit_weights(X, y, group_index)
        # Call the fit method of the parent class (BaseModel) to perform the main fitting logic.
        super().fit(X, y, group_index)
        return self
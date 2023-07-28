import sys
import functools
import itertools
import logging
import multiprocessing as mp
import joblib

import cvxpy
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import _num_features, _num_samples
from sklearn.utils import check_X_y
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.linear_model import Lasso as skLasso

from sklearn.decomposition import PCA, SparsePCA


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class BaseModel(BaseEstimator, RegressorMixin):
    """
    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term, controlling regularization
        strength. `l1_ratio` must be a non-negative float i.e. in `[0, inf)`.

        When `l1_ratio = 0`, the objective is equivalent to ordinary least
        squares, solved by the :class:`LinearRegression` object. For numerical
        reasons, using `l1_ratio = 0` with the `Lasso` object is not advised.
        Instead, you should use the :class:`LinearRegression` object.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``, see Notes below.
    solver: str, default="default"
        The solver to be used by CVXPY. default uses optimal alternative depending on the problem
    weight_calculator : str
        {'pca_1', 'pca_pct', 'pls_1', 'pls_pct',
        'unpenalized_','unpenalized_lm', 'unpenalized_qr', 'lasso', 'spca'}.

    Notes:
        https://arxiv.org/abs/2111.00472
        https://pypi.org/project/asgl/
    """

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        tol=1e-4,
        solver="default",
        max_iter=500,
        weight_calculator="pca_1",
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.weight_calculator = weight_calculator
        self.solver = solver

    # CVXPY SOLVER RELATED OPTIONS ###################################################################################

    def _cvxpy_solver_options(self, solver):
        if solver == "ECOS":
            solver_dict = dict(solver=solver, max_iters=self.max_iter)
        elif solver == "OSQP":
            solver_dict = dict(solver=solver, max_iter=self.max_iter)
        else:
            solver_dict = dict(solver=solver)
        return solver_dict

    # SOLVERS #########################################################################################################

    def _quantile_function(self, X):
        """
        Quantile function required for quantile regression models.
        """
        return 0.5 * cvxpy.abs(X) + (self.tau - 0.5) * X

    def _num_beta_var_from_group_index(self, group_index):
        """
        Internal function used in group based penalizations (gl, sgl, asgl, asgl_lasso, asgl_gl)
        """
        group_sizes = []
        beta_var = []
        unique_group_index = np.unique(group_index)
        # Define the objective function
        for idx in unique_group_index:
            group_sizes.append(len(np.where(group_index == idx)[0]))
            beta_var.append(cvxpy.Variable(len(np.where(group_index == idx)[0])))
        return group_sizes, beta_var

    def fit(self, X, y, sample_weight=None, *, group_index=None):
        """
        Main function of the module. Given a model, penalization and parameter values specified in the class definition,
        this function solves the model and produces the regression coefficients
        """
        return NotImplementedError

    # PREDICTION METHOD ###############################################################################################

    def predict(self, X):
        """
        To be executed after fitting a model. Given a new dataset, this function produces predictions for that data
        considering the different model coefficients output provided by function fit
        """
        if self.fit_intercept:
            X = np.c_[np.ones(_num_samples(X)), X]

        return np.dot(X, self.coef_)


class LinearRegression(BaseModel):
    def __init__(
        self,
        alpha=0.0,
        *,
        fit_intercept=True,
        quantile=False,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
        )
        self.quantile = quantile

    def fit(self, X, y, sample_weight=None, *, group_index=None):
        X, y = check_X_y(X, y)
        n_samples = _num_samples(X)
        n_features = _num_features(X)
        # If we want an fit_intercept, it adds a column of ones to the matrix X
        if self.fit_intercept:
            n_features = n_features + 1
            X = np.c_[np.ones(n_samples), X]
        # Define the objective function
        beta_var = cvxpy.Variable(n_features)
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(y - X @ beta_var)
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - X @ beta_var))
            )
        objective = cvxpy.Minimize(objective_function)
        problem = cvxpy.Problem(objective)
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")
        self.coef_ = beta_var.value
        self.coef_[np.abs(self.coef_) < self.tol] = 0
        return self


class QuantileRegression(LinearRegression):
    """Quantile Regression

    Parameters
    ----------
    tau : float, default=1e-4
        The quantile level in quantile regression models
    """

    def __init__(
        self,
        alpha=0.0,
        *,
        fit_intercept=True,
        quantile=True,
        tau=0.5,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            quantile=quantile,
            solver=solver,
            max_iter=max_iter,
        )
        self.tau = tau


class Lasso(BaseModel):
    """Lasso

    Parameters
    ----------
    alpha : float, default=1e-4
        The parameter value that controls the level of shrinkage applied on penalizations
    tau : float, default=1e-4
        The quantile level in quantile regression models
    lasso_weights: lasso weights in adaptive penalizations
    group_lasso_weights: group lasso weights in adaptive penalizations"""

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        quantile=False,
        tol=1e-5,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
        )

        self.quantile = quantile

    def fit(self, X, y, sample_weight=None, *, group_index=None):
        """
        Lasso penalized solver
        """
        # n_samples, n_features = X.shape
        X, y = check_X_y(X, y)

        n_samples = _num_samples(X)
        n_features = _num_features(X)
        # If we want an intercept, it adds a column of ones to the matrix X.
        # Init_pen controls when the penalization starts, this way the intercept is not penalized
        if self.fit_intercept:
            n_features = n_features + 1
            X = np.c_[np.ones(n_samples), X]
            init_pen = 1
        else:
            init_pen = 0
        # Define the objective function
        lambda_param = cvxpy.Parameter(nonneg=True)
        beta_var = cvxpy.Variable(n_features)
        lasso_penalization = lambda_param * cvxpy.norm(beta_var[init_pen:], 1)
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(y - X @ beta_var)
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - X @ beta_var))
            )
        objective = cvxpy.Minimize(objective_function + lasso_penalization)
        problem = cvxpy.Problem(objective)

        # Solve the problem iteratively for each parameter value
        lambda_param.value = self.alpha

        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")

        self.coef_ = beta_var.value
        self.coef_[np.abs(self.coef_) < self.tol] = 0

        return self


class QuantileLasso(Lasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        quantile=True,
        tol=1e-5,
        tau=0.5,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            quantile=quantile,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
        )
        self.tau = tau


class AdaptiveModel:
    def weight_calculate(
        self,
        X,
        y,
        *,
        weight_calculator=None,
        pca=PCA(),
        pls=PLSRegression(),
        linear_model=QuantileLasso(),
        variability_percent=1.0,
    ):
        if weight_calculator == "pca":
            pca.n_components = 1
            weight = self._pca_components(X, y, pca=pca)
        elif weight_calculator == "pcar":
            weight = self._pcar_coef_(X, y, pca=pca, linear_model=linear_model)
        elif weight_calculator == "pls":
            weight = self._pls(X, y)
        elif weight_calculator == "pls_percent":
            weight = self._pls_percent(
                X, y, pls=pls, variability_percent=variability_percent
            )
        elif weight_calculator == "unpenalized":
            setattr(linear_model, "quantile", False)
            weight = self._linear_model(X, y, linear_model=linear_model)

        elif weight_calculator == "unpenalized_lm":
            setattr(linear_model, "quantile", False)
            weight = self._linear_model(X, y, linear_model=linear_model)

        elif weight_calculator == "unpenalized_qr":
            setattr(linear_model, "quantile", True)
            setattr(linear_model, "tau", 0.5)
            weight = self._linear_model(X, y, linear_model=linear_model)

        elif weight_calculator == "lasso":
            setattr(linear_model, "alpha", self.alpha)
            weight = self._linear_model(X, y, linear_model=linear_model)

        elif weight_calculator == "spca":
            weight = self._sparse_pca(X, y, variability_percent=variability_percent)

        elif weight_calculator is None:
            weight = np.array([1.0] * _num_features(X))

        else:
            raise ValueError(
                r"The 'weight_calculator' parameter must be a str among \
                                {'pca_1', 'pca_pct', 'pls_1', 'pls_pct', ''unpenalized_', \
                                'unpenalized_lm', 'unpenalized_qr', 'lasso', 'spca'}. Got  instead."
            )

        return weight

    def _pca_components(self, X, y, *, pca=PCA()):
        """
        Computes the adpative weights based on the first principal component
        """
        pca.fit(X)
        weight = np.abs(pca.components_).flatten()
        return weight

    def _pcar_coef_(
        self, X, y, *, pca=PCA(), linear_model=sklearn.linear_model.LinearRegression()
    ):
        """
        Computes the adpative weights based on principal component analysis
        """
        pipe = Pipeline([("pca", pca), ("linear_model", linear_model)])
        pipe.fit(X, y)
        weight = np.abs(np.dot(pipe[0].components_.T, pipe[1].coef_[1:]).flatten())
        return weight

    def _pls(self, X, y, *, pls=PLSRegression(n_components=1, scale=False)):
        """
        Computes the adpative weights based on the first partial least squares component
        """
        # x_loadings_ is the pls equivalent to the PCs
        pls.fit(X, y)
        weight = np.abs(pls.x_rotations_).flatten()
        return weight

    def _pls_percent(
        self, X, y, *, pls=PLSRegression(scale=False), variability_percent=1.0
    ):
        """
        Computes the adpative weights based on partial least squares
        """
        total_variance_in_x = np.sum(np.var(X, axis=0))
        pls.fit(X, y)
        variance_in_pls = np.var(pls.x_scores_, axis=0)
        fractions_of_explained_variance = np.cumsum(
            variance_in_pls / total_variance_in_x
        )
        # Update variability_pct
        variability_percent = np.min(
            (variability_percent, np.max(fractions_of_explained_variance))
        )
        n_components = (
            np.argmax(fractions_of_explained_variance >= variability_percent) + 1
        )
        pls.n_components = n_components
        pls.fit(X, y)
        weight = np.abs(np.asarray(pls.coef_).flatten())
        return weight

    def _linear_model(self, X, y, *, linear_model=QuantileLasso()):
        linear_model.fit(X=X, y=y)
        weight = np.abs(linear_model.coef_[1:])
        return weight

    def _sparse_pca(self, X, y, *, variability_percent=1.0):
        """
        Computes the adpative weights based on sparse principal component analysis.
        """
        # Compute sparse pca
        x_center = X - X.mean(axis=0)
        total_variance_in_x = np.sum(np.var(X, axis=0))
        spca = SparsePCA(
            n_components=np.min((X.shape[0], X.shape[1])),
            alpha=self.sparsepca_alpha,
            ridge_alpha=self.sparsepca_ridge_alpha,
        )
        t = spca.fit_transform(x_center)
        p = spca.components_.T
        # Obtain explained variance using spca as explained in the original paper (based on QR decomposition)
        t_spca_qr_decomp = np.linalg.qr(t)
        # QR decomposition of modified PCs
        r_spca = t_spca_qr_decomp[1]
        t_spca_variance = np.diag(r_spca) ** 2 / X.shape[0]
        # compute variance_ratio
        fractions_of_explained_variance = np.cumsum(
            t_spca_variance / total_variance_in_x
        )
        # Update variability_pct
        variability_percent = np.min(
            (variability_percent, np.max(fractions_of_explained_variance))
        )
        n_components = (
            np.argmax(fractions_of_explained_variance >= variability_percent) + 1
        )
        unpenalized_model = sklearn.linear_model.LinearRegression()
        unpenalized_model.fit(X=t[:, 0:n_components], y=y)
        beta_qr = unpenalized_model.coef_
        # Recover an estimation of the beta parameters and use it as weight
        weight = np.abs(np.dot(p[:, 0:n_components], beta_qr)).flatten()
        return weight

    def _gl_weights_calculation(self, weight, group_index):
        self.gl_power_weight = self._preprocessing(self.gl_power_weight)
        unique_index = np.unique(group_index)
        gl_weights = []
        for glpw in self.gl_power_weight:
            tmp_list = [
                1
                / (
                    np.linalg.norm(
                        weight[np.where(group_index == unique_index[i])[0]], 2
                    )
                    ** glpw
                    + self.weight_tol
                )
                for i in range(len(unique_index))
            ]
            tmp_list = np.asarray(tmp_list)
            gl_weights.append(tmp_list)
        return gl_weights


class AdaptiveLasso(BaseModel, AdaptiveModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        quantile=False,
        tol=1e-5,
        solver="default",
        max_iter=500,
        weight_calculator="pca",
        pca=PCA(),
        pls=PLSRegression(),
        linear_model=QuantileLasso(),
        variability_percent=1.0,
        gamma=1.0,
        group_gamma=1.0,
        alpha_weights=1e-1,
        sparsepca_alpha=1e-5,
        sparsepca_ridge_alpha=1e-2,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
        )
        self.quantile = quantile
        self.weight_calculator = weight_calculator
        self.pca = pca
        self.pls = pls
        self.linear_model = linear_model
        self.variability_percent = variability_percent
        self.gamma = gamma
        self.group_gamma = group_gamma
        self.alpha_weights = alpha_weights
        self.sparsepca_alpha = sparsepca_alpha
        self.sparsepca_ridge_alpha = sparsepca_ridge_alpha

    def fit(self, X, y, sample_weight=None, *, group_index=None):
        """
        Lasso penalized solver
        """
        X, y = check_X_y(X, y)

        n_samples = _num_samples(X)
        n_features = _num_features(X)
        # If we want an intercept, it adds a column of ones to the matrix X.
        # Init_pen controls when the penalization starts, this way the intercept is not penalized

        if self.fit_intercept:
            n_features = n_features + 1
            X = np.c_[np.ones(n_samples), X]
            init_pen = 1
        else:
            init_pen = 0

        # Define the objective function
        l_weights_param = cvxpy.Parameter(n_features, nonneg=True)
        beta_var = cvxpy.Variable(n_features)
        lasso_penalization = cvxpy.norm(
            l_weights_param[init_pen:].T @ cvxpy.abs(beta_var[init_pen:]), 1
        )

        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(y - X @ beta_var)

        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - X @ beta_var))
            )

        objective = cvxpy.Minimize(objective_function + lasso_penalization)
        problem = cvxpy.Problem(objective)

        # Solve the problem iteratively for each parameter value
        if self.weight_calculator is not None:
            weight = self.weight_calculate(
                X,
                y,
                weight_calculator=self.weight_calculator,
                pca=self.pca,
                pls=self.pls,
                linear_model=self.linear_model,
                variability_percent=self.variability_percent,
            )

            lasso_weights = 1 / (weight**self.gamma + 1e-5)

        else:
            if isinstance(self.alpha, float):
                lasso_weights = np.array([self.alpha] * _num_features(X))
            elif isinstance(self.alpha, list) or isinstance(self.alpha, np.ndarray):
                lasso_weights = self.alpha

        l_weights_param.value = lasso_weights * self.alpha
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options

        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)

        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue

        # self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")

        self.coef_ = beta_var.value
        self.coef_[np.abs(self.coef_) < self.tol] = 0

        return self


class QuantileAdaptiveLasso(AdaptiveLasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        tol=1e-5,
        tau=0.5,
        solver="default",
        max_iter=500,
        weight_calculator="pca",
        pca=PCA(),
        pls=PLSRegression(),
        linear_model=QuantileLasso(),
        variability_percent=1.0,
        gamma=1.0,
        alpha_weights=1e-1,
        sparsepca_alpha=1e-5,
        sparsepca_ridge_alpha=1e-2,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
            weight_calculator=weight_calculator,
            pca=pca,
            pls=pls,
            linear_model=linear_model,
            variability_percent=variability_percent,
            gamma=gamma,
            alpha_weights=alpha_weights,
            sparsepca_alpha=sparsepca_alpha,
            sparsepca_ridge_alpha=sparsepca_ridge_alpha,
        )

        self.tau = tau


class GroupLasso(BaseModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        quantile=False,
        tol=1e-5,
        lasso_weights=None,
        group_lasso_weights=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
        )
        self.quantile = quantile
        self.lasso_weights = lasso_weights
        self.group_lasso_weights = group_lasso_weights
        self.quantile = quantile

    def fit(self, X, y, sample_weight=None, *, group_index=None):
        """
        Group lasso penalized solver
        """
        X, y = check_X_y(X, y)

        n_samples = _num_samples(X)
        n_features = _num_features(X)

        if group_index is None:
            group_index = np.ones(n_features)

        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        if (group_index == 0).any():
            group_index += 1

        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.fit_intercept:
            # Adds an element (referring to the intercept) to group_index, group_sizes, num groups
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            X = np.c_[np.ones(n_samples), X]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # Compute model prediction for the intercept with no penalization
            model_prediction = (
                X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            )
            inf_lim = 1

        for i in range(inf_lim, num_groups):
            model_prediction += (
                X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            )
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(
                beta_var[i], 2
            )

        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(
                y - model_prediction
            )

        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - model_prediction))
            )
        lambda_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(
            objective_function + (lambda_param * group_lasso_penalization)
        )
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        lambda_param.value = self.alpha
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue

        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")
        self.coef_ = np.concatenate([b.value for b in beta_var], axis=0)
        self.coef_[np.abs(self.coef_) < self.tol] = 0

        return self


class QuantileGroupLasso(GroupLasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        quantile=True,
        tol=1e-5,
        tau=0.5,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            quantile=quantile,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
        )
        self.tau = tau


class SparseGroupLasso(BaseModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        quantile=False,
        tol=1e-5,
        l1_ratio=1.0,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
        )

        self.quantile = quantile
        self.l1_ratio = l1_ratio

    def fit(self, X, y, sample_weight=None, *, group_index=None):
        """
        Sparse group lasso penalized solver
        """
        X, y = check_X_y(X, y)

        n_samples = _num_samples(X)
        n_features = _num_features(X)

        if group_index is None:
            group_index = np.ones(n_features)

        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        if (group_index == 0).any():
            group_index += 1

        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        lasso_penalization = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0

        if self.fit_intercept:
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            X = np.c_[np.ones(n_samples), X]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            model_prediction = (
                X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            )
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += (
                X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            )
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(
                beta_var[i], 2
            )
            lasso_penalization += cvxpy.norm(beta_var[i], 1)
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(
                y - model_prediction
            )
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - model_prediction))
            )
        lasso_param = cvxpy.Parameter(nonneg=True)
        group_lasso_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(
            objective_function
            + (group_lasso_param * group_lasso_penalization)
            + (lasso_param * lasso_penalization)
        )
        problem = cvxpy.Problem(objective)
        # Solve the problem iteratively for each parameter value

        lasso_param.value = self.alpha * self.l1_ratio
        group_lasso_param.value = self.alpha * (1 - self.l1_ratio)

        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")
        self.coef_ = np.concatenate([b.value for b in beta_var], axis=0)
        self.coef_[np.abs(self.coef_) < self.tol] = 0

        return self


class QuantileSparseGroupLasso(SparseGroupLasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        quantile=True,
        tol=1e-5,
        l1_ratio=0.5,
        tau=0.5,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            quantile=quantile,
            tol=tol,
            l1_ratio=l1_ratio,
            solver=solver,
            max_iter=max_iter,
        )
        self.tau = tau


class AdaptiveSparseGroupLasso(BaseModel, AdaptiveModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        quantile=False,
        tol=1e-5,
        l1_ratio=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        solver="default",
        max_iter=500,
        weight_calculator="pca",
        pca=PCA(),
        pls=PLSRegression(),
        linear_model=QuantileLasso(),
        variability_percent=1.0,
        gamma=1.0,
        group_gamma=1.0,
        alpha_weights=1e-1,
        sparsepca_alpha=1e-5,
        sparsepca_ridge_alpha=1e-2,
        penalty="asgl",
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
        )

        self.quantile = quantile

        self.l1_ratio = l1_ratio
        self.lasso_weights = lasso_weights
        self.group_lasso_weights = group_lasso_weights

        self.weight_calculator = weight_calculator
        self.pca = pca
        self.pls = pls
        self.linear_model = linear_model
        self.variability_percent = variability_percent
        self.gamma = gamma
        self.group_gamma = group_gamma
        self.alpha_weights = alpha_weights
        self.sparsepca_alpha = sparsepca_alpha
        self.sparsepca_ridge_alpha = sparsepca_ridge_alpha

        self.penalty = penalty

    def fit(self, X, y, sample_weight=None, *, group_index=None):
        """
        adaptive sparse group lasso penalized solver
        """
        X, y = check_X_y(X, y)

        n_samples = _num_samples(X)
        n_features = _num_features(X)

        if group_index is None:
            group_index = np.ones(n_features)

        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        if (group_index == 0).any():
            group_index += 1

        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)

        model_prediction = 0
        alasso_penalization = 0
        a_group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0

        if self.fit_intercept:
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            X = np.c_[np.ones(n_samples), X]
            n_features = n_features + 1
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            model_prediction = (
                X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            )
            inf_lim = 1

        l_weights_param = cvxpy.Parameter(n_features, nonneg=True)

        group_lasso_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += (
                X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            )
            a_group_lasso_penalization += (
                cvxpy.sqrt(group_sizes[i])
                * group_lasso_weights_param[i]
                * cvxpy.norm(beta_var[i], 2)
            )
            alasso_penalization += l_weights_param[
                np.where(group_index == unique_group_index[i])[0]
            ].T @ cvxpy.abs(beta_var[i])

        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(
                y - model_prediction
            )
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - model_prediction))
            )
        objective = cvxpy.Minimize(
            objective_function + a_group_lasso_penalization + alasso_penalization
        )
        problem = cvxpy.Problem(objective)
        # Solve the problem iteratively for each parameter value

        if self.weight_calculator is not None:
            weight = self.weight_calculate(
                X,
                y,
                weight_calculator=self.weight_calculator,
                pca=self.pca,
                pls=self.pls,
                linear_model=self.linear_model,
                variability_percent=self.variability_percent,
            )
            if self.penalty == "asgl":
                lasso_weights = lasso_weights = 1 / (weight**self.gamma + 1e-5)
                group_lasso_weights = np.array(
                    [
                        1
                        / (
                            np.linalg.norm(
                                weight[
                                    np.where(group_index == np.unique(group_index)[i])[
                                        0
                                    ]
                                ],
                                2,
                            )
                            ** self.group_gamma
                            + 1e-5
                        )
                        for i in range(len(np.unique(group_index)))
                    ]
                )

            elif self.penalty == "asgl_lasso":
                lasso_weights = lasso_weights = 1 / (weight**self.gamma + 1e-5)
                group_lasso_weights = np.ones(len(np.unique(group_index)))

            elif self.penalty == "asgl_gl":
                lasso_weights = np.ones(X.shape[1])
                group_lasso_weights = np.array(
                    [
                        1
                        / (
                            np.linalg.norm(
                                weight[
                                    np.where(group_index == np.unique(group_index)[i])[
                                        0
                                    ]
                                ],
                                2,
                            )
                            ** self.group_gamma
                            + 1e-5
                        )
                        for i in range(len(np.unique(group_index)))
                    ]
                )
            else:
                lasso_weights = lasso_weights = 1 / (weight**self.gamma + 1e-5)
                group_lasso_weights = np.array(
                    [
                        1
                        / (
                            np.linalg.norm(
                                weight[
                                    np.where(group_index == np.unique(group_index)[i])[
                                        0
                                    ]
                                ],
                                2,
                            )
                            ** self.group_gamma
                            + 1e-5
                        )
                        for i in range(len(np.unique(group_index)))
                    ]
                )

        else:
            if self.lasso_weights is None:
                lasso_weights = np.array([self.alpha] * (n_features)).reshape(
                    -1,
                )

            if self.group_lasso_weights is None:
                group_lasso_weights = np.array([self.alpha] * (num_groups)).reshape(
                    -1,
                )

        l_weights_param.value = lasso_weights * self.alpha * self.l1_ratio
        group_lasso_weights_param.value = (
            group_lasso_weights * self.alpha * (1 - self.l1_ratio)
        )
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats

        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")

        self.coef_ = np.concatenate([b.value for b in beta_var], axis=0)
        self.coef_[np.abs(self.coef_) < self.tol] = 0

        return self


class QuantileAdaptiveSparseGroupLasso(AdaptiveSparseGroupLasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        quantile=True,
        fit_intercept=True,
        tol=1e-5,
        l1_ratio=0.5,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        solver="default",
        max_iter=500,
        weight_calculator="pca",
        pca=PCA(),
        pls=PLSRegression(),
        linear_model=QuantileLasso(),
        variability_percent=1.0,
        gamma=1.0,
        group_gamma=1.0,
        alpha_weights=1e-1,
        sparsepca_alpha=1e-5,
        sparsepca_ridge_alpha=1e-2,
        penalty="asgl",
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            quantile=quantile,
            tol=tol,
            solver=solver,
            max_iter=max_iter,
            l1_ratio=l1_ratio,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            pca=pca,
            pls=pls,
            linear_model=linear_model,
            variability_percent=variability_percent,
            gamma=gamma,
            alpha_weights=alpha_weights,
            sparsepca_alpha=sparsepca_alpha,
            sparsepca_ridge_alpha=sparsepca_ridge_alpha,
        )
        self.tau = tau

        self.lasso_weights = lasso_weights
        self.group_lasso_weights = group_lasso_weights

        self.weight_calculator = weight_calculator
        self.pca = pca
        self.pls = pls
        self.linear_model = linear_model
        self.variability_percent = variability_percent
        self.gamma = gamma
        self.group_gamma = group_gamma
        self.penalty = penalty


def _quantile_function(y_true, y_pred, tau):
    """
    Quantile function required for error computation
    """
    return (1.0 / len(y_true)) * np.sum(
        0.5 * np.abs(y_true - y_pred) + (tau - 0.5) * (y_true - y_pred)
    )


def error_calculator(y_true, prediction_list, error_type="MSE", tau=None):
    """
    Computes the error between the predicted value and the true value of the response variable
    """
    error_dict = dict(
        MSE=mean_squared_error,
        MAE=mean_absolute_error,
        MDAE=median_absolute_error,
        QRE=_quantile_function,
    )
    valid_error_types = error_dict.keys()
    # Check that the error_type is a valid error type considered
    if error_type not in valid_error_types:
        raise ValueError(
            f"invalid error type. Valid error types are {error_dict.keys()}"
        )
    if y_true.shape[0] != len(prediction_list[0]):
        logging.error("Dimension of test data does not match dimension of prediction")
        raise ValueError(
            "Dimension of test data does not match dimension of prediction"
        )
    # For each prediction, store the error associated to that prediction in a list
    error_list = []
    if error_type == "QRE":
        for y_pred in prediction_list:
            error_list.append(
                error_dict[error_type](y_true=y_true, y_pred=y_pred, tau=tau)
            )
    else:
        for y_pred in prediction_list:
            error_list.append(error_dict[error_type](y_true=y_true, y_pred=y_pred))
    return error_list

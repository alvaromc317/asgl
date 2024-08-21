import warnings

import cvxpy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.utils import check_X_y


class BaseModel(BaseEstimator, RegressorMixin):
    def __init__(self, model='lm', penalization='lasso', quantile=0.5, fit_intercept=True, lambda1=0.1, alpha=0.5,
                 solver='default', tol=1e-5):
        self.model = model
        self.penalization = penalization
        self.quantile = quantile
        self.fit_intercept = fit_intercept
        self.lambda1 = lambda1
        self.alpha = alpha
        self.solver = solver
        self.tol = tol
        self.coef_ = None
        self.solver_stats = None
        self.non_adaptive_penalizations = ['lasso', 'ridge', 'gl', 'sgl']
        self.adaptive_penalizations = ['alasso', 'aridge', 'agl', 'asgl']
        self.grouped_penalizations = ['gl', 'agl', 'sgl', 'asgl']

    def _quantile_function(self, X):
        return 0.5 * cvxpy.abs(X) + (self.quantile - 0.5) * X

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def _prepare_data(self, X, group_index=None):
        n, m = X.shape
        if group_index is not None:
            group_index = np.asarray(group_index).astype(int)
        if self.fit_intercept:
            m += 1
            X = np.c_[np.ones(n), X]
            if group_index is not None:
                group_index = np.append(0, group_index)
        return X, m, group_index

    def _define_objective_function(self, y, model_prediction):
        if self.model == 'lm':
            return (1.0 / y.shape[0]) * cvxpy.sum_squares(y - model_prediction)
        elif self.model == 'qr':
            return (1.0 / y.shape[0]) * cvxpy.sum(self._quantile_function(X=(y - model_prediction)))
        elif self.model in ['logit', 'logit_raw', 'logit_proba']:
            return (-1.0 / y.shape[0]) * cvxpy.sum(cvxpy.multiply(y - 1, model_prediction) - cvxpy.logistic(-model_prediction))
        else:
            raise ValueError('Invalid value for model parameter.')

    def _solve_problem(self, problem):
        try:
            if self.solver == 'default':
                problem.solve()
            else:
                problem.solve(solver=self.solver)
        except (ValueError, cvxpy.error.SolverError):
            warnings.warn(f'Solver {self.solver} failed. Using alternative options from {cvxpy.installed_solvers()}', Warning, stacklevel=2)
            for alt_solver in cvxpy.installed_solvers():
                try:
                    problem.solve(solver=alt_solver)
                    if 'optimal' in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            warnings.warn('Optimization problem status failure', Warning, stacklevel=2)

    def _unpenalized(self, X, y):
        X, m, _ = self._prepare_data(X)
        beta_var = cvxpy.Variable(m)
        model_prediction = X @ beta_var
        objective_function = self._define_objective_function(y, model_prediction)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function))
        self._solve_problem(problem)
        beta_sol = beta_var.value
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return beta_sol

    def _ridge(self, X, y, group_index):
        X, m, _ = self._prepare_data(X)
        beta_var = cvxpy.Variable(m)
        model_prediction = X @ beta_var
        objective_function = self._define_objective_function(y, model_prediction)
        lambda_param = cvxpy.Parameter(nonneg=True, value=self.lambda1)
        if self.fit_intercept:
            individual_penalization = lambda_param * cvxpy.sum_squares(beta_var[1:])
        else:
            individual_penalization = lambda_param * cvxpy.sum_squares(beta_var)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + individual_penalization))
        self._solve_problem(problem)
        beta_sol = beta_var.value
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return beta_sol

    def _lasso(self, X, y, group_index):
        X, m, _ = self._prepare_data(X)
        beta_var = cvxpy.Variable(m)
        model_prediction = X @ beta_var
        objective_function = self._define_objective_function(y, model_prediction)
        lambda_param = cvxpy.Parameter(nonneg=True, value=self.lambda1)
        if self.fit_intercept:
            individual_penalization = lambda_param * cvxpy.norm(beta_var[1:], 1)
        else:
            individual_penalization = lambda_param * cvxpy.norm(beta_var, 1)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + individual_penalization))
        self._solve_problem(problem)
        beta_sol = beta_var.value
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return beta_sol

    def _gl(self, X, y, group_index):
        X, _, group_index = self._prepare_data(X, group_index)
        unique_group_index = np.unique(group_index)
        group_sizes = [np.sum(group_index == grp) for grp in unique_group_index]
        beta_var = [cvxpy.Variable(size) for size in group_sizes]
        num_groups = len(group_sizes)
        model_prediction = 0
        group_penalization = 0
        inf_lim = 0
        if self.fit_intercept:
            model_prediction = X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(beta_var[i], 2)
        objective_function = self._define_objective_function(y, model_prediction)
        lambda_param = cvxpy.Parameter(nonneg=True, value=self.lambda1)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + lambda_param * group_penalization))
        self._solve_problem(problem)
        beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return beta_sol

    def _sgl(self, X, y, group_index):
        X, _, group_index = self._prepare_data(X, group_index)
        unique_group_index = np.unique(group_index)
        group_sizes = [np.sum(group_index == grp) for grp in unique_group_index]
        beta_var = [cvxpy.Variable(size) for size in group_sizes]
        num_groups = len(group_sizes)
        model_prediction = 0
        individual_penalization = 0
        group_penalization = 0
        inf_lim = 0
        if self.fit_intercept:
            model_prediction = X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(beta_var[i], 2)
            individual_penalization += cvxpy.norm(beta_var[i], 1)
        objective_function = self._define_objective_function(y, model_prediction)
        lasso_param = cvxpy.Parameter(nonneg=True, value=self.lambda1 * self.alpha)
        grp_lasso_param = cvxpy.Parameter(nonneg=True, value=self.lambda1 * (1 - self.alpha))
        problem = cvxpy.Problem(cvxpy.Minimize(
            objective_function + grp_lasso_param * group_penalization + lasso_param * individual_penalization))
        self._solve_problem(problem)
        beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return beta_sol

    def fit(self, X, y, group_index=None, sample_weight=None):
        X, y = check_X_y(X, y)
        if self.penalization in self.grouped_penalizations and group_index is None:
            raise ValueError(f'The penalization provided requires fitting the model with a group_index parameter but no group_index was detected.')
        if self.penalization is None:
            self.coef_ = self._unpenalized(X=X, y=y)
        elif self.penalization in self.non_adaptive_penalizations:
            self.coef_ = getattr(self, '_' + self.penalization)(X=X, y=y, group_index=group_index)
        else:
            raise ValueError('Invalid value for penalization parameter.')

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        predictions = np.dot(X, self.coef_)
        if self.model == 'logit_proba':  # Compute probabilities in classification case
            return self._sigmoid(predictions)
        elif self.model == 'logit':
            return (self._sigmoid(predictions) > 0.5).astype(int)  # Assign probabilities larger than 0.5 to class 1
        else:
            return predictions


class AdaptiveWeights:
    def __init__(self, model='lm', penalization='alasso', quantile=0.5, weight_technique='pca_pct',
                 individual_power_weight=1, group_power_weight=1, variability_pct=0.9, lambda1_weights=0.1,
                 spca_alpha=1e-5, spca_ridge_alpha=1e-2, individual_weights=None, group_weights=None, weight_tol=1e-4):
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

    def _wpca_1(self, X, y):
        """
        Weights based on the first principal component
        """
        pca = PCA(n_components=1)
        pca.fit(X)
        tmp_weight = np.abs(pca.components_).flatten()
        return tmp_weight

    def _wpca_pct(self, X, y):
        """
        Weights based on principal component analysis
        """
        # If var_pct is equal to one, the algorithm selects just 1 component, not 100% of the variability.
        if self.variability_pct == 1:
            var_pct2 = np.min((X.shape[0], X.shape[1]))
        else:
            var_pct2 = self.variability_pct
        pca = PCA(n_components=var_pct2)
        # t is the matrix of "scores" (the projection of X into the PC subspace)
        # p is the matrix of "loadings" (the PCs, the eigen vectors)
        t = pca.fit_transform(X)
        p = pca.components_.T
        unpenalized_model = BaseModel(model=self.model, penalization=None, fit_intercept=True, quantile=self.quantile)
        unpenalized_model.fit(X=t, y=y)
        beta_sol = unpenalized_model.coef_[1:]  # Remove intercept
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p, beta_sol)).flatten()
        return tmp_weight

    def _wpls_1(self, X, y):
        """
        Weights based on the first partial least squares component
        """
        # x_loadings_ is the pls equivalent to the PCs
        pls = PLSRegression(n_components=1, scale=False)
        pls.fit(X, y)
        tmp_weight = np.abs(pls.x_rotations_).flatten()
        return tmp_weight

    def _wpls_pct(self, X, y):
        """
        Weights based on partial least squares
        """
        total_variance_in_x = np.sum(np.var(X, axis=0))
        pls = PLSRegression(n_components=np.min((X.shape[0], X.shape[1])), scale=False)
        pls.fit(X, y)
        variance_in_pls = np.var(pls.x_scores_, axis=0)
        fractions_of_explained_variance = np.cumsum(variance_in_pls / total_variance_in_x)
        # Update variability_pct
        self.variability_pct = np.min((self.variability_pct, np.max(fractions_of_explained_variance)))
        n_comp = np.argmax(fractions_of_explained_variance >= self.variability_pct) + 1
        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X, y)
        tmp_weight = np.abs(np.asarray(pls.coef_).flatten())
        return tmp_weight

    def _wunpenalized(self, X, y):
        """
        Only for low dimensional frameworks. Weights based on an unpenalized regression model
        """
        unpenalized_model = BaseModel(model=self.model, penalization=None, fit_intercept=True, quantile=self.quantile)
        unpenalized_model.fit(X=X, y=y)
        tmp_weight = np.abs(unpenalized_model.coef_[1:])  # Remove intercept
        return tmp_weight

    def _wsparse_pca(self, X, y):
        """
        Weights based on sparse principal component analysis.
        """
        x_center = X - X.mean(axis=0)
        total_variance_in_x = np.sum(np.var(X, axis=0))
        spca = SparsePCA(n_components=np.min((X.shape[0], X.shape[1])), alpha=self.spca_alpha,
                         ridge_alpha=self.spca_ridge_alpha)
        t = spca.fit_transform(x_center)
        p = spca.components_.T
        # Obtain explained variance using spca as explained in the original paper (based on QR decomposition)
        t_spca_qr_decomp = np.linalg.qr(t)
        r_spca = t_spca_qr_decomp[1]
        t_spca_variance = np.diag(r_spca) ** 2 / X.shape[0]
        fractions_of_explained_variance = np.cumsum(t_spca_variance / total_variance_in_x)
        # Update variability_pct
        self.variability_pct = np.min((self.variability_pct, np.max(fractions_of_explained_variance)))
        n_comp = np.argmax(fractions_of_explained_variance >= self.variability_pct) + 1
        unpenalized_model = BaseModel(model=self.model, penalization=None, fit_intercept=True, quantile=self.quantile)
        unpenalized_model.fit(X=t[:, 0:n_comp], y=y)
        beta_sol = unpenalized_model.coef_[1:]
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p[:, 0:n_comp], beta_sol)).flatten()
        return tmp_weight

    def _wlasso(self, X, y):
        lasso_model = BaseModel(model=self.model, penalization='lasso', lambda1=self.lambda1_weights, fit_intercept=True, quantile=self.quantile)
        lasso_model.fit(X=X, y=y)
        tmp_weight = np.abs(lasso_model.coef_[1:])
        return tmp_weight

    def _wridge(self, X, y):
        ridge_model = BaseModel(model=self.model, penalization='ridge', lambda1=self.lambda1_weights, fit_intercept=True, quantile=self.quantile)
        ridge_model.fit(X=X, y=y)
        tmp_weight = np.abs(ridge_model.coef_[1:])
        return tmp_weight

    def _check_type_penalization(self):
        bool_individual = False
        bool_group = False
        if self.penalization in ['alasso', 'asgl', 'aridge']:
            bool_individual = True
        if self.penalization in ['agl', 'asgl']:
            bool_group = True
        return bool_individual, bool_group

    def fit_weights(self, X, y, group_index=None):
        X, y = check_X_y(X, y)
        bool_individual, bool_group = self._check_type_penalization()
        if bool_group and (group_index is None):
            raise ValueError('A penalization including group weights was provided but no group_index was provided.\n'
                             'Either provide a group_index or switch to an individual-type penalization like lasso')
        tmp_weight = None
        if bool_individual and self.individual_weights is None:
            tmp_weight = getattr(self, '_w' + self.weight_technique)(X=X, y=y)
            self.individual_weights = 1 / (tmp_weight ** self.individual_power_weight + self.weight_tol)
        if bool_group and self.group_weights is None:
            if tmp_weight is None:
                tmp_weight = getattr(self, '_w' + self.weight_technique)(X=X, y=y)
            unique_index = np.unique(group_index)
            group_weights = [1 / (np.linalg.norm(tmp_weight[np.where(group_index == unique_index[i])[0]],
                                                 2) ** self.group_power_weight + self.weight_tol) for i in
                             range(len(unique_index))]
            self.group_weights = np.asarray(group_weights)
        if bool_individual and (len(self.individual_weights) != X.shape[1]):
            raise ValueError('Number of individual weights does not match the number of columns in X')
        if bool_group and (len(self.group_weights) != len(np.unique(group_index))):
            raise ValueError('Number of group weights does not match the number of groups in group_index')


class Regressor(BaseModel, AdaptiveWeights):
    """
    Parameters
    ----------
    model: str, default = 'lm'
        Model to be fit. Currently, accepts:
            - 'lm': linear regression models.
            - 'qr': quantile regression models.
            - 'logit': logistic regression for binary classification, output binary classification.
            - 'logit_proba': logistic regression for binary classification, output probability.
            - 'logit_raw': logistic regression for binary classification, output score before logistic transformation.
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
    quantile: float, defaul=0.5
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
    solver: str, defaul='defaul'
        Solver to be used by CVXPY. Default uses optimal alternative depending on the problem.
        Users can check available solvers via the command `cvxpy.installed_solvers()`.
    weight_technique: str, default='pca_pct'
        Weight technique used to fit the adaptive weights. Currently, accepts:
            - pca_1: Builds the weights using the first component from PCA.
            - pca_pct: Builds the weights using as many components from PCAas required to achieve the
            ``variability_pct``.
            - pls_1: Builds the weights using the first component from PLS.
            - pls_pct:  Builds the weights using as many components from PLS as indicated to achieve the
            ``variability_pct``.
            - lasso: Builds the weights using the lasso model.
            - ridge: Builds the weights using the ridge model.
            - unpenalized: Builds the weights using the unpenalized model.
            - sparse_pca: Similar to 'pca_pct' but it builds the weights using sparse PCA components.
    individual_power_weight: float, default=1
        Power at which the individual weights are risen.
    group_power_weight: float, default=1
        Power at which the group weights are risen.
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
    tol: float, default=1e-4
        The tolerance for a coefficient in the model to be considered as 0. Values smaller than ``tol`` are assumed to
        be 0.
    weight_tol: float, default=1e-4
        Tolerance value used to avoid ZeroDivision errors when computing the weights.
    """

    def __init__(self, model='lm', penalization='lasso', quantile=0.5, fit_intercept=True, lambda1=0.1, alpha=0.5,
                 solver='default', weight_technique='pca_pct', individual_power_weight=1, group_power_weight=1,
                 variability_pct=0.9, lambda1_weights=0.1, spca_alpha=1e-5, spca_ridge_alpha=1e-2,
                 individual_weights=None, group_weights=None, weight_tol=1e-4, tol=1e-4):
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

    def _aridge(self, X, y, group_index):
        X, m, _ = self._prepare_data(X)
        beta_var = cvxpy.Variable(m)
        model_prediction = X @ beta_var
        objective_function = self._define_objective_function(y, model_prediction)
        individual_weights_param = cvxpy.Parameter(m, nonneg=True)
        if self.fit_intercept:
            init_pen = 1
            individual_weights_param.value = np.sqrt(self.lambda1 * np.insert(self.individual_weights, 0, 0))
        else:
            init_pen = 0
            individual_weights_param.value = np.sqrt(self.lambda1 * self.individual_weights)
        individual_penalization = cvxpy.sum_squares(cvxpy.multiply(individual_weights_param[init_pen:], beta_var[init_pen:]))
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + individual_penalization))
        self._solve_problem(problem)
        beta_sol = beta_var.value
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return beta_sol


    def _alasso(self, X, y, group_index):
        X, m, _ = self._prepare_data(X)
        beta_var = cvxpy.Variable(m)
        model_prediction = X @ beta_var
        objective_function = self._define_objective_function(y, model_prediction)
        individual_weights_param = cvxpy.Parameter(m, nonneg=True)
        if self.fit_intercept:
            init_pen = 1
            individual_weights_param.value = self.lambda1 * np.insert(self.individual_weights, 0, 0)
        else:
            init_pen = 0
            individual_weights_param.value = self.lambda1 * self.individual_weights
        individual_penalization = individual_weights_param[init_pen:].T @ cvxpy.abs(beta_var[init_pen:])
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + individual_penalization))
        self._solve_problem(problem)
        beta_sol = beta_var.value
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return beta_sol

    def _agl(self, X, y, group_index):
        X, _, group_index = self._prepare_data(X, group_index)
        unique_group_index = np.unique(group_index)
        group_sizes = [np.sum(group_index == grp) for grp in unique_group_index]
        beta_var = [cvxpy.Variable(size) for size in group_sizes]
        num_groups = len(group_sizes)
        group_penalization = 0
        group_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        if self.fit_intercept:
            inf_lim = 1
            model_prediction = X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            group_weights_param.value = self.lambda1 * np.insert(self.group_weights, 0, 0)
        else:
            inf_lim = 0
            model_prediction = 0
            group_weights_param.value = self.lambda1 * self.group_weights
        for i in range(inf_lim, num_groups):
            model_prediction += X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_penalization += cvxpy.sqrt(group_sizes[i]) * group_weights_param[i] * cvxpy.norm(beta_var[i], 2)
        objective_function = self._define_objective_function(y, model_prediction)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + group_penalization))
        self._solve_problem(problem)
        beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return beta_sol

    def _asgl(self, X, y, group_index):
        X, m, group_index = self._prepare_data(X, group_index)
        unique_group_index = np.unique(group_index)
        group_sizes = [np.sum(group_index == grp) for grp in unique_group_index]
        beta_var = [cvxpy.Variable(size) for size in group_sizes]
        num_groups = len(group_sizes)
        individual_penalization = 0
        group_penalization = 0
        individual_weights_param = cvxpy.Parameter(m, nonneg=True)
        group_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        if self.fit_intercept:
            inf_lim = 1
            model_prediction = X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            group_weights_param.value = self.lambda1 * (1 - self.alpha) * np.insert(self.group_weights, 0, 0)
            individual_weights_param.value = self.lambda1 * self.alpha * np.insert(self.individual_weights, 0, 0)
        else:
            inf_lim = 0
            model_prediction = 0
            group_weights_param.value = self.lambda1 * (1 - self.alpha) * self.group_weights
            individual_weights_param.value = self.lambda1 * self.alpha * self.individual_weights
        for i in range(inf_lim, num_groups):
            model_prediction += X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_penalization += cvxpy.sqrt(group_sizes[i]) * group_weights_param[i] * cvxpy.norm(beta_var[i], 2)
            individual_penalization += individual_weights_param[np.where(group_index == unique_group_index[i])[0]].T @ cvxpy.abs(beta_var[i])
        objective_function = self._define_objective_function(y, model_prediction)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + group_penalization + individual_penalization))
        self._solve_problem(problem)
        beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return beta_sol

    def fit(self, X, y, group_index=None, sample_weight=None):
        X, y = check_X_y(X, y)
        if self.penalization in self.grouped_penalizations and group_index is None:
            raise ValueError(f'The penalization provided requires fitting the model with a group_index parameter but '
                             f'no group_index was detected.')
        if self.penalization is None:
            self.coef_ = self._unpenalized(X=X, y=y)
        elif self.penalization in self.non_adaptive_penalizations:
            self.coef_ = getattr(self, '_' + self.penalization)(X=X, y=y, group_index=group_index)
        elif self.penalization in self.adaptive_penalizations:
            self.fit_weights(X=X, y=y, group_index=group_index)
            self.coef_ = getattr(self, '_' + self.penalization)(X=X, y=y, group_index=group_index)
        else:
            raise ValueError('Invalid value for penalization parameter.')

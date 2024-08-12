# NEW FILE

import functools
import itertools
import logging
import multiprocessing as mp
import sys

import cvxpy
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ASGL:
    """
    Parameters
    ----------
    model: str, default = 'lm'
        Model to be fit. Currently, accepts:
            - 'lm': linear regression models.
            - 'qr': quantile regression models.
    penalization: str or None, default = 'lasso'
        Penalization to use. Currently, accepts:
            - None: unpenalized model.
            - 'lasso': lasso model.
            - 'gl': group lasso model.
            - 'sgl': sparse group lasso model.
            - 'alasso': adaptive lasso model.
            - 'agl': adaptive group lasso model.
            - 'asgl': adaptive sparse group lasso model.
    intercept: bool, default=True,
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
    tol: float, default=1e-5
        The tolerance for a coefficient in the model to be considered as 0. Values smaller than ``tol`` are assumed to
        be 0.
    lambda1: float or array, defaul=0.1
        Controls the level of shrinkage applied on penalizations. Smaller values specify weaker regularization.
        If it is float, it solves the model for the specific value. If it is an array, it solves the model for all
        specified values in the array.
    alpha: float or array, default=0.5
        Tradeoff between lasso and group lasso in sgl and asgl penalizations. ``alpha=1`` enforces a lasso penalization
        while ``alpha=0`` enforces a group lasso penalization. If it is float, it solves the model for the specific
        value. If it is an array, it solves the model for all specified values in the array.
    tau: float, defaul=0.5
        quantile level in quantile regression models. Valid values are between 0 and 1. It only has effect if
        ``model='qr'``
    lasso_weights: array, lost of arrays or None, default=None
        An array containing the values of lasso weights in adaptive penalizations.
        If it is an array, it solves the model for the specific set of weights. If it is a list of arrays, it solves
        the model for all specified arrays in the list. Each array must have length equal to the number of variables.
    gl_weights: array, list of arrays or None default=None
        An array containing the values of group lasso weights in adaptive penalizations.
        If it is an array, it solves the model for the specific set of weights. If it is a list of arrays, it solves
        the model for all specified arrays in the list. Each array must have length equal to the number of groups.
    parallel: bool, default=False
        Should te models be solved in parallel or sequentially.
    num_cores: int or None, default=None
        If ``parallel=True``, ``num_cores`` indicates the number of cores to use in the execution. If it has value
        None, it takes the value of maximum number of cores -1
    solver: str, defaul='defaul'
        Solver to be used by CVXPY. Default uses optimal alternative depending on the problem.
    """

    def __init__(self, model='lm', penalization='lasso', intercept=True, tol=1e-5, lambda1=0.1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver='default'):
        self.valid_models = ['lm', 'qr']
        self.valid_penalizations = ['lasso', 'gl', 'sgl', 'alasso', 'agl', 'asgl']
        self.model = model
        self.penalization = penalization
        self.intercept = intercept
        self.tol = tol
        self.lambda1 = lambda1
        self.alpha = alpha
        self.tau = tau
        self.lasso_weights = lasso_weights
        self.gl_weights = gl_weights
        self.parallel = parallel
        self.num_cores = num_cores
        self.coef_ = None
        # CVXPY solver parameters
        self.solver_stats = None
        self.solver = solver

    # Model checker related functions ---------------------------------------------------------------------------------

    def _model_checker(self):
        """
        Checks if the input model is one of the valid options:
         - lm for linear models
         - qr for quantile regression models
        """
        if self.model in self.valid_models:
            return True
        else:
            logging.error(f'{self.model} is not a valid model. Valid models are {self.valid_models}')
            return False

    def _penalization_checker(self):
        """
        Checks if the penalization is one of the valid options:
        """
        if (self.penalization in self.valid_penalizations) or (self.penalization is None):
            return True
        else:
            logging.error(f'{self.penalization} is not a valid penalization. '
                          f'Valid penalizations are {self.valid_penalizations} or None')
            return False

    def _dtype_checker(self):
        """
        Checks if some of the inputs are in the correct format
        """
        response_1 = False
        response_2 = False
        if isinstance(self.intercept, bool):
            response_1 = True
        if isinstance(self.tol, float):
            response_2 = True
        response = response_1 and response_2
        return response

    def _input_checker(self):
        """
        Checks that every input parameter for the model solvers has the expected format
        """
        response_list = [self._model_checker(), self._penalization_checker(), self._dtype_checker()]
        return False not in response_list

    # Preprocessing related functions ---------------------------------------------------------------------------------

    def _preprocessing_lambda(self):
        """
        Processes the input lambda1 parameter and transforms it as required by the solver package functions
        """
        n_lambda = None
        lambda_vector = None
        if self.penalization is not None:
            if isinstance(self.lambda1, (float, int)):
                lambda_vector = [self.lambda1]
            else:
                lambda_vector = self.lambda1
            n_lambda = len(lambda_vector)
        return n_lambda, lambda_vector

    def _preprocessing_alpha(self):
        """
        Processes the input alpha parameter from sgl and asgl penalizations and transforms it as required by the solver
        package functions
        """
        n_alpha = None
        alpha_vector = None
        if 'sgl' in self.penalization:
            if self.alpha is not None:
                if isinstance(self.alpha, (float, int)):
                    alpha_vector = [self.alpha]
                else:
                    alpha_vector = self.alpha
                n_alpha = len(alpha_vector)
        return n_alpha, alpha_vector

    def _preprocessing_weights(self, weights):
        """
        Converts weights into a list of lists. Each list inside weights defines a set of weights for a model
        """
        n_weights = None
        weights_list = None
        if self.penalization in ['asgl', 'alasso', 'agl']:
            if weights is not None:
                if isinstance(weights, list):
                    # If weights is a list of lists -> convert to list of arrays
                    if isinstance(weights[0], list):
                        weights_list = [np.asarray(elt) for elt in weights]
                    # If weights is a list of numbers -> store in a list
                    elif isinstance(weights[0], (float, int)):
                        weights_list = [np.asarray(weights)]
                    else:
                        # If it is a list of arrays, maintain this way
                        weights_list = weights
                # If weights is a ndarray -> store in a list and convert into list
                elif isinstance(weights, np.ndarray):
                    weights_list = [weights]
                if self.intercept:
                    weights_list = [np.insert(elt, 0, 0, axis=0) for elt in weights_list]
                n_weights = len(weights_list)
        return n_weights, weights_list

    def _preprocessing_itertools_param(self, lambda_vector, alpha_vector, lasso_weights_list, gl_weights_list):
        """
        Receives as input the results from preprocessing_lambda, preprocessing_alpha and preprocessing_weights
        Outputs an iterable list of parameter values "param"
        """
        if self.penalization in ['lasso', 'gl']:
            param = lambda_vector
        elif self.penalization == 'sgl':
            param = itertools.product(lambda_vector, alpha_vector)
        elif self.penalization == 'alasso':
            param = itertools.product(lambda_vector, lasso_weights_list)
        elif self.penalization == 'agl':
            param = itertools.product(lambda_vector, gl_weights_list)
        elif 'asgl' in self.penalization:
            param = itertools.product(lambda_vector, alpha_vector, lasso_weights_list, gl_weights_list)
        else:
            param = None
            logging.error(f'Error preprocessing input parameters')
        param = list(param)
        return param

    def _preprocessing(self):
        """
        Receives all the parameters of the models and creates tuples of the parameters to be executed in the penalized
        model solvers
        """
        # Run the input_checker to verify that the inputs have the correct format
        if self._input_checker() is False:
            logging.error('incorrect input parameters')
            raise ValueError('incorrect input parameters')
        # Defines param as None for the unpenalized model
        if self.penalization is None:
            param = None
        else:
            # Reformat parameter vectors
            n_lambda, lambda_vector = self._preprocessing_lambda()
            n_alpha, alpha_vector = self._preprocessing_alpha()
            n_lasso_weights, lasso_weights_list = self._preprocessing_weights(self.lasso_weights)
            n_gl_weights, gl_weights_list = self._preprocessing_weights(self.gl_weights)
            param = self._preprocessing_itertools_param(lambda_vector, alpha_vector, lasso_weights_list,
                                                        gl_weights_list)
        return param

    def _quantile_function(self, x):
        """
        Quantile function required for quantile regression models.
        """
        return 0.5 * cvxpy.abs(x) + (self.tau - 0.5) * x

    def _num_beta_var_from_group_index(self, group_index):
        """
        Internal function used in group based penalizations
        """
        unique_groups = np.unique(group_index)
        group_sizes = [np.sum(group_index == grp) for grp in unique_groups]
        beta_var = [cvxpy.Variable(size) for size in group_sizes]
        return group_sizes, beta_var

    # Model implementations -------------------------------------------------------------------------------------------

    def _prepare_data(self, x, y, group_index=None):
        n, m = x.shape
        if group_index is not None:
            group_index = np.asarray(group_index).astype(int)
        if self.intercept:
            m += 1
            x = np.c_[np.ones(n), x]
            if group_index is not None:
                group_index = np.append(0, group_index)
        return x, y, n, m, group_index

    def _define_objective_function(self, y, model_prediction):
        if self.model == 'lm':
            return (1.0 / y.shape[0]) * cvxpy.sum_squares(y - model_prediction)
        else:
            return (1.0 / y.shape[0]) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))

    def _solve_problem(self, problem):
        try:
            if self.solver == 'default':
                problem.solve(warm_start=True)
            else:
                problem.solve(solver=self.solver)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(f'Solver {self.solver} failed. Using alternative options from {cvxpy.installed_solvers()}')
            for alt_solver in cvxpy.installed_solvers():
                try:
                    problem.solve(solver=alt_solver)
                    if 'optimal' in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning('Optimization problem status failure')

    def _unpenalized(self, x, y):
        x, y, n, m, _ = self._prepare_data(x, y)
        beta_var = cvxpy.Variable(m)
        model_prediction = x @ beta_var
        objective_function = self._define_objective_function(y, model_prediction)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function))
        self._solve_problem(problem)
        beta_sol = beta_var.value
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return [beta_sol]

    def _lasso(self, x, y, param):
        x, y, n, m, _ = self._prepare_data(x, y)
        lambda_param = cvxpy.Parameter(nonneg=True)
        beta_var = cvxpy.Variable(m)
        model_prediction = x @ beta_var
        objective_function = self._define_objective_function(y, model_prediction)
        if self.intercept:
            lasso_penalization = lambda_param * cvxpy.norm(beta_var[1:], 1)
        else:
            lasso_penalization = lambda_param * cvxpy.norm(beta_var, 1)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + lasso_penalization))
        beta_sol_list = []
        for lam in param:
            lambda_param.value = lam
            self._solve_problem(problem)
            beta_sol = beta_var.value
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def _gl(self, x, y, group_index, param):
        x, y, n, _, group_index = self._prepare_data(x, y, group_index)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        inf_lim = 0
        if self.intercept:
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(beta_var[i], 2)
        objective_function = self._define_objective_function(y, model_prediction)
        lambda_param = cvxpy.Parameter(nonneg=True)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + lambda_param * group_lasso_penalization))
        beta_sol_list = []
        for lam in param:
            lambda_param.value = lam
            self._solve_problem(problem)
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def _sgl(self, x, y, group_index, param):
        x, y, n, _, group_index = self._prepare_data(x, y, group_index)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        lasso_penalization = 0
        group_lasso_penalization = 0
        inf_lim = 0
        if self.intercept:
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(beta_var[i], 2)
            lasso_penalization += cvxpy.norm(beta_var[i], 1)
        objective_function = self._define_objective_function(y, model_prediction)
        lasso_param = cvxpy.Parameter(nonneg=True)
        grp_lasso_param = cvxpy.Parameter(nonneg=True)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + grp_lasso_param * group_lasso_penalization + lasso_param * lasso_penalization))
        beta_sol_list = []
        for lam, al in param:
            lasso_param.value = lam * al
            grp_lasso_param.value = lam * (1 - al)
            self._solve_problem(problem)
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def _alasso(self, x, y, param):
        x, y, n, m, _ = self._prepare_data(x, y)
        l_weights_param = cvxpy.Parameter(m, nonneg=True)
        beta_var = cvxpy.Variable(m)
        model_prediction = x @ beta_var
        objective_function = self._define_objective_function(y, model_prediction)
        init_pen = 1 if self.intercept else 0
        lasso_penalization = cvxpy.norm(l_weights_param[init_pen:].T @ cvxpy.abs(beta_var[init_pen:]), 1)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + lasso_penalization))
        beta_sol_list = []
        for lam, lw in param:
            l_weights_param.value = lam * lw
            self._solve_problem(problem)
            beta_sol = beta_var.value
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def _agl(self, x, y, group_index, param):
        x, y, n, _, group_index = self._prepare_data(x, y, group_index)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        inf_lim = 0
        if self.intercept:
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        gl_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * gl_weights_param[i] * cvxpy.norm(beta_var[i], 2)
        objective_function = self._define_objective_function(y, model_prediction)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + group_lasso_penalization))
        beta_sol_list = []
        for lam, gl in param:
            gl_weights_param.value = lam * gl
            self._solve_problem(problem)
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def _asgl(self, x, y, group_index, param):
        x, y, n, m, group_index = self._prepare_data(x, y, group_index)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        lasso_penalization = 0
        group_lasso_penalization = 0
        inf_lim = 0
        if self.intercept:
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        l_weights_param = cvxpy.Parameter(m, nonneg=True)
        gl_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * gl_weights_param[i] * cvxpy.norm(beta_var[i], 2)
            lasso_penalization += l_weights_param[np.where(group_index == unique_group_index[i])[0]].T @ cvxpy.abs(beta_var[i])
        objective_function = self._define_objective_function(y, model_prediction)
        problem = cvxpy.Problem(cvxpy.Minimize(objective_function + group_lasso_penalization + lasso_penalization))
        beta_sol_list = []
        for lam, al, lw, glw in param:
            l_weights_param.value = lw * lam * al
            gl_weights_param.value = glw * lam * (1 - al)
            self._solve_problem(problem)
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    # PARALLEL CODE ---------------------------------------------------------------------------------------------------

    def _parallel_execution(self, x, y, param, group_index=None):
        """
        Parallel implementation of the solvers
        """
        if self.num_cores is None:
            # If the number of cores is not defined by user, use all available minus 1
            num_chunks = mp.cpu_count() - 1
        else:
            num_chunks = np.min((self.num_cores, mp.cpu_count()))
        # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
        tmp_group_index_chunks = np.array_split(range(len(param)), num_chunks)
        # If the number of parameters is shorter than the number of threads, delete the empty groups
        group_index_chunks = []
        for elt in tmp_group_index_chunks:
            if elt.size != 0:
                group_index_chunks.append(elt)
        # chunks is a list with as many elements as num_chunks
        # Each element of the list is another list of tuples of parameter values
        chunks = []
        for elt in group_index_chunks:
            chunks.append(param[elt[0]:(1 + elt[-1])])
        # Solve problem in parallel
        pool = mp.Pool(num_chunks)
        if self.penalization in ['lasso', 'alasso']:
            global_results = pool.map(functools.partial(getattr(self, self._get_solver_names()), x, y), chunks)
        else:
            global_results = pool.map(functools.partial(getattr(self, self._get_solver_names()), x, y, group_index), chunks)
        pool.close()
        pool.join()
        # Re-build the output of the function
        beta_sol_list = []
        if len(param) < num_chunks:
            limit = len(param)
        else:
            limit = num_chunks
        for i in range(limit):
            beta_sol_list.extend(global_results[i])
        return beta_sol_list

    # FIT METHOD ------------------------------------------------------------------------------------------------------

    def _get_solver_names(self):
        return '_' + self.penalization

    def fit(self, x, y, group_index=None):
        """
        Main function of the module. Given a model, penalization and parameter values specified in the class definition,
        this function solves the model and produces the regression coefficients
        """
        param = self._preprocessing()
        if self.penalization is None:
            self.coef_ = self._unpenalized(x=x, y=y)
        else:
            if self.parallel is False:
                if self.penalization in ['lasso', 'alasso']:
                    self.coef_ = getattr(self, self._get_solver_names())(x=x, y=y, param=param)
                else:
                    self.coef_ = getattr(self, self._get_solver_names())(x=x, y=y, param=param,
                                                                         group_index=group_index)
            else:
                self.coef_ = self._parallel_execution(x=x, y=y, param=param, group_index=group_index)

    # PREDICTION METHOD -----------------------------------------------------------------------------------------------

    def predict(self, x_new):
        """
        To be executed after fitting a model. Given a new dataset, this function produces predictions for that data
        considering the different model coefficients output provided by function fit
        """
        if self.intercept:
            x_new = np.c_[np.ones(x_new.shape[0]), x_new]
        if x_new.shape[1] != len(self.coef_[0]):
            logging.error('Model dimension and new data dimension does not match')
            raise ValueError('Model dimension and new data dimension does not match')
        # Store predictions in a list
        prediction_list = []
        for elt in self.coef_:
            prediction_list.append(np.dot(x_new, elt))
        return prediction_list

    # NUMBER OF PARAMETERS --------------------------------------------------------------------------------------------

    def _num_parameters(self):
        """
        retrieves the number of parameters to be considered in a model
        Output: tuple [num_models, n_lambda, n_alpha, n_l_weights, n_gl_weights] where
        - num_models: total number of models to be solved for the grid of parameters given
        - n_lambda: number of different lambda1 values
        - n_alpha: number of different alpha values
        - n_l_weights: number of different weights for the lasso part of the adaptive penalizations
        - n_gl_weights: number of different weights for the group lasso part of the adaptive penalizations
        """
        # Run the input_checker to verify that the inputs have the correct format
        if self._input_checker() is False:
            logging.error('incorrect input parameters')
            raise ValueError('incorrect input parameters')
        if self.penalization is None:
            # See meaning of each element in the "else" result statement.
            result = dict(num_models=1,
                          n_lambda=None,
                          n_alpha=None,
                          n_lasso_weights=None,
                          n_gl_weights=None)
        else:
            n_lambda, drop = self._preprocessing_lambda()
            n_alpha, drop = self._preprocessing_alpha()
            n_lasso_weights, drop = self._preprocessing_weights(self.lasso_weights)
            n_gl_weights, drop = self._preprocessing_weights(self.gl_weights)
            list_param = [n_lambda, n_alpha, n_lasso_weights, n_gl_weights]
            list_param_no_none = [elt for elt in list_param if elt is not None]
            num_models = np.prod(list_param_no_none)
            result = dict(num_models=num_models,
                          n_lambda=n_lambda,
                          n_alpha=n_alpha,
                          n_lasso_weights=n_lasso_weights,
                          n_gl_weights=n_gl_weights)
        return result

    def _retrieve_parameters_idx(self, param_index):
        """
        Given an index for the param iterable output from _preprocessing function, this function returns a tupple
        with the index of the value of each parameter.
        Example: Solving an adaptive sparse group lasso model with 5 values for lambda1, 4 values for alpha,
                 3 possible lasso weights and 3 possible group lasso weights yields in a grid search on
                 5*4*3*3=180 parameters.
                 Inputing param_index=120 (out of the 180 possible values)in this function will output the
                 lambda, alpha, and weights index for such value
        If the penalization under consideration does not include any of the required parameters (for example, if we are
        solving a sparse group lasso model, we do not consider adaptive weights), the output regarding the non-used
        parameters are set to be None.
        """
        number_parameters = self._num_parameters()
        n_models, n_lambda, n_alpha, n_l_weights, n_gl_weights = [number_parameters[elt] for elt in number_parameters]
        if param_index > n_models:
            string = f'param_index should be smaller or equal than the number of models solved. n_models={n_models}, ' \
                     f'param_index={param_index}'
            logging.error(string)
            raise ValueError(string)
        # If penalization is None, all parameters are set to None
        if self.penalization is None:
            result = [None, None, None, None]
        # If penalization is lasso or gl, there is only one parameter, so param_index = position of that parameter
        elif self.penalization in ['lasso', 'gl']:
            result = [param_index, None, None, None]
        # If penalization is sgl, there are two parameters and two None
        elif self.penalization == 'sgl':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_alpha))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], parameter_idx[1][0], None, None]
        elif self.penalization == 'alasso':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_l_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], None, parameter_idx[1][0], None]
        elif self.penalization == 'agl':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_gl_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], None, None, parameter_idx[1][0]]
        else:
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_alpha, n_l_weights, n_gl_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], parameter_idx[1][0],
                      parameter_idx[2][0], parameter_idx[3][0]]
        return result

    def retrieve_parameters_value(self, param_index):
        """
        Converts the index output from retrieve_parameters_idx into the actual numerical value of the parameters.
        Outputs None if the parameter was not used in the model formulation.
        To be executed after fit method.
        """
        param_index = self._retrieve_parameters_idx(param_index)
        result = [param[idx] if idx is not None else None for idx, param in
                  zip(param_index, [self.lambda1, self.alpha, self.lasso_weights, self.gl_weights])]
        result = dict(lambda1=result[0],
                      alpha=result[1],
                      lasso_weights=result[2],
                      gl_weights=result[3])
        return result


# ERROR CALCULATOR METHOD ---------------------------------------------------------------------------------------------

def _quantile_function(y_true, y_pred, tau):
    """
    Quantile function required for error computation
    """
    return (1.0 / len(y_true)) * np.sum(0.5 * np.abs(y_true - y_pred) + (tau - 0.5) * (y_true - y_pred))


def error_calculator(y_true, prediction_list, error_type="MSE", tau=None):
    """
    Computes the error between the predicted value and the true value of the response variable
    """
    error_dict = dict(
        MSE=mean_squared_error,
        MAE=mean_absolute_error,
        MDAE=median_absolute_error,
        QRE=_quantile_function)
    valid_error_types = error_dict.keys()
    # Check that the error_type is a valid error type considered
    if error_type not in valid_error_types:
        raise ValueError(f'invalid error type. Valid error types are {error_dict.keys()}')
    if y_true.shape[0] != len(prediction_list[0]):
        logging.error('Dimension of test data does not match dimension of prediction')
        raise ValueError('Dimension of test data does not match dimension of prediction')
    # For each prediction, store the error associated to that prediction in a list
    error_list = []
    if error_type == 'QRE':
        for y_pred in prediction_list:
            error_list.append(error_dict[error_type](y_true=y_true, y_pred=y_pred, tau=tau))
    else:
        for y_pred in prediction_list:
            error_list.append(error_dict[error_type](y_true=y_true, y_pred=y_pred))
    return error_list
